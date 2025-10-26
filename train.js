// train.js - Run this once to train and save the model
import * as tf from '@tensorflow/tfjs-node';
import Database from 'better-sqlite3';
import fs from 'fs';

// Load data from SQLite database
function loadDataFromDB(dbPath) {
  const db = new Database(dbPath);
  const rows = db.prepare('SELECT * FROM data').all();
  db.close();
  return rows;
}

// Apply log1p transform to length features and prepare features for scaling
function transformFeatures(row) {
  // Apply log1p to length-based features
  const text_length = Math.log1p(parseFloat(row.text_length));
  const word_count = Math.log1p(parseFloat(row.word_count));
  const num_sentences = Math.log1p(parseFloat(row.num_sentences));
  
  return [
    text_length,
    word_count,
    parseFloat(row.avg_word_length),
    num_sentences,
    parseFloat(row.punctuation_density),
    parseFloat(row.readability_score),
    parseFloat(row.phrase_repetition || 0),
    parseFloat(row.sentence_complexity || 0),
    parseFloat(row.sentence_diversity || 0)
  ];
}

// Preprocess data with log-transformed features and scaling
async function preprocessData(data) {
  // Filter out rows with missing or invalid data
  const validData = [];
  const featureArray = [];
  
  // First pass: transform features and collect valid data
  for (const row of data) {
    try {
      const generatedValue = Number(row.generated);
      if (isNaN(generatedValue) || (generatedValue !== 0 && generatedValue !== 1)) {
        continue;
      }
      
      const features = transformFeatures(row);
      if (features.every(val => !isNaN(val))) {
        validData.push({
          features,
          label: generatedValue
        });
        featureArray.push(features);
      }
    } catch (e) {
      console.warn('Error processing row:', e);
    }
  }
  
  console.log(`Filtered ${data.length - validData.length} invalid rows`);
  console.log(`Using ${validData.length} valid samples\n`);
  
  if (validData.length === 0) {
    throw new Error('No valid samples found in the dataset after preprocessing');
  }
  
  // Convert to tensors
  const featureTensor = tf.tensor2d(featureArray);
  
  // Calculate mean and std for z-score normalization
  const mean = featureTensor.mean(0);
  const std = tf.sqrt(featureTensor.sub(mean).square().mean(0));
  
  // Apply z-score normalization with clipping to prevent extreme values
  const normalizedFeatures = featureTensor.sub(mean).div(std.add(1e-7));
  
  // Clip values to reasonable range to prevent numerical instability
  const clippedFeatures = normalizedFeatures.clipByValue(-10, 10);
  
  // Get the normalized features as an array
  const normalizedArray = await clippedFeatures.array();
  
  // Clean up intermediate tensors
  featureTensor.dispose();
  normalizedFeatures.dispose();
  clippedFeatures.dispose();
  
  // Prepare final dataset
  const features = normalizedArray;
  const labels = validData.map(row => row.label);
  
  // Log feature statistics
  console.log('Feature Statistics (after normalization):');
  const featureNames = [
    'text_length_log', 'word_count_log', 'avg_word_length',
    'num_sentences_log', 'punctuation_density', 'readability_score',
    'phrase_repetition', 'sentence_complexity', 'sentence_diversity'
  ];
  
  const means = await mean.array();
  const stds = await std.array();
  
  // Log statistics
  featureNames.forEach((name, i) => {
    console.log(`${name}: mean=${means[i].toFixed(4)}, std=${stds[i].toFixed(4)}`);
  });
  console.log('');
  
  // Return both the normalized features and the normalization parameters
  return {
    features: tf.tensor2d(features),
    labels: tf.tensor2d(labels, [labels.length, 1]),
    mean,
    std,
    featureNames
  };
}

// Create and compile the model
function createModel() {
  const model = tf.sequential();
  
  // Input layer with lighter L2 regularization
  model.add(tf.layers.dense({
    inputShape: [9],
    units: 64,
    activation: 'relu',
    kernelInitializer: 'glorotUniform',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
    biasRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.3 }));
  
  // Hidden layers with decreasing units
  [32, 16].forEach(units => {
    model.add(tf.layers.dense({
      units,
      activation: 'relu',
      kernelInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }),
      biasRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
  });
  
  // Output layer with sigmoid activation for binary classification
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
    kernelInitializer: 'glorotNormal'
  }));
  
  // Use Adam optimizer with lower learning rate
  const optimizer = tf.train.adam(0.0001);
  
  model.compile({
    optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

// Train model with early stopping and learning rate scheduling
async function trainModel(model, features, labels) {
  console.log('Training model...\n');
  
  // Calculate class weights to handle imbalance
  const labelArray = await labels.array();
  const aiCount = labelArray.flat().filter(x => x === 1).length;
  const humanCount = labelArray.flat().filter(x => x === 0).length;
  const total = aiCount + humanCount;
  
  // More balanced class weights
  const classWeight = {
    0: (total / (2 * humanCount)) * 1.1,
    1: total / (2 * aiCount)
  };
  
  console.log(`Class distribution: ${humanCount} human, ${aiCount} AI`);
  console.log(`Class weights: Human=${classWeight[0].toFixed(2)}, AI=${classWeight[1].toFixed(2)}\n`);
  
  // Track best validation loss for early stopping
  let bestValLoss = Infinity;
  let patienceCount = 0;
  const patience = 15;
  
  // Learning rate scheduling variables
  let currentLr = 0.0001;
  const minLr = 1e-7;
  let reduceLrPatience = 0;
  const reduceLrPatienceMax = 5;
  
  // Training parameters
  const batchSize = 32;
  const valSplit = 0.2;
  const numEpochs = 100;
  
  // Split data into training and validation sets
  const splitIndex = Math.floor(features.shape[0] * (1 - valSplit));
  const trainFeatures = features.slice(0, splitIndex);
  const trainLabels = labels.slice(0, splitIndex);
  const valFeatures = features.slice(splitIndex);
  const valLabels = labels.slice(splitIndex);
  
  const numBatches = Math.ceil(trainFeatures.shape[0] / batchSize);
  
  const history = {
    loss: [],
    val_loss: [],
    acc: [],
    val_acc: []
  };
  
  for (let epoch = 0; epoch < numEpochs; epoch++) {
    // Shuffle the training data
    const indices = tf.util.createShuffledIndices(trainFeatures.shape[0]);
    const indicesTensor = tf.tensor1d(Array.from(indices), 'int32');
    const shuffledFeatures = trainFeatures.gather(indicesTensor);
    const shuffledLabels = trainLabels.gather(indicesTensor);
    
    // Clean up the indices tensor
    indicesTensor.dispose();
    
    let epochLoss = 0;
    let epochAcc = 0;
    
    // Training loop
    for (let i = 0; i < numBatches; i++) {
      const startIdx = i * batchSize;
      const batchSize_actual = Math.min(batchSize, trainFeatures.shape[0] - startIdx);
      
      const batchX = shuffledFeatures.slice(startIdx, batchSize_actual);
      const batchY = shuffledLabels.slice(startIdx, batchSize_actual);
      
      // trainOnBatch can return tensors or numbers depending on tfjs version
      const result = await model.trainOnBatch(batchX, batchY);
      
      // Handle return value - check if it's a tensor or plain value
      if (Array.isArray(result)) {
        // Array of values (could be tensors or numbers)
        if (result[0] && typeof result[0].data === 'function') {
          // Tensors
          epochLoss += (await result[0].data())[0];
          if (result.length > 1) {
            epochAcc += (await result[1].data())[0];
          }
          result.forEach(t => t.dispose());
        } else {
          // Plain numbers
          epochLoss += result[0];
          if (result.length > 1) {
            epochAcc += result[1];
          }
        }
      } else {
        // Single value (could be tensor or number)
        if (result && typeof result.data === 'function') {
          // Tensor
          epochLoss += (await result.data())[0];
          result.dispose();
        } else {
          // Plain number
          epochLoss += result;
        }
      }
      
      // Clean up batch tensors
      tf.dispose([batchX, batchY]);
    }
    
    // Calculate validation metrics
    const valResults = model.evaluate(valFeatures, valLabels, {batchSize: 32});
    
    let valLossValue, valAccValue;
    if (Array.isArray(valResults)) {
      valLossValue = (await valResults[0].data())[0];
      valAccValue = valResults.length > 1 ? (await valResults[1].data())[0] : 0;
      // Dispose the result tensors
      valResults.forEach(t => t.dispose());
    } else {
      valLossValue = (await valResults.data())[0];
      valAccValue = 0;
      valResults.dispose();
    }
    
    // Update history
    const avgLoss = epochLoss / numBatches;
    const avgAcc = epochAcc / numBatches;
    
    history.loss.push(avgLoss);
    history.acc.push(avgAcc);
    history.val_loss.push(valLossValue);
    history.val_acc.push(valAccValue);
    
    // Check for NaN values in both loss and weights
    if (isNaN(avgLoss) || isNaN(valLossValue)) {
      console.log('\n❌ NaN detected during training! Stopping...');
      console.log(`Epoch ${epoch + 1}: loss=${avgLoss}, val_loss=${valLossValue}`);
      tf.dispose([shuffledFeatures, shuffledLabels]);
      break;
    }
    
    // Early stopping and learning rate adjustment
    if (valLossValue < bestValLoss) {
      bestValLoss = valLossValue;
      patienceCount = 0;
      reduceLrPatience = 0;
    } else {
      patienceCount++;
      reduceLrPatience++;
      
      // Reduce learning rate on plateau
      if (reduceLrPatience >= reduceLrPatienceMax && currentLr > minLr) {
        const oldLr = currentLr;
        currentLr = Math.max(minLr, currentLr * 0.5);
        console.log(`\nReducing learning rate from ${oldLr.toExponential(2)} to ${currentLr.toExponential(2)}`);
        
        // Create new optimizer with updated learning rate
        const newOptimizer = tf.train.adam(currentLr);
        model.compile({
          optimizer: newOptimizer,
          loss: 'binaryCrossentropy',
          metrics: ['accuracy']
        });
        reduceLrPatience = 0;
      }
      
      // Early stopping
      if (patienceCount >= patience) {
        console.log(`\nEarly stopping after ${patience} epochs without improvement`);
        tf.dispose([shuffledFeatures, shuffledLabels]);
        break;
      }
    }
    
    // Log progress every 5 epochs or on the last epoch
    if ((epoch + 1) % 5 === 0 || epoch === numEpochs - 1) {
      console.log(`Epoch ${epoch + 1}/${numEpochs} - ` +
                 `loss: ${avgLoss.toFixed(4)}, acc: ${avgAcc.toFixed(4)}, ` +
                 `val_loss: ${valLossValue.toFixed(4)}, val_acc: ${valAccValue.toFixed(4)}, ` +
                 `lr: ${currentLr.toExponential(2)}`);
    }
    
    // Clean up
    tf.dispose([shuffledFeatures, shuffledLabels]);
  }
  
  // Clean up
  tf.dispose([trainFeatures, trainLabels, valFeatures, valLabels]);
  
  return history;
}

// Save normalization parameters
function saveNormalization(mean, std, featureNames, outputPath = './model') {
  if (!fs.existsSync(outputPath)) {
    fs.mkdirSync(outputPath, { recursive: true });
  }
  
  const normData = {
    mean: Array.from(mean.dataSync()),
    std: Array.from(std.dataSync()),
    featureNames,
    timestamp: new Date().toISOString()
  };
  
  fs.writeFileSync(
    `${outputPath}/normalization.json`,
    JSON.stringify(normData, null, 2)
  );
}

// Main training function
async function main() {
  const DB_PATH = 'AiHumanTextCombined.db';
  const MODEL_DIR = './model';
  
  try {
    console.log('Loading data from database...');
    const data = loadDataFromDB(DB_PATH);
    console.log(`Loaded ${data.length} samples\n`);
    
    console.log('Preprocessing data...');
    const { features, labels, mean, std, featureNames } = await preprocessData(data);
    
    console.log('Creating model...');
    const model = createModel();
    model.summary();
    console.log('');
    
    console.log('Starting training...');
    await trainModel(model, features, labels);
    
    console.log('\nSaving model and normalization parameters...');
    if (!fs.existsSync(MODEL_DIR)) {
      fs.mkdirSync(MODEL_DIR, { recursive: true });
    }
    
    await model.save(`file://${MODEL_DIR}`);
    saveNormalization(mean, std, featureNames, MODEL_DIR);
    
    console.log('\n✅ Training completed successfully!');
    console.log(`Model and normalization data saved to ${MODEL_DIR}`);
    
    // Cleanup
    features.dispose();
    labels.dispose();
    mean.dispose();
    std.dispose();
    
  } catch (error) {
    console.error('\n❌ Error during training:', error);
    process.exit(1);
  }
}

main().catch(console.error);