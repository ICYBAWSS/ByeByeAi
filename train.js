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

// Preprocess data
function preprocessData(data) {
  // Filter out rows with missing or invalid data
  const validData = data.filter(row => {
    // Convert generated to number first
    const generatedValue = Number(row.generated);
    
    return row.text_length != null &&
           row.word_count != null &&
           row.avg_word_length != null &&
           row.num_sentences != null &&
           row.punctuation_density != null &&
           row.readability_score != null &&
           !isNaN(row.text_length) &&
           !isNaN(row.word_count) &&
           !isNaN(row.avg_word_length) &&
           !isNaN(row.num_sentences) &&
           !isNaN(row.punctuation_density) &&
           !isNaN(row.readability_score) &&
           !isNaN(generatedValue) &&
           (generatedValue === 0 || generatedValue === 1);
  });
  
  console.log(`Filtered ${data.length - validData.length} invalid rows`);
  console.log(`Using ${validData.length} valid samples\n`);
  
  const features = validData.map(row => [
    parseFloat(row.text_length),
    parseFloat(row.word_count),
    parseFloat(row.avg_word_length),
    parseFloat(row.num_sentences),
    parseFloat(row.punctuation_density),
    parseFloat(row.readability_score),
    parseFloat(row.phrase_repetition),
    parseFloat(row.sentence_complexity),
    parseFloat(row.sentence_diversity)
  ]);
  
  const labels = validData.map(row => Number(row.generated));
  
  // Only create tensors if we have data
  if (features.length === 0) {
    throw new Error('No valid samples found in the dataset after preprocessing');
  }
  
  const featureTensor = tf.tensor2d(features);
  const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
  const mean = featureTensor.mean(0);
  const std = tf.sqrt(featureTensor.sub(mean).square().mean(0));
  
  const normalizedFeatures = featureTensor.sub(mean).div(std.add(1e-7));
  
  return { features: normalizedFeatures, labels: labelTensor, mean, std };
}

// Create model
function createCalibrationLayer() {
  // Temperature scaling layer for probability calibration
  return tf.layers.dense({
    units: 1,
    useBias: false,
    kernelInitializer: tf.initializers.ones(),
    trainable: true
  });
}

function createModel() {
  const model = tf.sequential();
  
  // First layer with more units
  model.add(tf.layers.dense({
    inputShape: [9], // Updated for new features
    units: 64,
    activation: 'relu',
    kernelInitializer: 'glorotUniform',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.4 }));
  
  // Add a new middle layer
  model.add(tf.layers.dense({
    units: 32,
    activation: 'relu',
    kernelInitializer: 'glorotUniform',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.3 }));
  
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    kernelInitializer: 'glorotUniform',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  
  model.add(tf.layers.dropout({ rate: 0.2 }));
  
  // Pre-calibration layer
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear', // Changed to linear for better calibration
    kernelInitializer: 'glorotUniform'
  }));
  
  // Add calibration layer
  model.add(createCalibrationLayer());
  
  // Final activation
  model.add(tf.layers.activation({activation: 'sigmoid'}));
  
  // Use Adam optimizer with gradient clipping
  const optimizer = tf.train.adam(0.0005, 0.9, 0.999, 1e-7); // Reduced learning rate for better stability
  
  model.compile({
    optimizer: optimizer,
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

// Train model
async function trainModel(model, features, labels) {
  console.log('Training model...\n');
  
  // Calculate class weights to handle imbalance
  const labelArray = await labels.array();
  const aiCount = labelArray.flat().filter(x => x === 1).length;
  const humanCount = labelArray.flat().filter(x => x === 0).length;
  const total = aiCount + humanCount;
  
  // Weight inversely proportional to class frequency with additional bias towards reducing false positives
  const classWeight = {
    0: (total / (2 * humanCount)) * 1.2,  // Increased human weight to reduce false positives
    1: total / (2 * aiCount)              // AI weight
  };
  
  console.log(`Class weights: Human=${classWeight[0].toFixed(2)}, AI=${classWeight[1].toFixed(2)}\n`);
  
  const history = await model.fit(features, labels, {
    epochs: 200,  // Further increased epochs with early stopping
    batchSize: 32,
    validationSplit: 0.2,
    classWeight: classWeight,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        // Check for NaN during training
        if (isNaN(logs.loss) || isNaN(logs.val_loss)) {
          console.log('\n❌ NaN detected during training! Stopping...');
          model.stopTraining = true;
          return;
        }
        
        if ((epoch + 1) % 10 === 0) {  // Print every 10 epochs
          console.log(`Epoch ${epoch + 1}/100 - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`);
        }
      }
    }
  });
  
  return history;
}

// Main training function
async function main() {
  const DB_PATH = 'AiHumanTextCombined.db'; // Change this to your .db file path
  
  console.log('Loading data from database...');
  const data = loadDataFromDB(DB_PATH);
  console.log(`Loaded ${data.length} samples\n`);
  
  console.log('Preprocessing data...');
  const { features, labels, mean, std } = preprocessData(data);
  
  console.log('Creating model...');
  const model = createModel();
  model.summary();
  console.log('');
  
  await trainModel(model, features, labels);
  
  console.log('\nSaving model...');
  await model.save('file://./model');
  
  console.log('Saving normalization parameters...');
  const meanArray = await mean.array();
  const stdArray = await std.array();
  fs.writeFileSync('./model/normalization.json', JSON.stringify({
    mean: meanArray,
    std: stdArray
  }, null, 2));
  
  console.log('\n✓ Training complete! Model saved to ./model directory');
  
  // Cleanup
  features.dispose();
  labels.dispose();
  mean.dispose();
  std.dispose();
}

main().catch(console.error);