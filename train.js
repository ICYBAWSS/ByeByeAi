// train-academic.js - Training with research-based features
import * as tf from '@tensorflow/tfjs-node';
import Database from 'better-sqlite3';
import fs from 'fs';

// [Include all the feature extraction functions from previous artifact]
// calculatePerplexity, calculateBurstiness, calculateEntropy, etc.

function calculatePerplexity(text) {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (words.length < 2) return { perplexity: 0, normalizedPerplexity: 0 };
  
  const bigramFreq = {};
  const unigramFreq = {};
  
  for (let i = 0; i < words.length - 1; i++) {
    const bigram = `${words[i]} ${words[i + 1]}`;
    const unigram = words[i];
    bigramFreq[bigram] = (bigramFreq[bigram] || 0) + 1;
    unigramFreq[unigram] = (unigramFreq[unigram] || 0) + 1;
  }
  
  let logProb = 0;
  for (let i = 0; i < words.length - 1; i++) {
    const bigram = `${words[i]} ${words[i + 1]}`;
    const unigram = words[i];
    const bigramCount = bigramFreq[bigram] || 0;
    const unigramCount = unigramFreq[unigram] || 1;
    const prob = (bigramCount + 1) / (unigramCount + Object.keys(unigramFreq).length);
    logProb += Math.log2(prob);
  }
  
  const perplexity = Math.pow(2, -logProb / (words.length - 1));
  return { perplexity, normalizedPerplexity: perplexity / words.length };
}

function calculateBurstiness(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  if (sentences.length < 2) return { burstiness: 0, sentenceLengthVariance: 0 };
  
  const sentLengths = sentences.map(s => s.trim().split(/\s+/).length);
  const mean = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
  const variance = sentLengths.reduce((sum, len) => sum + Math.pow(len - mean, 2), 0) / sentLengths.length;
  const stdDev = Math.sqrt(variance);
  const burstiness = (stdDev - mean) / (stdDev + mean);
  
  return { 
    burstiness: Math.max(-1, Math.min(1, burstiness)),
    sentenceLengthVariance: variance 
  };
}

function calculateEntropy(text) {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (words.length === 0) return { entropy: 0, normalizedEntropy: 0 };
  
  const freq = {};
  words.forEach(word => freq[word] = (freq[word] || 0) + 1);
  
  let entropy = 0;
  Object.values(freq).forEach(count => {
    const prob = count / words.length;
    entropy -= prob * Math.log2(prob);
  });
  
  const maxEntropy = Math.log2(Object.keys(freq).length);
  const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;
  
  return { entropy, normalizedEntropy };
}

function calculateLexicalDiversity(text) {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (words.length === 0) return { ttr: 0, mtld: 0, vocdD: 0 };
  
  const uniqueWords = new Set(words);
  const ttr = uniqueWords.size / words.length;
  
  let mtld = 0;
  const windowSize = 50;
  if (words.length >= windowSize) {
    for (let i = 0; i <= words.length - windowSize; i++) {
      const window = words.slice(i, i + windowSize);
      const uniqueInWindow = new Set(window);
      mtld += uniqueInWindow.size / windowSize;
    }
    mtld /= (words.length - windowSize + 1);
  } else {
    mtld = ttr;
  }
  
  const vocdD = uniqueWords.size / Math.sqrt(words.length);
  
  return { ttr, mtld, vocdD };
}

function analyzeNGrams(text, n = 3) {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (words.length < n) return { ngramRepetition: 0 };
  
  const ngrams = {};
  for (let i = 0; i <= words.length - n; i++) {
    const ngram = words.slice(i, i + n).join(' ');
    ngrams[ngram] = (ngrams[ngram] || 0) + 1;
  }
  
  const frequencies = Object.values(ngrams);
  const repeated = frequencies.filter(f => f > 1).length;
  const ngramRepetition = repeated / Object.keys(ngrams).length;
  
  return { ngramRepetition };
}

function extractStylometricFeatures(text) {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  const functionWords = new Set([
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
  ]);
  
  const functionWordCount = words.filter(w => 
    functionWords.has(w.toLowerCase())
  ).length;
  const functionWordRatio = functionWordCount / words.length;
  
  const punctuationMarks = text.match(/[.,;:!?]/g) || [];
  const punctuationDiversity = new Set(punctuationMarks).size / Math.max(1, punctuationMarks.length);
  
  const starters = sentences.map(s => s.trim().split(/\s+/)[0]?.toLowerCase());
  const uniqueStarters = new Set(starters.filter(Boolean));
  const starterDiversity = uniqueStarters.size / Math.max(1, starters.length);
  
  const contractions = text.match(/\b\w+'\w+\b/g) || [];
  const contractionRatio = contractions.length / words.length;
  
  const avgWordLength = words.reduce((sum, w) => sum + w.length, 0) / words.length;
  
  return {
    functionWordRatio,
    punctuationDiversity,
    starterDiversity,
    contractionRatio,
    avgWordLength
  };
}

function calculateCoherence(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  if (sentences.length < 2) return { coherenceScore: 0, transitionWordRatio: 0 };
  
  const transitions = [
    'however', 'therefore', 'moreover', 'furthermore', 'consequently',
    'meanwhile', 'nevertheless', 'thus', 'hence', 'additionally'
  ];
  
  let transitionCount = 0;
  const lowerText = text.toLowerCase();
  transitions.forEach(trans => {
    const regex = new RegExp(`\\b${trans}\\b`, 'gi');
    const matches = lowerText.match(regex);
    if (matches) transitionCount += matches.length;
  });
  
  const transitionWordRatio = transitionCount / sentences.length;
  
  let overlapSum = 0;
  for (let i = 0; i < sentences.length - 1; i++) {
    const words1 = new Set(sentences[i].toLowerCase().split(/\s+/));
    const words2 = new Set(sentences[i + 1].toLowerCase().split(/\s+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const overlap = intersection.size / Math.max(words1.size, words2.size);
    overlapSum += overlap;
  }
  
  const coherenceScore = overlapSum / (sentences.length - 1);
  
  return { coherenceScore, transitionWordRatio };
}

function extractAcademicFeatures(text) {
  const { perplexity, normalizedPerplexity } = calculatePerplexity(text);
  const { burstiness, sentenceLengthVariance } = calculateBurstiness(text);
  const { entropy, normalizedEntropy } = calculateEntropy(text);
  const { ttr, mtld, vocdD } = calculateLexicalDiversity(text);
  const bigrams = analyzeNGrams(text, 2);
  const trigrams = analyzeNGrams(text, 3);
  const stylometric = extractStylometricFeatures(text);
  const coherence = calculateCoherence(text);
  
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  return [
    normalizedPerplexity,
    burstiness,
    normalizedEntropy,
    mtld,
    ttr,
    vocdD,
    bigrams.ngramRepetition,
    trigrams.ngramRepetition,
    stylometric.functionWordRatio,
    stylometric.punctuationDiversity,
    stylometric.starterDiversity,
    stylometric.contractionRatio,
    stylometric.avgWordLength,
    coherence.coherenceScore,
    coherence.transitionWordRatio,
    Math.log1p(text.length),
    Math.log1p(words.length),
    Math.log1p(sentences.length),
    sentenceLengthVariance
  ];
}

function loadDataFromDB(dbPath) {
  const db = new Database(dbPath);
  const rows = db.prepare('SELECT * FROM data').all();
  db.close();
  return rows;
}

async function preprocessData(data) {
  console.log('Extracting research-based linguistic features...\n');
  
  const validData = [];
  const featureArray = [];
  
  for (const row of data) {
    try {
      const generatedValue = Number(row.generated);
      if (isNaN(generatedValue) || (generatedValue !== 0 && generatedValue !== 1)) continue;
      
      const features = extractAcademicFeatures(row.text || '');
      if (features.every(val => !isNaN(val) && isFinite(val))) {
        validData.push({ features, label: generatedValue });
        featureArray.push(features);
      }
    } catch (e) {
      console.warn('Error processing row:', e.message);
    }
  }
  
  console.log(`Valid samples: ${validData.length}/${data.length}\n`);
  
  const featureTensor = tf.tensor2d(featureArray);
  const mean = featureTensor.mean(0);
  const std = tf.sqrt(featureTensor.sub(mean).square().mean(0));
  const normalized = featureTensor.sub(mean).div(std.add(1e-7)).clipByValue(-10, 10);
  
  const features = await normalized.array();
  const labels = validData.map(row => row.label);
  
  featureTensor.dispose();
  normalized.dispose();
  
  return {
    features: tf.tensor2d(features),
    labels: tf.tensor2d(labels, [labels.length, 1]),
    mean,
    std,
    featureNames: [
      'perplexity', 'burstiness', 'entropy', 'lexicalDiversity',
      'typeTokenRatio', 'vocabularyRichness', 'bigramRepetition',
      'trigramRepetition', 'functionWordRatio', 'punctuationDiversity',
      'starterDiversity', 'contractionRatio', 'avgWordLength',
      'coherenceScore', 'transitionWordRatio', 'textLength',
      'wordCount', 'sentenceCount', 'sentenceLengthVariance'
    ]
  };
}

function createModel() {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({
    inputShape: [19],
    units: 64,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  
  model.add(tf.layers.dense({
    units: 32,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
  }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));
  
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

async function trainModel(model, features, labels) {
  console.log('Training model with academic features...\n');
  
  const history = await model.fit(features, labels, {
    epochs: 50,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 5 === 0) {
          console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, ` +
                     `acc=${logs.acc.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}, ` +
                     `val_acc=${logs.val_acc.toFixed(4)}`);
        }
      }
    }
  });
  
  return history;
}

async function main() {
  const DB_PATH = 'AiHumanTextCombined.db';
  const MODEL_DIR = './model-academic';
  
  console.log('═══════════════════════════════════════════════════');
  console.log('  AI Text Detector - Academic Implementation');
  console.log('  Based on: ACL 2025 GenAI Detection Research');
  console.log('═══════════════════════════════════════════════════\n');
  
  const data = loadDataFromDB(DB_PATH);
  console.log(`Loaded ${data.length} samples\n`);
  
  const { features, labels, mean, std, featureNames } = await preprocessData(data);
  
  console.log('Creating model...');
  const model = createModel();
  model.summary();
  
  await trainModel(model, features, labels);
  
  console.log('\nSaving model...');
  if (!fs.existsSync(MODEL_DIR)) {
    fs.mkdirSync(MODEL_DIR, { recursive: true });
  }
  
  await model.save(`file://${MODEL_DIR}`);
  
  const normData = {
    mean: Array.from(mean.dataSync()),
    std: Array.from(std.dataSync()),
    featureNames
  };
  fs.writeFileSync(`${MODEL_DIR}/normalization.json`, JSON.stringify(normData, null, 2));
  
  console.log(`\n✅ Model saved to ${MODEL_DIR}`);
  console.log('\nFeatures used:');
  featureNames.forEach((name, i) => console.log(`  ${i + 1}. ${name}`));
  
  features.dispose();
  labels.dispose();
  mean.dispose();
  std.dispose();
}

main().catch(console.error);