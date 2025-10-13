// diagnose.js - Check your database and model
import * as tf from '@tensorflow/tfjs-node';
import Database from 'better-sqlite3';
import fs from 'fs';

const CONFIDENCE_THRESHOLD = 0.75; // Increased confidence threshold to reduce false positives

const DB_PATH = 'AiHumanTextCombined.db'; // Update this to your .db file path
const TABLE_NAME = 'data'; // Update this to your table name

function analyzeDatabase() {
  const db = new Database(DB_PATH);
  
  // Get sample rows
  console.log('═══════════════════════════════════════'); 
  console.log('DATABASE ANALYSIS');
  console.log('═══════════════════════════════════════\n');
  
  const totalRows = db.prepare(`SELECT COUNT(*) as count FROM ${TABLE_NAME}`).get();
  console.log(`Total rows: ${totalRows.count}\n`);
  
  // Check AI vs Human distribution
  const aiCount = db.prepare(`SELECT COUNT(*) as count FROM ${TABLE_NAME} WHERE generated = 1`).get();
  const humanCount = db.prepare(`SELECT COUNT(*) as count FROM ${TABLE_NAME} WHERE generated = 0`).get();
  console.log('Label Distribution:');
  console.log(`AI-generated: ${aiCount.count}`);
  console.log(`Human-written: ${humanCount.count}\n`);
  
  // Get statistics for each feature
  const stats = db.prepare(`
    SELECT 
      AVG(text_length) as avg_text_length,
      MIN(text_length) as min_text_length,
      MAX(text_length) as max_text_length,
      AVG(word_count) as avg_word_count,
      MIN(word_count) as min_word_count,
      MAX(word_count) as max_word_count,
      AVG(avg_word_length) as avg_avg_word_length,
      MIN(avg_word_length) as min_avg_word_length,
      MAX(avg_word_length) as max_avg_word_length,
      AVG(num_sentences) as avg_num_sentences,
      MIN(num_sentences) as min_num_sentences,
      MAX(num_sentences) as max_num_sentences,
      AVG(punctuation_density) as avg_punctuation_density,
      MIN(punctuation_density) as min_punctuation_density,
      MAX(punctuation_density) as max_punctuation_density,
      AVG(readability_score) as avg_readability_score,
      MIN(readability_score) as min_readability_score,
      MAX(readability_score) as max_readability_score
    FROM ${TABLE_NAME}
  `).get();
  
  console.log('Feature Statistics:');
  console.log('─'.repeat(60));
  console.log('text_length:');
  console.log(`  Range: ${stats.min_text_length} - ${stats.max_text_length}`);
  console.log(`  Average: ${stats.avg_text_length.toFixed(2)}`);
  console.log('\nword_count:');
  console.log(`  Range: ${stats.min_word_count} - ${stats.max_word_count}`);
  console.log(`  Average: ${stats.avg_word_count.toFixed(2)}`);
  console.log('\navg_word_length:');
  console.log(`  Range: ${stats.min_avg_word_length} - ${stats.max_avg_word_length}`);
  console.log(`  Average: ${stats.avg_avg_word_length.toFixed(2)}`);
  console.log('\nnum_sentences:');
  console.log(`  Range: ${stats.min_num_sentences} - ${stats.max_num_sentences}`);
  console.log(`  Average: ${stats.avg_num_sentences.toFixed(2)}`);
  console.log('\npunctuation_density:');
  console.log(`  Range: ${stats.min_punctuation_density} - ${stats.max_punctuation_density}`);
  console.log(`  Average: ${stats.avg_punctuation_density.toFixed(4)}`);
  console.log('\nreadability_score:');
  console.log(`  Range: ${stats.min_readability_score} - ${stats.max_readability_score}`);
  console.log(`  Average: ${stats.avg_readability_score.toFixed(2)}`);
  console.log('─'.repeat(60));
  
  // Show 5 sample AI rows
  console.log('\n\nSample AI-generated rows:');
  console.log('─'.repeat(60));
  const aiSamples = db.prepare(`SELECT * FROM ${TABLE_NAME} WHERE generated = 1 LIMIT 5`).all();
  aiSamples.forEach((row, i) => {
    console.log(`\n${i + 1}. Text length: ${row.text_length}, Words: ${row.word_count}, Avg word len: ${row.avg_word_length}`);
    console.log(`   Sentences: ${row.num_sentences}, Punct: ${row.punctuation_density}, Readability: ${row.readability_score}`);
  });
  
  // Show 5 sample Human rows
  console.log('\n\nSample Human-written rows:');
  console.log('─'.repeat(60));
  const humanSamples = db.prepare(`SELECT * FROM ${TABLE_NAME} WHERE generated = 0 LIMIT 5`).all();
  humanSamples.forEach((row, i) => {
    console.log(`\n${i + 1}. Text length: ${row.text_length}, Words: ${row.word_count}, Avg word len: ${row.avg_word_length}`);
    console.log(`   Sentences: ${row.num_sentences}, Punct: ${row.punctuation_density}, Readability: ${row.readability_score}`);
  });
  
  db.close();
  
  // Check normalization params
  console.log('\n\n═══════════════════════════════════════');
  console.log('MODEL NORMALIZATION PARAMETERS');
  console.log('═══════════════════════════════════════\n');
  
  if (fs.existsSync('./model/normalization.json')) {
    const normData = JSON.parse(fs.readFileSync('./model/normalization.json', 'utf8'));
    console.log('Mean:', normData.mean);
    console.log('Std:', normData.std);
  } else {
    console.log('⚠️  No normalization file found. Train the model first.');
  }
}

async function analyzeModel() {
  if (!fs.existsSync('./model/model.json')) {
    console.log('⚠️  No model found. Train the model first.');
    return;
  }

  console.log('\n═══════════════════════════════════════');
  console.log('MODEL PREDICTION ANALYSIS');
  console.log('═══════════════════════════════════════\n');

  const model = await tf.loadLayersModel('file://./model/model.json');
  const normData = JSON.parse(fs.readFileSync('./model/normalization.json', 'utf8'));
  const db = new Database(DB_PATH);
  
  // Get a balanced test set
  const aiSamples = db.prepare(`SELECT * FROM ${TABLE_NAME} WHERE generated = 1 LIMIT 100`).all();
  const humanSamples = db.prepare(`SELECT * FROM ${TABLE_NAME} WHERE generated = 0 LIMIT 100`).all();
  const testData = [...aiSamples, ...humanSamples];
  
  // Prepare features
  const features = testData.map(row => [
    parseFloat(row.text_length),
    parseFloat(row.word_count),
    parseFloat(row.avg_word_length),
    parseFloat(row.num_sentences),
    parseFloat(row.punctuation_density),
    parseFloat(row.readability_score)
  ]);
  
  // Normalize features
  const featuresTensor = tf.tensor2d(features);
  const mean = tf.tensor1d(normData.mean);
  const std = tf.tensor1d(normData.std);
  const normalizedFeatures = featuresTensor.sub(mean).div(std);
  
  // Get predictions
  const predictions = model.predict(normalizedFeatures);
  const predArray = await predictions.array();
  
  // Analyze results
  let truePos = 0, trueNeg = 0, falsePos = 0, falseNeg = 0;
  let lowConfidenceCount = 0;
  let lowConfidenceMistakes = 0;
  
  testData.forEach((row, i) => {
    const prob = predArray[i][0];
    const predicted = prob >= CONFIDENCE_THRESHOLD ? 1 : 0;
    const actual = Number(row.generated);
    
    if (prob > 0.5 && prob < CONFIDENCE_THRESHOLD) {
      lowConfidenceCount++;
      if (predicted !== actual) lowConfidenceMistakes++;
    }
    
    if (predicted === 1 && actual === 1) truePos++;
    if (predicted === 0 && actual === 0) trueNeg++;
    if (predicted === 1 && actual === 0) falsePos++;
    if (predicted === 0 && actual === 1) falseNeg++;
  });
  
  console.log('Model Performance Analysis:');
  console.log(`Using confidence threshold: ${CONFIDENCE_THRESHOLD}`);
  console.log(`\nLow confidence predictions (0.5-${CONFIDENCE_THRESHOLD}): ${lowConfidenceCount}`);
  console.log(`Low confidence mistakes: ${lowConfidenceMistakes}`);
  
  console.log('\nConfusion Matrix:');
  console.log('─'.repeat(40));
  console.log(`True Positives (AI correctly identified): ${truePos}`);
  console.log(`True Negatives (Human correctly identified): ${trueNeg}`);
  console.log(`False Positives (Human mistaken as AI): ${falsePos}`);
  console.log(`False Negatives (AI mistaken as Human): ${falseNeg}`);
  
  const accuracy = (truePos + trueNeg) / testData.length;
  const precision = truePos / (truePos + falsePos);
  const recall = truePos / (truePos + falseNeg);
  const f1 = 2 * (precision * recall) / (precision + recall);
  
  console.log('\nMetrics:');
  console.log('─'.repeat(40));
  console.log(`Accuracy: ${(accuracy * 100).toFixed(2)}%`);
  console.log(`Precision: ${(precision * 100).toFixed(2)}%`);
  console.log(`Recall: ${(recall * 100).toFixed(2)}%`);
  console.log(`F1 Score: ${(f1 * 100).toFixed(2)}%`);
  
  // Show some example predictions
  console.log('\nSample Predictions:');
  console.log('─'.repeat(40));
  for (let i = 0; i < 5; i++) {
    const row = testData[i];
    const prob = predArray[i][0];
    const predicted = prob >= CONFIDENCE_THRESHOLD ? 'AI' : 'Human';
    const actual = row.generated === 1 ? 'AI' : 'Human';
    console.log(`\nSample ${i + 1}:`);
    console.log(`Confidence: ${(prob * 100).toFixed(2)}%`);
    console.log(`Prediction: ${predicted} (${actual === predicted ? '✓' : '✗'})`);
    console.log(`Actual: ${actual}`);
  }
  
  // Cleanup
  featuresTensor.dispose();
  predictions.dispose();
  mean.dispose();
  std.dispose();
  db.close();
}

// Run both analyses
analyzeDatabase();
analyzeModel().catch(console.error);