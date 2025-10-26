// test.js - Interactive model testing with automatic feature extraction
import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import readline from 'readline';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(prompt) {
  return new Promise((resolve) => {
    rl.question(prompt, resolve);
  });
}

// Calculate phrase repetition
function calculatePhraseRepetition(text, minPhraseLen = 3, maxPhraseLen = 6) {
  const words = text.toLowerCase().split(/\s+/);
  if (words.length < minPhraseLen) return 0.0;

  const phraseCounts = {};
  for (let phraseLen = minPhraseLen; phraseLen <= maxPhraseLen; phraseLen++) {
    for (let i = 0; i <= words.length - phraseLen; i++) {
      const phrase = words.slice(i, i + phraseLen).join(' ');
      phraseCounts[phrase] = (phraseCounts[phrase] || 0) + 1;
    }
  }

  const repeatedPhrases = Object.values(phraseCounts).filter(count => count > 1).length;
  const totalPhrases = Object.keys(phraseCounts).length;
  return totalPhrases > 0 ? repeatedPhrases / totalPhrases : 0.0;
}

// Calculate sentence complexity and diversity
function calculateSentenceComplexity(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
  if (sentences.length === 0) return [0.0, 0.0];

  // Calculate sentence length variation
  const sentLengths = sentences.map(s => s.trim().split(/\s+/).length);
  const avgLen = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
  const variance = sentLengths.reduce((sum, len) => sum + Math.pow(len - avgLen, 2), 0) / sentLengths.length;
  const complexityScore = avgLen > 0 ? variance / avgLen : 0.0;

  // Calculate sentence structure diversity
  const structurePatterns = sentences.map(sent => {
    const punctCount = (sent.match(/[,.;:]/g) || []).length;
    return `${sent.trim().split(/\s+/).length}-${punctCount}`;
  });
  
  const uniquePatterns = new Set(structurePatterns).size;
  const diversityScore = uniquePatterns / sentences.length;

  return [complexityScore, diversityScore];
}

// Simple syllable counter
function countSyllables(word) {
  word = word.toLowerCase();
  if (word.length <= 3) return 1;
  word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
  word = word.replace(/^y/, '');
  const matches = word.match(/[aeiouy]{1,2}/g);
  return matches ? matches.length : 1;
}

// Calculate text features with log-transformed length features
function extractFeatures(text) {
  // Text length (log-transformed)
  const text_length = Math.log1p(text.length);
  
  // Word count (log-transformed)
  const words = text.trim().split(/\s+/).filter(w => w.length > 0);
  const word_count_log = Math.log1p(words.length);
  
  // Average word length
  const avg_word_length = words.reduce((sum, word) => sum + word.length, 0) / words.length;
  
  // Number of sentences (log-transformed)
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  const num_sentences_log = Math.log1p(sentences.length);
  
  // Punctuation density
  const punctuation = text.match(/[.,!?;:'"()-]/g) || [];
  const punctuation_density = punctuation.length / text.length;
  
  // Readability score (Flesch Reading Ease approximation)
  const syllables = words.reduce((sum, word) => sum + countSyllables(word), 0);
  const readability_score = 206.835 - 1.015 * (words.length / sentences.length) - 84.6 * (syllables / words.length);
  
  // Calculate additional features
  const phrase_repetition = calculatePhraseRepetition(text);
  const [sentence_complexity, sentence_diversity] = calculateSentenceComplexity(text);
  
  return {
    text_length_log: text_length,
    word_count_log: word_count_log,
    avg_word_length: parseFloat(avg_word_length.toFixed(2)),
    num_sentences_log: num_sentences_log,
    punctuation_density: parseFloat(punctuation_density.toFixed(4)),
    readability_score: Math.max(0, Math.min(100, parseFloat(readability_score.toFixed(2)))),
    phrase_repetition: parseFloat(phrase_repetition.toFixed(4)),
    sentence_complexity: parseFloat(sentence_complexity.toFixed(4)),
    sentence_diversity: parseFloat(sentence_diversity.toFixed(4))
  };
}

// Log feature importance (to be called after model loading)
function logFeatureImportance(weights, featureNames) {
  console.log('\n─── Feature Importance ───');
  const importance = featureNames.map((name, i) => ({
    name,
    weight: Math.abs(weights[i])
  }));
  
  // Sort by absolute weight (descending)
  importance.sort((a, b) => b.weight - a.weight);
  
  // Log top 10 features
  importance.slice(0, 10).forEach((feat, i) => {
    console.log(`${i + 1}. ${feat.name.padEnd(25)}: ${feat.weight.toFixed(6)}`);
  });
  
  // Check if length features are dominating
  const lengthFeatures = ['text_length_log', 'word_count_log', 'num_sentences_log'];
  const top3LengthFeatures = importance
    .slice(0, 3)
    .filter(feat => lengthFeatures.includes(feat.name));
    
  if (top3LengthFeatures.length > 0) {
    console.log('\n⚠️  Note: Length-related features are among the top predictors.');
    console.log('   Consider collecting more diverse training data with balanced lengths.');
  }
}

// Load normalization parameters
function loadNormalizationParams(modelDir = './model') {
  try {
    const normData = JSON.parse(fs.readFileSync(`${modelDir}/normalization.json`, 'utf8'));
    return { 
      mean: tf.tensor1d(normData.mean), 
      std: tf.tensor1d(normData.std), 
      featureNames: normData.featureNames
    };
  } catch (error) {
    console.error('Error loading normalization parameters:', error);
    throw new Error('Failed to load model normalization parameters');
  }
}

// Load model with L2 regularization and normalization parameters
async function loadModel() {
  try {
    // Load the entire model from the saved files
    const model = await tf.loadLayersModel('file://./model/model.json');
    
    // Load normalization parameters
    const { mean, std, featureNames } = await loadNormalizationParams();
    
    return { model, mean, std, featureNames };
  } catch (error) {
    console.error('Error loading model:', error);
    throw new Error('Failed to load model. Make sure to train the model first.');
  }
}

// Predict function with proper feature scaling
async function predict(model, features, mean, std) {
  try {
    // Ensure features are in the correct order
    const featureArray = [
      features.text_length_log,
      features.word_count_log,
      features.avg_word_length,
      features.num_sentences_log,
      features.punctuation_density,
      features.readability_score,
      features.phrase_repetition,
      features.sentence_complexity,
      features.sentence_diversity
    ];

    // Convert features to tensor and normalize
    const inputTensor = tf.tensor2d([featureArray]);
    const normalizedInput = inputTensor.sub(mean).div(std.add(1e-7));
    
    // Make prediction
    const prediction = model.predict(normalizedInput);
    const score = (await prediction.data())[0];
    
    // Clean up
    inputTensor.dispose();
    normalizedInput.dispose();
    prediction.dispose();
    
    return score;
  } catch (error) {
    console.error('Error during prediction:', error);
    return 0.5; // Return neutral score on error
  }
}

// Main function
async function main() {
  console.log('═══════════════════════════════════════');
  console.log('   AI Text Detector - Interactive Test');
  console.log('═══════════════════════════════════════\n');
  
  // Load model and normalization parameters
  console.log('Loading model...');
  const { model, mean, std, featureNames } = await loadModel();
  console.log('✓ Model loaded successfully\n');
  
  // Get text input from user
  const text = await question('Enter text to analyze (or "quit" to exit):\n> ');
  
  if (text.toLowerCase() === 'quit') {
    console.log('Goodbye!');
    rl.close();
    return;
  }
  
  if (!text || text.trim().length === 0) {
    console.log('⚠️  Please enter some text to analyze.');
    rl.close();
    return;
  }
  
  console.log('\n─── Extracted Features ───');
  const textFeatures = extractFeatures(text);
  
  console.log(`Text Length (log): ${textFeatures.text_length_log.toFixed(4)}`);
  console.log(`Word Count (log): ${textFeatures.word_count_log.toFixed(4)}`);
  console.log(`Avg Word Length: ${textFeatures.avg_word_length}`);
  console.log(`Num Sentences (log): ${textFeatures.num_sentences_log.toFixed(4)}`);
  console.log(`Punctuation Density: ${textFeatures.punctuation_density}`);
  console.log(`Readability Score: ${textFeatures.readability_score}`);
  console.log(`Phrase Repetition: ${textFeatures.phrase_repetition}`);
  console.log(`Sentence Complexity: ${textFeatures.sentence_complexity}`);
  console.log(`Sentence Diversity: ${textFeatures.sentence_diversity}`);
  
  // Features in the same order as the model expects
  const features = [
    textFeatures.text_length_log,
    textFeatures.word_count_log,
    textFeatures.avg_word_length,
    textFeatures.num_sentences_log,
    textFeatures.punctuation_density,
    textFeatures.readability_score,
    textFeatures.phrase_repetition,
    textFeatures.sentence_complexity,
    textFeatures.sentence_diversity
  ];
  
  console.log('\n─── Normalized Features ───');
  const meanArray = mean.arraySync();
  const stdArray = std.arraySync();
  
  // Log feature values with their normalized versions
  featureNames.forEach((name, i) => {
    const normalized = (features[i] - meanArray[i]) / (stdArray[i] + 1e-7);
    console.log(`${name.padEnd(20)}: ${features[i].toFixed(4)} -> ${normalized.toFixed(4)} (mean: ${meanArray[i].toFixed(2)}, std: ${stdArray[i].toFixed(2)})`);
  });
  
  // Log feature importance
  try {
    // Get weights from the first dense layer (assuming it's the input layer)
    const weights = model.layers[0].getWeights()[0].arraySync();
    const avgWeights = weights[0].map((_, i) => 
      Math.abs(weights.reduce((sum, w) => sum + Math.abs(w[i]), 0) / weights.length)
    );
    logFeatureImportance(avgWeights, featureNames);
  } catch (e) {
    console.log('\n⚠️  Could not calculate feature importance:', e.message);
  }
  
  console.log('\nAnalyzing...\n');
  
  const probability = await predict(model, textFeatures, mean, std);
  
  console.log('═══════════════════════════════════════');
  console.log('         PREDICTION RESULT');
  console.log('═══════════════════════════════════════');
  
  if (isNaN(probability)) {
    console.log('⚠️  ERROR: Model returned NaN');
    console.log('This usually means:');
    console.log('- Model wasn\'t trained properly');
    console.log('- Features are too different from training data');
    console.log('- Normalization issue');
  } else {
    const isAI = probability > 0.5;
    const confidence = (isAI ? probability : 1 - probability) * 100;
    
    console.log(`Result: ${isAI ? '🤖 AI-GENERATED' : '✍️  HUMAN-WRITTEN'}`);
    console.log(`Confidence: ${confidence.toFixed(2)}%`);
    console.log(`AI Probability: ${(probability * 100).toFixed(2)}%`);
    
    // Provide more nuanced length guidance
    const wordCount = Math.exp(textFeatures.word_count_log) - 1;
    const sentenceCount = Math.exp(textFeatures.num_sentences_log) - 1;
    
    if (wordCount > 500 || sentenceCount > 30) {
      console.log('\nℹ️  Note: This is a longer text');
      console.log('   The model uses log-scaled length features to better handle');
      console.log('   varying text lengths, but extreme values may still affect accuracy.');
    }
  }
  console.log('═══════════════════════════════════════\n');
  
  // Clean up
  rl.close();
}

main().catch(console.error);