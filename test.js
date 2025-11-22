// test-academic.js - Interactive testing with academic model
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

// Feature extraction functions (same as training)
function calculatePerplexity(text) {
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  if (words.length < 2) return { perplexity: 0, normalizedPerplexity: 0 };
  
  const maxWords = 500;
  const processWords = words.length > maxWords ? words.slice(0, maxWords) : words;
  
  const bigramFreq = {};
  const unigramFreq = {};
  
  for (let i = 0; i < processWords.length - 1; i++) {
    const bigram = `${processWords[i]} ${processWords[i + 1]}`;
    const unigram = processWords[i];
    bigramFreq[bigram] = (bigramFreq[bigram] || 0) + 1;
    unigramFreq[unigram] = (unigramFreq[unigram] || 0) + 1;
  }
  
  let logProb = 0;
  for (let i = 0; i < processWords.length - 1; i++) {
    const bigram = `${processWords[i]} ${processWords[i + 1]}`;
    const unigram = processWords[i];
    const bigramCount = bigramFreq[bigram] || 0;
    const unigramCount = unigramFreq[unigram] || 1;
    const prob = (bigramCount + 1) / (unigramCount + Object.keys(unigramFreq).length);
    logProb += Math.log2(prob);
  }
  
  const perplexity = Math.pow(2, -logProb / (processWords.length - 1));
  return { perplexity, normalizedPerplexity: perplexity / processWords.length };
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
  
  let mtld = ttr;
  const windowSize = Math.min(50, Math.floor(words.length / 2));
  
  if (words.length >= windowSize * 2) {
    const numSamples = Math.min(10, Math.floor(words.length / windowSize));
    let mtldSum = 0;
    
    for (let s = 0; s < numSamples; s++) {
      const startIdx = Math.floor((words.length - windowSize) * s / numSamples);
      const window = words.slice(startIdx, startIdx + windowSize);
      const uniqueInWindow = new Set(window);
      mtldSum += uniqueInWindow.size / windowSize;
    }
    mtld = mtldSum / numSamples;
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
  
  const maxSentences = 50;
  const sampledSentences = sentences.length > maxSentences 
    ? sentences.slice(0, maxSentences) 
    : sentences;
  
  let overlapSum = 0;
  for (let i = 0; i < sampledSentences.length - 1; i++) {
    const words1 = new Set(sampledSentences[i].toLowerCase().split(/\s+/));
    const words2 = new Set(sampledSentences[i + 1].toLowerCase().split(/\s+/));
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const overlap = intersection.size / Math.max(words1.size, words2.size);
    overlapSum += overlap;
  }
  
  const coherenceScore = overlapSum / (sampledSentences.length - 1);
  
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
  
  return {
    perplexity: normalizedPerplexity,
    burstiness: burstiness,
    entropy: normalizedEntropy,
    lexicalDiversity: mtld,
    typeTokenRatio: ttr,
    vocabularyRichness: vocdD,
    bigramRepetition: bigrams.ngramRepetition,
    trigramRepetition: trigrams.ngramRepetition,
    functionWordRatio: stylometric.functionWordRatio,
    punctuationDiversity: stylometric.punctuationDiversity,
    starterDiversity: stylometric.starterDiversity,
    contractionRatio: stylometric.contractionRatio,
    avgWordLength: stylometric.avgWordLength,
    coherenceScore: coherence.coherenceScore,
    transitionWordRatio: coherence.transitionWordRatio,
    textLength: Math.log1p(text.length),
    wordCount: Math.log1p(words.length),
    sentenceCount: Math.log1p(sentences.length),
    sentenceLengthVariance: sentenceLengthVariance
  };
}

function displayFeatures(features) {
  console.log('\n═══════════════════════════════════════════════════');
  console.log('        LINGUISTIC FEATURE ANALYSIS');
  console.log('═══════════════════════════════════════════════════\n');
  
  console.log('📊 Core AI Detection Metrics:');
  console.log(`   Perplexity (lower = AI):           ${features.perplexity.toFixed(4)}`);
  console.log(`   Burstiness (lower = AI):           ${features.burstiness.toFixed(4)}`);
  console.log(`   Entropy (lower = AI):              ${features.entropy.toFixed(4)}`);
  console.log(`   Lexical Diversity (lower = AI):    ${features.lexicalDiversity.toFixed(4)}`);
  
  console.log('\n📝 Stylometric Features:');
  console.log(`   Type-Token Ratio:                  ${features.typeTokenRatio.toFixed(4)}`);
  console.log(`   Function Word Ratio:               ${features.functionWordRatio.toFixed(4)}`);
  console.log(`   Contraction Usage:                 ${features.contractionRatio.toFixed(4)}`);
  console.log(`   Sentence Starter Diversity:        ${features.starterDiversity.toFixed(4)}`);
  console.log(`   Punctuation Diversity:             ${features.punctuationDiversity.toFixed(4)}`);
  
  console.log('\n🔄 Pattern Analysis:');
  console.log(`   Bigram Repetition:                 ${features.bigramRepetition.toFixed(4)}`);
  console.log(`   Trigram Repetition:                ${features.trigramRepetition.toFixed(4)}`);
  console.log(`   Coherence Score:                   ${features.coherenceScore.toFixed(4)}`);
  console.log(`   Transition Word Usage:             ${features.transitionWordRatio.toFixed(4)}`);
  
  console.log('\n📏 Text Statistics:');
  console.log(`   Text Length (log):                 ${features.textLength.toFixed(4)}`);
  console.log(`   Word Count (log):                  ${features.wordCount.toFixed(4)}`);
  console.log(`   Sentence Count (log):              ${features.sentenceCount.toFixed(4)}`);
  console.log(`   Sentence Length Variance:          ${features.sentenceLengthVariance.toFixed(4)}`);
  
  const signals = [];
  if (features.perplexity < 0.05) signals.push('🔴 Very low perplexity (highly predictable)');
  if (features.burstiness < -0.3) signals.push('🔴 Low burstiness (uniform patterns)');
  if (features.entropy < 0.7) signals.push('🔴 Low entropy (limited vocabulary)');
  if (features.lexicalDiversity < 0.4) signals.push('🔴 Low lexical diversity');
  if (features.contractionRatio < 0.01) signals.push('🔴 Lack of contractions (overly formal)');
  if (features.starterDiversity < 0.5) signals.push('🔴 Repetitive sentence starters');
  if (features.bigramRepetition > 0.3) signals.push('🔴 High bigram repetition');
  
  console.log('\n🔍 AI Detection Signals:');
  if (signals.length === 0) {
    console.log('   ✅ No strong AI indicators detected');
  } else {
    signals.forEach(signal => console.log(`   ${signal}`));
  }
  
  console.log('\n═══════════════════════════════════════════════════\n');
}

async function loadModel() {
  try {
    const model = await tf.loadLayersModel('file://./model-academic/model.json');
    const normData = JSON.parse(fs.readFileSync('./model-academic/normalization.json', 'utf8'));
    const mean = tf.tensor1d(normData.mean);
    const std = tf.tensor1d(normData.std);
    return { model, mean, std, featureNames: normData.featureNames };
  } catch (error) {
    console.error('❌ Error loading model:', error.message);
    console.log('\nMake sure you have trained the model first by running:');
    console.log('  npm run train\n');
    throw error;
  }
}

async function predict(model, features, mean, std) {
  try {
    const featureArray = [
      features.perplexity,
      features.burstiness,
      features.entropy,
      features.lexicalDiversity,
      features.typeTokenRatio,
      features.vocabularyRichness,
      features.bigramRepetition,
      features.trigramRepetition,
      features.functionWordRatio,
      features.punctuationDiversity,
      features.starterDiversity,
      features.contractionRatio,
      features.avgWordLength,
      features.coherenceScore,
      features.transitionWordRatio,
      features.textLength,
      features.wordCount,
      features.sentenceCount,
      features.sentenceLengthVariance
    ];

    const inputTensor = tf.tensor2d([featureArray]);
    const normalizedInput = inputTensor.sub(mean).div(std.add(1e-7));
    
    const prediction = model.predict(normalizedInput);
    const score = (await prediction.data())[0];
    
    inputTensor.dispose();
    normalizedInput.dispose();
    prediction.dispose();
    
    return score;
  } catch (error) {
    console.error('❌ Error during prediction:', error);
    return 0.5;
  }
}

async function main() {
  console.log('═══════════════════════════════════════════════════');
  console.log('  AI Text Detector - Academic Implementation');
  console.log('  Based on: ACL 2025 GenAI Detection Research');
  console.log('═══════════════════════════════════════════════════\n');
  
  console.log('Loading model...');
  const { model, mean, std, featureNames } = await loadModel();
  console.log('✅ Model loaded successfully\n');
  
  const text = await question('Enter text to analyze (or "quit" to exit):\n> ');
  
  if (text.toLowerCase() === 'quit' || !text || text.trim().length === 0) {
    console.log('Goodbye!');
    rl.close();
    return;
  }
  
  console.log('\nExtracting features...');
  const features = extractAcademicFeatures(text);
  displayFeatures(features);
  
  console.log('Running AI detection model...\n');
  const probability = await predict(model, features, mean, std);
  
  console.log('═══════════════════════════════════════════════════');
  console.log('              PREDICTION RESULT');
  console.log('═══════════════════════════════════════════════════');
  
  if (isNaN(probability)) {
    console.log('⚠️  ERROR: Model returned NaN');
  } else {
    const isAI = probability > 0.5;
    const confidence = (isAI ? probability : 1 - probability) * 100;
    
    console.log(`\nResult: ${isAI ? '🤖 AI-GENERATED' : '✍️  HUMAN-WRITTEN'}`);
    console.log(`Confidence: ${confidence.toFixed(2)}%`);
    console.log(`AI Probability: ${(probability * 100).toFixed(2)}%`);
  }
  console.log('\n═══════════════════════════════════════════════════\n');
  
  rl.close();
}

main().catch(console.error);