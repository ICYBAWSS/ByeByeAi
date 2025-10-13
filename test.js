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

// Calculate text features
function extractFeatures(text) {
  // Text length
  const text_length = text.length;
  
  // Word count
  const words = text.trim().split(/\s+/).filter(w => w.length > 0);
  const word_count = words.length;
  
  // Average word length
  const avg_word_length = words.reduce((sum, word) => sum + word.length, 0) / word_count;
  
  // Number of sentences
  const num_sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
  
  // Punctuation density
  const punctuation = text.match(/[.,!?;:'"()-]/g) || [];
  const punctuation_density = punctuation.length / text_length;
  
  // Readability score (Flesch Reading Ease approximation)
  const syllables = words.reduce((sum, word) => sum + countSyllables(word), 0);
  const readability_score = 206.835 - 1.015 * (word_count / num_sentences) - 84.6 * (syllables / word_count);
  
  // Calculate additional features
  const phrase_repetition = calculatePhraseRepetition(text);
  const [sentence_complexity, sentence_diversity] = calculateSentenceComplexity(text);
  
  return {
    text_length,
    word_count,
    avg_word_length: parseFloat(avg_word_length.toFixed(2)),
    num_sentences,
    punctuation_density: parseFloat(punctuation_density.toFixed(4)),
    readability_score: Math.max(0, Math.min(100, parseFloat(readability_score.toFixed(2)))),
    phrase_repetition: parseFloat(phrase_repetition.toFixed(4)),
    sentence_complexity: parseFloat(sentence_complexity.toFixed(4)),
    sentence_diversity: parseFloat(sentence_diversity.toFixed(4))
  };
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

// Load model
async function loadModel() {
  const model = await tf.loadLayersModel('file://./model/model.json');
  const normData = JSON.parse(fs.readFileSync('./model/normalization.json', 'utf8'));
  return {
    model,
    mean: tf.tensor1d(normData.mean),
    std: tf.tensor1d(normData.std)
  };
}

// Predict function
function predict(model, features, mean, std) {
  const featureTensor = tf.tensor2d([features]);
  const normalized = featureTensor.sub(mean).div(std.add(1e-7));
  const prediction = model.predict(normalized);
  const probability = prediction.dataSync()[0];
  
  featureTensor.dispose();
  normalized.dispose();
  prediction.dispose();
  
  return probability;
}

// Main
async function main() {
  console.log('Loading model...\n');
  const { model, mean, std } = await loadModel();
  console.log('✓ Model loaded successfully!\n');
  
  console.log('═══════════════════════════════════════');
  console.log('     AI TEXT DETECTION TEST');
  console.log('═══════════════════════════════════════\n');
  
  // Get text input
  console.log('Paste your text below (press Enter twice when done):');
  let text = '';
  let emptyLines = 0;
  
  rl.on('line', (line) => {
    if (line === '') {
      emptyLines++;
      if (emptyLines >= 2 || text.length > 0) {
        rl.close();
      }
    } else {
      emptyLines = 0;
      text += line + '\n';
    }
  });
  
  await new Promise(resolve => rl.on('close', resolve));
  
  text = text.trim();
  
  if (!text) {
    console.log('No text provided. Exiting.');
    return;
  }
  
  console.log('\nExtracting features...');
  const textFeatures = extractFeatures(text);
  
  console.log('\n─── Extracted Features ───');
  console.log(`Text Length: ${textFeatures.text_length}`);
  console.log(`Word Count: ${textFeatures.word_count}`);
  console.log(`Avg Word Length: ${textFeatures.avg_word_length}`);
  console.log(`Sentences: ${textFeatures.num_sentences}`);
  console.log(`Punctuation Density: ${textFeatures.punctuation_density}`);
  console.log(`Readability Score: ${textFeatures.readability_score}`);
  
  const features = [
    textFeatures.text_length,
    textFeatures.word_count,
    textFeatures.avg_word_length,
    textFeatures.num_sentences,
    textFeatures.punctuation_density,
    textFeatures.readability_score,
    textFeatures.phrase_repetition,
    textFeatures.sentence_complexity,
    textFeatures.sentence_diversity
  ];
  
  console.log('\n─── Normalized Features ───');
  const meanArray = mean.arraySync();
  const stdArray = std.arraySync();
  
  features.forEach((f, i) => {
    const normalized = (f - meanArray[i]) / (stdArray[i] + 1e-7);
    console.log(`Feature ${i}: ${f.toFixed(4)} -> ${normalized.toFixed(4)} (mean: ${meanArray[i].toFixed(2)}, std: ${stdArray[i].toFixed(2)})`);
  });
  
  console.log('\nAnalyzing...\n');
  
  const probability = predict(model, features, mean, std);
  
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
    
    // Warning if text is very different from training data
    if (textFeatures.word_count > 500 || textFeatures.num_sentences > 30) {
      console.log('\n⚠️  WARNING: This text is much longer than training data');
      console.log('   Training average: 309 words, 16 sentences');
      console.log('   This may reduce accuracy.');
    }
  }
  console.log('═══════════════════════════════════════\n');
}

main().catch(console.error);