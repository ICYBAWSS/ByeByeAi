// check-model.js - Verify the model works
import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';

async function checkModel() {
  console.log('Checking if model exists...');
  
  if (!fs.existsSync('./model/model.json')) {
    console.log('❌ Model not found. Run "npm run train" first.');
    return;
  }
  
  console.log('✓ Model file exists\n');
  console.log('Loading model...');
  
  const model = await tf.loadLayersModel('file://./model/model.json');
  console.log('✓ Model loaded\n');
  
  model.summary();
  
  console.log('\nLoading normalization parameters...');
  const normData = JSON.parse(fs.readFileSync('./model/normalization.json', 'utf8'));
  const mean = tf.tensor1d(normData.mean);
  const std = tf.tensor1d(normData.std);
  console.log('✓ Normalization loaded\n');
  
  console.log('Testing with a simple input...');
  
  // Use the mean values (should give ~0.5 probability)
  const testInput = tf.tensor2d([normData.mean]);
  console.log('Input (using mean values):', testInput.arraySync()[0]);
  
  const prediction = model.predict(testInput);
  const prob = prediction.arraySync()[0][0];
  
  console.log('Raw prediction output:', prediction.arraySync());
  console.log('Probability:', prob);
  
  if (isNaN(prob)) {
    console.log('\n❌ Model is returning NaN - model is broken');
    console.log('Try deleting the ./model folder and retraining:');
    console.log('  rm -rf ./model');
    console.log('  npm run train');
  } else {
    console.log('\n✓ Model is working correctly!');
  }
  
  testInput.dispose();
  prediction.dispose();
  mean.dispose();
  std.dispose();
}

checkModel().catch(console.error);