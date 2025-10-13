// check-data.js - Check for bad data in database
import fs from 'fs';
import Database from 'better-sqlite3';

// Use the combined DB that actually contains the table and features
const DB_PATH = './AiHumanTextCombined.db';
const TABLE_NAME = 'data';

function checkData() {
  if (!fs.existsSync(DB_PATH)) {
    console.error(`Database file not found at ${DB_PATH}`);
    process.exit(1);
  }

  const db = new Database(DB_PATH);

  console.log('Checking database for problematic values...\n');

  // quick sanity: ensure the expected table exists
  const tbl = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?").get(TABLE_NAME);
  if (!tbl) {
    const available = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all().map(r => r.name).join(', ');
    console.error(`Table \"${TABLE_NAME}\" not found in ${DB_PATH}. Available tables: ${available}`);
    db.close();
    process.exit(1);
  }
  
  // Check for NULL values
  const nullChecks = [
    'text_length',
    'word_count',
    'avg_word_length',
    'num_sentences',
    'punctuation_density',
    'readability_score',
    'generated'
  ];
  
  nullChecks.forEach(col => {
    const result = db.prepare(`SELECT COUNT(*) as count FROM ${TABLE_NAME} WHERE ${col} IS NULL`).get();
    if (result.count > 0) {
      console.log(`❌ Found ${result.count} NULL values in ${col}`);
    }
  });
  
  // Check for infinity values
  const rows = db.prepare(`SELECT * FROM ${TABLE_NAME}`).all();

  if (!rows || rows.length === 0) {
    console.log(`No rows found in table ${TABLE_NAME}. Nothing to check.`);
    console.log('\n✓ Data check complete');
    db.close();
    return;
  }
  
  let nanCount = 0;
  let infCount = 0;
  let zeroStdFeatures = {};
  
  rows.forEach(row => {
    const features = [
      row.text_length,
      row.word_count,
      row.avg_word_length,
      row.num_sentences,
      row.punctuation_density,
      row.readability_score
    ];
    
    features.forEach((val, i) => {
      if (isNaN(val)) nanCount++;
      if (!isFinite(val)) infCount++;
    });
  });
  
  if (nanCount > 0) console.log(`❌ Found ${nanCount} NaN values in data`);
  if (infCount > 0) console.log(`❌ Found ${infCount} Infinity values in data`);
  
  // Check if any feature has zero variance (all same value)
  const featureNames = ['text_length', 'word_count', 'avg_word_length', 'num_sentences', 'punctuation_density', 'readability_score'];
  
  featureNames.forEach(name => {
    const minMax = db.prepare(`SELECT MIN(${name}) as min, MAX(${name}) as max FROM ${TABLE_NAME}`).get();
    if (minMax.min === minMax.max) {
      console.log(`❌ Feature ${name} has zero variance (all values are ${minMax.min})`);
    }
  });
  
  // Check for extreme outliers in readability_score
  const readabilityCheck = db.prepare(`
    SELECT COUNT(*) as count 
    FROM ${TABLE_NAME} 
    WHERE readability_score < -1000 OR readability_score > 1000
  `).get();
  
  if (readabilityCheck.count > 0) {
    console.log(`⚠️  Warning: ${readabilityCheck.count} rows have extreme readability scores (< -1000 or > 1000)`);
  }
  
  if (nanCount === 0 && infCount === 0) {
    console.log('✓ No NaN or Infinity values found');
  }
  
  console.log('\n✓ Data check complete');
  
  db.close();
}

checkData();