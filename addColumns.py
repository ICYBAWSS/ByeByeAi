import sqlite3
import string
import re

def calculate_readability_score(text):
    """
    Calculate Flesch Reading Ease score.
    Score ranges: 90-100 (very easy), 60-70 (standard), 0-30 (very difficult)
    """
    # Count sentences
    sentences = re.split(r'[.!?]+', text)
    num_sentences = len([s for s in sentences if s.strip()])
    
    if num_sentences == 0:
        return 0.0
    
    # Count words
    words = text.split()
    num_words = len(words)
    
    if num_words == 0:
        return 0.0
    
    # Count syllables (approximation)
    syllables = 0
    for word in words:
        word = word.lower().strip(string.punctuation)
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        # Every word has at least one syllable
        if syllable_count == 0:
            syllable_count = 1
            
        syllables += syllable_count
    
    # Flesch Reading Ease formula
    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (syllables / num_words)
    
    return round(score, 2)

def calculate_phrase_repetition(text, min_phrase_len=3, max_phrase_len=6):
    """Calculate phrase repetition score."""
    words = text.lower().split()
    if len(words) < min_phrase_len:
        return 0.0
    
    # Count repeated phrases
    phrase_counts = {}
    for phrase_len in range(min_phrase_len, max_phrase_len + 1):
        for i in range(len(words) - phrase_len + 1):
            phrase = ' '.join(words[i:i + phrase_len])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    
    # Calculate repetition score
    repeated_phrases = sum(1 for count in phrase_counts.values() if count > 1)
    total_phrases = len(phrase_counts)
    repetition_score = repeated_phrases / total_phrases if total_phrases > 0 else 0.0
    return round(repetition_score, 4)

def calculate_sentence_complexity(text):
    """Calculate sentence complexity metrics."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0, 0.0
    
    # Calculate sentence length variation
    sent_lengths = [len(s.split()) for s in sentences]
    avg_len = sum(sent_lengths) / len(sent_lengths)
    variance = sum((x - avg_len) ** 2 for x in sent_lengths) / len(sent_lengths)
    complexity_score = (variance / avg_len) if avg_len > 0 else 0.0
    
    # Calculate sentence structure diversity
    structure_patterns = []
    for sent in sentences:
        # Simple pattern based on sentence length and punctuation
        pattern = f"{len(sent.split())}-{sum(1 for c in sent if c in ',.;:')}"
        structure_patterns.append(pattern)
    
    unique_patterns = len(set(structure_patterns))
    diversity_score = unique_patterns / len(sentences)
    
    return round(complexity_score, 4), round(diversity_score, 4)

def analyze_text(text):
    """Analyze a single text and return all metrics."""
    if text is None or not isinstance(text, str):
        return (0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Text length
    text_length = len(text)
    
    # Word count and average word length
    words = text.split()
    word_count = len(words)
    
    if word_count > 0:
        # Calculate average word length (excluding punctuation)
        cleaned_words = [word.strip(string.punctuation) for word in words]
        total_word_chars = sum(len(word) for word in cleaned_words if word)
        avg_word_length = total_word_chars / word_count if word_count > 0 else 0.0
    else:
        avg_word_length = 0.0
    
    # Number of sentences
    sentences = re.split(r'[.!?]+', text)
    num_sentences = len([s for s in sentences if s.strip()])
    
    # Punctuation density
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    punctuation_density = punctuation_count / text_length if text_length > 0 else 0.0
    
    # Readability score
    readability_score = calculate_readability_score(text)
    
    # Calculate new metrics
    phrase_repetition = calculate_phrase_repetition(text)
    sentence_complexity, sentence_diversity = calculate_sentence_complexity(text)
    
    return (
        text_length,
        word_count,
        round(avg_word_length, 2),
        num_sentences,
        round(punctuation_density, 4),
        readability_score,
        phrase_repetition,
        sentence_complexity,
        sentence_diversity
    )

def add_analysis_columns(db_path, table_name='your_table_name'):
    """Add analysis columns to the database table and populate them."""
    
    # Connect to the database
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists and show current columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f"\nCurrent columns in '{table_name}':")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Add new columns if they don't exist
    new_columns = [
        ('text_length', 'INTEGER'),
        ('word_count', 'INTEGER'),
        ('avg_word_length', 'REAL'),
        ('num_sentences', 'INTEGER'),
        ('punctuation_density', 'REAL'),
        ('readability_score', 'REAL'),
        ('phrase_repetition', 'REAL'),
        ('sentence_complexity', 'REAL'),
        ('sentence_diversity', 'REAL')
    ]
    
    print("\nAdding new columns...")
    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
            print(f"  Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"  Column '{col_name}' already exists, skipping")
            else:
                raise
    
    conn.commit()
    
    # Get all rows with ROWID
    print("\nFetching data...")
    cursor.execute(f"SELECT ROWID, text FROM {table_name}")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} rows to analyze")
    
    # Analyze and update each row
    print("\nAnalyzing texts and updating database...")
    for idx, (rowid, text) in enumerate(rows):
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(rows)}")
        
        analysis = analyze_text(text)
        
        cursor.execute(f"""
            UPDATE {table_name}
            SET text_length = ?,
                word_count = ?,
                avg_word_length = ?,
                num_sentences = ?,
                punctuation_density = ?,
                readability_score = ?,
                phrase_repetition = ?,
                sentence_complexity = ?,
                sentence_diversity = ?
            WHERE ROWID = ?
        """, (*analysis, rowid))
    
    # Commit all updates
    conn.commit()
    
    # Show sample results
    print("\nSample results (first 5 rows):")
    cursor.execute(f"""
        SELECT text_length, word_count, avg_word_length, 
               num_sentences, punctuation_density, readability_score,
               phrase_repetition, sentence_complexity, sentence_diversity
        FROM {table_name}
        LIMIT 5
    """)
    
    print("\ntext_length | word_count | avg_word_length | num_sentences | punct_density | readability | repetition | complexity | diversity")
    print("-" * 120)
    for row in cursor.fetchall():
        print(f"{row[0]:11} | {row[1]:10} | {row[2]:15.2f} | {row[3]:13} | {row[4]:13.4f} | {row[5]:10.2f} | {row[6]:9.4f} | {row[7]:9.4f} | {row[8]:8.4f}")
    
    # Show statistics
    print("\nOverall statistics:")
    cursor.execute(f"""
        SELECT 
            AVG(text_length) as avg_text_length,
            AVG(word_count) as avg_word_count,
            AVG(avg_word_length) as avg_avg_word_length,
            AVG(num_sentences) as avg_num_sentences,
            AVG(punctuation_density) as avg_punct_density,
            AVG(readability_score) as avg_readability,
            AVG(phrase_repetition) as avg_repetition,
            AVG(sentence_complexity) as avg_complexity,
            AVG(sentence_diversity) as avg_diversity
        FROM {table_name}
    """)
    stats = cursor.fetchone()
    print(f"  Average text length: {stats[0]:.1f}")
    print(f"  Average word count: {stats[1]:.1f}")
    print(f"  Average word length: {stats[2]:.2f}")
    print(f"  Average sentences: {stats[3]:.1f}")
    print(f"  Average punctuation density: {stats[4]:.4f}")
    print(f"  Average readability score: {stats[5]:.2f}")
    print(f"  Average phrase repetition: {stats[6]:.4f}")
    print(f"  Average sentence complexity: {stats[7]:.4f}")
    print(f"  Average sentence diversity: {stats[8]:.4f}")
    
    # Close connection
    conn.close()
    print("\nDone! Database updated successfully.")

# Example usage
if __name__ == "__main__":
    # Replace these with your actual values
    db_path = "AiHumanTextCombined.db"
    table_name = "data"  # Replace with your actual table name
    
    # Run the analysis
    add_analysis_columns(db_path, table_name)