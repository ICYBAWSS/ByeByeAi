import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas display options for better output
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.width', 1000)

# Load the dataset
try:
    df = pd.read_csv('linkedin_posts_database.csv')
    print(f"Successfully loaded {len(df)} posts")
    
    # Check if required columns exist
    if 'post' not in df.columns or 'category' not in df.columns:
        raise ValueError("CSV must contain 'post' and 'category' columns")
    
    # Clean and prepare data
    df = df.dropna()  # Remove any rows with missing values
    df['category'] = df['category'].str.lower().str.strip()
    
    # Check if we have both AI and human examples
    if len(df['category'].unique()) < 2:
        raise ValueError("Dataset must contain both 'ai' and 'human' categories")
    
    # Split data into features (X) and target (y)
    X = df['post']
    y = df['category']
    
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train a classifier
    print("Training classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_tfidf, y_train)
    
    # Get prediction probabilities
    y_pred_proba = clf.predict_proba(X_test_tfidf)
    y_pred = clf.predict(X_test_tfidf)
    
    # Create a DataFrame for analysis
    results = pd.DataFrame({
        'post': X_test,
        'true_label': y_test,
        'predicted': y_pred,
        'confidence': np.max(y_pred_proba, axis=1),
        'ai_prob': y_pred_proba[:, 0] if clf.classes_[0] == 'ai' else y_pred_proba[:, 1],
        'human_prob': y_pred_proba[:, 1] if clf.classes_[1] == 'human' else y_pred_proba[:, 0]
    })
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Model Accuracy: {accuracy:.2f}")
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Show confidence statistics
    print("\n🎯 Confidence Analysis:")
    print(f"Average confidence: {results['confidence'].mean():.1%}")
    print(f"Confidence when correct: {results[results['true_label'] == results['predicted']]['confidence'].mean():.1%}")
    print(f"Confidence when wrong: {results[results['true_label'] != results['predicted']]['confidence'].mean():.1%}")
    
    # Show examples with confidence levels
    print("\n🔍 Sample Predictions with Confidence:")
    sample_results = results.sample(min(5, len(results)), random_state=42)
    
    for _, row in sample_results.iterrows():
        is_correct = "✅" if row['true_label'] == row['predicted'] else "❌"
        print(f"\n{is_correct} Post: {row['post'][:100]}...")
        print(f"   True: {row['true_label']:6} | Predicted: {row['predicted']:6} | Confidence: {row['confidence']:.1%}")
        print(f"   AI: {row['ai_prob']:.1%} | Human: {row['human_prob']:.1%}")
    
    # Show most uncertain predictions
    results['uncertainty'] = 1 - results['confidence']
    uncertain = results.nlargest(3, 'uncertainty')
    
    if len(uncertain) > 0:
        print("\n🤔 Most Uncertain Predictions:")
        for _, row in uncertain.iterrows():
            print(f"\n⁉️  Post: {row['post'][:100]}...")
            print(f"   True: {row['true_label']:6} | Predicted: {row['predicted']:6} | Confidence: {row['confidence']:.1%}")
    
    # Show most confident mistakes
    mistakes = results[results['true_label'] != results['predicted']].nlargest(3, 'confidence')
    if len(mistakes) > 0:
        print("\n❌ Most Confident Mistakes:")
        for _, row in mistakes.iterrows():
            print(f"\n💥 Post: {row['post'][:100]}...")
            print(f"   True: {row['true_label']:6} | Predicted: {row['predicted']:6} | Confidence: {row['confidence']:.1%}")
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
    print(cm_df)
    
except FileNotFoundError:
    print("Error: Could not find 'linkedin_posts_database.csv' in the current directory.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
