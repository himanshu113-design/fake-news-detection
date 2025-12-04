# backend/train_tfidf.py
"""
Train a TF-IDF + Logistic Regression model on a news dataset.
Usage:
    python train_tfidf.py --data ../data/news.csv --output models/tfidf_lr.joblib
"""

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression for Fake News Detection")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save trained model")
    args = parser.parse_args()

    # 1. Load dataset
    print(f"📂 Loading dataset from {args.data} ...")
    df = pd.read_csv(args.data)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have 'text' and 'label' columns!")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # 2. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build pipeline (TF-IDF + Logistic Regression)
    print("⚙️ Training model...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", LogisticRegression(max_iter=300))
    ])

    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 5. Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"💾 Model saved to {args.output}")

if __name__ == "__main__":
    main()
