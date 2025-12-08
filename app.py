"""
Train a model to predict `is_ai_fear` from tweet text and save the pipeline.
Produces: `ai_fear_model.joblib` and `ai_fear_metrics.json`.
"""
import json
import os
import joblib
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

DATA_PATH = 'comp3800F25_tweets_cleaned.csv'
MODEL_PATH = 'ai_fear_model.joblib'
METRICS_PATH = 'ai_fear_metrics.json'

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Ensure text is string
    df['text'] = df['text'].astype(str)
    # If target column not present, create it using keyword/regex matching
    if 'is_ai_fear' not in df.columns:
        print('`is_ai_fear` column not found — creating from text using regex patterns')
        text_lower = df['text'].str.lower().fillna('')
        ai_fear_keywords = [
            r'\bai\s+(taking|replacing|steal(ing)?)\s+jobs?\b',
            r'\bautomation\s+(fear|anxiety|threat)\b',
            r'\b(fear|afraid|worried|anxious)\s+.*\s+ai\b',
            r'\bjob\s+security.*ai\b', r'\bai.*unemployment\b',
            r'\breplaced?\s+by\s+(ai|automation|robots?|machines?)\b',
            r'\bai\s+threat\b', r'\bautomation\s+crisis\b'
        ]
        pattern = '|'.join([f'({p})' for p in ai_fear_keywords])
        df['is_ai_fear'] = text_lower.str.contains(pattern, case=False, regex=True, na=False).astype(int)
    else:
        # Drop rows with missing target and ensure integer type
        df = df.dropna(subset=['is_ai_fear'])
        df['is_ai_fear'] = df['is_ai_fear'].astype(int)
    return df


def build_pipeline():
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1,2),
        max_features=20000
    )
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    pipe = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf)
    ])
    return pipe


def train_and_evaluate():
    df = load_data()
    X = df['text'].values
    y = df['is_ai_fear'].values

    # Train/test split with stratification to avoid class imbalance leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    print('Training model...')
    pipe.fit(X_train, y_train)

    print('Predicting on test set...')
    y_pred = pipe.predict(X_test)
    y_proba = None
    if hasattr(pipe, 'predict_proba'):
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    # Evaluation metrics
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    prec = metrics.precision_score(y_test, y_pred, zero_division=0)
    rec = metrics.recall_score(y_test, y_pred, zero_division=0)
    roc_auc = None
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            roc_auc = float(metrics.roc_auc_score(y_test, y_proba))
        except Exception:
            roc_auc = None

    metrics_dict = {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(prec),
        'recall': float(rec),
        'roc_auc': roc_auc
    }

    # Save model pipeline
    joblib.dump(pipe, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

    # Save metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f'Metrics written to {METRICS_PATH}')

    # Print summary
    print('\nEvaluation metrics:')
    for k, v in metrics_dict.items():
        print(f' - {k}: {v}')

    return pipe, metrics_dict


if __name__ == '__main__':
    pipe, metrics = train_and_evaluate()
