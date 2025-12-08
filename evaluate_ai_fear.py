"""
Evaluate the trained ai_fear model: ROC, PR, confusion matrices, threshold tuning.
Saves: `ai_fear_evaluation.json`, `roc_curve.png`, `pr_curve.png`, `cm_default.png`, `cm_best.png`.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

MODEL_PATH = 'ai_fear_model.joblib'
DATA_PATH = 'comp3800F25_tweets_cleaned.csv'
OUT_METRICS = 'ai_fear_evaluation.json'

# Utility: create is_ai_fear if missing
import re

def create_ai_fear_column(df):
    if 'is_ai_fear' not in df.columns:
        text_lower = df['text'].astype(str).str.lower().fillna('')
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
        df['is_ai_fear'] = df['is_ai_fear'].fillna(0).astype(int)
    return df


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df = create_ai_fear_column(df)
    X = df['text'].astype(str).values
    y = df['is_ai_fear'].astype(int).values
    return X, y


def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

    pipe = joblib.load(MODEL_PATH)
    X, y = load_data()

    # Recreate the same stratified split used for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Predictions & probabilities
    y_proba = None
    if hasattr(pipe, 'predict_proba'):
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None
    y_pred_default = pipe.predict(X_test)

    # Metrics at default threshold
    acc = float(metrics.accuracy_score(y_test, y_pred_default))
    f1 = float(metrics.f1_score(y_test, y_pred_default, zero_division=0))
    prec = float(metrics.precision_score(y_test, y_pred_default, zero_division=0))
    rec = float(metrics.recall_score(y_test, y_pred_default, zero_division=0))

    results = {
        'default': {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}
    }

    # ROC & PR curves
    if y_proba is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
        roc_auc = float(metrics.roc_auc_score(y_test, y_proba))
        precision, recall, pr_thresh = metrics.precision_recall_curve(y_test, y_proba)
        pr_auc = float(metrics.auc(recall, precision))

        # Save ROC
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        plt.plot([0,1],[0,1],'k--', alpha=0.5)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.close()

        # Save PR
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig('pr_curve.png')
        plt.close()

        results['roc_auc'] = roc_auc
        results['pr_auc'] = pr_auc

        # Threshold tuning: choose threshold that maximizes F1
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = -1
        best_t = 0.5
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            f1_t = metrics.f1_score(y_test, y_pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t = float(t)
        results['best_threshold'] = best_t
        results['best_f1'] = float(best_f1)

        # Confusion matrices
        from sklearn.metrics import confusion_matrix
        cm_default = confusion_matrix(y_test, y_pred_default)
        cm_best = confusion_matrix(y_test, (y_proba >= best_t).astype(int))

        # Plot & save confusion matrices
        import seaborn as sns
        plt.figure(figsize=(5,4))
        sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix (threshold=0.5)')
        plt.tight_layout(); plt.savefig('cm_default.png'); plt.close()

        plt.figure(figsize=(5,4))
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens')
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion Matrix (best threshold={best_t:.2f})')
        plt.tight_layout(); plt.savefig('cm_best.png'); plt.close()

        results['cm_default'] = cm_default.tolist()
        results['cm_best'] = cm_best.tolist()

    # Save results
    with open(OUT_METRICS, 'w') as f:
        json.dump(results, f, indent=2)

    print('Evaluation complete. Results saved to', OUT_METRICS)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    evaluate()
