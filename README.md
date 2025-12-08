# comp3800P3_georgemoore

## Project Summary

This repository contains an exploratory analysis and a small machine learning pipeline for studying job-anxiety narratives on Twitter/X. Work performed in this project includes data cleaning, EDA, feature engineering, modeling, evaluation, and a minimal model-serving API.

Key components and artifacts:

- `tweets_analysis.ipynb`: Jupyter notebook with full exploratory data analysis (EDA). Includes:
	- Data loading and cleaning of the provided CSV dataset.
	- Binary indicator creation for job-anxiety themes (layoffs, hiring, AI fear, burnout, upskilling, remote work).
	- Sentiment analysis using TextBlob and VADER.
	- Temporal analysis (time series, hourly patterns) and word-level analysis (word clouds, distinctive words).

- `comp3800F25_tweets_cleaned.csv`: Cleaned dataset used for analysis and modeling.

- `train_ai_fear.py`: Training script implementing a TF-IDF (uni+bi) + Logistic Regression pipeline to predict `is_ai_fear`. The script performs a stratified train/test split, trains the model, and saves the pipeline (`ai_fear_model.joblib`) and training metrics (`ai_fear_metrics.json`).

- `evaluate_ai_fear.py`: Evaluation script to compute ROC/PR curves, tune decision threshold for best F1, and save confusion matrices and evaluation JSON (`ai_fear_evaluation.json`).

- `app.py`: Simple Flask REST API exposing the trained model with endpoints:
	- `GET /health` — health check
	- `POST /predict` — accepts JSON `{"text": "..."}` and returns prediction + probability

- `requirements.txt` and `README_MODEL.md`: Reproducible environment and instructions for training, evaluating, and serving the model.

## Quick Results (initial run)

- Model artifacts produced: `ai_fear_model.joblib`, `ai_fear_metrics.json`.
- Initial evaluation (stratified 80/20 split) reported:
	- accuracy: 0.9990
	- F1: 0.000
	- precision: 0.000
	- recall: 0.000
	- ROC AUC: 0.9829

	These results indicate a highly imbalanced dataset: the trained model largely predicts the negative class by default, yielding high accuracy but zero positive-class recall/precision at the default threshold. The ROC AUC is high, meaning the model probabilities are useful for ranking; adjusting the decision threshold or rebalancing the training data should improve positive-class detection.

## Next recommended steps

- Address class imbalance (upsampling, SMOTE, or threshold tuning using the PR curve).
- Use stronger text representations (sentence-transformer embeddings or transformer fine-tuning).
- Run cross-validated hyperparameter search and produce a final evaluation with confusion matrix, ROC, and PR curves in the notebook.

---

See `README_MODEL.md` for detailed training/evaluation/deployment commands.