# SmartRail — Predictive Maintenance & Compressor Health Intelligence

End-to-end machine learning system for predicting compressor failures on a metro train, using real sensor data from the [MetroPT-3 dataset (UCI)](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).

## Problem
Given live sensor data (pressure, temperature, motor current) from a metro train's air compressor, predict failure risk before it happens — enabling proactive maintenance instead of reactive repairs.

## Pipeline
1. **Data Engineering** — Cleaned 1.5M+ sensor readings, engineered 40+ rolling/lag/statistical features (5-min windows)
2. **Dimensionality Reduction** — PCA (10 components, 93% variance retained)
3. **Unsupervised Operating-State Discovery** — KMeans, Agglomerative Clustering, DBSCAN to identify normal vs high-risk operating states
4. **Classification** — 10 models trained and compared: Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, XGBoost
5. **Hyperparameter Tuning** — Optuna-based search on XGBoost (20 trials)
6. **Ensemble** — Stacking (LR + RF + XGBoost with meta-learner)
7. **Data Leakage Investigation** — Identified and fixed a temporal leakage bug from random train/test splitting on time-series rolling features; implemented block-wise time-aware splitting for realistic evaluation

## Key Results
| Model | AUC |
|---|---|
| Logistic Regression | 0.9973 |
| SVM | 0.9993 |
| Decision Tree | 0.9991 |
| XGBoost (tuned, time-split) | 0.9997 |
| Stacking Ensemble | 0.9953 |

## Key Finding
An unsupervised KMeans cluster (representing high motor-current + high oil-temperature operating states) showed a 45% failure rate vs <0.05% in other clusters — validating that unsupervised feature engineering meaningfully improved downstream supervised classification.

## Tech Stack
Python, scikit-learn, XGBoost, Optuna, Pandas, Matplotlib

## Structure
- `notebooks/` — 10 sequential notebooks covering the full pipeline
- `models/` — Trained model files (KNN excluded due to file size)
- `results.json` — All model performance metrics

## Note
This project deliberately investigates and documents a data leakage issue found during development, as a demonstration of rigorous ML evaluation practice — not every model reporting near-perfect AUC is trustworthy without checking *why*.
