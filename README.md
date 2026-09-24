# SmartRail — Predictive Maintenance & Compressor Health Intelligence

An end-to-end machine learning system for predicting compressor failure risk in metro trains using real-world sensor data from the MetroPT-3 dataset.

🔗 **Live Demo:** https://smart rail-predictive-maintenance.streamlit.app/

## 🚆 Problem

Air compressors are critical components in metro trains. Unexpected compressor failures can lead to maintenance issues and operational downtime.

SmartRail uses historical sensor data such as pressure, oil temperature, and motor current to estimate compressor failure risk and support proactive maintenance.

## 🔧 ML Pipeline

1. **Data Engineering**
   - Cleaned 1.5M+ sensor readings
   - Created rolling statistical features
   - Engineered lag and difference features
   - Generated 40+ engineered features

2. **Dimensionality Reduction**
   - StandardScaler
   - PCA with 10 components
   - ~93% variance retained

3. **Unsupervised Learning**
   - KMeans
   - Agglomerative Clustering
   - DBSCAN
   - Identified different compressor operating states

4. **Classification**
   - Logistic Regression
   - Naive Bayes
   - KNN
   - SVM
   - Decision Tree
   - Random Forest
   - Bagging
   - AdaBoost
   - Gradient Boosting
   - XGBoost

5. **Hyperparameter Optimization**
   - Optuna
   - XGBoost tuning with 20 trials

6. **Ensemble Learning**
   - Stacking Ensemble
   - Logistic Regression + Random Forest + XGBoost

7. **Model Evaluation**
   - Compared models using ROC-AUC
   - Investigated temporal data leakage
   - Replaced random splitting with time-aware block splitting for more realistic evaluation

## 📊 Model Results

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.9973 |
| SVM | 0.9993 |
| Decision Tree | 0.9991 |
| Tuned XGBoost | 0.9997 |
| Stacking Ensemble | 0.9953 |

> The extremely high scores were investigated for potential temporal leakage rather than being accepted blindly.

## 🔍 Key Finding

KMeans clustering identified an operating state associated with significantly higher failure frequency.

One cluster showed approximately **45% failure rate**, compared with **less than 0.05%** in the other clusters.

This provided an additional unsupervised signal for understanding compressor operating conditions.

## 🌐 Streamlit Application

The project includes an interactive Streamlit application where users can enter compressor sensor readings and receive an estimated failure-risk probability.

### Input Sensors

- TP2 Pressure
- TP3 Pressure
- H1 Pressure
- DV Pressure
- Reservoirs
- Oil Temperature
- Motor Current

The application performs the same feature transformation pipeline used during model development before generating the prediction.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- Matplotlib
- Streamlit
- Joblib

## 📁 Project Structure

```text
SmartRail-Predictive-Maintenance/
│
├── models/
│   ├── classification models
│   ├── scaler
│   ├── PCA model
│   ├── KMeans model
│   └── stacking model
│
├── notebooks/
│   └── ML pipeline notebooks
│
├── streamlit_app/
│   └── app.py
│
├── README.md
└── requirements.txt
