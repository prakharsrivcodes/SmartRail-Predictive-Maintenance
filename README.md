
# SmartRail — Predictive Maintenance & Compressor Health Intelligence

An end-to-end machine learning system for predicting compressor failure risk in metro trains, built on real-world industrial sensor data from the MetroPT-3 dataset. The project spans the complete ML lifecycle — data engineering, unsupervised analysis, supervised modeling, rigorous time-aware evaluation, and deployment as a live interactive application.

🔗 **Live Demo:** https://smartrail-predictive-maintenance.streamlit.app/
🔗 **Dataset:** https://archive.ics.uci.edu/dataset/791/metropt+3+dataset

## Problem

Air compressors are critical components in metro train operations. Unexpected compressor failures lead to costly downtime, disrupted service, and safety risk, while fixed-schedule maintenance wastes resources servicing equipment that is still healthy. SmartRail addresses this by using live sensor readings — pressure, oil temperature, motor current — to estimate failure risk in real time, enabling condition-based maintenance instead of reactive or calendar-driven servicing.

## Project Highlights

End-to-end predictive maintenance pipeline built from raw sensor data through to a deployed, publicly accessible application. Trained on a real-world industrial dataset of 1.5 million-plus sensor readings, with over 40 engineered rolling, lag, and difference features capturing temporal compressor behavior. Combines supervised and unsupervised machine learning, including PCA-based dimensionality reduction, Optuna-driven hyperparameter optimization, and a stacking ensemble across three distinct model families. Employs a time-aware evaluation methodology after identifying and resolving a temporal data leakage issue during development. Fully deployed as a live, interactive Streamlit application connected directly to the project repository.

## Machine Learning Pipeline

**Data Engineering:** Cleaned over 1.5 million raw sensor readings and removed duplicate records. For each of seven major compressor sensors, engineered rolling statistical features over a 5-minute window — mean, standard deviation, minimum, and maximum — along with lag-1 values and first-order differences, producing over 40 features that capture how compressor behavior evolves over time rather than relying on isolated point readings.

**Dimensionality Reduction:** Applied StandardScaler followed by Principal Component Analysis on the resulting 42-dimensional engineered feature space, compressing it into 10 principal components while retaining approximately 93 percent of the original variance.

**Unsupervised Learning:** Applied KMeans, Agglomerative Clustering, and DBSCAN without using any failure labels, in order to let the data reveal its own operating structure. Agglomerative clustering independently validated the KMeans cluster structure with 98.5 percent agreement, confirming the discovered operating states were genuinely structural rather than arbitrary.

**Classification:** Trained and rigorously compared ten distinct classification algorithms — Logistic Regression, Naive Bayes, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, and XGBoost — to understand real tradeoffs between precision, recall, and interpretability on this highly imbalanced, safety-relevant classification problem.

**Hyperparameter Optimization:** Tuned the strongest-performing model, XGBoost, using Optuna across a 20-trial Bayesian search over max depth, learning rate, subsample ratio, column sampling, and regularization parameters.

**Ensemble Learning:** Constructed a stacking ensemble combining Logistic Regression, Random Forest, and the Optuna-tuned XGBoost model, with a Logistic Regression meta-learner, to evaluate whether blending predictions across model families outperformed the single best model.

**Evaluation and Leakage Investigation:** This was the most critical stage of the project. Several tree-based models initially returned a suspicious, implausibly perfect 1.0000 AUC score — a result treated with skepticism rather than reported at face value. Initial investigation into feature dominance did not explain the issue. Root-cause analysis revealed that although the train/test split was randomized, the underlying data was time-series in nature with rolling-window features, meaning adjacent time points were nearly identical — a random split was allowing the model to effectively see near-duplicates of test data during training, a subtle form of temporal leakage. This was resolved by implementing a block-wise, time-aware train/test split that enforces genuine temporal separation between training and evaluation data, producing a realistic and fully defensible final AUC of 99.97 percent.

## Model Results

Logistic Regression achieved an AUC of 0.9973. Naive Bayes achieved 0.9859. K-Nearest Neighbors achieved 0.9971. Support Vector Machine achieved 0.9993. Decision Tree achieved 0.9991. Random Forest and Gradient Boosting each achieved 1.0000 prior to the time-aware split correction and are retained here for comparison with that leakage caveat explicitly noted. The final tuned XGBoost model, evaluated under the corrected time-aware split, achieved 0.9997 AUC. The stacking ensemble achieved 0.9953 AUC. Every unusually high score in this table was investigated for potential temporal leakage rather than accepted uncritically, in line with the evaluation methodology described above.

## Key Finding

Unsupervised KMeans clustering identified a distinct operating state characterized by elevated motor current and elevated oil temperature, which showed an approximate 45 percent failure rate — compared with under 0.05 percent in every other cluster. This result confirmed that the unsupervised stage of the pipeline discovered a genuinely predictive signal rather than functioning as a purely exploratory or cosmetic step, and that it meaningfully contributed usable signal to downstream supervised classification.

## Streamlit Application

The trained system is deployed as a fully interactive Streamlit web application, allowing users to enter current compressor sensor readings and receive an estimated failure-risk probability in real time, computed through the same feature transformation pipeline used throughout model development. Input sensors include TP2 pressure, TP3 pressure, H1 pressure, DV pressure, reservoir pressure, oil temperature, and motor current. A sample output returns a failure risk percentage alongside a plain-language operating status such as normal operating range or elevated risk requiring inspection. The application currently accepts manually entered, single-point sensor readings and assumes steady-state conditions for feature computation, rather than ingesting a continuous real-time sensor stream — a deliberate simplification for demonstration purposes. The application is deployed through Streamlit Community Cloud, connected directly to this GitHub repository, and bundles the trained classification models, fitted scaler, PCA transform, KMeans model, and final stacking classifier.

## Tech Stack

Python for the core language. Pandas and NumPy for data processing. Scikit-learn, XGBoost, and Optuna for machine learning and hyperparameter optimization. Matplotlib for visualization. Streamlit and Joblib for deployment and model serialization.

## Project Structure

The repository is organized into a notebooks directory containing ten sequential notebooks that walk through the complete pipeline from raw data ingestion through final stacking model — covering the data pipeline, PCA and clustering, anomaly detection, baseline classification, additional baselines, SVM and tree-based models, ensemble methods, gradient boosting and XGBoost, Optuna tuning and stacking, and the final stacking model. A models directory contains all trained classification models, the fitted scaler, the PCA transform, the KMeans model, and the final stacking classifier. A streamlit_app directory contains the deployed application code. The repository also includes a results.json file with performance metrics for every model evaluated, a requirements.txt file, and this README.

## What's Next

Planned extensions include formulating a Remaining Useful Life regression target using the dataset's timestamped failure logs, adding SHAP-based explainability for individual model predictions, and connecting the deployed application to a rolling sensor buffer instead of relying on single-point steady-state input.

## Why This Project

Most early-stage machine learning portfolios showcase a model that works. This project instead documents a model that initially appeared to work suspiciously well — and the disciplined process of recognizing that, investigating the root cause, and correcting it properly. That process, more than any individual performance metric, is representative of how real applied machine learning work is actually done.

## Author

Prakhar Srivastava
B.Tech, Information Technology
GitHub: https://github.com/prakharsrivcodes