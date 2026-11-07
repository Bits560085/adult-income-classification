# Adult Income Classification
## Machine Learning Assignment 2 – BITS M.Tech AIML

### 1. Problem Statement
The objective of this project is to classify whether an individual's annual income exceeds $50K based on demographic and employment-related attributes using machine learning classification algorithms.

### 2. Dataset Description
- **Dataset:** Adult Income Dataset
- **Source:** UCI Machine Learning Repository
- **Total Instances:** 48,842
- **Features:** 14 input features
- **Target Variable:** income
- **Classes:** <=50K and >50K

The dataset was preprocessed by handling missing values, cleaning categorical values, and preparing the data for model training.

### 3. GitHub Repository Link
[Add your GitHub repository URL here]

### 4. Models Used and Evaluation Metrics

The following five classification models were implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8507 | 0.9042 | 0.7314 | 0.5941 | 0.6557 | 0.5666 |
| Decision Tree | 0.8102 | 0.7435 | 0.6011 | 0.6155 | 0.6082 | 0.4831 |
| KNN | 0.8299 | 0.8601 | 0.6606 | 0.5945 | 0.6258 | 0.5173 |
| Naive Bayes | 0.5715 | 0.7949 | 0.3517 | 0.9376 | 0.5116 | 0.3497 |
| Support Vector Machine | 0.8511 | 0.9033 | 0.7401 | 0.5821 | 0.6517 | 0.5652 |

### 5. Model Performance Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Achieved strong AUC and balanced classification performance. |
| Decision Tree | Lower overall accuracy and AUC compared to the other models. |
| KNN | Moderate performance with reasonable accuracy and AUC. |
| Naive Bayes | High recall but low precision, resulting in more false positives. |
| Support Vector Machine | Achieved the highest accuracy and precision among the five mandatory models. |

### Overall Winner
Support Vector Machine achieved the highest accuracy (0.8511) and precision (0.7401). Logistic Regression achieved the highest AUC (0.9042). The overall model selection depends on the desired balance between accuracy, precision, and recall.

### 6. Streamlit Application
The Streamlit application provides:
- CSV test dataset upload
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix
- Classification report
- Comparison of all five models

The application loads pre-trained models and evaluates them on uploaded test data without retraining.

### 7. Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Joblib

### 8. Deployment
Streamlit Community Cloud deployment link:

[Add your live Streamlit app URL here]
