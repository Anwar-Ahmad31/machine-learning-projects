# 🫀 Heart Disease Prediction ML Project

An end-to-end Machine Learning classification project built with **Scikit-Learn** for predicting coronary artery disease presence using the **UCI Cleveland Heart Disease dataset**.

---

## 📌 Project Overview

This project implements a modular, reproducible ML classification architecture to assist clinical risk assessment:
- **Dataset**: UCI Cleveland Heart Disease Dataset (303 records, 14 clinical features including age, resting BP, cholesterol, max heart rate, chest pain type, etc.).
- **Target Variable**: Binary classification (`0` = No heart disease, `1` = Heart disease present).
- **Preprocessing Pipeline**: Missing value imputation (`SimpleImputer`), numerical standardization (`StandardScaler`), categorical encoding (`OneHotEncoder`), unified via `ColumnTransformer`.
- **Model Evaluation Benchmark**: 7 distinct machine learning algorithms were trained and benchmarked across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and 5-Fold Stratified Cross-Validation.
- **Model Serialization & Inference**: Production model and preprocessing pipeline serialized via `joblib` (`model.pkl` & `pipeline.pkl`), supporting automatic train vs. inference execution modes (`input.csv` ➡️ `output.csv`).

---

## 📊 Model Comparison & Recruiter Rationale

| Model Name | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV Mean | Status | Selection / Rejection Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | **88.52%** | **83.87%** | **92.86%** | **0.8814** | **0.9665** | **85.47% ± 4.8%** | **SELECTED (Production)** | **Top CV score (85.47%)**, highest ROC-AUC (96.65%), high recall (92.86%) minimizing clinical false negatives, zero overfitting risk on 303 samples, and full linear feature interpretability (odds ratios). |
| **AdaBoost** | 88.52% | 81.82% | 96.43% | 0.8852 | 0.9632 | 84.15% ± 3.3% | Rejected | Slightly lower 5-Fold CV stability compared to Logistic Regression. |
| **Support Vector Machine (SVC)** | 88.52% | 83.87% | 92.86% | 0.8814 | 0.9643 | 84.48% ± 4.5% | Rejected | Comparable performance, but acts as a black-box model without linear feature weight interpretability for medical decision-making. |
| **K-Nearest Neighbors** | 88.52% | 86.21% | 89.29% | 0.8772 | 0.9529 | 84.16% ± 3.7% | Rejected | Sensitive to distance metric selection and scale noise in clinical attributes. |
| **Random Forest** | 86.89% | 81.25% | 92.86% | 0.8667 | 0.9426 | 83.50% ± 1.0% | Rejected | Lower Cross-Validation score; ensemble tree complexity adds unnecessary overhead without performance gain over linear models on 303 rows. |
| **Gradient Boosting** | 86.89% | 81.25% | 92.86% | 0.8667 | 0.9470 | 79.86% ± 3.6% | Rejected | Substantially lower 5-Fold CV score (79.86%), showing vulnerability to variance on small medical datasets. |
| **Decision Tree** | 73.77% | 67.65% | 82.14% | 0.7419 | 0.7440 | 77.19% ± 5.9% | Rejected | Prone to severe overfitting on small tabular samples with poor generalization. |

---

## 📂 Project File Structure

```text
heart disease prediction project/
├── heart.csv                      # Raw UCI Cleveland dataset with 14 attributes
├── main.py                        # Executable ML pipeline (Training & Inference engine)
├── input.csv                       # Unseen patient features test set for inference
├── output.csv                      # Prediction output with risk probabilities and diagnosis
├── model.pkl                      # Serialized Logistic Regression production model artifact
├── pipeline.pkl                   # Serialized ColumnTransformer preprocessing pipeline artifact
├── model_comparison_results.csv   # Benchmark results across all 7 candidate models
└── README.md                      # Project documentation and engineering decisions
```

---

## 🛠️ Feature Breakdown

1. `age`: Patient age in years
2. `sex`: `1` = Male, `0` = Female
3. `cp`: Chest pain type (`1`: typical angina, `2`: atypical angina, `3`: non-anginal, `4`: asymptomatic)
4. `trestbps`: Resting blood pressure (mm Hg)
5. `chol`: Serum cholesterol (mg/dl)
6. `fbs`: Fasting blood sugar > 120 mg/dl (`1` = True, `0` = False)
7. `restecg`: Resting ECG results (`0`: normal, `1`: ST-T wave abnormality, `2`: left ventricular hypertrophy)
8. `thalach`: Maximum heart rate achieved
9. `exang`: Exercise-induced angina (`1` = Yes, `0` = No)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: Peak exercise ST segment slope (`1`: upsloping, `2`: flat, `3`: downsloping)
12. `ca`: Major vessels colored by fluoroscopy (`0`–`3`)
13. `thal`: Thalassemia (`3`: normal, `6`: fixed defect, `7`: reversible defect)
14. `target_prediction`: Predicted heart disease presence (`0`: No Disease, `1`: Disease Present)
15. `heart_disease_risk_prob`: Calculated risk probability (`0.0000` to `1.0000`)

---

## 🚀 Quickstart Guide

### Run Model Pipeline:
```bash
python main.py
```
- **First Run**: Evaluates all candidate models, exports `model_comparison_results.csv`, trains production model, and saves `model.pkl` & `pipeline.pkl`.
- **Subsequent Runs**: Automatically executes **Inference Mode**, loading pre-trained artifacts to generate risk probabilities and predictions in `output.csv`.
