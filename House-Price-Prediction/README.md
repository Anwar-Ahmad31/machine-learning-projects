# 🏠 California House Price Prediction

A machine learning pipeline for predicting median house values in California block groups using Scikit-Learn's `RandomForestRegressor`.

---

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline using `housing.csv` (California Housing dataset):
- **Stratified Data Splitting**: Uses `StratifiedShuffleSplit` on `income_cat` to create representative training and test sets (`input.csv`).
- **Feature Engineering & Preprocessing**:
  - `SimpleImputer` (median strategy) for missing numerical data.
  - `StandardScaler` for feature scaling.
  - `OneHotEncoder` for categorical feature `ocean_proximity`.
  - `ColumnTransformer` & `Pipeline` for clean modular preprocessing.
- **Model**: `RandomForestRegressor` trained and saved using `joblib`.
- **Inference Mode**: Automatically loads pre-trained model/pipeline when available to generate predictions on unseen input data (`input.csv`) and save results (`output.csv`).

---

## 📂 File Structure

- `main.py`: Main executable Python script (trains model or runs inference).
- `housing.csv`: Raw California housing dataset.
- `input.csv`: Unseen test dataset generated via stratified split for inference.
- `output.csv`: Predicted median house values output file.
- `pipeline.pkl`: Saved Scikit-Learn ColumnTransformer pipeline artifact.

---

## 🚀 How to Run

```bash
python main.py
```
