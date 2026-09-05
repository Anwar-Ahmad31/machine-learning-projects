# Iris Flower Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#license)

An end-to-end multi-class classification machine learning project predicting iris flower species using the famous **Iris Dataset** from the UCI Machine Learning Repository.

---

## 🌟 Project Overview

This individual machine learning project demonstrates data loading, feature extraction, and multi-class classification model training using **Scikit-Learn** (`RandomForestClassifier` & `DecisionTreeClassifier`).

### 📊 Dataset Attributes
The dataset contains **150 total samples** categorized across **3 Iris species**:
- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

#### 4 Input Features:
1. `sepallength` (Sepal length in cm)
2. `sepalwidth` (Sepal width in cm)
3. `petallength` (Petal length in cm)
4. `petalwidth` (Petal width in cm)

---

## 📁 Repository Structure

```text
iris-project/
├── Iris_main.ipynb         # Main Jupyter Notebook for model training & evaluation
├── i try to build .ipynb    # Experimental feature exploration & testing notebook
├── iris-train.xlsx         # Training dataset (116 samples)
├── iris-test.xlsx          # Testing dataset (34 samples)
└── README.md               # Project documentation
```

---

## 🛠️ Tech Stack

- **Language**: Python
- **Machine Learning**: `scikit-learn` (`RandomForestClassifier`, `DecisionTreeClassifier`)
- **Data Manipulation**: `pandas`, `numpy`, `openpyxl`
- **Dataset Source**: UCI Machine Learning Repository

---

## 🚀 How to Run Locally

### 1. Clone Repository
```bash
git clone https://github.com/Anwar-Ahmad31/iris-project.git
cd iris-project
```

### 2. Install Required Packages
```bash
pip install pandas numpy scikit-learn openpyxl jupyter
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook Iris_main.ipynb
```

---

## 📄 License
This project is open-source under the MIT License.
