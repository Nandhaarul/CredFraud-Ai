# Credit Card Fraud Detection Project

This project demonstrates an end-to-end machine learning workflow for detecting fraudulent credit card transactions. It leverages a highly imbalanced and anonymized dataset to showcase real-world data handling, feature engineering, model comparison, and optimization techniques.

The final output is a trained Random Forest model and a Python script that can be used to predict fraud on new, unseen transaction data.

---

## 📋 Project Features

- **Exploratory Data Analysis (EDA):**  
  Deep dive into anonymized features (`V1-V28`), `Time`, and `Amount` to uncover fraud patterns.

- **Creative Feature Engineering:**  
  Creation of new, impactful features (`Hour`, `Log_Amount`, `Business_Hours`) to improve model performance.

- **Advanced Modeling:**  
  Comparison of multiple algorithms (`Logistic Regression`, `Random Forest`, `XGBoost`) and selection of the best performer.

- **Threshold Optimization:**  
  Fine-tuning the model's decision threshold to maximize fraud detection (Recall) for better business outcomes.

- **Production-Ready Script:**  
  A reusable Python script (`src/predict.py`) to make predictions on new data.

---

## 📂 Project Structure
```bash
fraud-detection-project/
│
├── data/
│ ├── raw/
│ │ └── creditcard.csv
│ └── processed/
│   └── creditcard_processed.csv
│
├── models/
│ └── best_model.pkl
│
├── notebooks/
│ ├── 01_eda_time_amount.ipynb
│ ├── 02_eda_v_features.ipynb
│ ├── 03_feature_engineering.ipynb
│ └── 04_modeling.ipynb
│
├── src/
│ ├── __init__.py
│ └── predict.py
│
├── reports/
│ └── figures/
│
└── requirements.txt
```
---

## 🚀 Getting Started

Follow these instructions to set up the project environment and run the notebooks.

### Prerequisites

- Python 3.9+
- A virtual environment tool (like `venv` or `conda`)

### Installation

Clone the repository:

```bash
git clone <https://github.com/parthivkola/Credit-Card-Fraud-Detection.git>
cd fraud-detection-project
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
#or use conda
```  

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Workflow and Notebook Guide

The project is structured into a series of Jupyter notebooks that walk through the entire process from data exploration to modeling.

### 1. Exploratory Data Analysis (EDA)

- **`notebooks/01_eda_time_amount.ipynb`**  
  Focuses on the non-anonymized features. Analyzes how transaction `Amount` and `Time` correlate with fraudulent activity, establishing key patterns like the time of day when fraud is most likely to occur.

- **`notebooks/02_eda_v_features.ipynb`**  
  Analyzes the anonymized PCA features (`V1-V28`). Uses correlation matrices and violin plots to identify which features are the strongest predictors of fraud.

### 2. Feature Engineering

- **`notebooks/03_feature_engineering.ipynb`**  
  Details the creation and validation of new features. Transforms the raw data into a format more suitable for machine learning. Functions for these transformations are stored in `src/feature_engineering.py`.

### 3. Modeling and Optimization

- Multiple classification models were trained and compared.  
- **Random Forest** was selected for its high F1-Score and Precision.  
- The decision threshold was optimized to maximize Recall, resulting in a model that catches over **90% of fraudulent transactions**.  
- Trained model saved as `models/best_model.pkl`.

---

## 🔮 Running Predictions on Your Own Data

### Data Format Requirement

To use the prediction script, your input CSV file must have the same structure as the original `creditcard.csv` dataset, including the following columns:

- `Time`  
- `Amount`  
- `V1, V2, ..., V28`  

The script will automatically perform the necessary feature engineering.

### How to Run the Script

1. Place your new data file (e.g., `new_data.csv`) in the `data/raw/` directory.  
2. Navigate to the project's root folder in your terminal.  
3. Run:

```bash
python src/predict.py --input data/raw/new_data.csv --output data/processed/my_predictions.csv
```

The script will generate a new CSV file containing the original data along with:

- `predicted_class`  
- `fraud_probability`  

---

## ⚙️ A Note on Preprocessing

Feature engineering steps are defined in `src/feature_engineering.py`. The included features (`hour`, `log_amount`, `business_hours`) are based on initial analysis and are designed as a strong starting point.

You can modify this file to add or change features. If you do, **retrain the model** to ensure it can leverage your new features.
