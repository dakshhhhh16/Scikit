# Customer Churn Prediction Using Logistic Regression

A machine learning project that predicts whether a customer will churn (leave) or stay with a telecom company based on their service usage and demographic information.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [How the Project Works](#how-the-project-works)
- [Code Explanation](#code-explanation)
  - [Training Notebook (app.ipynb)](#training-notebook-appipynb)
  - [Streamlit App (lg_app.py)](#streamlit-app-lg_apppy)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)

---

## Overview

**Customer Churn** refers to when customers stop doing business with a company. This project uses **Logistic Regression** (a classification algorithm) to predict whether a customer will churn based on features like:

- Gender, age (senior citizen status)
- Partner and dependent status
- Tenure (how long they've been a customer)
- Phone service and multiple lines
- Contract type
- Total charges

This is a **supervised learning** problem because we have labeled data (the `Churn` column tells us Yes/No for each customer).

It's a **classification** problem (not regression) because we're predicting a category (Churn or Not Churn), not a continuous number.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Machine learning (Logistic Regression, preprocessing, metrics) |
| **Pickle** | Saving/loading the trained model |
| **Streamlit** | Building the interactive web application |
| **VS Code Notebooks** | Training and experimentation environment |

---

## Project Structure

```
Churn_Prediction/
├── app.ipynb                 # Jupyter notebook for training the model
├── lg_app.py                 # Streamlit web application for predictions
├── Churn_prediction.pkl      # Saved trained model
├── README.md                 # This documentation file
└── ../Sample_Data/
    └── churn.csv             # Dataset (in parent folder)
```

---

## Dataset

The dataset (`churn.csv`) contains customer information from a telecom company. Key columns used:

| Column | Description | Type |
|--------|-------------|------|
| `customerID` | Unique customer identifier | String (dropped before training) |
| `gender` | Male or Female | Categorical |
| `SeniorCitizen` | Whether customer is a senior citizen (0/1) | Binary |
| `Partner` | Whether customer has a partner | Yes/No |
| `Dependents` | Whether customer has dependents | Yes/No |
| `tenure` | Number of months with the company | Numeric |
| `PhoneService` | Whether customer has phone service | Yes/No |
| `MultipleLines` | Multiple phone lines status | Yes/No/No phone service |
| `Contract` | Contract type | Month-to-month/One year/Two year |
| `TotalCharges` | Total amount charged to customer | Numeric |
| `Churn` | **Target variable** - Did customer leave? | Yes/No |

---

## How the Project Works

### Two-Stage Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: TRAINING (app.ipynb)                                  │
│                                                                 │
│  Load Data → Clean → Encode → Split → Scale → Train → Save     │
│                                                    ↓            │
│                                          Churn_prediction.pkl   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: DEPLOYMENT (lg_app.py)                                │
│                                                                 │
│  Load Model → User Input → Encode → Scale → Predict → Display  │
└─────────────────────────────────────────────────────────────────┘
```

1. **Training Phase**: The Jupyter notebook trains a Logistic Regression model and saves it as a `.pkl` file
2. **Deployment Phase**: The Streamlit app loads the saved model and makes predictions on new customer data

---

## Code Explanation

### Training Notebook (app.ipynb)

#### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
```
- **NumPy**: For numerical operations
- **Pandas**: For loading and manipulating the dataset

#### Step 2: Load Dataset
```python
df = pd.read_csv("../Sample_Data/churn.csv")
df.head()
```
- Loads the CSV file into a DataFrame
- `df.head()` displays the first 5 rows to verify the data

#### Step 3: Check for Missing Values
```python
df.isnull().sum()
```
- Counts null values in each column
- Important for data quality assessment

#### Step 4: Feature Engineering (Select Important Columns)
```python
columns_to_keep = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 
                   'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
                   'Contract', 'TotalCharges', 'Churn']
df = df[columns_to_keep]
```
- Keeps only the columns that are relevant for prediction
- Reduces noise and improves model performance

#### Step 5: Clean TotalCharges Column
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
```
- `TotalCharges` contains some blank values (stored as strings)
- `pd.to_numeric(..., errors='coerce')` converts to numbers, turning invalid values to NaN
- `fillna(mean)` replaces missing values with the column average

#### Step 6: Encode Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                    'MultipleLines', 'Contract', 'Churn']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
```
- **Why encode?** Machine learning models need numbers, not text
- **LabelEncoder** converts categories to numbers:
  - `Male/Female` → `1/0`
  - `Yes/No` → `1/0`
  - `Month-to-month/One year/Two year` → `0/1/2`

#### Step 7: Split Features and Target
```python
X = df.drop(['Churn', 'customerID'], axis=1)  # Features
y = df['Churn']                                # Target
```
- **X (Features)**: All columns except `Churn` and `customerID`
- **y (Target)**: The `Churn` column (what we want to predict)
- `customerID` is dropped because it's just an identifier, not a predictive feature

#### Step 8: Train-Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Splits data into **80% training** and **20% testing**
- `random_state=42` ensures reproducible results
- Training data teaches the model; test data evaluates it

#### Step 9: Standardize Features
```python
from sklearn.preprocessing import StandardScaler

X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
- **Why standardize?** Features have different scales (e.g., tenure: 1-72, TotalCharges: 0-8000+)
- **StandardScaler** transforms features to have:
  - Mean = 0
  - Standard Deviation = 1
- `fit_transform` on training data, `transform` on test data (to avoid data leakage)

#### Step 10: Train Logistic Regression Model
```python
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
```
- **Logistic Regression** is ideal for binary classification (Churn/Not Churn)
- `fit()` trains the model on the training data
- `predict()` generates predictions on the test data

#### Step 11: Evaluate Model Accuracy
```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```
- Compares predictions (`y_pred`) with actual values (`y_test`)
- Returns accuracy as a percentage (e.g., 0.775 = 77.5%)

#### Step 12: Save the Model
```python
import pickle
pickle.dump(lg, open('Churn_prediction.pkl', 'wb'))
```
- **Pickle** serializes the trained model to a file
- `'wb'` = write binary mode
- This saved model is loaded by the Streamlit app

#### Step 13: Prediction Function (for testing)
```python
def prediction(gender, Seniorcitizen, Partner, Dependents, tenure, 
               Phoneservice, multiline, contact, totalcharge):
    data = {
        'gender': [gender],
        'SeniorCitizen': [Seniorcitizen],
        # ... other features
    }
    df = pd.DataFrame(data)
    
    # Encode categorical columns
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    
    # Scale features
    df = scaler.fit_transform(df)
    
    # Predict
    result = lg.predict(df).reshape(1, -1)
    return result[0]
```
- Creates a reusable function to make predictions on new customer data
- Follows the same preprocessing steps as training

---

### Streamlit App (lg_app.py)

#### Loading the Model and Data
```python
import streamlit as st
import pickle
import os

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'Churn_prediction.pkl')
data_path = os.path.join(os.path.dirname(base_dir), 'Sample_Data', 'churn.csv')

model = pickle.load(open(model_path, 'rb'))
df = pd.read_csv(data_path)
```
- Uses `os.path` for robust file path handling
- Loads the pre-trained model from the pickle file
- `'rb'` = read binary mode

#### Creating the User Interface
```python
st.title("Customer Churn Prediction Using Logistic Regression")

gender = st.selectbox("Select Gender", options=['Female', 'Male'])
SeniorCitizen = st.selectbox("Are you a senior citizen?", options=['Yes', 'No'])
Partner = st.selectbox("Do you have a partner?", options=['Yes', 'No'])
# ... more input fields

tenure = st.text_input("Enter Your tenure?")
TotalCharges = st.text_input("Enter your Total charges?")
```
- `st.selectbox()` creates dropdown menus for categorical inputs
- `st.text_input()` creates text fields for numeric inputs
- Streamlit automatically creates a clean web interface

#### Making Predictions
```python
if st.button("Predict churn or not"):
    result = prediction(gender, SeniorCitizen, Partner, ...)
    
    if result == 1:
        st.title("Churn")
        st.dataframe(churn_tips_df, use_container_width=True)
    else:
        st.title("Not Churn")
        st.dataframe(retention_tips_df, use_container_width=True)
```
- Button triggers the prediction
- Displays "Churn" or "Not Churn" based on model output
- Shows relevant tips for either preventing churn or retaining customers

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn streamlit
```

### Step 1: Train the Model (if needed)
1. Open `app.ipynb` in VS Code or Jupyter
2. Run all cells from top to bottom
3. This creates `Churn_prediction.pkl`

### Step 2: Run the Streamlit App
```bash
cd "/path/to/Machine Learning"
streamlit run Churn_Prediction/lg_app.py
```

### Step 3: Use the App
1. Open the URL shown in terminal (usually `http://localhost:8501`)
2. Fill in customer details using the dropdowns and text fields
3. Click "Predict churn or not"
4. View the prediction and actionable tips

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | Logistic Regression |
| **Train/Test Split** | 80% / 20% |
| **Accuracy** | ~77.5% |

### Possible Improvements
- Try other algorithms (Random Forest, XGBoost)
- Feature engineering (create new features from existing ones)
- Handle class imbalance (if churn is rare)
- Hyperparameter tuning
- Cross-validation for more robust evaluation

---

## Key Concepts Explained

### What is Logistic Regression?
Despite its name, Logistic Regression is used for **classification**, not regression. It predicts the probability that an instance belongs to a particular class (0 or 1).

The model outputs a probability between 0 and 1:
- If probability > 0.5 → Predict "Churn" (1)
- If probability ≤ 0.5 → Predict "Not Churn" (0)

### Why Use StandardScaler?
Features with larger values can dominate the model. Standardization ensures all features contribute equally:

| Feature | Before Scaling | After Scaling |
|---------|----------------|---------------|
| tenure | 1 - 72 | -1.5 to +2.0 |
| TotalCharges | 0 - 8000+ | -1.0 to +3.0 |

### What is Label Encoding?
Converts text categories to numbers:
```
"Male" → 1, "Female" → 0
"Yes" → 1, "No" → 0
"Month-to-month" → 0, "One year" → 1, "Two year" → 2
```

---

## Author

Built as a learning project for Machine Learning with Python.

---

*Happy Learning! 🚀*
