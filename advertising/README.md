# Advertising Sales Prediction Project

## Overview

This project predicts **sales** based on money spent on **TV, Radio, and Newspaper** advertising.

You use **machine learning (Linear Regression)** to learn the relationship between advertising budgets and sales from past data, and then build a **simple web app** where a user can enter new budgets and get a predicted sales value.

Project pieces:
- **Training notebook**: `app.ipynb`
- **Web application**: `lr_app.py`
- **Dataset**: `Sample_Data/advertising.csv`

---

## Tech Stack

**Language**
- Python

**Libraries**
- `pandas` – work with tabular data (like Excel in code)
- `numpy` – numeric arrays and math operations
- `matplotlib`, `seaborn` – plotting and data visualization
- `scikit-learn (sklearn)` – machine learning:
  - `LinearRegression` for the model
  - `train_test_split` for splitting data
  - error metrics like `mean_absolute_error`
- `pickle` – save and load the trained model as a `.pkl` file
- `streamlit` – build a simple web app for user interaction

**Environment / Tools**
- VS Code with Python and Jupyter extensions

You are using VS Code for **both** the notebook and the app, which is equivalent to using Jupyter Notebook + PyCharm separately.

---

## How the Notebook Works (Training Phase)

File: `app.ipynb`

This notebook is your **lab** where you explore the data and train the model.

1. **Import libraries**  
   Import `pandas`, `numpy`, `matplotlib`, `seaborn`, and scikit-learn utilities.

2. **Load the dataset**  
   ```python
   df = pd.read_csv('Sample_Data/advertising.csv')
   df.head()
   ```
   - Each row in the dataset contains:
     - TV budget
     - Radio budget
     - Newspaper budget
     - Sales (actual result)

3. **Visualize relationships**  
   Example:
   ```python
   plt.scatter(df['TV'], df['Sales'])
   ```
   This helps you see how TV spending relates to sales.

4. **Split into training and test sets**  
   ```python
   from sklearn.model_selection import train_test_split
   x_train, x_test, y_train, y_test = train_test_split(
       df.iloc[:, :-1],  # all columns except the last (features)
       df.iloc[:, -1],   # last column (target: Sales)
       test_size=0.2,
       random_state=42
   )
   ```
   - Training set (80%) is used to teach the model.
   - Test set (20%) is used to check how well the model learned.

5. **Train a Linear Regression model**  
   ```python
   from sklearn.linear_model import LinearRegression

   lr = LinearRegression()
   lr.fit(x_train, y_train)
   y_pred = lr.predict(x_test)
   ```
   The model learns a formula like:

   $$
   \text{Sales} = w_1 \cdot TV + w_2 \cdot Radio + w_3 \cdot Newspaper + b
   $$

6. **Evaluate the model**  
   ```python
   from sklearn.metrics import mean_absolute_error

   mean_absolute_error(y_test, y_pred)
   ```
   This tells you, on average, how far off the predictions are from the true sales values.

7. **Create a helper function for predictions**  
   ```python
   def predict_sales(tv_budget, radio_budget, newspaper_budget):
       features = np.array([[tv_budget, radio_budget, newspaper_budget]])
       results = lr.predict(features).reshape(1, -1)
       return results[0]
   ```
   This wraps the model prediction into a simple function you can reuse.

8. **Save the trained model with pickle**  
   ```python
   import pickle
   pickle.dump(lr, open('linear_regression_model.pkl', 'wb'))
   ```
   This creates `linear_regression_model.pkl` – a file containing your trained model, so you **don’t have to retrain it every time**.

---

## How the Streamlit App Works (Prediction Phase)

File: `lr_app.py`

This script is your **user-facing web app** built with Streamlit.

1. **Imports**  
   ```python
   import streamlit as st
   import numpy as np
   import pickle
   import sklearn
   import pandas as pd
   ```

2. **Load the saved model**  
   ```python
   model = pickle.load(open('linear_regression_model.pkl', 'rb'))
   ```
   This loads the trained `LinearRegression` model you created in the notebook.

3. **Set up the app title**  
   ```python
   st.title('Advertising Sales Prediction App (Linear-Regression)')
   ```

4. **User input fields**  
   ```python
   tv = st.text_input('Enter TV sales...')
   radio = st.text_input('Enter Radio sales...')
   newspaper = st.text_input('Enter Newspaper sales...')
   ```
   These show three text boxes where the user types the advertising budgets.

5. **Predict when the button is clicked**  
   ```python
   if st.button('Predict Sales'):
       features = np.array([[tv, radio, newspaper]], dtype=np.float64)
       results = model.predict(features).reshape(1, -1)
       st.write("Predicted sales is: ", results[0])
   ```
   - Takes the inputs from the text boxes.
   - Converts them into a `numpy` array of floats.
   - Calls `model.predict(features)` to get the predicted sales.
   - Shows the result in the app.

In short: the Streamlit app is a **simple interface** to use the model you trained in the notebook.

---

## End-to-End Flow

1. **Train the model** in `app.ipynb` using `advertising.csv`:
   - Load and explore data.
   - Split into train/test sets.
   - Train a `LinearRegression` model.
   - Evaluate performance.
   - Save the trained model to `linear_regression_model.pkl`.

2. **Run the app** with Streamlit using `lr_app.py`:
   - Load `linear_regression_model.pkl` with `pickle`.
   - Ask the user for TV, Radio, Newspaper budgets.
   - Use the model to predict sales.
   - Display the prediction on a web page.

This gives you a complete mini machine learning project pipeline:

**Data → Training (Notebook) → Saved Model → Web App (Streamlit) → Live Predictions**.
