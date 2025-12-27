import streamlit as st
import numpy as np
import pickle
import sklearn
import pandas as pd

model = pickle.load(open('linear_regression_model.pkl','rb'))

st.title('Advertising Sales Prediction App (Linear-Regression)')

tv = st.text_input ('Enter TV sales...')
radio = st.text_input ('Enter Radio sales...')
newspaper = st.text_input ('Enter Newspaper sales...')

if st.button ('Predict Sales'):
    features = np.array([[tv, radio, newspaper]], dtype = np. float64)
    results = model.predict(features).reshape(1,-1)
    st.write ("Predicted sales is: ", results[0])