import streamlit as st
import numpy as np
import pickle
import sklearn
import pandas as pd

model = pickle.load(open('linear_regression_model.pkl','rb'))
