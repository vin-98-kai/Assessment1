# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:19:40 2022

@author: Calvin
"""

import os
import pickle
import numpy as np
import streamlit as st

# Static
MODEL_PATH = os.path.join(os.getcwd(),'model', 'best_pipe.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

#%% Form Creation with Streamlit

with st.form("Risk of CVDs"):
    age = st.number_input('Age') # assign input to variables
    sex = st.number_input('Sex')
    cp = st.number_input('Chestpain')
    trtbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Cholesterol')
    fbs = st.number_input('Fasting Blood Sugar')
    restecg = st.number_input('Resting Electrocardiographic Results')
    thalachh = st.number_input('Maximum Heart Rate Achieved')
    exng = st.number_input('Exercise Induced Angina')
    oldpeak = st.number_input('Old Peak')
    slp = st.number_input('Slope')
    caa = st.number_input('No. of Major Vessels')
    thall = st.number_input('Thallasemia')
     
    #Every form must have a submit button
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,
                 caa,thall] # create list and put the variable inside
        outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
        outcome_dict = {0:'Less risk of heart disease',
                        1:'High risk of heart disease'}
        st.write(outcome_dict[outcome[0]])