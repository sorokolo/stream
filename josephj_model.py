import streamlit as st
from sklearn.linear_model import LogisticRegression 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle


scaler = StandardScaler()

# --- Streamlit app ---
st.title("COVID-19 Risk Prediction")

training_cols = ['USMER_1', 'USMER_2', 'SEX_1', 'SEX_2', 'PATIENT_TYPE_1',
       'PATIENT_TYPE_2', 'PNEUMONIA_1', 'PNEUMONIA_2', 'PREGNANT_1',
       'PREGNANT_2', 'DIABETES_1', 'DIABETES_2', 'COPD_1', 'COPD_2',
       'ASTHMA_1', 'ASTHMA_2', 'INMSUPR_1', 'INMSUPR_2', 'HIPERTENSION_1',
       'HIPERTENSION_2', 'OTHER_DISEASE_1', 'OTHER_DISEASE_2',
       'CARDIOVASCULAR_1', 'CARDIOVASCULAR_2', 'OBESITY_1', 'OBESITY_2',
       'RENAL_CHRONIC_1', 'RENAL_CHRONIC_2', 'TOBACCO_1', 'TOBACCO_2', 'AGE']

# Create user input form
st.sidebar.header("Input Parameters")
user_input = {
    "USMER": st.sidebar.selectbox("USMER", [1, 2]),
    "MEDICAL_UNIT": st.sidebar.selectbox("MEDICAL_UNIT", [1, 2]),
    "SEX": st.sidebar.selectbox("SEX", [1, 2]),
    "PATIENT_TYPE": st.sidebar.selectbox("PATIENT_TYPE", [1, 2]),
    "PNEUMONIA": st.sidebar.selectbox("PNEUMONIA", [1, 2]),
    "PREGNANT": st.sidebar.selectbox("PREGNANT", [1, 2]),
    "DIABETES": st.sidebar.selectbox("DIABETES", [1, 2]),
    "COPD": st.sidebar.selectbox("COPD", [1, 2]),
    "ASTHMA": st.sidebar.selectbox("ASTHMA", [1, 2]),
    "INMSUPR": st.sidebar.selectbox("INMSUPR", [1, 2]),
    "HIPERTENSION": st.sidebar.selectbox("HIPERTENSION", [1, 2]),
    "OTHER_DISEASE": st.sidebar.selectbox("OTHER_DISEASE", [1, 2]),
    "CARDIOVASCULAR": st.sidebar.selectbox("CARDIOVASCULAR", [1, 2]),
    "OBESITY": st.sidebar.selectbox("OBESITY", [1, 2]),
    "RENAL_CHRONIC": st.sidebar.selectbox("RENAL_CHRONIC", [1, 2]),
    "TOBACCO": st.sidebar.selectbox("TOBACCO", [1, 2]),
    "AGE": st.sidebar.slider("AGE", 0, 120, 42),
}


input_df = pd.DataFrame([user_input])
input_df = input_df.drop(columns=["MEDICAL_UNIT"])
user_data = input_df.drop(columns=['AGE'])

user_data_dummies = pd.get_dummies(user_data.astype("category"), drop_first=False).astype(int)

user_data_dummies['AGE'] = input_df['AGE']

user_data_dummies['AGE'] = scaler.fit_transform(user_data_dummies[['AGE']])

for col in training_cols:
    if col not in user_data_dummies:
        user_data_dummies[col] = 0

user_data_dummies = user_data_dummies[training_cols]

with open('model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

prediction = lr_model.predict(user_data_dummies)

if prediction[0] == 1:
    st.write("High Risk")
else:
    st.write("Low Risk")
