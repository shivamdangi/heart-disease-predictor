import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Split the data into features and target
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Streamlit app
st.title('Heart Disease Prediction')
st.write("This app predicts if a person has heart disease based on their medical attributes.")

# Input fields for user to provide data
age = st.number_input('Age', min_value=1, max_value=120, value=41)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=130)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=204)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=172)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.4)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3, 4])
thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

# Button to submit and predict
if st.button('Predict'):
    # Create a numpy array with the input data
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_nparr = np.asarray(input_data)
    input_data_reshape = input_data_nparr.reshape(1, -1)

    # Predict the output
    predict_output = model.predict(input_data_reshape)

    # Display the result
    if predict_output[0] == 0:
        st.write('The person does not have heart disease.')
    else:
        st.write('The person has heart disease.')
