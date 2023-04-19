import streamlit as st
import joblib as jb
import numpy as np

st.title('Insurance Claim Rate Predictor')
st.image("""https://img.freepik.com/free-photo/cute-family-concept-arrangement-with-copy-space_23-2148485832.jpg?w=1060&t=st=1681572511~exp=1681573111~hmac=35bf4e88b23c3ebe88de14872b3c9626bab4991bc89f79c57206e244ae1694d9""")
st.header('Enter the details of the insurance holder:')

placeholder = st.empty()

age = st.number_input('Age of the person:', min_value=10, max_value=100, value=20)

sex = st.selectbox('Gender:', ['Male', 'Female'])

diabetic_patient = st.selectbox('Are you Diabetic:', ['Yes', 'No'])

smoker = st.selectbox('Are you smoker:', ['Yes', 'No'])

bmi = st.number_input('Body Mas Index:', min_value=16.0, max_value=54.0, value=18.9)

blood_pressure = st.number_input('Blood Pressure rate:', min_value=80, max_value=140, value=120)

region = st.selectbox('Select your region:', ['southeast', 'northwest', 'southwest', 'northeast'])

if sex == 'Male':
    gender = 1
else:
    gender = 0

if smoker == 'Yes':
    smoke = 1
else:
    smoke = 0

if diabetic_patient == 'Yes':
    diabetic = 1
else:
    diabetic = 0

if region == 'southeast':
    reg = 0
elif region == 'northwest':
    reg = 1
elif region == 'southwest':
    reg = 2
else:
    reg = 3

features = [age, gender, bmi, blood_pressure, diabetic, smoke, reg]

int_features = [int(x) for x in features]

min_scaler = jb.load("model_scaler.pkl")
loaded_model = jb.load("gradient_model.pkl")

final_features = min_scaler.transform([int_features])

if st.button('Predict'):
    placeholder.empty()
    prediction = loaded_model.predict(final_features)
    st.balloons()
    st.success(f'Your insurance charges would be: Rs:{round(prediction[0],2)}')
    
