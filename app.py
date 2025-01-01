import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.preprocessing import LabelEncoder
import pickle

# Load the model
with open("customer_churn_model.pkl","rb") as f:
    model_data = pickle.load(f)
    
with open("encoders.pkl","rb") as f:
    encoders = pickle.load(f)

loaded_model = model_data['model']
feature_names = model_data['feature_names']

# Initialize encoders for categorical fields
categorical_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Create the Streamlit interface
st.title("Customer Churn Prediction")
st.write("Fill in the details below to predict customer churn:")

# Collect user inputs
def user_input():
    inputs = {}
    inputs['gender'] = st.selectbox("Gender", ['Male', 'Female'])
    inputs['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
    inputs['Partner'] = st.selectbox("Partner", ['Yes', 'No'])
    inputs['Dependents'] = st.selectbox("Dependents", ['Yes', 'No'])
    inputs['tenure'] = st.slider("Tenure (in months)", 0, 72, 12)
    inputs['PhoneService'] = st.selectbox("Phone Service", ['Yes', 'No'])
    inputs['MultipleLines'] = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
    inputs['InternetService'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    inputs['OnlineSecurity'] = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    inputs['OnlineBackup'] = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    inputs['DeviceProtection'] = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    inputs['TechSupport'] = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    inputs['StreamingTV'] = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    inputs['StreamingMovies'] = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    inputs['Contract'] = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    inputs['PaperlessBilling'] = st.selectbox("Paperless Billing", ['Yes', 'No'])
    inputs['PaymentMethod'] = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    inputs['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    inputs['TotalCharges'] = st.number_input("Total Charges", min_value=0.0, value=500.0)
    return inputs

user_inputs = user_input()
input_data = pd.DataFrame([user_inputs])
# Encode categorical fields
for column, encoder in encoders.items():
    input_data[column] = encoder.transform(input_data[column])


# Predict churn
if st.button("Predict Churn"):
    prediction = loaded_model.predict(input_data)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.subheader(f"Prediction: {result}")
