import streamlit as st
import pandas as pd
import joblib

# Load the model [54:20]
model = joblib.load('extra_trees_credit_model.pkl')

# Load the Label Encoders [54:40]
# Note: 'Job' is excluded as it's a numeric input (0-3) in the app,
# and 'Purpose' was excluded from the final features.
encoders = {
    col: joblib.load(f'{col}_encoder.pkl')
    for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account']
}

# --- Streamlit App Layout ---
st.title('Credit Risk Prediction App')
st.write('Enter applicant information to predict if the credit risk is good or bad.')

# --- Input Fields [55:45] ---
age = st.number_input('Age', min_value=18, max_value=80, value=30)
sex = st.selectbox('Sex', ['male', 'female'])
job = st.number_input('Job (0-3)', min_value=0, max_value=3, value=1)
housing = st.selectbox('Housing', ['own', 'rent', 'free'])
saving_accounts = st.selectbox(
    'Saving accounts', ['little', 'moderate', 'rich', 'quite rich'])
checking_account = st.selectbox(
    'Checking account', ['little', 'moderate', 'rich'])
credit_amount = st.number_input('Credit amount', min_value=0, value=1000)
duration = st.number_input('Duration (months)', min_value=1, value=12)


# --- Prediction Logic [01:01:03] ---
if st.button('Predict Risk'):

    # Prepare input data for the model [01:01:03]
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': encoders['Sex'].transform([sex])[0],
        'Job': [job],
        'Housing': encoders['Housing'].transform([housing])[0],
        'Saving accounts': encoders['Saving accounts'].transform([saving_accounts])[0],
        'Checking account': encoders['Checking account'].transform([checking_account])[0],
        'Credit amount': [credit_amount],
        'Duration': [duration]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.success('The predicted credit risk is **Good**')
    else:
        st.error('The predicted credit risk is **Bad**')

# To run the app, save this file as app.py and run the command:
# streamlit run app.py
