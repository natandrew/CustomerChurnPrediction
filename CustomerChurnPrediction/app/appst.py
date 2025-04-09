#Streamlit App(can be run from the terminal with : streamlit run appst.py)
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
lda = joblib.load('lda.pkl')

st.title("Customer Churn Prediction Dashboard")
st.sidebar.header("User Input Features")

subscription_cat = ['Premium', 'Standard']
contract_cat= ['Monthly', 'Quarterly']
num_fields = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

age = st.sidebar.number_input("Age", min_value=0, step=1, format='%d')
tenure = st.sidebar.number_input("Tenure", min_value=0, step=1, format='%d')
usage_frequency = st.sidebar.number_input("Usage Frequency", min_value=0, step=1, format='%d')
support_calls = st.sidebar.number_input("Number of Support Calls", min_value=0, step=1, format='%d')
payment_delay = st.sidebar.number_input("Delay of Payment", min_value=0, step=1, format='%d')
total_spend = st.sidebar.number_input("Total Spend", min_value=0, step=10, format='%d')
last_interaction = st.sidebar.number_input("Last Interaction", min_value=0, step=1, format='%d')
gender_male = st.sidebar.selectbox("Male Gender", [False, True], format_func=lambda x: "No" if not x else "Yes")
subscription_type = st.sidebar.selectbox("Subscription Type", ['Basic', 'Standard', 'Premium'])
contract_length = st.sidebar.selectbox("Contract Length", ['Annual', 'Quarterly', 'Monthly'])

data = {
    'Age': age,
    'Tenure': tenure,
    'Usage Frequency': usage_frequency,
    'Support Calls': support_calls,
    'Payment Delay': payment_delay,
    'Total Spend': total_spend,
    'Last Interaction': last_interaction,
    'Gender_Male': gender_male,
    'Subscription Type': subscription_type,
    'Contract Length': contract_length
}

for field in num_fields:
        data[field] = float(data[field])

subscription = data.pop('Subscription Type')
one_hot_subscription = {f'Subscription Type_{cat}': subscription == cat for cat in subscription_cat}
data.update(one_hot_subscription)

contract = data.pop('Contract Length')
one_hot_contract = {f'Contract Length_{cat}': contract == cat for cat in contract_cat}
data.update(one_hot_contract)

input_data = pd.DataFrame([data])

input_data = scaler.transform(input_data)
input_data = lda.transform(input_data)

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    st.subheader("Prediction Results")
    st.write("Churn" if prediction == 1 else "Not Churn")
    st.write(f"Probability of Churn: **{probability * 100:.2f}%**")

    # Visualization
    st.subheader("Churn Probability Visualization")
    st.progress(probability)