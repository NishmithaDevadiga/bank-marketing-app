import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Marketing Prediction App")
st.write("Predict whether a bank customer will subscribe to a term deposit.")

# 2. Create a quick synthetic/sample dataset directly in code (No CSV needed!)
@st.cache_resource
def train_quick_model():
    # Sample training data representing bank customer features
    data = {
        'age': [30, 45, 28, 55, 39, 50, 32, 41],
        'balance': [1500, 3200, 400, 8500, 1200, 4300, 2100, 5000],
        'duration': [200, 150, 300, 100, 250, 180, 220, 290],
        'campaign': [1, 2, 1, 3, 2, 1, 4, 2],
        'subscribed': [0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    X = df[['age', 'balance', 'duration', 'campaign']]
    y = df['subscribed']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_quick_model()

# 3. Customer Input Form
st.subheader("Enter Customer Details")

with st.form("prediction_form"):
    age = st.slider("Age", 18, 90, 35)
    balance = st.number_input("Account Balance (Euros)", value=1000)
    duration = st.number_input("Last Contact Duration (seconds)", value=250)
    campaign = st.slider("Campaign Contacts", 1, 10, 1)
    
    submitted = st.form_submit_button("Predict Subscription", type="primary")

# 4. Make Prediction
if submitted:
    input_data = pd.DataFrame([[age, balance, duration, campaign]], 
                              columns=['age', 'balance', 'duration', 'campaign'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"🎉 High chance! The customer is likely to subscribe (Probability: {probability * 100:.1f}%)")
    else:
        st.info(f"ℹ️ Low chance. The customer is unlikely to subscribe (Probability: {probability * 100:.1f}%)")