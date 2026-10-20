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

# 2. Train a quick model with expanded sample features
@st.cache_resource
def train_model():
    data = {
        'balance': [1500, 3200, 400, 8500, 1200, 4300, 2100, 5000],
        'housing': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        'loan': ['no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no'],
        'duration': [200, 150, 300, 100, 250, 180, 220, 290],
        'campaign': [1, 2, 1, 3, 2, 1, 4, 2],
        'subscribed': [0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Convert categorical text to numeric for the random forest model
    df_encoded = pd.get_dummies(df, columns=['housing', 'loan'])
    
    X = df_encoded.drop(columns=['subscribed'])
    y = df_encoded['subscribed']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_columns = train_model()

# 3. Customer Input Form (Matching your friend's style!)
st.subheader("Enter Customer Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        balance = st.number_input("Average Yearly Balance", value=1000)
        housing = st.selectbox("Housing Loan", options=["yes", "no"])
        loan = st.selectbox("Personal Loan", options=["yes", "no"])
        
    with col2:
        duration = st.number_input("Last Contact Duration (seconds)", value=250)
        campaign = st.number_input("Previous Contacts / Campaign", value=1, min_value=1, max_value=10)
    
    submitted = st.form_submit_button("Predict Subscription", type="primary")

# 4. Make Prediction
if submitted:
    # Create input dataframe and encode it identically to training columns
    input_data = pd.DataFrame({
        'balance': [balance],
        'duration': [duration],
        'campaign': [campaign],
        'housing_yes': [1 if housing == 'yes' else 0],
        'housing_no': [1 if housing == 'no' else 0],
        'loan_yes': [1 if loan == 'yes' else 0],
        'loan_no': [1 if loan == 'no' else 0]
    })
    
    # Ensure columns match what the model expects
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"🎉 High chance! The customer is likely to subscribe (Probability: {probability * 100:.1f}%)")
    else:
        st.info(f"ℹ️ Low chance. The customer is unlikely to subscribe (Probability: {probability * 100:.1f}%)")