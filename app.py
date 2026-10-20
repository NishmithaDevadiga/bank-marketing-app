import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Prediction App")
st.write("Predict whether a bank customer will subscribe to a term deposit using complete customer details.")

# 2. Train a self-contained model with all requested features
@st.cache_resource
def train_model():
    data = {
        'age': [30, 45, 28, 55, 39, 50, 32, 41],
        'job': ['management', 'technician', 'blue-collar', 'admin.', 'management', 'services', 'admin.', 'technician'],
        'marital': ['married', 'single', 'single', 'married', 'divorced', 'married', 'single', 'married'],
        'education': ['tertiary', 'secondary', 'secondary', 'tertiary', 'tertiary', 'secondary', 'secondary', 'tertiary'],
        'default': ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
        'balance': [1500, 3200, 400, 8500, 1200, 4300, 2100, 5000],
        'housing': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no'],
        'loan': ['no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no'],
        'contact': ['cellular', 'unknown', 'telephone', 'cellular', 'cellular', 'unknown', 'cellular', 'telephone'],
        'day': [5, 15, 22, 10, 18, 3, 14, 25],
        'month': ['may', 'jun', 'jul', 'aug', 'may', 'jun', 'jul', 'aug'],
        'duration': [200, 150, 300, 100, 250, 180, 220, 290],
        'campaign': [1, 2, 1, 3, 2, 1, 4, 2],
        'pdays': [-1, 120, -1, 90, -1, 150, -1, 30],
        'previous': [0, 2, 0, 1, 0, 3, 0, 2],
        'poutcome': ['unknown', 'failure', 'unknown', 'success', 'unknown', 'other', 'unknown', 'success'],
        'subscribed': [0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    X = df_encoded.drop(columns=['subscribed'])
    y = df_encoded['subscribed']
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X.columns

model, feature_columns = train_model()

# 3. Customer Input Form (Organized into 3 neat columns)
st.subheader("Enter Complete Customer Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Personal Information")
        age = st.slider("Age", 18, 95, 35)
        job = st.selectbox("Job", ["admin.", "blue-collar", "technician", "services", "management", "retired", "entrepreneur", "self-employed", "unemployed", "student", "housemaid", "unknown"])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
        default = st.selectbox("Credit in Default", ["no", "yes"])
        
    with col2:
        st.markdown("### Financial & Loans")
        balance = st.number_input("Average Yearly Balance", value=1500)
        housing = st.selectbox("Housing Loan", ["yes", "no"])
        loan = st.selectbox("Personal Loan", ["no", "yes"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
        
    with col3:
        st.markdown("### Campaign & Call Info")
        day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
        month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        duration = st.number_input("Last Call Duration (seconds)", value=250)
        campaign = st.number_input("Contacts During This Campaign", min_value=1, max_value=50, value=1)
        pdays = st.number_input("Days Since Previous Contact (-1 if none)", value=-1)
        previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "success", "other"])
    
    submitted = st.form_submit_button("Predict Subscription", type="primary")

# 4. Make Prediction
if submitted:
    input_data = pd.DataFrame({
        'age': [age],
        'balance': [balance],
        'day': [day],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'poutcome': [poutcome]
    })
    
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    input_encoded = pd.get_dummies(input_data, columns=categorical_cols)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"🎉 High chance! The customer is likely to subscribe (Probability: {probability * 100:.1f}%)")
    else:
        st.info(f"ℹ️ Low chance. The customer is unlikely to subscribe (Probability: {probability * 100:.1f}%)")