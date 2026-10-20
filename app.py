import streamlit as st
import pandas as pd
import joblib

# Load trained model and fitted preprocessor
model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Prediction")
st.write(
    "Predict whether a customer is likely to subscribe "
    "to a term deposit using a Random Forest classifier."
)

st.divider()

with st.form("customer_form"):
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )
        job = st.selectbox(
            "Job",
            [
                "admin.", "blue-collar", "entrepreneur",
                "housemaid", "management", "retired",
                "self-employed", "services", "student",
                "technician", "unemployed", "unknown"
            ]
        )
        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced"]
        )
        education = st.selectbox(
            "Education",
            ["secondary", "tertiary", "primary", "unknown"]
        )
        default = st.selectbox(
            "Credit in Default", ["no", "yes"]
        )
        balance = st.number_input(
            "Average Yearly Balance", value=1000
        )
        housing = st.selectbox(
            "Housing Loan", ["yes", "no"]
        )
        loan = st.selectbox(
            "Personal Loan", ["no", "yes"]
        )

    with col2:
        contact = st.selectbox(
            "Contact Type",
            ["cellular", "telephone", "unknown"]
        )
        day = st.number_input(
            "Last Contact Day",
            min_value=1, max_value=31, value=15
        )
        month = st.selectbox(
            "Last Contact Month",
            [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ]
        )
        duration = st.number_input(
            "Last Call Duration (seconds)",
            min_value=0, value=180
        )
        campaign = st.number_input(
            "Contacts During This Campaign",
            min_value=1, value=1
        )
        pdays = st.number_input(
            "Days Since Previous Contact",
            min_value=-1, value=-1
        )
        previous = st.number_input(
            "Previous Contacts",
            min_value=0, value=0
        )
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "failure", "other", "success"]
        )

    submitted = st.form_submit_button(
        "Predict Subscription",
        use_container_width=True
    )

if submitted:
    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    try:
        # Apply the same preprocessing used during training
        customer_processed = preprocessor.transform(customer)

        # Predict the class
        prediction = model.predict(customer_processed)[0]

        # Get probability for class 1 (subscription = yes)
        probabilities = model.predict_proba(customer_processed)[0]
        class_index = list(model.classes_).index(1)
        probability = probabilities[class_index] * 100

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "The model predicts that the customer "
                "is likely to subscribe."
            )
        else:
            st.warning(
                "The model predicts that the customer "
                "is unlikely to subscribe."
            )

        st.metric(
            "Estimated Subscription Probability",
            f"{probability:.2f}%"
        )

        st.caption(
            "Predictions are estimates and do not guarantee "
            "actual customer behavior."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")