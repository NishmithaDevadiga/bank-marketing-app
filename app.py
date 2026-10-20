
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("Bank Marketing Prediction")
st.write(
    "Predict whether a bank customer will subscribe "
    "to a term deposit."
)


# --------------------------------------------------
# 2. LOAD DATASET: bank.csv
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("bank.csv", sep=None, engine="python")


try:
    df = load_data()

except Exception as e:
    st.error(
        "Could not load bank.csv. Keep bank.csv in the "
        "same folder as app.py."
    )
    st.code(str(e))
    st.stop()


# Clean column names.
df.columns = df.columns.astype(str).str.strip()

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)


# --------------------------------------------------
# 3. DATASET INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", int(df.isna().sum().sum()))

with st.expander("View Dataset Details"):
    st.write("Column Names")
    st.write(df.columns.tolist())

    st.write("Descriptive Statistics")
    st.dataframe(
        df.describe(include="all").fillna(""),
        use_container_width=True
    )

    st.write("Missing Values")
    st.dataframe(df.isna().sum().rename("Missing Values"))


# --------------------------------------------------
# 4. FIND THE TARGET COLUMN
# --------------------------------------------------

possible_targets = [
    "y",
    "deposit",
    "subscribed",
    "term_deposit",
    "target"
]

target_column = next(
    (col for col in possible_targets if col in df.columns),
    None
)

if target_column is None:
    st.error(
        "Target column not found. Expected one of: "
        + ", ".join(possible_targets)
    )
    st.write("Available columns:", df.columns.tolist())
    st.stop()


st.success(f"Target column detected: {target_column}")

df = df.dropna(subset=[target_column]).copy()

X = df.drop(columns=[target_column]).copy()

y_raw = (
    df[target_column]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_mapping = {
    "yes": 1,
    "no": 0,
    "1": 1,
    "0": 0,
    "true": 1,
    "false": 0
}

y = y_raw.map(target_mapping)

if y.isna().any():
    st.error(
        "The target column must contain yes/no or 1/0 values."
    )
    st.write("Target values found:")
    st.write(y_raw.value_counts())
    st.stop()

y = y.astype(int)

if y.nunique() != 2:
    st.error("The dataset must contain two target classes.")
    st.stop()

if y.value_counts().min() < 2:
    st.error("Each target class must contain at least two records.")
    st.stop()


# --------------------------------------------------
# 5. PREPARE NUMERIC AND CATEGORICAL FEATURES
# --------------------------------------------------

# Convert numeric-looking text columns to numeric columns.
for column in X.columns:
    if X[column].dtype == "object":
        converted = pd.to_numeric(X[column], errors="coerce")

        if X[column].notna().sum() == converted.notna().sum():
            X[column] = converted

# Remove completely empty input columns.
empty_columns = X.columns[X.isna().all()].tolist()

if empty_columns:
    X.drop(columns=empty_columns, inplace=True)
    st.warning(
        "Removed empty columns: " + ", ".join(empty_columns)
    )

if X.empty or X.shape[1] == 0:
    st.error("No usable input features were found.")
    st.stop()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# --------------------------------------------------
# 6. SPLIT THE DATASET
# --------------------------------------------------

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

except ValueError as e:
    st.error("Unable to split the dataset.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# 7. BUILD PREPROCESSING PIPELINES
# --------------------------------------------------

def create_preprocessor(numeric_cols, categorical_cols):

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append(
            ("numeric", numeric_pipeline, list(numeric_cols))
        )

    if categorical_cols:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_cols))
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# --------------------------------------------------
# 8. TRAIN THE COMPLETE MODEL
# --------------------------------------------------

@st.cache_resource
def train_model(X_train, y_train, numeric_cols, categorical_cols):

    preprocessor = create_preprocessor(
        numeric_cols,
        categorical_cols
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    # Fit preprocessing and classifier together.
    model_pipeline.fit(X_train, y_train)

    return model_pipeline


model = train_model(
    X_train,
    y_train,
    tuple(numeric_columns),
    tuple(categorical_columns)
)


# --------------------------------------------------
# 9. EVALUATE THE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Test Records",
        len(y_test)
    )

with st.expander("Classification Report"):
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Subscription", "Subscription"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )

with st.expander("Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    st.write(
        pd.DataFrame(
            cm,
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"]
        )
    )


# --------------------------------------------------
# 10. CUSTOMER INPUT FORM
# --------------------------------------------------

st.subheader("Predict for a New Customer")
st.write("Enter customer information below.")

with st.form("customer_form"):

    customer_values = {}

    col1, col2 = st.columns(2)

    for index, column in enumerate(X.columns):

        parent_col = col1 if index % 2 == 0 else col2

        with parent_col:

            if column in numeric_columns:

                values = pd.to_numeric(
                    X[column],
                    errors="coerce"
                ).dropna()

                if values.empty:
                    customer_values[column] = st.number_input(
                        column,
                        value=0.0
                    )

                else:
                    minimum = float(values.min())
                    maximum = float(values.max())
                    default = float(values.median())

                    if minimum == maximum:
                        customer_values[column] = st.number_input(
                            column,
                            value=minimum,
                            disabled=True
                        )
                    else:
                        customer_values[column] = st.number_input(
                            column,
                            min_value=minimum,
                            max_value=maximum,
                            value=min(max(default, minimum), maximum)
                        )

            else:

                options = sorted(
                    X[column].dropna().astype(str).unique().tolist()
                )

                if not options:
                    options = ["Unknown"]

                customer_values[column] = st.selectbox(
                    column,
                    options=options
                )

    submitted = st.form_submit_button(
        "Predict Customer Subscription",
        type="primary"
    )


# --------------------------------------------------
# 11. PREDICT CUSTOMER SUBSCRIPTION
# --------------------------------------------------

if submitted:

    customer = pd.DataFrame(
        [customer_values],
        columns=X.columns
    )

    for column in numeric_columns:
        customer[column] = pd.to_numeric(
            customer[column],
            errors="coerce"
        )

    try:
        # The fitted model performs preprocessing automatically.
        # Do not call preprocessor.transform() separately.
        prediction = int(model.predict(customer)[0])

        probabilities = model.predict_proba(customer)[0]
        classes = model.named_steps["classifier"].classes_

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "Prediction: The customer may subscribe "
                "to a term deposit."
            )
        else:
            st.info(
                "Prediction: The customer may not subscribe "
                "to a term deposit."
            )

        st.write("Prediction Probabilities")

        for class_label, probability in zip(classes, probabilities):

            label = (
                "Subscribe (Yes)"
                if class_label == 1
                else "Not Subscribe (No)"
            )

            st.write(f"{label}: {probability * 100:.2f}%")
            st.progress(float(probability))

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Bank Marketing Prediction | Python | Pandas | "
    "Scikit-learn | Streamlit"
)
