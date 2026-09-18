
import streamlit as st
import joblib

model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("churn_preprocessor.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction & Retention Analytics")

st.write(
    "An interactive machine learning application "
    "for customer churn analysis and retention."
)
import pandas as pd

df = pd.read_csv("final_customer_churn_retention_report.csv")
st.subheader("📌 Customer Overview")

total_customers = len(df)
churned_customers = (df["Churn_Prediction"] == 1).sum()
high_risk_customers = (df["Risk_Category"] == "High Risk").sum()
medium_risk_customers = (df["Risk_Category"] == "Medium Risk").sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Predicted Churn", churned_customers)
col3.metric("High Risk", high_risk_customers)
col4.metric("Medium Risk", medium_risk_customers)
st.subheader("📊 Customer Risk Distribution")

risk_counts = df["Risk_Category"].value_counts()

st.bar_chart(risk_counts)
st.subheader("📉 Churn Prediction Distribution")

churn_counts = df["Churn_Prediction"].value_counts()

churn_counts.index = churn_counts.index.map({
    0: "No Churn",
    1: "Churn"
})

st.bar_chart(churn_counts)
st.subheader("📄 Churn Rate by Contract Type")

contract_churn = (
    df.groupby("Contract")["Churn_Prediction"]
    .mean()
    .mul(100)
    .round(2)
)

st.bar_chart(contract_churn)
st.bar_chart(contract_churn)

st.subheader("🌐 Churn Rate by Internet Service")

internet_churn = (
    df.groupby("InternetService")["Churn_Prediction"]
    .mean()
    .mul(100)
    .round(2)
)

st.bar_chart(internet_churn)

st.subheader("💰 Monthly Charges vs Predicted Churn")


charge_bins = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 40, 60, 80, 100, 150],
    labels=["$0–40", "$40–60", "$60–80", "$80–100", "$100+"]
)

charge_churn = (
    df.groupby(charge_bins, observed=True)["Churn_Prediction"]
    .mean()
    .mul(100)
    .round(2)
)

st.bar_chart(charge_churn)
st.subheader("🚨 High-Risk Customers")

high_risk = df[df["Risk_Category"] == "High Risk"].copy()

high_risk = high_risk.sort_values(
    "Churn_Probability",
    ascending=False
)

display_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "Churn_Probability",
    "Risk_Category"
]

st.dataframe(
    high_risk[display_columns].head(20),
    width="stretch"
)
st.subheader("💡 Retention Recommendations")

def get_recommendation(row):
    if row["Risk_Category"] == "High Risk":
        if row["Contract"] == "Month-to-month":
            return "Offer conuse_container_width=Truetract upgrade or retention discount"
        elif row["PaymentMethod"] == "Electronic check":
            return "Offer automatic payment option"
        else:
            return "Contact customer with personalized retention offer"

    elif row["Risk_Category"] == "Medium Risk":
        return "Offer loyalty benefits and service support"

    else:
        return "Maintain regular customer engagement"


high_risk["Retention_Recommendation"] = high_risk.apply(
    get_recommendation,
    axis=1
)

recommendation_columns = [
    "tenure",
    "MonthlyCharges",
    "Contract",
    "PaymentMethod",
    "Churn_Probability",
    "Retention_Recommendation"
]

st.dataframe(
    high_risk[recommendation_columns].head(20),
    
)
st.subheader("📥 Download High-Risk Customer Report")

csv_data = high_risk.to_csv(index=False)

st.download_button(
    label="Download High-Risk Customers CSV",
    data=csv_data,
    file_name="high_risk_customers.csv",
    mime="text/csv"
)
st.subheader("🤖 Predict Customer Churn Risk")

st.write(
    "Enter customer information below and the trained "
    "machine learning model will estimate churn risk."
)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )


if st.button("🔍 Analyze Churn Risk"):

    customer_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    customer_processed = preprocessor.transform(customer_data)

    prediction = model.predict(customer_processed)[0]

    probability = model.predict_proba(customer_processed)[0][1]

    st.write("### 🔎 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.70:
        st.error("🔴 High Churn Risk")
    elif probability >= 0.40:
        st.warning("🟡 Medium Churn Risk")
    else:
        st.success("🟢 Low Churn Risk")

    if prediction == 1:
        st.error("The model predicts that this customer may churn.")
    else:
        st.success("The model predicts that this customer is likely to stay.")

    st.write("### Customer Information")

    st.dataframe(
        customer_data,
        width="stretch"
       
    )
