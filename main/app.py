import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_churn_model.pkl")

# Page config
st.set_page_config(page_title="SecureBank", layout="wide")

# Custom CSS for bank-style UI
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.navbar {
    background-color: #0a2540;
    padding: 15px;
    color: white;
    font-size: 24px;
    font-weight: bold;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.stButton>button {
    background-color: #0a2540;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown('<div class="navbar">SecureBank – Customer Insights Portal</div>', unsafe_allow_html=True)

st.write("")

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Customer Profile")

    credit_score = st.number_input("Credit Score", 300, 900, 650)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
    num_products = st.number_input("Number of Products", 1, 4, 2)
    has_card = st.selectbox("Has Credit Card", [0, 1])
    is_active = st.selectbox("Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

    # NEW FIELD: Card Type
    card_type = st.selectbox("Card Type", ["Silver", "Gold", "Platinum", "Diamond"])

    predict_btn = st.button("Analyze Customer")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if predict_btn:
        input_data = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
            "Card Type": card_type   # added here
        }])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠ High Risk of Churn")
            st.write("Recommended action: Offer retention incentives.")
        else:
            st.success("✓ Customer Likely to Stay")
            st.write("No immediate action required.")

    st.markdown('</div>', unsafe_allow_html=True)
