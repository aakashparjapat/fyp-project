import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("best_churn_pipeline.pkl")

model = load_model()

st.title("Customer Churn Prediction Dashboard")

age = st.slider("Age",18,80,35)
credit_score = st.slider("Credit Score",300,900,650)
balance = st.number_input("Balance",0.0,250000.0,50000.0)
products = st.selectbox("Products Number",[1,2,3,4])
active = st.selectbox("Active Member",[0,1])

if st.button("Predict"):

    sample = pd.DataFrame([{
        'credit_score': credit_score,
        'country': 'France',
        'gender': 'Male',
        'age': age,
        'tenure': 3,
        'balance': balance,
        'products_number': products,
        'credit_card': 1,
        'active_member': active,
        'estimated_salary': 60000,
        'balance_per_product': balance/products,
        'salary_balance_ratio': 60000/max(balance,1),
        'age_group': '35-44',
        'tenure_bucket': '3-5',
        'high_balance': int(balance>50000)
    }])

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0,1]

    st.write("Prediction:", "Churn" if pred==1 else "No Churn")
    st.write("Probability:", round(prob,3))
