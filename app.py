
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('best_churn_pipeline.pkl')

st.title('Customer Churn Prediction Dashboard')

age = st.slider('Age', 18, 80, 35)
credit_score = st.slider('Credit Score', 300, 900, 650)
balance = st.number_input('Balance', 0.0, 250000.0, 50000.0)
products = st.selectbox('Products Number', [1,2,3,4])
active = st.selectbox('Active Member', [0,1])

if st.button('Predict'):
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
        'balance_per_product': balance / products,
        'salary_balance_ratio': 60000 / max(balance,1),
        'age_group': '35-44',
        'tenure_bucket': '3-5',
        'high_balance': int(balance > 50000)
    }])

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0,1]

    if st.button('Predict', key='predict_button'):
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
        'balance_per_product': balance / products,
        'salary_balance_ratio': 60000 / max(balance,1),
        'age_group': '35-44',
        'tenure_bucket': '3-5',
        'high_balance': int(balance > 50000)
    }])

    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0,1]

    st.write('Prediction:', 'Churn' 
             if pred == 1 else 'No Churn')
    st.write('Probability:', round(prob, 3))

    st.subheader('Customer Risk Analysis')

    if prob < 0.30:
        st.success('Low Risk Customer')
        st.write('Recommendation: Maintain normal communication and offers.')

    elif prob < 0.70:
        st.warning('Medium Risk Customer')
        st.write('Recommendation: Give discount, cashback or special service.')

    else:
        st.error('High Risk Customer')

        reasons = []

        if age > 50:
            reasons.append('Customer age is high')

        if credit_score < 500:
            reasons.append('Credit score is low')

        if products == 1:
            reasons.append('Customer uses only one product')

        if active == 0:
            reasons.append('Customer is not an active member')

        st.write('Why customer may churn:')
        for r in reasons:
            st.write('-', r)

        st.write('Recommendation:')
        st.write('- Assign relationship manager')
        st.write('- Offer cashback or loan benefits')
        st.write('- Contact customer within 24 hours')
