import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Constants and Configuration ---
MODEL_PATH = 'decision_tree_model.pkl'
st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load the model using Streamlit's cache
@st.cache_resource
def load_model():
    """Loads the pickled model pipeline from the specified path."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.info("Please ensure your 'decision_tree_model.pkl' file (saved from your Jupyter notebook) is in the same directory as this 'app.py' file. If you haven't saved your model yet, please do so.")
        st.stop()
    try:
        with open(MODEL_PATH, 'rb') as file:
            pipeline = pickle.load(file)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        st.stop()

# Load the model globally
model_pipeline = load_model()

# --- Streamlit UI Design ---
st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Enter the customer details below to predict their likelihood of leaving the bank (Exited = 1, Not Exited = 0).")

# Input Form Layout
with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        st.subheader("Personal & Geographic")
        credit_score = st.number_input("Credit Score (e.g., 650)", min_value=300, max_value=850, value=650)
        gender = st.selectbox("Gender", ('Male', 'Female'))
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.slider("Tenure (Years at Bank)", 0, 10, 5)

    # Column 2
    with col2:
        st.subheader("Financial Metrics")
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, format="%.2f")
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0, format="%.2f")
        geography = st.selectbox("Country of Residence", ('France', 'Germany', 'Spain'))

    # Column 3
    with col3:
        st.subheader("Bank Relationship")
        num_of_products = st.selectbox("Number of Products", (1, 2, 3, 4))
        has_cr_card = st.selectbox("Has Credit Card?", ('Yes', 'No'))
        is_active_member = st.selectbox("Is Active Member?", ('Yes', 'No'))
        
        # Convert Yes/No to 0/1 for DataFrame construction
        has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
        is_active_member_val = 1 if is_active_member == 'Yes' else 0

# --- Prediction Logic ---
if st.button("Predict Churn Likelihood", type="primary"):
    
    # 1. Create a DataFrame from the inputs
    # NOTE: The order and naming of columns MUST match the data *before* it enters the ColumnTransformer in the pipeline.
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_val],
        'IsActiveMember': [is_active_member_val],
        'EstimatedSalary': [estimated_salary]
    })

    try:
        # 2. Make Prediction (The pipeline handles all preprocessing/OHE internally)
        prediction = model_pipeline.predict(input_data)[0]
        prediction_proba = model_pipeline.predict_proba(input_data)
        churn_proba = prediction_proba[0][1] # Probability of Exited (Class 1)

        # 3. Display Results
        
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"❌ Customer is **PREDICTED TO CHURN**.")
            st.metric(label="Probability of Churn (Exit)", value=f"{churn_proba:.2%}")
        else:
            st.success(f"✅ Customer is **PREDICTED NOT TO CHURN**.")
            st.metric(label="Probability of Retention (No Exit)", value=f"{1 - churn_proba:.2%}")

        st.markdown(f"*(The model predicts the outcome is {prediction}. A prediction of 1 means the customer is likely to churn.)*")

    except Exception as e:
        st.error(f"An error occurred during prediction. This is often caused by a mismatch between the features expected by the loaded model and the features provided by the app. Error details: {e}")

# Footer for instructions
st.sidebar.markdown("### Deployment Instructions")
st.sidebar.markdown(f"1. **Save Model:** Ensure you have saved your model pipeline to `{MODEL_PATH}` in your Jupyter file.")
st.sidebar.markdown("2. **Place Files:** Make sure both `app.py` and `decision_tree_model.pkl` are in the same folder.")
st.sidebar.markdown("3. **Run App:** Open your terminal in that folder and run: `streamlit run app.py`")
