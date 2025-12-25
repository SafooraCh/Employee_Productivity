# ============================================================================
# Streamlit App for Employee Productivity Prediction
# Name: SAFOORA | Roll No: 2330-0022
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go

# ------------------ 1️⃣ PAGE CONFIGURATION ------------------
st.set_page_config(
    page_title="Employee Productivity Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Employee Productivity Prediction System")
st.markdown("### AI-Powered Productivity Classification | By Safoora (2330-0022)")
st.markdown("---")

# ------------------ 2️⃣ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    try:
        with open('logistic_regression_model.pkl', 'rb') as f:
            log_reg = pickle.load(f)
        with open('random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('knn_model.pkl', 'rb') as f:
            knn_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            target_le = pickle.load(f)
        return log_reg, rf_model, knn_model, scaler, target_le
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure all .pkl files are in the app directory.")
        return None, None, None, None, None

log_reg, rf_model, knn_model, scaler, target_le = load_models()

# ------------------ 3️⃣ SIDEBAR ------------------
with st.sidebar:
    st.header("📋 Project Information")
    st.info("""
    **Project:** Employee Productivity Prediction
    
    **Student:** Safoora
    
    **Roll No:** 2330-0022
    
    **Models Used:**
    - Logistic Regression
    - Random Forest
    - K-Nearest Neighbors
    """)
    st.markdown("---")
    st.header("🎯 Select Model")
    model_choice = st.radio(
        "Choose the model for prediction:",
        ["Random Forest (Recommended)", "Logistic Regression", "KNN"]
    )

# ------------------ 4️⃣ INPUT FORM ------------------
st.subheader("🛠️ Input Employee Features")

# Replace these features with actual columns from your dataset
# Example numeric features:
col1, col2, col3 = st.columns(3)

with col1:
    hours_worked = st.number_input("Hours Worked per Week", min_value=0, max_value=100, value=40)
    experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=3)

with col2:
    tasks_completed = st.number_input("Tasks Completed", min_value=0, max_value=200, value=50)
    age = st.number_input("Employee Age", min_value=18, max_value=65, value=30)

with col3:
    overtime_hours = st.number_input("Overtime Hours", min_value=0, max_value=50, value=5)
    # Add more features if needed

# ------------------ 5️⃣ PREDICTION ------------------
if st.button("🔮 Predict Productivity", type="primary"):
    if log_reg and rf_model and knn_model and scaler and target_le:
        # Prepare input
        input_data = np.array([[hours_worked, experience, tasks_completed, age, overtime_hours]])
        input_scaled = scaler.transform(input_data)

        # Predict
        if model_choice == "Random Forest (Recommended)":
            pred = rf_model.predict(input_scaled)[0]
            prob = rf_model.predict_proba(input_scaled)[0]
        elif model_choice == "Logistic Regression":
            pred = log_reg.predict(input_scaled)[0]
            prob = log_reg.predict_proba(input_scaled)[0]
        else:
            pred = knn_model.predict(input_scaled)[0]
            # KNN predict_proba may fail if classes are not probability enabled
            try:
                prob = knn_model.predict_proba(input_scaled)[0]
            except:
                prob = [0,1] if pred==1 else [1,0]

        pred_label = target_le.inverse_transform([pred])[0]

        # ------------------ 6️⃣ DISPLAY RESULTS ------------------
        col1, col2 = st.columns([2,3])

        with col1:
            st.markdown(f"""
                <div style='background-color: #1e3a8a; padding: 2rem; 
                border-radius: 1rem; text-align: center;'>
                    <h1 style='color: white; margin: 0;'>💼 {pred_label}</h1>
                    <p style='color: #93c5fd; font-size: 1.2rem; margin-top: 1rem;'>
                        Predicted Employee Productivity
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.subheader("📊 Prediction Confidence")
        fig = go.Figure(go.Bar(
            x=target_le.classes_,
            y=[p*100 for p in prob],
            text=[f'{p*100:.1f}%' for p in prob],
            textposition='auto',
            marker_color=['#10b981', '#3b82f6']
        ))
        fig.update_layout(
            title="Probability Distribution",
            yaxis_title="Probability (%)",
            xaxis_title="Productivity Class",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Input summary
        st.subheader("🔍 Input Summary")
        summary_data = {
            'Feature': ['Hours Worked', 'Experience', 'Tasks Completed', 'Age', 'Overtime Hours'],
            'Value': [hours_worked, experience, tasks_completed, age, overtime_hours]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280;'>
    <p>Employee Productivity Prediction System | AI Semester Project | Safoora (2330-0022)</p>
    <p>Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
