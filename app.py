import streamlit as st
import joblib
import numpy as np
import os

st.title("Employee Performance Predictor")

# Correct model path (important fix)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl"))

st.write("Model path:", model_path)  # debug

if not os.path.exists(model_path):
    st.error("❌ Model not found! Run main.py first.")
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")

        # Inputs
        age = st.slider("Age", 20, 60)
        exp = st.slider("Experience", 1, 20)
        salary = st.slider("Salary", 20000, 100000)
        training = st.slider("Training Hours", 10, 100)
        projects = st.slider("Projects", 1, 10)

        dept = st.selectbox("Department", ["HR", "IT", "Sales"])
        dept_map = {"HR": 0, "IT": 1, "Sales": 2}

        if st.button("Predict"):
            data = np.array([[age, exp, dept_map[dept], salary, training, projects, 3]])
            prediction = model.predict(data)

            st.success(f"🎯 Predicted Performance: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")