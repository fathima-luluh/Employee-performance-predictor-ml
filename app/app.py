import streamlit as st
import joblib
import pandas as pd
import os

# Page config (MUST be first Streamlit command)
st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

st.title("👨‍💼 Employee Performance Predictor")

# Model path
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl"))

# Check model
if not os.path.exists(model_path):
    st.error("❌ Model not found! Run main.py first.")

else:
    try:
        model = joblib.load(model_path)
        st.success("✅ Model loaded successfully!")

        # Layout
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 20, 60)
            exp = st.slider("Experience", 1, 20)
            salary = st.slider("Salary", 20000, 100000)

        with col2:
            training = st.slider("Training Hours", 10, 100)
            projects = st.slider("Projects", 1, 10)
            dept = st.selectbox("Department", ["HR", "IT", "Sales"])

        dept_map = {"HR": 0, "IT": 1, "Sales": 2}

        # Predict button
        if st.button("Predict"):
            input_data = pd.DataFrame({
                'age': [age],
                'experience': [exp],
                'department': [dept_map[dept]],
                'salary': [salary],
                'training_hours': [training],
                'projects': [projects],
                'performance_score': [3]
            })

            prediction = model.predict(input_data)[0]

            # Output
            if prediction == "High":
                st.success("🌟 High Performer")
            elif prediction == "Medium":
                st.warning("⚖️ Medium Performer")
            else:
                st.error("⚠️ Low Performer")

        # 📊 Add Graph
        st.subheader("📊 Performance Distribution")
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "employee_data.csv"))
        st.bar_chart(df['performance'].value_counts())

        # 📌 About section
        st.markdown("""
        ### 📌 About Project
        This system predicts employee performance using Machine Learning.

        It helps HR teams:
        - Identify high performers
        - Detect low performers early
        - Improve training decisions
        """)

    except Exception as e:
        st.error(f"❌ Error: {e}")