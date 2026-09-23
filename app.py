import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/student_performance_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)


# Title
st.title("🎓 Student Performance Predictor")
st.write(
    "Enter the student's academic information to predict their expected final score."
)


st.divider()


# Input fields
st.subheader("📊 Student Information")

study_hours = st.slider(
    "📚 Study Hours per Day",
    min_value=0,
    max_value=12,
    value=4
)

attendance = st.slider(
    "🏫 Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.slider(
    "📝 Previous Exam Score",
    min_value=0,
    max_value=100,
    value=65
)

assignment_score = st.slider(
    "📖 Assignment Score",
    min_value=0,
    max_value=100,
    value=70
)

sleep_hours = st.slider(
    "😴 Sleep Hours per Day",
    min_value=0,
    max_value=12,
    value=7
)

extracurricular = st.selectbox(
    "🏆 Participates in Extracurricular Activities?",
    ["No", "Yes"]
)


# Convert Yes/No to 0/1
extracurricular_value = 1 if extracurricular == "Yes" else 0


st.divider()


# Prediction button
if st.button("🔮 Predict Performance", use_container_width=True):

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_score": [previous_score],
        "assignment_score": [assignment_score],
        "sleep_hours": [sleep_hours],
        "extracurricular": [extracurricular_value]
    })

    prediction = model.predict(input_data)[0]

    # Keep score between 0 and 100
    prediction = max(0, min(100, prediction))

    st.subheader("🎯 Prediction Result")

    st.metric(
        "Predicted Final Score",
        f"{prediction:.1f}/100"
    )

    # Performance category
    if prediction >= 85:
        category = "🌟 Excellent"
        message = "Great performance! Keep maintaining your current study habits."

    elif prediction >= 70:
        category = "🟢 Good"
        message = "Good performance. A little more consistency can improve your score."

    elif prediction >= 50:
        category = "🟡 Average"
        message = "There is room for improvement. Focus on attendance and study time."

    else:
        category = "🔴 Needs Improvement"
        message = "Consider increasing study time, attendance and assignment performance."

    st.write("### Performance Level")
    st.success(category)

    st.info(message)