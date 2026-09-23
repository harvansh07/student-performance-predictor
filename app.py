import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("models/student_performance_model.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #666;
}

.card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">🎓 Student Performance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered prediction of student academic performance'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("📊 Student Information")

study_hours = st.sidebar.slider(
    "📚 Study Hours per Day",
    0,
    12,
    4
)

attendance = st.sidebar.slider(
    "🏫 Attendance (%)",
    0,
    100,
    75
)

previous_score = st.sidebar.slider(
    "📝 Previous Score",
    0,
    100,
    65
)

assignment_score = st.sidebar.slider(
    "📖 Assignment Score",
    0,
    100,
    70
)

sleep_hours = st.sidebar.slider(
    "😴 Sleep Hours",
    0,
    12,
    7
)

extracurricular = st.sidebar.selectbox(
    "🏆 Extracurricular Activities",
    ["No", "Yes"]
)

extracurricular_value = (
    1 if extracurricular == "Yes" else 0
)


# --------------------------------------------------
# Main dashboard
# --------------------------------------------------

st.subheader("📋 Student Profile")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Study Hours", f"{study_hours} hrs")

with col2:
    st.metric("Attendance", f"{attendance}%")

with col3:
    st.metric("Previous Score", f"{previous_score}/100")


col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Assignment Score", f"{assignment_score}/100")

with col5:
    st.metric("Sleep", f"{sleep_hours} hrs")

with col6:
    st.metric(
        "Extracurricular",
        "Yes" if extracurricular_value else "No"
    )


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict Student Performance",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "previous_score": [previous_score],
        "assignment_score": [assignment_score],
        "sleep_hours": [sleep_hours],
        "extracurricular": [extracurricular_value]
    })

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))


    # --------------------------------------------------
    # Performance category
    # --------------------------------------------------

    if prediction >= 85:

        category = "🌟 Excellent"
        recommendation = (
            "Excellent performance! Maintain your current "
            "study habits and consistency."
        )

    elif prediction >= 70:

        category = "🟢 Good"
        recommendation = (
            "Good performance. Increasing study consistency "
            "could help you reach an excellent score."
        )

    elif prediction >= 50:

        category = "🟡 Average"
        recommendation = (
            "There is room for improvement. Focus on "
            "attendance, assignments and study hours."
        )

    else:

        category = "🔴 Needs Improvement"
        recommendation = (
            "Consider increasing study hours and improving "
            "attendance and assignment performance."
        )


    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    st.subheader("🎯 Prediction Result")

    result1, result2 = st.columns(2)

    with result1:

        st.metric(
            "Predicted Final Score",
            f"{prediction:.1f}/100"
        )

    with result2:

        st.metric(
            "Performance Level",
            category
        )


    st.info(
        f"💡 **Personalized Recommendation:** "
        f"{recommendation}"
    )


    # --------------------------------------------------
    # Performance chart
    # --------------------------------------------------

    st.subheader("📊 Performance Overview")

    labels = [
        "Previous Score",
        "Assignment Score",
        "Predicted Score"
    ]

    values = [
        previous_score,
        assignment_score,
        prediction
    ]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylim(0, 100)

    ax.set_ylabel("Score")

    ax.set_title("Academic Performance Comparison")

    st.pyplot(fig)


    # --------------------------------------------------
    # Prediction details
    # --------------------------------------------------

    st.subheader("🧠 Prediction Factors")

    factor_data = pd.DataFrame({
        "Factor": [
            "Study Hours",
            "Attendance",
            "Previous Score",
            "Assignment Score",
            "Sleep Hours"
        ],

        "Value": [
            study_hours,
            attendance,
            previous_score,
            assignment_score,
            sleep_hours
        ]
    })

    st.dataframe(
        factor_data,
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# About project
# --------------------------------------------------

st.divider()

st.subheader("ℹ️ About This Project")

st.write(
    """
    Student Performance Predictor is a machine learning project
    that estimates a student's expected final score using academic
    and lifestyle-related factors.

    The project uses a Random Forest Regression model trained on
    student performance data.
    """
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit • GitHub"
)
