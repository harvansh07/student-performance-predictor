import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = joblib.load(
    "models/student_performance_model.pkl"
)


# --------------------------------------------------
# Load model comparison results
# --------------------------------------------------

comparison_path = "models/model_comparison.csv"

try:
    model_results = pd.read_csv(comparison_path)
except FileNotFoundError:
    model_results = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎓 Student Performance Predictor")

st.markdown(
    """
    ### AI/ML Dashboard
    Predict a student's expected final score using academic,
    attendance, study, and lifestyle factors.
    """
)

st.divider()


# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------

st.sidebar.header("📋 Student Information")

study_hours = st.sidebar.slider(
    "Study Hours per Day",
    min_value=0.0,
    max_value=12.0,
    value=4.0,
    step=0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)

previous_score = st.sidebar.slider(
    "Previous Score",
    min_value=0,
    max_value=100,
    value=70
)

assignment_score = st.sidebar.slider(
    "Assignment Score",
    min_value=0,
    max_value=100,
    value=70
)

sleep_hours = st.sidebar.slider(
    "Sleep Hours per Day",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.5
)

extracurricular = st.sidebar.selectbox(
    "Extracurricular Activities",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


# --------------------------------------------------
# Student summary
# --------------------------------------------------

st.subheader("📌 Student Profile")

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Study Hours", study_hours)
col2.metric("Attendance", f"{attendance}%")
col3.metric("Previous Score", previous_score)
col4.metric("Assignment", assignment_score)
col5.metric("Sleep", f"{sleep_hours} hrs")
col6.metric(
    "Activities",
    "Yes" if extracurricular == 1 else "No"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Final Score",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            study_hours,
            attendance,
            previous_score,
            assignment_score,
            sleep_hours,
            extracurricular
        ]],
        columns=[
            "study_hours",
            "attendance",
            "previous_score",
            "assignment_score",
            "sleep_hours",
            "extracurricular"
        ]
    )

    prediction = model.predict(input_data)[0]

    prediction = round(prediction, 2)


    # --------------------------------------------------
    # Performance category
    # --------------------------------------------------

    if prediction >= 85:
        category = "Excellent 🌟"
        recommendation = (
            "Excellent performance! Keep maintaining "
            "your study consistency."
        )

    elif prediction >= 70:
        category = "Good 👍"
        recommendation = (
            "Good performance. Increasing study consistency "
            "and attendance may help further."
        )

    elif prediction >= 50:
        category = "Average 📚"
        recommendation = (
            "There is room for improvement. Focus on study "
            "hours, assignments, and attendance."
        )

    else:
        category = "Needs Improvement ⚠️"
        recommendation = (
            "Consider improving study habits, attendance, "
            "assignments, and sleep routine."
        )


    # --------------------------------------------------
    # Prediction result
    # --------------------------------------------------

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Predicted Final Score",
        f"{prediction}/100"
    )

    result_col2.metric(
        "Performance Category",
        category
    )

    st.info(recommendation)


    # --------------------------------------------------
    # Score comparison
    # --------------------------------------------------

    st.subheader("📊 Score Comparison")

    score_data = pd.DataFrame(
        {
            "Score Type": [
                "Previous Score",
                "Assignment Score",
                "Predicted Final Score"
            ],
            "Score": [
                previous_score,
                assignment_score,
                prediction
            ]
        }
    )

    st.bar_chart(
        score_data.set_index("Score Type")
    )


# --------------------------------------------------
# Model comparison
# --------------------------------------------------

st.divider()

st.header("🤖 Machine Learning Model Comparison")

if model_results is not None:

    st.dataframe(
        model_results.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 RMSE Comparison")

    chart_data = model_results.set_index("Model")[
        ["RMSE"]
    ]

    st.bar_chart(chart_data)

    best_model = model_results.loc[
        model_results["RMSE"].idxmin(),
        "Model"
    ]

    st.success(
        f"🏆 Selected Model: **{best_model}**"
    )

else:

    st.warning(
        "Model comparison results are not available."
    )


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

st.divider()

st.header("🧠 Feature Importance")

feature_names = [
    "Study Hours",
    "Attendance",
    "Previous Score",
    "Assignment Score",
    "Sleep Hours",
    "Extracurricular"
]

if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance
        }
    ).sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.dataframe(
        importance_df.round(3),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Feature importance visualization is not available "
        "for the selected model."
    )


# --------------------------------------------------
# About
# --------------------------------------------------

st.divider()

st.header("ℹ️ About This Project")

st.markdown(
    """
    This project demonstrates an end-to-end machine learning
    workflow for predicting student academic performance.

    **Models compared:**
    - Linear Regression
    - Random Forest Regression
    - Gradient Boosting Regression

    **Selected model:**
    Random Forest Regression based on the lowest RMSE.

    **Dataset:**
    A small synthetic/practice dataset containing 48 student
    records.

    ⚠️ **Important:** The dataset is intended for educational
    and portfolio demonstration purposes. The predictions
    should not be used for real academic decisions.
    """
)