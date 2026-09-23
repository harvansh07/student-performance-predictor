import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL & RESULTS
# --------------------------------------------------

model = joblib.load(
    "models/student_performance_model.pkl"
)

model_results = pd.read_csv(
    "models/model_comparison.csv"
)
feature_importance = pd.read_csv(
    "models/feature_importance.csv"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎓 Student Performance Predictor")

st.markdown(
    """
    Predict a student's **final mathematics grade (G3)** 
    using academic, demographic, family and lifestyle factors.
    
    **Machine Learning Model:** Random Forest Regressor
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("👤 Student Information")

st.sidebar.subheader("Basic Information")

school = st.sidebar.selectbox(
    "School",
    ["GP", "MS"]
)

sex = st.sidebar.selectbox(
    "Sex",
    ["F", "M"]
)

age = st.sidebar.slider(
    "Age",
    min_value=15,
    max_value=22,
    value=17
)

address = st.sidebar.selectbox(
    "Address",
    ["U", "R"]
)

famsize = st.sidebar.selectbox(
    "Family Size",
    ["GT3", "LE3"]
)

Pstatus = st.sidebar.selectbox(
    "Parent Cohabitation Status",
    ["A", "T"]
)


# --------------------------------------------------
# EDUCATION
# --------------------------------------------------

st.sidebar.subheader("📚 Education")

Medu = st.sidebar.slider(
    "Mother's Education",
    0,
    4,
    2
)

Fedu = st.sidebar.slider(
    "Father's Education",
    0,
    4,
    2
)

Mjob = st.sidebar.selectbox(
    "Mother's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

Fjob = st.sidebar.selectbox(
    "Father's Job",
    ["teacher", "health", "services", "at_home", "other"]
)

reason = st.sidebar.selectbox(
    "Reason for Choosing School",
    ["home", "reputation", "course", "other"]
)

guardian = st.sidebar.selectbox(
    "Guardian",
    ["mother", "father", "other"]
)


# --------------------------------------------------
# STUDY HABITS
# --------------------------------------------------

st.sidebar.subheader("📖 Study Habits")

traveltime = st.sidebar.slider(
    "Travel Time",
    1,
    4,
    2
)

studytime = st.sidebar.slider(
    "Weekly Study Time",
    1,
    4,
    2
)

failures = st.sidebar.slider(
    "Past Class Failures",
    0,
    4,
    0
)

absences = st.sidebar.number_input(
    "Number of Absences",
    min_value=0,
    max_value=100,
    value=5
)


# --------------------------------------------------
# SUPPORT & ACTIVITIES
# --------------------------------------------------

st.sidebar.subheader("🏫 Support & Activities")

schoolsup = st.sidebar.selectbox(
    "School Support",
    ["yes", "no"]
)

famsup = st.sidebar.selectbox(
    "Family Support",
    ["yes", "no"]
)

paid = st.sidebar.selectbox(
    "Extra Paid Classes",
    ["yes", "no"]
)

activities = st.sidebar.selectbox(
    "Extra-curricular Activities",
    ["yes", "no"]
)

nursery = st.sidebar.selectbox(
    "Attended Nursery School",
    ["yes", "no"]
)

higher = st.sidebar.selectbox(
    "Wants Higher Education",
    ["yes", "no"]
)

internet = st.sidebar.selectbox(
    "Internet Access",
    ["yes", "no"]
)

romantic = st.sidebar.selectbox(
    "In a Romantic Relationship",
    ["yes", "no"]
)


# --------------------------------------------------
# LIFESTYLE
# --------------------------------------------------

st.sidebar.subheader("🌱 Lifestyle")

famrel = st.sidebar.slider(
    "Family Relationship Quality",
    1,
    5,
    4
)

freetime = st.sidebar.slider(
    "Free Time",
    1,
    5,
    3
)

goout = st.sidebar.slider(
    "Going Out With Friends",
    1,
    5,
    3
)

Dalc = st.sidebar.slider(
    "Workday Alcohol Consumption",
    1,
    5,
    1
)

Walc = st.sidebar.slider(
    "Weekend Alcohol Consumption",
    1,
    5,
    1
)

health = st.sidebar.slider(
    "Current Health Status",
    1,
    5,
    3
)


# --------------------------------------------------
# CREATE INPUT DATAFRAME
# --------------------------------------------------

input_data = pd.DataFrame({
    "school": [school],
    "sex": [sex],
    "age": [age],
    "address": [address],
    "famsize": [famsize],
    "Pstatus": [Pstatus],
    "Medu": [Medu],
    "Fedu": [Fedu],
    "Mjob": [Mjob],
    "Fjob": [Fjob],
    "reason": [reason],
    "guardian": [guardian],
    "traveltime": [traveltime],
    "studytime": [studytime],
    "failures": [failures],
    "schoolsup": [schoolsup],
    "famsup": [famsup],
    "paid": [paid],
    "activities": [activities],
    "nursery": [nursery],
    "higher": [higher],
    "internet": [internet],
    "romantic": [romantic],
    "famrel": [famrel],
    "freetime": [freetime],
    "goout": [goout],
    "Dalc": [Dalc],
    "Walc": [Walc],
    "health": [health],
    "absences": [absences]
})


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

st.subheader("📊 Student Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Study Time",
        f"{studytime}/4"
    )

with col2:
    st.metric(
        "Absences",
        absences
    )

with col3:
    st.metric(
        "Past Failures",
        failures
    )

with col4:
    st.metric(
        "Family Support",
        famsup.upper()
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.subheader("🔮 Final Grade Prediction")

if st.button(
    "🚀 Predict Student Performance",
    use_container_width=True
):

    prediction = model.predict(
        input_data
    )[0]

    prediction = max(
        0,
        min(20, prediction)
    )

    prediction = round(
        prediction,
        2
    )


    # Performance category

    if prediction >= 15:
        category = "Excellent 🌟"
    elif prediction >= 12:
        category = "Good 👍"
    elif prediction >= 10:
        category = "Average 📚"
    else:
        category = "Needs Improvement ⚠️"


    # Results

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Final Grade",
            f"{prediction}/20"
        )

    with col2:
        st.metric(
            "Performance Category",
            category
        )


    # Progress bar

    st.progress(
        prediction / 20
    )


    st.success(
        f"Predicted final mathematics grade: **{prediction}/20**"
    )


    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------

    st.subheader("📈 Prediction Visualization")

    chart_data = pd.DataFrame({
        "Metric": [
            "Predicted Grade",
            "Maximum Grade"
        ],
        "Score": [
            prediction,
            20
        ]
    })

    st.bar_chart(
        chart_data.set_index("Metric")
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.subheader("🤖 Model Performance")

st.dataframe(
    model_results.round(2),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# MODEL COMPARISON CHART
# --------------------------------------------------

st.subheader("📊 Model Comparison")

chart_data = model_results.set_index(
    "Model"
)[["MAE", "RMSE"]]

st.bar_chart(
    chart_data
)


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------
# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()

st.subheader("🧠 What Influences the Prediction?")

st.caption(
    "Permutation importance shows how much model performance "
    "changes when each feature is shuffled. Higher values indicate "
    "greater importance to this model on the test data."
)

importance_chart = feature_importance.head(10).copy()

importance_chart = importance_chart.sort_values(
    "Importance",
    ascending=True
)

st.bar_chart(
    importance_chart.set_index("Feature")
)

st.dataframe(
    feature_importance.head(10).round(4),
    use_container_width=True,
    hide_index=True
)
st.divider()

st.subheader("🧠 About This Project")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        **Dataset**

        UCI Student Performance Dataset

        **Target**

        Final Mathematics Grade (G3)

        **Models Tested**

        - Linear Regression
        - Random Forest
        - Gradient Boosting

        **Selected Model**

        Random Forest
        """
    )


with col2:

    st.markdown(
        """
        **Input Categories**

        - Student demographics
        - Education
        - Study habits
        - Family support
        - School support
        - Lifestyle
        - Absences

        **Important**

        This project is an educational machine-learning
        application. Predictions should not be treated
        as guaranteed academic outcomes.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python • Scikit-learn • Pandas • Streamlit"
)