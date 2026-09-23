import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

data = pd.read_csv("data/student_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "sleep_hours",
    "extracurricular"
]

X = data[features]
y = data["final_score"]


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# Define models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# --------------------------------------------------
# Train and evaluate models
# --------------------------------------------------

results = []

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"\n{name}")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 2))


# --------------------------------------------------
# Results table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n==============================")
print("RESULTS")
print("==============================")

print(results_df.round(2))


# --------------------------------------------------
# Select model based on RMSE
# --------------------------------------------------

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

best_model = models[best_model_name]

print("\n==============================")
print("BEST MODEL")
print("==============================")

print("Selected model:", best_model_name)


# --------------------------------------------------
# Train selected model on full dataset
# --------------------------------------------------

best_model.fit(X, y)


# --------------------------------------------------
# Save model
# --------------------------------------------------

joblib.dump(
    best_model,
    "models/student_performance_model.pkl"
)

print("\nModel saved successfully!")

print(
    "Saved model:",
    best_model_name
)