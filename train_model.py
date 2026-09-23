import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("data/student_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# Features and target
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


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train
model.fit(X_train, y_train)

print("Model training completed!")


# Predictions
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 2))


# Feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# Save model
joblib.dump(
    model,
    "models/student_performance_model.pkl"
)

print("\nModel saved successfully!")
print("Feature importance chart saved successfully!")
print(importance)