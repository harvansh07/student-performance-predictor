import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load dataset
data = pd.read_csv("data/student_data.csv")

print("Dataset loaded successfully!")
print("Number of students:", len(data))


# 2. Select features
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


# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Create ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 5. Train model
model.fit(X_train, y_train)

print("Model training completed!")


# 6. Make predictions
predictions = model.predict(X_test)


# 7. Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("----------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# 8. Save trained model
joblib.dump(model, "models/student_performance_model.pkl")

print("\nModel saved successfully!")