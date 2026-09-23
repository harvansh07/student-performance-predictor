import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


# --------------------------------------------------
# Load real UCI dataset
# --------------------------------------------------

data = pd.read_csv(
    "data/student-mat.csv",
    sep=";"
)

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)


# --------------------------------------------------
# Select features
# --------------------------------------------------

features = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences"
]

X = data[features]

y = data["G3"]


# --------------------------------------------------
# Identify feature types
# --------------------------------------------------

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


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
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}


# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------

results = []

trained_models = {}


print("\n==============================")
print("MODEL COMPARISON")
print("==============================")


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

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

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    trained_models[name] = pipeline

    print(f"\n{name}")
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 2))


# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

print("\n==============================")
print("RESULTS")
print("==============================")

print(
    results_df.round(2)
)


# --------------------------------------------------
# Save comparison results
# --------------------------------------------------

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# --------------------------------------------------
# Select best model
# --------------------------------------------------

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]


best_model = trained_models[
    best_model_name
]


print("\n==============================")
print("BEST MODEL")
print("==============================")

print(
    "Selected model:",
    best_model_name
)
# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

print("\nCalculating feature importance...")

importance = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_root_mean_squared_error"
)

feature_importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance.importances_mean
})

feature_importance_df = feature_importance_df.sort_values(
    "Importance",
    ascending=False
)

feature_importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nTop Feature Importance:")
print(
    feature_importance_df.head(10).round(4)
)


# --------------------------------------------------
# Train selected model on full dataset
# --------------------------------------------------

best_model.fit(
    X,
    y
)


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

print(
    "\nModel comparison saved successfully!"
)