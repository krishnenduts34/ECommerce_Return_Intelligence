import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Load the prepared dataset
# --------------------------------------------------

df = pd.read_csv("data/prepared_data.csv")


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop("Return_Flag", axis=1)
y = df["Return_Flag"]


# --------------------------------------------------
# 3. Split the data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Identify columns
# --------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("Numerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)


# --------------------------------------------------
# 6. Create models
# --------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)


decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            DecisionTreeClassifier(
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)


# --------------------------------------------------
# 7. Train models
# --------------------------------------------------

print("\nTraining Logistic Regression...")
logistic_model.fit(X_train, y_train)
print("Logistic Regression trained successfully!")


print("\nTraining Decision Tree...")
decision_tree_model.fit(X_train, y_train)
print("Decision Tree trained successfully!")


print("\nTraining Random Forest...")
random_forest_model.fit(X_train, y_train)
print("Random Forest trained successfully!")


print("\nAll three models have been trained successfully!")


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

logistic_pred = logistic_model.predict(X_test)
decision_tree_pred = decision_tree_model.predict(X_test)
random_forest_pred = random_forest_model.predict(X_test)


# --------------------------------------------------
# 9. Evaluate models
# --------------------------------------------------

def evaluate_model(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n" + name)
    print("-" * 30)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred
)

evaluate_model(
    "Decision Tree",
    y_test,
    decision_tree_pred
)

evaluate_model(
    "Random Forest",
    y_test,
    random_forest_pred
)


# --------------------------------------------------
# 10. Confusion matrices
# --------------------------------------------------

print("\nConfusion Matrix - Logistic Regression")
print(confusion_matrix(y_test, logistic_pred))


print("\nConfusion Matrix - Decision Tree")
print(confusion_matrix(y_test, decision_tree_pred))


print("\nConfusion Matrix - Random Forest")
print(confusion_matrix(y_test, random_forest_pred))
# --------------------------------------------------
# 11. Model Comparison
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, logistic_pred),
        accuracy_score(y_test, decision_tree_pred),
        accuracy_score(y_test, random_forest_pred)
    ],
    "Precision": [
        precision_score(y_test, logistic_pred),
        precision_score(y_test, decision_tree_pred),
        precision_score(y_test, random_forest_pred)
    ],
    "Recall": [
        recall_score(y_test, logistic_pred),
        recall_score(y_test, decision_tree_pred),
        recall_score(y_test, random_forest_pred)
    ],
    "F1 Score": [
        f1_score(y_test, logistic_pred),
        f1_score(y_test, decision_tree_pred),
        f1_score(y_test, random_forest_pred)
    ]
})

print("\nModel Comparison")
print("=" * 60)
print(comparison.round(4))


# --------------------------------------------------
# 12. Performance Chart
# --------------------------------------------------

import matplotlib.pyplot as plt

comparison.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig("model_comparison.png")

print("\nPerformance chart saved as: model_comparison.png")

plt.show()
# --------------------------------------------------
# 13. Save the best model
# --------------------------------------------------

import joblib

joblib.dump(
    logistic_model,
    "model/return_prediction_model.pkl"
)

print("\nBest model saved successfully!")
print("Saved as: model/return_prediction_model.pkl")