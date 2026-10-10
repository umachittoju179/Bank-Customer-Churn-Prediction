
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "European_Bank.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Exited"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False))
    print("\nDuplicate rows:", int(df.duplicated().sum()))

    if TARGET not in df.columns:
        raise ValueError(f"Required target column '{TARGET}' not found.")

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=[TARGET]).copy()

    target_values = set(pd.to_numeric(df[TARGET], errors="coerce").dropna().unique())
    if not target_values.issubset({0, 1}) or len(target_values) != 2:
        raise ValueError("Exited must contain both binary classes 0 and 1.")

    df[TARGET] = pd.to_numeric(df[TARGET]).astype(int)

    # Remove columns that identify customers rather than generalizable behavior.
    drop_cols = [
        col for col in ["CustomerId", "Surname"]
        if col in df.columns
    ]

    # A year field requires investigation. Exclude it from the initial model
    # until its meaning and availability at prediction time are confirmed.
    if "Year" in df.columns:
        drop_cols.append("Year")

    X = df.drop(columns=[TARGET] + drop_cols)
    y = df[TARGET]

    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain.")

    # Remove columns that are entirely missing.
    X = X.dropna(axis=1, how="all")

    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_cols = [
        col for col in X.columns if col not in categorical_cols
    ]

    # Numeric conversion prevents malformed numeric columns from being
    # silently treated as meaningful continuous predictors.
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Remove any columns that became entirely missing after conversion.
    X = X.dropna(axis=1, how="all")
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_cols = [
        col for col in X.columns if col not in categorical_cols
    ]

    if X.shape[1] == 0:
        raise ValueError("No valid features remain after preprocessing.")

    print("\n========== TARGET DISTRIBUTION ==========")
    print(y.value_counts().sort_index())
    print("\nChurn rate:", round(float(y.mean()), 4))
    print("\nExcluded columns:", drop_cols)
    print("Numeric features:", numerical_cols)
    print("Categorical features:", categorical_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    transformers = []
    if numerical_cols:
        transformers.append(("num", numeric_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    minimum_class_count = int(y_train.value_counts().min())
    n_splits = min(5, minimum_class_count)
    if n_splits < 2:
        raise ValueError("Not enough examples per class for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    results = []
    fitted_models = {}

    print("\n========== CROSS-VALIDATION ==========")

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", clone(classifier)),
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"Model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"CV_{metric}"] = float(np.mean(values))
            row[f"CV_{metric}_std"] = float(np.std(values))

        results.append(row)
        print(
            f"{name}: "
            f"CV F1={row['CV_f1']:.4f}, "
            f"CV ROC-AUC={row['CV_roc_auc']:.4f}, "
            f"CV Recall={row['CV_recall']:.4f}"
        )

        fitted_models[name] = pipeline

    cv_results = pd.DataFrame(results).sort_values(
        by="CV_roc_auc", ascending=False
    )
    cv_results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Select by cross-validated ROC-AUC, then evaluate the chosen model
    # exactly once on the held-out test set.
    best_name = cv_results.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_model.fit(X_train, y_train)

    probabilities = best_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "selected_model": best_name,
        "test_rows": int(len(y_test)),
        "threshold": 0.50,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    with open(OUTPUT_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        y_test, predictions, zero_division=0
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    # Save the full fitted pipeline, including preprocessing.
    joblib.dump(best_model, OUTPUT_DIR / "churn_model.pkl")

    # Save held-out predictions for audit and plotting.
    prediction_output = pd.DataFrame({
        "actual_Exited": y_test.to_numpy(),
        "churn_probability": probabilities,
        "predicted_Exited_at_0_50": predictions,
    })
    prediction_output.to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False
    )

    print("\n========== MODEL COMPARISON ==========")
    print(cv_results.to_string(index=False))

    print("\n========== SELECTED MODEL ==========")
    print(best_name)

    print("\n========== HELD-OUT TEST METRICS ==========")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    print("\nSaved files in:", OUTPUT_DIR)
    print("- churn_model.pkl")
    print("- model_comparison.csv")
    print("- evaluation_metrics.json")
    print("- classification_report.txt")
    print("- test_predictions.csv")


if __name__ == "__main__":
    main()
