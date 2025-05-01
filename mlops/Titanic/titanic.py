import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_SAVE_PATH = "/teamspace/studios/this_studio/mlops/MLOPS/mlops/Titanic/saved_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def load_data(
    data_path: str = "/teamspace/studios/this_studio/mlops/MLOPS/mlops/Titanic/titanic_train.csv",
) -> pd.DataFrame:
    """Load and return Titanic dataset"""
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess data and return features/target"""
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Separate features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    return X, y


def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline"""
    numeric_features = ["Age", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_models(X_train, y_train):
    """Train and return multiple models"""
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
    }

    trained_models = {}
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", create_preprocessor()), ("classifier", model)]
        )
        pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline

    return trained_models


def evaluate_model(model, X_test, y_test):
    """Evaluate and print model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba),
    }

    return metrics


def save_pipeline(pipeline, model_name):
    """Save the entire preprocessing+model pipeline"""
    joblib.dump(pipeline, os.path.join(MODEL_SAVE_PATH, f"{model_name}_pipeline.pkl"))


def main():
    # Load data
    df = load_data()

    # Preprocess
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate and save models
    for model_name, pipeline in models.items():
        print(f"\nEvaluating {model_name}:")
        metrics = evaluate_model(pipeline, X_test, y_test)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        save_pipeline(pipeline, model_name)
        print(f"Saved {model_name} pipeline to {MODEL_SAVE_PATH}/")


if __name__ == "__main__":
    main()
