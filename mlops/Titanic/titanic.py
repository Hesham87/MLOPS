import os

import joblib
import pandas as pd
from omegaconf import DictConfig
import hydra
from pathlib import Path

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
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

script_dir = Path(__file__).parent

def load_data(
    cfg: DictConfig
) -> pd.DataFrame:
    # data_path = os.path.join(project_root, f"{cfg.Pipeline.data.raw_data_path}", f"{cfg.Pipeline.data.file_name}")
    # """Load and return Titanic dataset"""
    
    data_path = f"{script_dir.parent}{cfg.Pipeline.data.raw_data_path}/{cfg.Pipeline.data.file_name}"
    df = pd.read_csv(data_path)
    return df


def preprocess_data(cfg: DictConfig, df: pd.DataFrame) -> tuple:
    """Preprocess data and return features/target"""
    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # Separate features and target
    X = df.drop(cfg.Pipeline.data.target_column, axis=1)
    y = df[cfg.Pipeline.data.target_column]

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


def train_models(cfg: DictConfig, X_train, y_train):
    """Train and return multiple models"""
    models = {
        "RandomForest": RandomForestClassifier(
            **cfg.Pipeline.model.Random_forest.optimization_params
        ),
        "LogisticRegression": LogisticRegression(
            **cfg.Pipeline.model.Logestic_regression.optimization_params
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


def save_pipeline(cfg, pipeline, model_name):
    """Save the entire preprocessing+model pipeline"""
    if(model_name == "RandomForest"):
        joblib.dump(pipeline, f"{script_dir.parent}{cfg.Pipeline.evaluate.Random_forest.trained_model_path}")
    elif(model_name == "LogisticRegression"):
        joblib.dump(pipeline, f"{script_dir.parent}{cfg.Pipeline.evaluate.Logestic_regression.trained_model_path}")
    


@hydra.main(config_path="../", config_name="Config")
def main(cfg: DictConfig):
    # Load data
    df = load_data(cfg)

    # Preprocess
    X, y = preprocess_data(cfg, df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.Pipeline.data.test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Train models
    models = train_models(cfg, X_train, y_train)

    # Evaluate and save models
    for model_name, pipeline in models.items():
        print(f"\nEvaluating {model_name}:")
        metrics = evaluate_model(pipeline, X_test, y_test)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        save_pipeline(cfg, pipeline, model_name)
        if(model_name == "RandomForest"):
            print(f"Saved {model_name} pipeline to {cfg.Pipeline.evaluate.Random_forest.trained_model_path}/")
        elif(model_name == "LogisticRegression"):
            print(f"Saved {model_name} pipeline to {cfg.Pipeline.evaluate.Logestic_regression.trained_model_path}/")


if __name__ == "__main__":
    main()
