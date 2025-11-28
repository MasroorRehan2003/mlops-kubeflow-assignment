import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ----------------- CONFIG -----------------
DATA_PATH = Path("data/raw_data.csv")
TARGET_COL = "medv"   # 🔴 Change this if your target column has a different name


def load_data(path: Path = DATA_PATH):
    """Load dataset from CSV and split into train/test."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Make sure DVC pulled it correctly.")

    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_log_with_mlflow():
    """Train a Random Forest model and log everything to MLflow."""
    # Create / select an experiment in MLflow
    mlflow.set_experiment("boston-housing-mlops")

    # Start a tracked run
    with mlflow.start_run(run_name="rf_baseline"):
        # 1) Load data
        X_train, X_test, y_train, y_test = load_data()

        # 2) Define model hyperparameters
        params = {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }

        # Log parameters to MLflow
        mlflow.log_params(params)

        # 3) Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # 4) Evaluate model
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # 5) Log artifacts: trained model + dataset snapshot
        mlflow.sklearn.log_model(model, artifact_path="model")
        if DATA_PATH.exists():
            mlflow.log_artifact(str(DATA_PATH))

        print(f"[MLflow] Training finished. RMSE={rmse:.4f}, R2={r2:.4f}")


if __name__ == "__main__":
    train_and_log_with_mlflow()
