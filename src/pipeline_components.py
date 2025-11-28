# src/pipeline_components.py

from kfp.v2.dsl import component, Output, Input, Dataset, Model, Metrics
import pandas as pd
from pathlib import Path
import dvc.api
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json
import joblib


# ---------------------------------------------------------
# 1) DATA EXTRACTION COMPONENT
# ---------------------------------------------------------
@component(
    base_image="python:3.10",
    packages_to_install=["dvc", "pandas"]
)
def data_extraction(
    repo_url: str,
    dvc_path: str,
    rev: str,
    output_dataset: Output[Dataset]
):
    """
    Fetch DVC-tracked dataset from GitHub.
    """
    with dvc.api.open(
        path=dvc_path,
        repo=repo_url,
        rev=rev,
        mode="r",
    ) as f:
        df = pd.read_csv(f)

    df.to_csv(output_dataset.path, index=False)
    print("Saved extracted dataset to:", output_dataset.path)


# ---------------------------------------------------------
# 2) DATA PREPROCESSING
# ---------------------------------------------------------
@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn"]
)
def data_preprocessing(
    input_dataset: Input[Dataset],
    x_train: Output[Dataset],
    x_test: Output[Dataset],
    y_train: Output[Dataset],
    y_test: Output[Dataset],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Convert regression target to classification (above/below median).
    """
    df = pd.read_csv(input_dataset.path)

    X = df.iloc[:, :-1]
    y_cont = df.iloc[:, -1]

    y_bin = (y_cont > y_cont.median()).astype(int)

    X_train, X_test, y_train_split, y_test_split = train_test_split(
        X, y_bin, test_size=test_size, random_state=random_state, stratify=y_bin
    )

    X_train.to_csv(x_train.path, index=False)
    X_test.to_csv(x_test.path, index=False)
    y_train_split.to_csv(y_train.path, index=False)
    y_test_split.to_csv(y_test.path, index=False)

    print("Preprocessing complete.")


# ---------------------------------------------------------
# 3) MODEL TRAINING
# ---------------------------------------------------------
@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def model_training(
    x_train: Input[Dataset],
    y_train: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int = 100,
    random_state: int = 42,
):
    X_train = pd.read_csv(x_train.path)
    y_train_df = pd.read_csv(y_train.path)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    clf.fit(X_train, y_train_df.values.ravel())

    joblib.dump(clf, model_output.path)
    print("Model saved:", model_output.path)


# ---------------------------------------------------------
# 4) MODEL EVALUATION
# ---------------------------------------------------------
@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn", "joblib"]
)
def model_evaluation(
    model: Input[Model],
    x_test: Input[Dataset],
    y_test: Input[Dataset],
    metrics: Output[Metrics],
):
    X_test = pd.read_csv(x_test.path)
    y_test_df = pd.read_csv(y_test.path).values.ravel()

    clf = joblib.load(model.path)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test_df, y_pred)
    f1 = f1_score(y_test_df, y_pred)

    metrics.log_metric("accuracy", float(acc))
    metrics.log_metric("f1_score", float(f1))

    print("Accuracy:", acc)
    print("F1 Score:", f1)


# ---------------------------------------------------------
# 5) GENERATE YAML WHEN RUN DIRECTLY
# ---------------------------------------------------------
if __name__ == "__main__":
    from kfp.v2 import compiler
    from kfp.v2.dsl import pipeline

    @pipeline(name="test-pipeline")
    def test_pipeline():
        data = data_extraction(
            repo_url="dummy",
            dvc_path="dummy",
            rev="main",
        )
    
    # Just compile a dummy pipeline to verify
    compiler.Compiler().compile(
        pipeline_func=test_pipeline,
        package_path="components/test_pipeline.yaml"
    )

    print("KFP v2 components are valid.")
