# pipeline.py

from kfp import dsl
from kfp.v2 import compiler
from src.pipeline_components import (
    data_extraction,
    data_preprocessing,
    model_training,
    model_evaluation
)


@dsl.pipeline(
    name="housing-classification-pipeline",
    description="End-to-end ML pipeline using DVC + KFP v2"
)
def housing_pipeline(
    repo_url: str = "https://github.com/MasroorRehan2003/mlops-kubeflow-assignment",
    dvc_path: str = "data/raw_data.csv",
    rev: str = "main",
    test_size: float = 0.2,
    n_estimators: int = 100,
):

    # -----------------------------------------
    # 1. Data Extraction
    # -----------------------------------------
    extract_task = data_extraction(
        repo_url=repo_url,
        dvc_path=dvc_path,
        rev=rev
    )

    # extract_task.outputs: {"output_dataset": <path>}
    # -----------------------------------------
    # 2. Data Preprocessing
    # -----------------------------------------
    preprocess_task = data_preprocessing(
        input_dataset=extract_task.outputs["output_dataset"],
        test_size=test_size
    )

    # preprocess_task.outputs:
    #   "x_train", "x_test", "y_train", "y_test"
    # -----------------------------------------
    # 3. Model Training
    # -----------------------------------------
    train_task = model_training(
        x_train=preprocess_task.outputs["x_train"],
        y_train=preprocess_task.outputs["y_train"],
        n_estimators=n_estimators
    )

    # -----------------------------------------
    # 4. Model Evaluation
    # -----------------------------------------
    eval_task = model_evaluation(
        model=train_task.outputs["model_output"],
        x_test=preprocess_task.outputs["x_test"],
        y_test=preprocess_task.outputs["y_test"]
    )



# ---------------------------------------------------------
# MAIN — Compile to pipeline.yaml
# ---------------------------------------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=housing_pipeline,
        package_path="pipeline.yaml"
    )
    print("\n✔✔ pipeline.yaml generated successfully.\n")
