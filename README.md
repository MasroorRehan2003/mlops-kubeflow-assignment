MLOps Assignment – MLflow Pipeline with DVC, CI/CD, and GitHub Actions

This repository contains the complete implementation of the MLOps Assignment (#4) for MLOps (AI) – Fall 2025.
It includes data versioning with DVC, model training with MLflow, automated experiment tracking, and CI/CD with GitHub Actions.

Since Kubeflow Pipelines caused cluster issues (as noted by the course instructor), this project uses MLflow instead of Kubeflow — while preserving all MLOps concepts such as reproducibility, versioning, orchestration, and automation.

 Project Structure
mlops-kubeflow-assignment/
│
├── data/
│   ├── raw_data.csv.dvc         # DVC-tracked dataset metadata
│   └── .gitignore               # Data ignored by Git
│
├── src/
│   ├── pipeline_mlflow.py       # MLflow training & logging pipeline
│   ├── pipeline_components.py   # Earlier Kubeflow-style components (not required)
│   └── model_training.py        # Model training helper functions
│
├── components/                  # Compiled components (Kubeflow version)
│
├── mlruns/                      # Local MLflow experiment logs (Git-ignored)
│
├── pipeline.py                  # (Optional) Kubeflow pipeline definition
├── pipeline.yaml                # Compiled pipeline (if Kubeflow used)
│
├── .github/workflows/
│   └── mlflow-ci.yml            # GitHub Actions workflow for CI/CD
│
├── .dvc/                        # DVC metadata
├── .dvcignore
├── Dockerfile
├── Jenkinsfile (optional)
│
├── requirements.txt
└── README.md

 1. Project Overview

This project demonstrates a full MLOps workflow by applying:

Data Versioning → using DVC

Experiment Tracking & Model Registry → using MLflow

CI/CD Automation → using GitHub Actions

Reproducibility → through pinned dependencies & a clean repository structure

The ML task used in the project is a Regression Model for the Boston Housing Dataset using Random Forest Regressor.

The pipeline:

Loads the dataset via DVC

Splits into train/test

Trains a Random Forest model

Logs parameters, metrics & artifacts to MLflow

Registers the trained model

 2. Setup Instructions
✅ Clone the Repository
git clone https://github.com/<your-username>/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment

✅ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate # Mac/Linux

✅ Install Dependencies
pip install -r requirements.txt

 3. DVC Setup (Data Versioning)
Initialize DVC:
dvc init

Add the dataset to DVC:
dvc add data/raw_data.csv

Commit DVC metadata:
git add data/raw_data.csv.dvc .gitignore .dvc/config
git commit -m "Track dataset with DVC"
git push

Push data to remote:

(Uses Google Drive or local folder depending on your setup)

dvc push

 4. MLflow Pipeline – Training & Experiment Logging

Run the MLflow pipeline:

python src/pipeline_mlflow.py


This will:

Start a new MLflow experiment

Train a Random Forest model

Log:

Parameters (n_estimators, max_depth, etc.)

Metrics (RMSE, R²)

Trained model artifact

Store everything inside mlruns/

Launch MLflow UI:

mlflow ui


Open browser at:

http://127.0.0.1:5000

 5. CI/CD – GitHub Actions Workflow

This repository includes a workflow:

.github/workflows/mlflow-ci.yml


Workflow stages:

Stage 1: Environment Setup

Checkout code

Install dependencies

Stage 2: Pipeline Validation

Run the MLflow training script end-to-end

Ensure no errors before merging

Stage 3: Optional Artifacts Upload

Metrics

Model files

Trigger:

The workflow runs automatically on:

push
pull_request


This ensures every commit is reproducible and testable.

 6. Pipeline Walkthrough
🔹 Step 1: Data Loading

The script loads the dataset from:

data/raw_data.csv


pulled via DVC.

🔹 Step 2: Preprocessing

Feature selection

Train/test split

Scaling (if required)

🔹 Step 3: Training

Random Forest Regressor is trained.

🔹 Step 4: Evaluation

Metrics logged:

RMSE

R²

🔹 Step 5: MLflow Logging

Everything is logged to:

mlruns/


including:

Model

Metrics

Parameters

Environment

 7. Running Everything from Scratch
 Full Pipeline (Local)
dvc pull
python src/pipeline_mlflow.py
mlflow ui

 Full Pipeline (GitHub Actions)

Just push any commit:

git add .
git commit -m "Trigger CI"
git push


GitHub Actions will automatically:

Install dependencies

Run pipeline

Validate outputs

 8. Technologies Used
Tool	Purpose
Git/GitHub	Source control
DVC	Data versioning
MLflow	Tracking, model registry
Scikit-learn	ML model
Python	Core scripting
GitHub Actions	CI/CD
Docker	Component containerization (optional)

Author

Masroor Bin Rehan – BS AI (FAST-NUCES)