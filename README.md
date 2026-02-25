# MLflow Experiment Tracking Demonstration

## Overview

This project demonstrates the practical use of MLflow for experiment tracking, parameter logging, metric logging, model persistence, and optional model registration.

The focus of this repository is to showcase how MLflow can be integrated into a machine learning workflow using command-line execution and local or remote tracking servers.

This project is designed purely to demonstrate MLflow application and workflow management.

## Objectives

This project demonstrates:

- Running machine learning experiments via command line
- Logging hyperparameters
- Logging evaluation metrics
- Saving trained models as artifacts
- Viewing experiments in the MLflow UI
- Using local tracking storage
- Configuring a remote tracking server
- Registering models in MLflow Model Registry

## Models Implemented

The following regression models are implemented for demonstration purposes:

- RandomForestRegressor
- Xgboost
- SVR
- GradientBoostingRegressor
- DecisionTreeRegressor

Each model:

- Accepts hyperparameters via command-line arguments
- Logs parameters to MLflow
- Logs evaluation metrics
- Logs trained models
- Optionally registers models (when remote tracking is enabled)

## Environment Setup

### Clone the Repository

```bash
git clone https://github.com/ramgopal-reddy/MLflow.git
cd MLflow
```

## Environment Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running Experiments

Run the script with default hyperparameters:

```bash
python GradientBoostingRegressor.py
```

Each execution automatically:

- Starts a new MLflow run
- Logs parameters
- Logs evaluation metrics
- Saves the trained model artifact

## Viewing Experiments in MLflow UI (Local)

After running experiments, start the MLflow UI:

```bash
mlflow ui
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

The UI allows you to:

- Compare multiple experiment runs
- Inspect logged parameters
- Analyze metrics
- Download artifacts
- Compare runs side-by-side
- Select best-performing models

By default, MLflow stores runs locally in:

```
./mlruns/
```

## Remote Tracking Server Configuration

To use a remote MLflow tracking server, configure the tracking URI:

```python
mlflow.set_tracking_uri("http://server:5000/")
```

This enables:

- Centralized experiment tracking
- Team collaboration
- Remote artifact storage
- Model registry usage

If no tracking URI is specified, MLflow uses local file storage.

## Model Registration

When using a remote tracking server, models can be registered:

```python
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="ModelName"
)
```

Model registration enables:

- Model versioning
- Stage transitions (e.g., Staging, Production)
- Lifecycle management

## Experiment Comparison Workflow

Multiple runs can be executed sequentially:

```bash
python GradientBoostingRegressor.py
```

Using the MLflow UI, you can:

- Compare RMSE, MAE, and R² across runs
- Sort experiments by performance
- Identify the best-performing configuration
- Register the selected model

## Project Structure

```
MLflow-Demo/
│
├── GradientBoostingRegressor.py
├── requirements.txt
├── README.md
└── mlruns/   (generated automatically)
```

## Key Learnings

This project demonstrates:

- Structured experiment tracking
- Reproducible machine learning workflows
- Centralized logging of parameters and metrics
- Model artifact management
- Seamless integration of MLflow with scikit-learn
- Local and remote experiment management

## Potential Extensions

- Hyperparameter tuning integration
- Cross-validation logging
- Dockerized MLflow server
- REST-based model deployment
- Cloud storage integration
- CI/CD integration for ML workflows

## Cloud Platform Integration

### AWS Integration

**MLflow on AWS Setup:**

1. Login to AWS console
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running `aws configure`
4. Create a S3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 5000 port

**Run the following command on EC2 machine:**

```bash
sudo apt update
sudo apt install python3-pip
sudo apt install pipenv
sudo apt install virtualenv

mkdir mlflow
cd mlflow

pipenv install mlflow
pipenv install awscli
pipenv install boto3
pipenv shell

# Then set AWS credentials
aws configure

# Finally start MLflow server
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-tracking-buc25 --allowed-hosts *

# Open Public IPv4 DNS to the port 5000

# Set URI in your local terminal and in your code
export MLFLOW_TRACKING_URI=http://ec2.compute-1.amazonaws.com:5000/
```

## Conclusion

This repository serves as a practical demonstration of MLflow's capabilities in managing the machine learning lifecycle. It provides a clean and extensible structure suitable for experimentation, learning, and portfolio presentation.
