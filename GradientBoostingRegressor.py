import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    return rmse_val, mae_val, r2_val


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Dataset URL
    dataset_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        df = pd.read_csv(dataset_url, sep=";")
    except Exception as e:
        logger.exception("Dataset loading failed. Error: %s", e)

    # Train-test split (75%-25%)
    train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)

    X_train = train_data.drop("quality", axis=1)
    X_test = test_data.drop("quality", axis=1)

    y_train = train_data["quality"]
    y_test = test_data["quality"]

    # Model hyperparameters (can be passed via command line)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    with mlflow.start_run():

        regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=3,
            random_state=42
        )

        regressor.fit(X_train, y_train)

        predictions = regressor.predict(X_test)

        rmse, mae, r2 = calculate_metrics(y_test, predictions)

        print("GradientBoostingRegressor Model")
        print(f"  n_estimators: {n_estimators}")
        print(f"  learning_rate: {learning_rate}")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # 🔹 MLflow logging (UNCHANGED STRUCTURE)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Remote tracking server (AWS)
        # remote_server_uri = "http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/"
        # mlflow.set_tracking_uri(remote_server_uri)

        tracking_store_type = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_store_type != "file":
            mlflow.sklearn.log_model(
                regressor,
                "model",
                registered_model_name="GradientBoostingWineModel"
            )
        else:
            mlflow.sklearn.log_model(regressor, "model")