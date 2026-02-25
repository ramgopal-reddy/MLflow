import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    dataset_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        df = pd.read_csv(dataset_url, sep=";")
    except Exception as e:
        logger.exception("Dataset loading failed: %s", e)
        sys.exit(1)

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    X_train = train_df.drop("quality", axis=1)
    X_test = test_df.drop("quality", axis=1)

    y_train = train_df["quality"]
    y_test = test_df["quality"]

    # Hyperparameters (optional CLI)
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    with mlflow.start_run():

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42,
            objective="reg:squarederror"
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse, mae, r2 = evaluate_model(y_test, predictions)

        print("\nXGBoost Regressor Performance")
        print(f"n_estimators: {n_estimators}")
        print(f"learning_rate: {learning_rate}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")

        # MLflow logging
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_depth", 5)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Optional Remote Tracking
        # remote_server_uri = "http://your-mlflow-server:5000/"
        # mlflow.set_tracking_uri(remote_server_uri)

        tracking_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_store != "file":
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="XGBoostWineModel"
            )
        else:
            mlflow.sklearn.log_model(model, "model")