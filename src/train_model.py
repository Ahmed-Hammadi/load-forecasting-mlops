import argparse
import copy
import os
import logging
from typing import Optional
import mlflow
import pandas as pd
import numpy as np
import yaml
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from mlflow.tracking import MlflowClient
from src.data_preprocessing import preprocess_load_data
from src.feature_engineering import apply_feature_engineering

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/config.yaml"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def plot_predictions(y_true, y_pred, title, filename, target_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue', alpha=0.6)
    plt.plot(y_pred, label='Predicted', color='orange', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def time_series_metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mape': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        'last_24_mae': float(mean_absolute_error(y_true[-24:], y_pred[-24:])),
        'last_7_rmse': float(np.sqrt(mean_squared_error(y_true[-168:], y_pred[-168:])))
    }

def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj

def promote_to_production():
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Retrieve and select the latest staging version for RandomForest
        rf_staging_versions = client.get_latest_versions("Electricity_RandomForest", stages=["Staging"])
        if not rf_staging_versions:
            logger.error("No staging versions found for Electricity_RandomForest.")
            return
        rf_staging_version = max(rf_staging_versions, key=lambda v: int(v.version))

        # Retrieve and select the latest staging version for XGBoost
        xgb_staging_versions = client.get_latest_versions("Electricity_XGBoost", stages=["Staging"])
        if not xgb_staging_versions:
            logger.error("No staging versions found for Electricity_XGBoost.")
            return
        xgb_staging_version = max(xgb_staging_versions, key=lambda v: int(v.version))

        # Get the run IDs and fetch metrics for both models
        rf_run_id = rf_staging_version.run_id
        xgb_run_id = xgb_staging_version.run_id

        rf_metrics = client.get_run(rf_run_id).data.metrics
        xgb_metrics = client.get_run(xgb_run_id).data.metrics

        # Retrieve the MAE metric (using infinity as default for missing values)
        rf_mae = rf_metrics.get("mae", float('inf'))
        xgb_mae = xgb_metrics.get("mae", float('inf'))

        # Determine the best staging model based on lower MAE
        best_model = "RandomForest" if rf_mae < xgb_mae else "XGBoost"
        best_model_version = rf_staging_version if best_model == "RandomForest" else xgb_staging_version

        # Retrieve production model versions for both models
        rf_prod_versions = client.get_latest_versions("Electricity_RandomForest", stages=["Production"])
        xgb_prod_versions = client.get_latest_versions("Electricity_XGBoost", stages=["Production"])
        production_models = rf_prod_versions + xgb_prod_versions

        if production_models:
            # If production models exist, select the one with the highest version number
            prod_model_version = max(production_models, key=lambda v: int(v.version))
            prod_run_id = prod_model_version.run_id
            prod_metrics = client.get_run(prod_run_id).data.metrics
            prod_mae = prod_metrics.get("mae", float('inf'))

            # Compare best staging model with current production model
            if best_model == "RandomForest" and rf_mae < prod_mae:
                client.transition_model_version_stage(
                    name="Electricity_RandomForest", 
                    version=best_model_version.version, 
                    stage="Production"
                )
                logger.info(f"Model Electricity_RandomForest v{best_model_version.version} promoted to 'Production'.")
            elif best_model == "XGBoost" and xgb_mae < prod_mae:
                client.transition_model_version_stage(
                    name="Electricity_XGBoost", 
                    version=best_model_version.version, 
                    stage="Production"
                )
                logger.info(f"Model Electricity_XGBoost v{best_model_version.version} promoted to 'Production'.")
            else:
                logger.info("Current production model performs better than or equal to the best staging model. No promotion executed.")
        else:
            # If no production model exists, promote the best staging model
            client.transition_model_version_stage(
                name=f"Electricity_{best_model}", 
                version=best_model_version.version, 
                stage="Production"
            )
            logger.info(f"Model Electricity_{best_model} v{best_model_version.version} promoted to 'Production'.")
    
    except Exception as e:
        logger.error(f"Error during promotion: {str(e)}")
        raise

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_smoke_overrides(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["validation"]["n_splits"] = min(cfg["validation"].get("n_splits", 2), 2)

    rf_params = cfg["models"].get("RandomForest", {}).copy()
    rf_params.setdefault("n_estimators", 100)
    rf_params["n_estimators"] = min(rf_params["n_estimators"], 50)
    rf_params["max_depth"] = min(rf_params.get("max_depth", 10), 8)
    rf_params["min_samples_leaf"] = max(rf_params.get("min_samples_leaf", 1), 2)
    cfg["models"]["RandomForest"] = rf_params

    xgb_params = cfg["models"].get("XGBoost", {}).copy()
    xgb_params.setdefault("n_estimators", 200)
    xgb_params["n_estimators"] = min(xgb_params["n_estimators"], 50)
    xgb_params["max_depth"] = min(xgb_params.get("max_depth", 6), 4)
    xgb_params["learning_rate"] = max(xgb_params.get("learning_rate", 0.05), 0.1)
    xgb_params["early_stopping_rounds"] = min(xgb_params.get("early_stopping_rounds", 10), 5)
    cfg["models"]["XGBoost"] = xgb_params

    return cfg


def train(config: dict, run_name: str = "Load_forecasting_run", smoke_run: bool = False, max_rows: Optional[int] = None):
    try:
        config_to_use = copy.deepcopy(config)
        if smoke_run:
            config_to_use = apply_smoke_overrides(config_to_use)
            max_rows = max_rows or 500

        mlflow.set_experiment(config_to_use["mlflow"]["experiment_name"])

        df = pd.read_csv(config_to_use["data"]["path"])
        if max_rows:
            df = df.head(max_rows)
        df = preprocess_load_data(
            df,
            columns_to_drop=config_to_use["data"]["columns_to_drop"],
            timestamp_column=config_to_use["data"]["timestamp_column"]
        )

        df = apply_feature_engineering(
            df,
            target_column=config_to_use["data"]["target_column"],
            lags=config_to_use["features"]["lags"],
            windows=config_to_use["features"]["rolling_windows"]
        )
        df.head(50).to_csv(os.path.join("datablabla.csv"), index=False)
        df = df.fillna(method='ffill').fillna(method='bfill')

        X = df.drop(columns=[config_to_use["data"]["target_column"]])
        y = df[config_to_use["data"]["target_column"]]

        models = {
            "RandomForest": RandomForestRegressor(
                **config_to_use["models"]["RandomForest"],
                random_state=config_to_use["models"]["common"]["random_state"]
            ),
            "XGBoost": XGBRegressor(
                **{k: v for k, v in config_to_use["models"]["XGBoost"].items() if k != 'early_stopping_rounds'},
                random_state=config_to_use["models"]["common"]["random_state"]
            )
        }

        tscv = TimeSeriesSplit(n_splits=config_to_use["validation"]["n_splits"])

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(convert_to_serializable(config_to_use["models"]["common"]))
            mlflow.log_dict(convert_to_serializable(config_to_use), "config.yaml")

            for model_name, model in models.items():
                with mlflow.start_run(nested=True, run_name=model_name):
                    metrics = {"mae": [], "rmse": [], "mape": []}

                    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                        model_clone = model.__class__(**model.get_params())
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        train_data = X_train.copy()
                        train_data[config_to_use["data"]["target_column"]] = y_train
                        train_data = train_data.head(100)
                        train_csv_path = f"train_dataset_fold_{fold}.csv"
                        train_data.to_csv(train_csv_path, index=False)
                        mlflow.log_artifact(train_csv_path)

                        eval_data = X_test.copy()
                        eval_data[config_to_use["data"]["target_column"]] = y_test
                        eval_data = eval_data.head(100)
                        eval_csv_path = f"eval_dataset_fold_{fold}.csv"
                        eval_data.to_csv(eval_csv_path, index=False)
                        mlflow.log_artifact(eval_csv_path)

                        if model_name == "XGBoost":
                            early_stop = config_to_use["models"]["XGBoost"].get("early_stopping_rounds")
                            if early_stop:
                                model_clone.set_params(early_stopping_rounds=early_stop)
                            model_clone.fit(
                                X_train, y_train,
                                eval_set=[(X_test, y_test)],
                                verbose=False
                            )
                        else:
                            model_clone.fit(X_train, y_train)

                        pred = model_clone.predict(X_test)
                        fold_metrics = time_series_metrics(y_test, pred)
                        for k, v in fold_metrics.items():
                            metrics.setdefault(k, []).append(v)

                        test_results = pd.DataFrame({'actual': y_test, 'predicted': pred})
                        test_results.to_csv(f"test_predictions_{model_name}_fold_{fold}.csv", index=False)
                        mlflow.log_artifact(f"test_predictions_{model_name}_fold_{fold}.csv")

                        plot_file = f"predictions_plot_{model_name}_fold_{fold}.png"
                        plot_predictions(
                            y_test.values,
                            pred,
                            f"{model_name} Fold {fold}",
                            plot_file,
                            config_to_use["data"]["target_column"]
                        )
                        mlflow.log_artifact(plot_file)

                    mlflow.log_metrics({
                        f"mean_{k}": convert_to_serializable(np.mean(v)) for k, v in metrics.items()
                    })
                    mlflow.log_metrics({
                        f"std_{k}": convert_to_serializable(np.std(v)) for k, v in metrics.items()
                    })

                    final_model = model.__class__(**model.get_params())
                    final_model.fit(X, y)

                    mlflow.sklearn.log_model(
                        sk_model=final_model,
                        artifact_path="model",
                        registered_model_name=f"Electricity_{model_name}",
                        input_example=X.iloc[:1],
                        signature=mlflow.models.infer_signature(X, y),
                        metadata=convert_to_serializable({
                            "metrics": {k: np.mean(v) for k, v in metrics.items()},
                            "best_params": model.get_params()
                        })
                    )
                    client = MlflowClient()
                    full_model_name = f"Electricity_{model_name}"
                    latest_version = client.get_latest_versions(full_model_name, stages=["None"])[-1].version

                    client.transition_model_version_stage(
                        name=full_model_name,
                        version=latest_version,
                        stage="Staging"
                    )
                    logger.info(f"Model {full_model_name} v{latest_version} transitioned to 'Staging'")
                    promote_to_production()

    except Exception as e:
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_param("error", str(e))
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Train load forecasting models")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to YAML config")
    parser.add_argument("--run-name", type=str, default="Load_forecasting_run", help="MLflow run name")
    parser.add_argument("--smoke-run", action="store_true", help="Enable lightweight smoke test")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(
        config=config,
        run_name=args.run_name,
        smoke_run=args.smoke_run,
        max_rows=args.max_rows
    )