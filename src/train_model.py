import mlflow
import pandas as pd
import numpy as np
import yaml
import warnings
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from data_preprocessing import preprocess_load_data
from feature_engineering import apply_feature_engineering

# Cleanup MLflow artifacts on each run
mlruns_path = Path("mlruns")
if mlruns_path.exists():
    shutil.rmtree(mlruns_path)

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# Load configuration
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(config["mlflow"]["experiment_name"])

def plot_predictions(y_true, y_pred, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue', alpha=0.6)
    plt.plot(y_pred, label='Predicted', color='orange', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel(config["data"]["target_column"])
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

def train(run_name="forecasting_run"):
    try:
        # Load and preprocess
        df = pd.read_csv(config["data"]["path"])
        df = preprocess_load_data(
            df,
            columns_to_drop=config["data"]["columns_to_drop"],
            timestamp_column=config["data"]["timestamp_column"]
        )

        # Feature engineering
        df = apply_feature_engineering(
            df,
            target_column=config["data"]["target_column"],
            lags=config["features"]["lags"],
            windows=config["features"]["rolling_windows"]
        )

        # Advanced NaN handling
        df = df.fillna(method='ffill').fillna(method='bfill')

        X = df.drop(columns=[config["data"]["target_column"]])
        y = df[config["data"]["target_column"]]

        models = {
            "RandomForest": RandomForestRegressor(
                **config["models"]["RandomForest"],
                random_state=config["models"]["common"]["random_state"]
            ),
            "XGBoost": XGBRegressor(
                **{k: v for k, v in config["models"]["XGBoost"].items() if k != 'early_stopping_rounds'},
                random_state=config["models"]["common"]["random_state"]
            )
        }

        tscv = TimeSeriesSplit(n_splits=config["validation"]["n_splits"])

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(convert_to_serializable(config["models"]["common"]))
            mlflow.log_dict(convert_to_serializable(config), "config.yaml")

            for model_name, model in models.items():
                with mlflow.start_run(nested=True, run_name=model_name):
                    metrics = {"mae": [], "rmse": [], "mape": []}

                    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                        model_clone = model.__class__(**model.get_params())
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        if model_name == "XGBoost":
                            model_clone.set_params(
                                early_stopping_rounds=config["models"]["XGBoost"]["early_stopping_rounds"]
                            )
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

                        # Save predictions
                        test_results = pd.DataFrame({
                            'actual': y_test,
                            'predicted': pred
                        })
                        test_results.to_csv(f"test_predictions_{model_name}_fold_{fold}.csv", index=False)
                        mlflow.log_artifact(f"test_predictions_{model_name}_fold_{fold}.csv")

                        # Save plots
                        plot_file = f"predictions_plot_{model_name}_fold_{fold}.png"
                        plot_predictions(y_test.values, pred, f"{model_name} Fold {fold}", plot_file)
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

    except Exception as e:
        mlflow.log_param("error", str(e))
        raise

if __name__ == "__main__":
    train()
