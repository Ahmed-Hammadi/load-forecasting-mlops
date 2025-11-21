import sys
import os
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="Production Model Loader", layout="wide")
import yaml
import mlflow.pyfunc
import pandas as pd
import logging
import numpy as np  # 🔄 ADDED
import matplotlib.pyplot as plt  # 🔄 ADDED
from datetime import datetime, timedelta  # 🔄 ADDED

# Setup paths and logging
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.path.join("config", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Path to the weather forecast file produced by your weather_fetcher
forecast_file = os.path.join("data", "weather_forecast.csv")
try:
    df_forecast = pd.read_csv(forecast_file)
    st.success("✅ Forecast input data loaded from CSV.")  # 🔄 ADDED
except Exception as e:
    st.error(f"❌ Error loading forecast CSV: {e}")  # 🔄 ADDED
    st.stop()  # 🔄 ADDED
# Read the CSV file (which should include your engineered features)
df_forecast = pd.read_csv(forecast_file)
# Import local modules
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_production_model(candidate_models):
    client = MlflowClient()
    prod_candidates = []
    
    for model_name in candidate_models:
        try:
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if versions and len(versions) > 0:
                # Append tuple (model_name, version_object)
                prod_candidates.append((model_name, versions[-1]))
        except Exception:
            continue

    if not prod_candidates:
        raise Exception("No candidate model found in Production stage.")

    selected_model_name, selected_version = prod_candidates[0]
    model_uri = f"models:/{selected_model_name}/Production"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    return loaded_model, selected_model_name, selected_version.version
import pandas as pd
import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace("'", "").str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
def clean_and_cast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    schema = {
        'airTemperature': 'float64',
        'dewTemperature': 'float64',
        'hour': 'int32',
        'day_of_week': 'int32',
        'month': 'int32',
        'Robin_office_Shirlene_lag_1': 'float64',
        'Robin_office_Shirlene_lag_2': 'float64',
        'Robin_office_Shirlene_lag_3': 'float64',
        'Robin_office_Shirlene_lag_4': 'float64',
        'Robin_office_Shirlene_lag_24': 'float64',
        'Robin_office_Shirlene_lag_168': 'float64',
        'rolling_4_mean': 'float64',
        'rolling_4_std': 'float64',
        'rolling_24_mean': 'float64',
        'rolling_24_std': 'float64',
        'rolling_168_mean': 'float64',
        'rolling_168_std': 'float64',
        'load_gradient': 'float64',
        'load_volatility_4h': 'float64',
        'hour_sin': 'float64',
        'hour_cos': 'float64',
        'day_sin': 'float64',
        'day_cos': 'float64',
        'is_business_day': 'bool',
        'hours_since_work_start': 'int32',
    }

    for col, dtype in schema.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("'", "").str.strip()
            if dtype == 'bool':
                df[col] = df[col].astype(bool)
            elif 'int' in dtype:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(dtype)
        else:
            raise ValueError(f"Missing expected column in DataFrame: {col}")
    return df

def reattach_timestamps(start_time: datetime, horizon: int, prediction_df: pd.DataFrame) -> pd.DataFrame:
    timestamps = [start_time + timedelta(hours=i + 1) for i in range(horizon)]
    prediction_df = prediction_df.copy()
    prediction_df["timestamp"] = pd.to_datetime(timestamps)
    return prediction_df

# Streamlit basic layout
st.title("🔄 Load Production Model from MLflow")

st.markdown("""
This app loads the currently deployed production model from the MLflow Model Registry.
Candidate models include:
- Electricity_XGBoost
- Electricity_RandomForest
""")

# List candidate models (adjust if needed)
candidate_models = ["Electricity_XGBoost", "Electricity_RandomForest"]

try:
    prod_model, model_name, version = load_production_model(candidate_models)
    st.success(f"✅ Loaded production model: **{model_name}** (version {version})")
    st.write("Model URI:", f"models:/{model_name}/Production")
except Exception as e:
    st.error(f"❌ Error loading production model: {e}")

# --- Prepare Model Input for Prediction ---
input_data = df_forecast.drop(columns=[config["data"]["target_column"]], errors="ignore")
input_data = clean_dataframe(input_data) 
input_data = clean_and_cast_dataframe(input_data) 
st.subheader("📥 Input Data Sent to Model")
st.dataframe(input_data.head(50))

# --- Make Prediction ---
try:
    predictions = prod_model.predict(input_data)  
    df_forecast["Forecasted_Load"] = predictions  
    st.success("✅ Predictions made successfully.")  
except Exception as e:
    st.error(f"❌ Error during forecasting: {e}")  
    st.stop()  

start_time = datetime (2017, 12, 31, 20, 0)
horizon = 48
df_forecast = reattach_timestamps(start_time, horizon, df_forecast)
# --- Visualize the Forecast ---
st.write("### 48-Hour Load Forecast")  # 🔄 ADDED
st.dataframe(df_forecast[[config["data"]["timestamp_column"], "Forecasted_Load"]].head(48)) 

# --- Create Matplotlib Plot ---
try:
    df_forecast[config["data"]["timestamp_column"]] = pd.to_datetime(df_forecast[config["data"]["timestamp_column"]])  # 🔄 ADDED
    fig, ax = plt.subplots(figsize=(12, 4))  # 🔄 ADDED
    ax.plot(df_forecast[config["data"]["timestamp_column"]], df_forecast["Forecasted_Load"], marker="o", linestyle="--", color="orange", label="Forecasted Load")  # 🔄 ADDED
    ax.set_xlabel("Time")  # 🔄 ADDED
    ax.set_ylabel("Load")  # 🔄 ADDED
    ax.set_title("48-Hour Load Forecast")  # 🔄 ADDED
    ax.legend()  # 🔄 ADDED
    ax.grid(True)  # 🔄 ADDED
    st.pyplot(fig)  # 🔄 ADDED
except Exception as e:
    st.error(f"❌ Error during plotting: {e}")  # 🔄 ADDED