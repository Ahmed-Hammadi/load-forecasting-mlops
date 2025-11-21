import os
import sys
import yaml
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# === Setup Paths and Logging ===
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Config ===
CONFIG_PATH = os.path.join("config", "config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# === Local Imports (after sys.path setup) ===
from data_preprocessing import preprocess_load_data
from feature_engineering import apply_feature_engineering


# === Weather API Fetch ===
def fetch_weather_forecast(lat, lon, forecast_days=7, hourly_vars="temperature_2m,dew_point_2m", timezone="UTC"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "forecast_days": forecast_days,
        "timezone": timezone
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


# === Weather Forecast to DataFrame ===
def process_weather_forecast(api_response, horizon=48, last_timestamp=None):
    hourly = api_response.get("hourly", {})
    timestamps = pd.to_datetime(hourly.get("time", []))
    temperatures = hourly.get("temperature_2m", [])
    dew_points = hourly.get("dew_point_2m", [])

    if len(timestamps) == 0:
        raise ValueError("No hourly forecast data returned from API.")

    df_forecast = pd.DataFrame({
        "timestamp": timestamps[:horizon],
        "temperature": temperatures[:horizon],
        "dew_point": dew_points[:horizon]
    })

    # Align forecast start with history end
    if last_timestamp:
        shift = last_timestamp - df_forecast["timestamp"].iloc[0] + pd.Timedelta(hours=1)
        df_forecast["timestamp"] += shift

    return df_forecast


# === Main Forecast + Feature Engineering Pipeline ===
def fetch_and_save_weather(lat, lon, output_file="weather_forecast.csv", horizon=48,
                           last_timestamp=None, forecast_days=7,
                           hourly_vars="temperature_2m,dew_point_2m", timezone="UTC"):

    # 1. Get weather forecast
    api_response = fetch_weather_forecast(lat, lon, forecast_days, hourly_vars, timezone)
    df_forecast = process_weather_forecast(api_response, horizon, last_timestamp)
    df_forecast[config["data"]["target_column"]] = np.nan
    df_forecast["hour"] = df_forecast[config["data"]["timestamp_column"]].dt.hour
    df_forecast["day_of_week"] = df_forecast[config["data"]["timestamp_column"]].dt.dayofweek
    df_forecast["month"] = df_forecast[config["data"]["timestamp_column"]].dt.month
    # 3. Historical data preprocessing
    df_hist = pd.read_csv(config["data"]["path"])
    df_hist = preprocess_load_data(
        df_hist,
        columns_to_drop=config["data"]["columns_to_drop"],
        timestamp_column=config["data"]["timestamp_column"]
        
    )
    print(df_hist)

    # 4. Combine history + forecast for lag/rolling continuity
    df_hist_tail = df_hist.tail(168).copy()
    for col in df_hist_tail.columns:
        if col not in df_forecast.columns:
            df_forecast[col] = np.nan
    df_forecast = df_forecast[df_hist_tail.columns]

    df_combined = pd.concat([df_hist_tail, df_forecast], ignore_index=True)

    # 5. Re-apply feature engineering on the combined dataset
    df_combined_fe = apply_feature_engineering(
        df_combined,
        target_column=config["data"]["target_column"],
        lags=config["features"]["lags"],
        windows=config["features"]["rolling_windows"]
    )

    df_forecast_fe = df_combined_fe.tail(horizon)

    # 6. Save output
    df_forecast_fe.to_csv(output_file, index=False)
    logger.info(f"Weather forecast + engineered features saved to {output_file}")

    return df_forecast_fe

if __name__ == "__main__":
    # Parameters for London
    LATITUDE = 51.5074
    LONGITUDE = -0.1278
    # Define the last_timestamp for alignment (adjust as needed based on your data)
    last_timestamp = datetime(2017, 12, 31, 20, 0)
    
    # Set the horizon to 48 hours
    horizon = 48
    
    # Define the output file path (saving to the 'data' folder)
    output_file = os.path.join("data", "weather_forecast.csv")
    
    # Fetch weather forecast and save the engineered features to the output file
    forecast_df = fetch_and_save_weather(
        lat=LATITUDE,
        lon=LONGITUDE,
        output_file=output_file,
        horizon=horizon,
        last_timestamp=last_timestamp,
        forecast_days=7,  # As defined in your default parameters
        hourly_vars="temperature_2m,dew_point_2m",
        timezone="UTC"
    )