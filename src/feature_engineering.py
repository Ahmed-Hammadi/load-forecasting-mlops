import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_lag_features(df: pd.DataFrame, target_column: str, lags: List[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target_column: str, windows: List[int]) -> pd.DataFrame:
    for window in windows:
        df[f"rolling_{window}_mean"] = df[target_column].rolling(window=window, min_periods=1, closed="left").mean()
        df[f"rolling_{window}_std"] = df[target_column].rolling(window=window, min_periods=1, closed="left").std()
    return df

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    if "day_of_week" in df.columns:
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df

def add_load_dynamics(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df["load_gradient"] = df[target_column].diff()
    df["load_volatility_4h"] = df[target_column].rolling(window=4, min_periods=1).std()
    return df

def add_operational_context(df: pd.DataFrame) -> pd.DataFrame:
    df["is_business_day"] = df["timestamp"].dt.weekday < 5
    df["hours_since_work_start"] = np.where(
        (df["is_business_day"]) & (df["hour"] >= 8),
        df["hour"] - 8,
        0
    )
    return df

def apply_feature_engineering(
    df: pd.DataFrame,
    target_column: str,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    try:
        logger.info("Starting feature engineering")

        lags = lags or [1, 2, 3, 4, 24, 168]
        windows = windows or [4, 24, 168]

        df = add_lag_features(df, target_column, lags)
        df = add_rolling_features(df, target_column, windows)
        df = add_load_dynamics(df, target_column)
        df = add_cyclical_features(df)
        df = add_operational_context(df)

        df = df.fillna(method='ffill').bfill()
        df = df.drop(columns=["timestamp"], errors="ignore")

        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise
