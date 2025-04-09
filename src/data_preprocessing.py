import pandas as pd
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_load_data(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
    timestamp_column: Optional[str] = None,
    timestamp_start: str = "2016-01-01 00:00:00",
    freq: str = "h"
) -> pd.DataFrame:
    """
    Preprocess time-series data:
      1. Drops specified columns and empty columns.
      2. Handles timestamps (existing or synthetic).
      3. Extracts basic temporal features.
    
    Args:
        df: Input DataFrame.
        columns_to_drop: Columns to remove.
        timestamp_column: Existing timestamp column name.
        timestamp_start: Start time for synthetic timestamps.
        freq: Frequency for synthetic timestamps.
    
    Returns:
        Processed DataFrame with temporal features.
    """
    try:
        # Define default columns to drop if none provided
        default_drop = ["precipDepth6HR", "seaLvlPressure", "precipDepth1HR", "cloudCoverage"]
        cols_to_drop = columns_to_drop if columns_to_drop else default_drop
        df = df.drop(columns=cols_to_drop, errors="ignore")
        df = df.dropna(axis=1, how="all")
        logger.info(f"Dropped columns: {cols_to_drop}")

        # Handle timestamps: If timestamp_column is provided, convert it; otherwise, generate synthetic timestamps.
        if timestamp_column:
            df["timestamp"] = pd.to_datetime(df[timestamp_column], errors="coerce")
        else:
            timestamps = pd.date_range(start=timestamp_start, periods=len(df), freq=freq)
            df["timestamp"] = timestamps

        # Basic temporal features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        logger.info(f"Final shape after preprocessing: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise
