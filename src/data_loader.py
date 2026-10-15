"""
Data Loader and Preprocessing Module.
Aggregates hourly monitoring stations into daily AQI time-series, handles missing values,
and partitions data into train (80%) and test (20%) sets.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple, Dict


def process_raw_station_data(city: str, raw_data_dir: str = "data") -> pd.Series:
    """
    Reads all station excel files for a given city, computes the station average per hour,
    and then calculates the 24-hour daily mean AQI series.
    """
    city_dir = os.path.join(raw_data_dir, city)
    xlsx_files = sorted([
        f for f in glob.glob(os.path.join(city_dir, "*.xlsx"))
        if not os.path.basename(f).startswith("~$")
    ])
    
    if not xlsx_files:
        raise FileNotFoundError(f"No valid .xlsx files found in {city_dir}")

    print(f"[{city.upper()}] Ingesting {len(xlsx_files)} monitoring stations...")
    station_dfs = []
    for fpath in xlsx_files:
        try:
            df = pd.read_excel(fpath, usecols=["year", "month", "day", "hour", "AQI"])
            station_dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed reading {fpath}: {e}")

    # Combine all stations to compute hourly mean across stations
    combined = pd.concat(station_dfs, ignore_index=True)
    hourly_mean = combined.groupby(["year", "month", "day", "hour"])["AQI"].mean().reset_index()

    # Calculate 24-hour daily mean AQI
    daily_mean = hourly_mean.groupby(["year", "month", "day"])["AQI"].mean().reset_index()

    # Construct complete date series
    daily_mean["date"] = pd.to_datetime(daily_mean[["year", "month", "day"]])
    daily_mean = daily_mean.sort_values("date").set_index("date")

    # Complete timeline reindexing
    full_idx = pd.date_range(start=daily_mean.index.min(), end=daily_mean.index.max(), freq="D")
    daily_series = daily_mean["AQI"].reindex(full_idx)

    # Impute missing days via linear interpolation
    missing_count = daily_series.isna().sum()
    if missing_count > 0:
        daily_series = daily_series.interpolate(method="linear").bfill().ffill()
        print(f"[{city.upper()}] Imputed {missing_count} missing days via linear interpolation.")

    print(f"[{city.upper()}] Successfully processed {len(daily_series)} daily observations "
          f"({daily_series.index.min().strftime('%Y-%m-%d')} to {daily_series.index.max().strftime('%Y-%m-%d')}).")
    return daily_series


def prepare_and_save_all_cities(
    cities: list = ("beijing", "chengdu", "guangzhou", "wuhan"),
    raw_data_dir: str = "data",
    processed_dir: str = "processed_data",
) -> Dict[str, pd.Series]:
    """Processes all cities and saves preprocessed daily series to processed_data/."""
    os.makedirs(processed_dir, exist_ok=True)
    all_series = {}

    for city in cities:
        csv_path = os.path.join(processed_dir, f"{city}_daily.csv")
        if os.path.exists(csv_path):
            print(f"[{city.upper()}] Loading cached preprocessed series from {csv_path}")
            df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
            series = df["AQI"]
        else:
            series = process_raw_station_data(city, raw_data_dir=raw_data_dir)
            df_out = pd.DataFrame({"AQI": series})
            df_out.to_csv(csv_path, index_label="date")
            print(f"[{city.upper()}] Saved preprocessed series to {csv_path}")
        all_series[city] = series

    return all_series


def load_city_split(
    city: str,
    train_ratio: float = 0.8,
    processed_dir: str = "processed_data",
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Loads daily series for a city and splits chronologically:
    80% train, 20% test.
    Returns:
        train_data: np.ndarray
        test_data: np.ndarray
        dates: pd.DatetimeIndex
    """
    csv_path = os.path.join(processed_dir, f"{city}_daily.csv")
    if not os.path.exists(csv_path):
        prepare_and_save_all_cities(cities=[city], processed_dir=processed_dir)

    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    series = df["AQI"].values
    dates = df.index

    split_idx = int(len(series) * train_ratio)
    train_data = series[:split_idx]
    test_data = series[split_idx:]

    return train_data, test_data, dates


if __name__ == "__main__":
    prepare_and_save_all_cities()
