#!/usr/bin/env python3

import argparse
import os
from datetime import datetime, timezone
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def detect_epoch_seconds(ts: float) -> float:
    """
    Normalize a numeric timestamp to seconds since epoch.

    Heuristics:
    - > 1e12: nanoseconds
    - > 1e10: milliseconds
    - else: seconds
    """
    if ts > 1e12:  # nanoseconds
        return ts / 1e9
    if ts > 1e10:  # milliseconds
        return ts / 1e3
    return ts


def to_datetime_utc(epoch_seconds: float) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


def find_temperature_columns(columns: List[str]) -> List[str]:
    lower = [c.lower() for c in columns]
    selected: List[str] = []
    for i, name in enumerate(lower):
        if "motor_temperature" in name or name.endswith("temperature"):
            selected.append(columns[i])
    return selected


def compute_time_info(first_col: pd.Series) -> Tuple[pd.Series, datetime, datetime]:
    # Drop NaNs for min/max detection
    non_na = first_col.dropna().astype(float)
    if non_na.empty:
        raise ValueError("Timestamp column is empty after dropping NaNs.")

    start_raw = float(non_na.iloc[0])
    end_raw = float(non_na.iloc[-1])

    start_sec = detect_epoch_seconds(start_raw)
    end_sec = detect_epoch_seconds(end_raw)

    # Convert entire column to seconds since start for plotting
    sec_series = first_col.astype(float).map(detect_epoch_seconds)
    rel_time = sec_series - start_sec

    start_dt = to_datetime_utc(start_sec)
    end_dt = to_datetime_utc(end_sec)
    return rel_time, start_dt, end_dt


def plot_temperatures(time_s: pd.Series, df: pd.DataFrame, temp_cols: List[str], out_path: str, show: bool = False) -> None:
    plt.figure(figsize=(12, 6))
    for col in temp_cols:
        plt.plot(time_s, df[col], label=col)

    plt.xlabel("Time since start (s)")
    plt.ylabel("Temperature")
    plt.title("Motor Temperatures vs Time")
    plt.legend(loc="best", ncol=2, fontsize=8)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print CSV start/end time and plot motor temperatures.")
    parser.add_argument("csv", help="Path to the CSV file. First column must be ROS timestamp.")
    parser.add_argument("--delimiter", "-d", default=",", help="CSV delimiter (default: ',').")
    parser.add_argument("--show", action="store_true", help="Show the plot window in addition to saving.")
    parser.add_argument("--output", "-o", help="Optional output image path. Defaults to <csv_dir>/motor_temperature_plot.png")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, delimiter=args.delimiter)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}") from exc

    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: timestamp and at least one data column.")

    timestamp_col = df.columns[0]
    rel_time_s, start_dt, end_dt = compute_time_info(df[timestamp_col])

    duration = (end_dt - start_dt).total_seconds()
    print(f"Start (UTC): {start_dt.isoformat()}")
    print(f"End   (UTC): {end_dt.isoformat()}")
    print(f"Duration(s): {duration:.3f}")

    temp_cols = find_temperature_columns(list(df.columns))
    if not temp_cols:
        # Fallback: try columns containing 'temp'
        temp_cols = [c for c in df.columns if "temp" in c.lower()]

    if not temp_cols:
        print("Warning: No temperature columns found (looking for 'motor_temperature' or '*temperature' or 'temp').")
    else:
        out_path = (
            args.output
            if args.output
            else os.path.join(os.path.dirname(csv_path), "motor_temperature_plot.png")
        )
        plot_temperatures(rel_time_s, df, temp_cols, out_path, show=args.show)
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()


