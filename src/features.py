"""Feature engineering: multi-label channel encoding, Profit_Flag, funnel ratios."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from . import config


def split_channels(s: str) -> list[str]:
    if not isinstance(s, str) or s == "Unknown":
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def encode_channels(series: pd.Series) -> pd.DataFrame:
    """Multi-label (multi-hot) encoding: 'Google, YouTube' -> Ch_Google=1, Ch_YouTube=1."""
    mlb = MultiLabelBinarizer(classes=config.CHANNELS)
    arr = mlb.fit_transform(series.map(split_channels))
    return pd.DataFrame(arr, columns=config.CHANNEL_COLS, index=series.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = pd.concat([out, encode_channels(out["Channel_Used"])], axis=1)
    out["Num_Channels"] = out[config.CHANNEL_COLS].sum(axis=1)
    out["Channel_Unknown"] = (out["Channel_Used"] == "Unknown").astype(int)

    out["Total_Spend"] = (out["Acquisition_Cost"] * out["Conversions"]).round(2)
    out["CTR"] = out["Clicks"] / out["Impressions"]
    out["Lead_Rate"] = out["Leads"] / out["Clicks"]
    out["Conversion_Rate"] = out["Conversions"] / out["Leads"]

    date = pd.to_datetime(out["Date"], errors="coerce")
    out["Month"] = date.dt.month
    out["Weekday"] = date.dt.dayofweek

    # Target for classification: 1 = Profit (ROI > 0), 0 = Loss / break-even
    out["Profit_Flag"] = (out["ROI"] > 0).astype(int)
    out["Profit_Label"] = np.where(out["Profit_Flag"] == 1, "Profit", "Loss")
    return out


def build_input_row(values: dict) -> pd.DataFrame:
    """Turn raw campaign inputs (as typed in the app) into a model-ready row,
    applying exactly the same feature engineering as training."""
    row = pd.DataFrame([values])
    if isinstance(row.at[0, "Channel_Used"], (list, tuple)):
        row.at[0, "Channel_Used"] = ", ".join(sorted(row.at[0, "Channel_Used"])) or "Unknown"
    if "ROI" not in row:
        row["ROI"] = np.nan
    row = add_features(row)
    return row


def run() -> pd.DataFrame:
    df = pd.read_csv(config.CLEAN_CSV, parse_dates=["Date"])
    feat = add_features(df)
    feat.to_csv(config.FEATURES_CSV, index=False)
    return feat


if __name__ == "__main__":
    f = run()
    print(f.shape, f["Profit_Flag"].mean().round(3))
    print(f[config.CHANNEL_COLS + ["Num_Channels"]].head())
