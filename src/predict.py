"""Load the saved models and score a campaign typed in by a user."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from . import config
from .features import build_input_row


@lru_cache(maxsize=8)
def load(task: str, variant: str) -> dict:
    name = "revenue" if task == "regression" else "profit"
    path = config.MODELS_DIR / f"{name}_model_{variant}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python run_pipeline.py` first.")
    return joblib.load(path)


def predict_campaign(values: dict, variant: str = "spec") -> dict:
    """values: raw campaign inputs (Channel_Used may be a list). Returns revenue + profit/loss.

    spec     – Revenue model uses ROI; Profit model uses Revenue (never ROI).
    forecast – neither model sees ROI or Revenue.
    """
    row = build_input_row(values)
    reg, clf = load("regression", variant), load("classification", variant)
    revenue = float(reg["pipeline"].predict(row[reg["features"]])[0])

    clf_row = row.copy()
    if "Revenue" in clf["numeric"] and pd.isna(clf_row.at[0, "Revenue"]):
        clf_row["Revenue"] = revenue  # no actual revenue given: use the regression estimate
    p_profit = float(clf["pipeline"].predict_proba(clf_row[clf["features"]])[0, 1])
    spend = float(row.at[0, "Total_Spend"])
    return {
        "predicted_revenue": revenue,
        "total_spend": spend,
        "implied_roi": (revenue - spend) / spend if spend else float("nan"),
        "profit_probability": p_profit,
        "prediction": "Profit" if p_profit >= 0.5 else "Loss",
        "revenue_model": reg["model_name"],
        "profit_model": clf["model_name"],
        "engineered": row[["Total_Spend", "CTR", "Lead_Rate", "Conversion_Rate", "Num_Channels"]
                          + config.CHANNEL_COLS].iloc[0].to_dict(),
    }


def predict_frame(df: pd.DataFrame, variant: str = "forecast") -> pd.DataFrame:
    """Batch scoring for a CSV with the raw campaign columns."""
    from .features import add_features
    feat = add_features(df.assign(ROI=df.get("ROI"), Revenue=df.get("Revenue")))
    reg, clf = load("regression", variant), load("classification", variant)
    out = df.copy()
    out["Predicted_Revenue"] = reg["pipeline"].predict(feat[reg["features"]]).round(0)
    cf = feat.copy()
    if "Revenue" in clf["numeric"]:
        cf["Revenue"] = cf["Revenue"].fillna(out["Predicted_Revenue"]) if "Revenue" in cf else \
            out["Predicted_Revenue"]
    proba = clf["pipeline"].predict_proba(cf[clf["features"]])[:, 1]
    out["Profit_Probability"] = proba.round(4)
    out["Predicted_Profit_Loss"] = pd.Series(proba >= 0.5).map({True: "Profit", False: "Loss"}).values
    return out


if __name__ == "__main__":
    demo = dict(Brand="Nykaa", Campaign_Type="Paid Ads", Target_Audience="Youth", Language="Hindi",
                Customer_Segment="Youth", Channel_Used=["Instagram", "YouTube"], Duration=15,
                Impressions=60000, Clicks=5000, Leads=2000, Conversions=1100, Acquisition_Cost=150,
                Engagement_Score=13.5, Date="2025-03-10", ROI=2.3, Revenue=None)
    for v in ("spec", "forecast"):
        print(v, predict_campaign(demo, v))
