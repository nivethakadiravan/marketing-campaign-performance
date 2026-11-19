"""Load the three brand CSVs into one DataFrame and clean it.

Steps (each logged in data/processed/cleaning_report.json):
 1. CSV -> DataFrame for Nykaa, Purplle, Tira, with a Brand column added
 2. Strip and standardise text, normalise the multi-channel strings
 3. Rebuild missing Campaign_IDs (they follow <prefix>-CMP-<1000 + row number>)
 4. Remove duplicate rows / IDs, parse dates, check value ranges and funnel order
 5. Validate ROI = (Revenue - Spend) / Spend with Spend = Acquisition_Cost x Conversions;
    correct wrong ROI values and use the identity to recover missing
    ROI / Revenue / Acquisition_Cost / Conversions
 6. Drop rows whose targets (Revenue, ROI) cannot be recovered
 7. Impute the remaining gaps: funnel-ratio imputation for Impressions/Clicks/Leads,
    a CTR-based fit for Engagement_Score, brand medians for the rest, "Unknown" for text
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

NUMERIC = ["Duration", "Impressions", "Clicks", "Leads", "Conversions", "Revenue",
           "Acquisition_Cost", "ROI", "Engagement_Score"]
TEXT = ["Campaign_Type", "Target_Audience", "Channel_Used", "Language", "Customer_Segment"]


def load_raw() -> pd.DataFrame:
    frames = []
    for brand, (fname, prefix) in config.BRAND_FILES.items():
        df = pd.read_csv(config.RAW_DIR / fname)
        df.insert(0, "Brand", brand)
        df["_row"] = np.arange(len(df))
        df["_prefix"] = prefix
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _normalise_channels(s) -> str | float:
    if not isinstance(s, str) or not s.strip():
        return np.nan
    known = {c.lower(): c for c in config.CHANNELS}
    parts = [known.get(p.strip().lower(), p.strip().title()) for p in s.split(",") if p.strip()]
    return ", ".join(sorted(dict.fromkeys(parts)))


def recompute_roi(df: pd.DataFrame) -> pd.Series:
    spend = df["Acquisition_Cost"] * df["Conversions"]
    return (df["Revenue"] - spend) / spend


def clean(raw: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    rep: dict = {"rows_raw": int(len(raw)),
                 "rows_per_brand": raw["Brand"].value_counts().to_dict(),
                 "missing_before": raw[NUMERIC + TEXT + ["Campaign_ID", "Date"]].isna().sum().to_dict()}
    df = raw.copy()

    # 2. text standardisation
    for c in TEXT + ["Campaign_ID", "Date"]:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)
    df["Channel_Used"] = df["Channel_Used"].map(_normalise_channels)

    # 3. rebuild Campaign_ID
    expected = df["_prefix"] + "-CMP-" + (1000 + df["_row"]).astype(str)
    present = df["Campaign_ID"].notna()
    rep["campaign_id_pattern_match_pct"] = round(float((df.loc[present, "Campaign_ID"]
                                                        == expected[present]).mean() * 100), 2)
    rep["campaign_id_rebuilt"] = int((~present).sum())
    df["Campaign_ID"] = df["Campaign_ID"].fillna(expected)

    # 4. duplicates, types, ranges
    rep["duplicate_rows"] = int(df.drop(columns=["_row"]).duplicated().sum())
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "_row"])
    rep["duplicate_campaign_ids"] = int(df["Campaign_ID"].duplicated().sum())
    df = df.drop_duplicates(subset="Campaign_ID")
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nonneg = [c for c in NUMERIC if c != "ROI"]
    rep["negative_values_set_to_nan"] = int((df[nonneg] < 0).sum().sum())
    df[nonneg] = df[nonneg].mask(df[nonneg] < 0)
    rep["funnel_violations"] = {
        "clicks_gt_impressions": int((df.Clicks > df.Impressions).sum()),
        "leads_gt_clicks": int((df.Leads > df.Clicks).sum()),
        "conversions_gt_leads": int((df.Conversions > df.Leads).sum()),
    }

    # 5. ROI validation + identity-based recovery
    calc = recompute_roi(df)
    both = calc.notna() & df["ROI"].notna()
    diff = (calc - df["ROI"]).abs()
    rep["roi_rows_checked"] = int(both.sum())
    rep["roi_max_abs_diff"] = round(float(diff[both].max()), 4)
    wrong = both & (diff > config.ROI_TOLERANCE)
    rep["roi_corrected"] = int(wrong.sum())
    df.loc[wrong, "ROI"] = calc[wrong].round(2)

    rec = {}
    m = df.ROI.isna() & calc.notna()
    df.loc[m, "ROI"] = calc[m].round(2); rec["ROI"] = int(m.sum())
    spend = df.Acquisition_Cost * df.Conversions
    m = df.Revenue.isna() & spend.notna() & df.ROI.notna()
    df.loc[m, "Revenue"] = (spend[m] * (1 + df.ROI[m])).round(0); rec["Revenue"] = int(m.sum())
    m = df.Acquisition_Cost.isna() & df[["Revenue", "Conversions", "ROI"]].notna().all(axis=1) & (df.ROI > -1)
    df.loc[m, "Acquisition_Cost"] = (df.Revenue[m] / (df.Conversions[m] * (1 + df.ROI[m]))).round(2)
    rec["Acquisition_Cost"] = int(m.sum())
    m = df.Conversions.isna() & df[["Revenue", "Acquisition_Cost", "ROI"]].notna().all(axis=1) & (df.ROI > -1)
    df.loc[m, "Conversions"] = (df.Revenue[m] / (df.Acquisition_Cost[m] * (1 + df.ROI[m]))).round(0)
    rec["Conversions"] = int(m.sum())
    rep["recovered_by_roi_identity"] = rec

    # 6. targets must exist
    unrec = df.Revenue.isna() | df.ROI.isna()
    rep["rows_dropped_target_unrecoverable"] = int(unrec.sum())
    df = df[~unrec].copy()

    # 7. impute remaining features
    imp = {}
    ratios = {
        "ctr": (df.Clicks / df.Impressions).groupby(df.Brand).median(),
        "lead": (df.Leads / df.Clicks).groupby(df.Brand).median(),
        "conv": (df.Conversions / df.Leads).groupby(df.Brand).median(),
    }
    r = {k: df.Brand.map(v) for k, v in ratios.items()}
    funnel = ["Impressions", "Clicks", "Leads", "Conversions"]
    bounds = {c: (df[c].min(), df[c].max()) for c in funnel}
    was_na = df[funnel].isna()
    # walk the funnel both ways using brand-median conversion ratios
    df["Leads"] = df.Leads.fillna((df.Conversions / r["conv"]).round(0))
    df["Clicks"] = df.Clicks.fillna((df.Leads / r["lead"]).round(0))
    df["Impressions"] = df.Impressions.fillna((df.Clicks / r["ctr"]).round(0))
    df["Clicks"] = df.Clicks.fillna((df.Impressions * r["ctr"]).round(0))
    df["Leads"] = df.Leads.fillna((df.Clicks * r["lead"]).round(0))
    # Conversions and Acquisition_Cost both missing: spend is still known from
    # Revenue / (1 + ROI), so estimate Conversions from Leads and back out the cost
    m = df.Conversions.isna() & df.Acquisition_Cost.isna() & df.Leads.notna() & (df.ROI > -1)
    df.loc[m, "Conversions"] = (df.Leads[m] * r["conv"][m]).round(0).clip(lower=1)
    df.loc[m, "Acquisition_Cost"] = (df.Revenue[m] / (1 + df.ROI[m]) / df.Conversions[m]).round(2)
    imp["conversions_and_cost_from_spend"] = int(m.sum())
    # imputed values stay inside the observed range, and the funnel stays ordered
    # (only imputed upstream values are raised, so observed numbers and ROI never change)
    for c in funnel:
        df.loc[was_na[c], c] = df.loc[was_na[c], c].clip(*bounds[c])
    for up, down in [("Leads", "Conversions"), ("Clicks", "Leads"), ("Impressions", "Clicks")]:
        fix = was_na[up] & (df[up] < df[down])
        df.loc[fix, up] = df.loc[fix, down]
    imp["funnel_ratio"] = {c: int((was_na[c] & df[c].notna()).sum()) for c in funnel[:3]}

    # Engagement_Score tracks CTR almost perfectly (r ~ 0.96): fit and use it
    ctr = df.Clicks / df.Impressions
    ok = df.Engagement_Score.notna() & ctr.notna()
    slope, intercept = np.polyfit(ctr[ok], df.Engagement_Score[ok], 1)
    m = df.Engagement_Score.isna() & ctr.notna()
    df.loc[m, "Engagement_Score"] = (intercept + slope * ctr[m]).round(2)
    imp["engagement_from_ctr"] = int(m.sum())
    rep["engagement_ctr_fit"] = {"slope": round(float(slope), 3), "intercept": round(float(intercept), 3)}

    med_cols = ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
                "Acquisition_Cost", "Engagement_Score"]
    n_med = df[med_cols].isna().sum()
    for c in med_cols:
        df[c] = df[c].fillna(df.groupby("Brand")[c].transform("median"))
    imp["brand_median"] = {k: int(v) for k, v in n_med.items() if v}
    n_txt = df[TEXT].isna().sum()
    df[TEXT] = df[TEXT].fillna("Unknown")
    imp["unknown_category"] = {k: int(v) for k, v in n_txt.items()}
    imp["date_missing_left_as_NaT"] = int(df.Date.isna().sum())
    rep["imputation"] = imp

    for c in ["Duration", "Impressions", "Clicks", "Leads", "Conversions", "Revenue"]:
        df[c] = df[c].round(0).astype(int)
    # ROI is re-derived from the final numbers so it is exactly consistent
    df["ROI"] = recompute_roi(df).round(4)

    df = df.drop(columns=["_row", "_prefix"]).reset_index(drop=True)
    rep["rows_final"] = int(len(df))
    rep["missing_after"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    if verbose:
        print(json.dumps({k: rep[k] for k in ("rows_raw", "roi_corrected", "recovered_by_roi_identity",
                                              "rows_dropped_target_unrecoverable", "rows_final")}, indent=2))
    return df, rep


def run() -> pd.DataFrame:
    df, rep = clean(load_raw())
    df.to_csv(config.CLEAN_CSV, index=False)
    with open(config.CLEANING_REPORT, "w") as fh:
        json.dump(rep, fh, indent=2, default=str)
    return df


if __name__ == "__main__":
    run()
