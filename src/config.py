"""Paths, column groups and modelling settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Raw files: brand name -> (file, Campaign_ID prefix)
BRAND_FILES = {
    "Nykaa": ("nykaa_campaign_data_with_nulls.csv", "NY"),
    "Purplle": ("purplle_campaign_data_with_nulls.csv", "PU"),
    "Tira": ("tira_campaign_data_with_nulls.csv", "TI"),
}

CLEAN_CSV = PROCESSED_DIR / "campaigns_clean.csv.gz"        # cleaned, missing values handled
FEATURES_CSV = PROCESSED_DIR / "campaigns_features.csv.gz"  # + encoded channels, Profit_Flag, ratios
CLEANING_REPORT = PROCESSED_DIR / "cleaning_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
ROI_TOLERANCE = 0.05  # |reported ROI - recomputed ROI| above this is treated as wrong

CATEGORICAL = ["Brand", "Campaign_Type", "Target_Audience", "Language", "Customer_Segment"]
CHANNELS = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]
CHANNEL_COLS = [f"Ch_{c}" for c in CHANNELS]

# Inputs a marketer knows once a campaign has run, before revenue is booked
BASE_NUMERIC = ["Duration", "Impressions", "Clicks", "Leads", "Conversions",
                "Acquisition_Cost", "Engagement_Score", "Total_Spend", "CTR",
                "Lead_Rate", "Conversion_Rate", "Month", "Weekday"]
# (Num_Channels is left out of the models: it is the exact sum of the channel columns)

# The brief's specification. ROI is NEVER a classification input (leakage rule).
FEATURE_SETS = {
    "regression": {
        "spec": BASE_NUMERIC + ["ROI"],   # brief: predict Revenue from campaign features incl. ROI
        "forecast": BASE_NUMERIC,         # stricter: no ROI (ROI is computed from Revenue)
    },
    "classification": {
        "spec": BASE_NUMERIC + ["Revenue"],  # brief: exclude ROI
        "forecast": BASE_NUMERIC,            # stricter: exclude ROI and Revenue
    },
}
# Skewed money/count columns get log1p inside the model pipeline
LOG_COLS = ["Impressions", "Clicks", "Leads", "Conversions", "Acquisition_Cost",
            "Total_Spend", "Revenue"]
