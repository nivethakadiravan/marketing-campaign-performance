"""End-to-end pipeline: 3 CSVs -> cleaning -> feature engineering -> EDA -> models.

Usage:
    python run_pipeline.py            # full run (~6 minutes)
    python run_pipeline.py --skip-eda
"""
import argparse
import time

from src import cleaning, eda, features, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eda", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    print("== Step 1/4: data collection & cleaning")
    clean = cleaning.run()
    print(f"   {len(clean):,} clean campaigns -> data/processed/campaigns_clean.csv.gz")

    print("== Step 2/4: feature engineering (multi-label channels, Profit_Flag, ratios)")
    feat = features.run()
    print(f"   {feat.shape[1]} columns, profit share {feat.Profit_Flag.mean():.1%} "
          "-> data/processed/campaigns_features.csv.gz")

    if not args.skip_eda:
        print("== Step 3/4: exploratory data analysis")
        s = eda.run_eda(feat)
        print(f"   {len(s['figures'])} figures -> reports/figures/")

    print("== Step 4/4: regression & classification models")
    res = train.run(feat)
    rs, cs = res["regression_spec"], res["classification_spec"]
    print(f"\nRevenue (spec):      {rs['best_model']}  R² = {rs['test_metrics']['R2']:.4f}")
    print(f"Profit/Loss (spec):  {cs['best_model']}  Accuracy = {cs['test_metrics']['Accuracy']:.4f}")
    print(f"\nDone in {time.time() - t0:.0f}s. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
