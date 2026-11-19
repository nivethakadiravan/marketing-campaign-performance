"""Builds and executes notebooks/01_cleaning_eda_and_modeling.ipynb.

    python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "01_cleaning_eda_and_modeling.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def img(name):
    return code(f"display(Image(config.FIGURES_DIR / '{name}'))")


cells = [
    md("# Multi-Brand Marketing Campaign Performance Analysis & Prediction\n"
       "**Nykaa · Purplle · Tira: cleaning → feature engineering → EDA → regression & classification**\n\n"
       "This notebook walks through the pipeline using the functions in `src/`. `run_pipeline.py` "
       "runs the same steps from the command line, and `app.py` serves the saved models."),
    code("import sys, json, warnings\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent))\n"
         "warnings.filterwarnings('ignore')\n\nimport pandas as pd, numpy as np\n"
         "from IPython.display import Image, display\n\n"
         "from src import config, cleaning, features, eda, train\n"
         "from src.predict import predict_campaign\n"
         "pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200)"),

    md("## 1. Data collection: CSV → DataFrame\nThree brand files are loaded and stacked, with a `Brand` column added."),
    code("raw = cleaning.load_raw().drop(columns=['_row', '_prefix'])\nprint(raw.shape)\n"
         "raw.groupby('Brand').size()"),
    code("raw.head()"),
    code("raw.dtypes.to_frame('dtype').T"),
    md("### Missing values\nEvery column (even `Campaign_ID` for Purplle and Tira) is about 5% empty, and "
       "**more than half of all rows have at least one gap**, so dropping incomplete rows is not an option."),
    code("miss = (raw.drop(columns='Brand').isna().groupby(raw['Brand']).mean() * 100).round(2).T\nmiss"),
    code("print('Rows with any missing value: %.1f%%' % (raw.isna().any(axis=1).mean() * 100))\n"
         "print('Exact duplicate rows:', raw.duplicated().sum())"),

    md("### Validating ROI\nROI turns out to follow exactly "
       "`ROI = (Revenue − Spend) / Spend`, where `Spend = Acquisition_Cost × Conversions`."),
    code("spend = raw.Acquisition_Cost * raw.Conversions\ncalc = (raw.Revenue - spend) / spend\n"
         "diff = (calc - raw.ROI).abs().dropna()\n"
         "print(f'rows checked: {len(diff):,}  max |diff|: {diff.max():.4f}  (rounding only)')\n"
         "print('Order value per conversion (Revenue / Conversions):')\n"
         "(raw.Revenue / raw.Conversions).describe().round(1)"),
    md("Because this identity holds, a missing ROI, Revenue, Acquisition_Cost or Conversions value can be "
       "**recovered exactly** from the other three, instead of being guessed."),

    md("## 2. Cleaning & preprocessing"),
    code("clean, rep = cleaning.clean(cleaning.load_raw())\nclean.to_csv(config.CLEAN_CSV, index=False)\n"
         "with open(config.CLEANING_REPORT, 'w') as fh: json.dump(rep, fh, indent=2, default=str)\n"
         "print(json.dumps({k: rep[k] for k in ['campaign_id_rebuilt', 'duplicate_rows', 'roi_corrected', "
         "'recovered_by_roi_identity', 'rows_dropped_target_unrecoverable', 'imputation', 'rows_final']}, "
         "indent=1))"),
    code("print('Missing after cleaning:', clean.isna().sum()[clean.isna().sum() > 0].to_dict())\nclean.head()"),

    md("## 3. Feature engineering\n* **Multi-label encoding** of `Channel_Used` (`'Google, YouTube'` → "
       "`Ch_Google=1, Ch_YouTube=1`)\n* **Profit_Flag** = 1 if ROI > 0, else 0\n"
       "* `Total_Spend`, `CTR`, `Lead_Rate`, `Conversion_Rate`, `Month`, `Weekday`"),
    code("feat = features.add_features(clean)\nfeat.to_csv(config.FEATURES_CSV, index=False)\n"
         "feat[['Channel_Used'] + config.CHANNEL_COLS + ['Num_Channels']].drop_duplicates('Channel_Used').head(8)"),
    code("feat[['ROI', 'Profit_Flag', 'Profit_Label', 'Total_Spend', 'CTR', 'Lead_Rate', "
         "'Conversion_Rate']].head()"),
    code("feat.Profit_Label.value_counts(normalize=True).round(3)"),

    md("## 4. Exploratory data analysis"),
    code("s = eda.run_eda(feat)\npd.DataFrame(s['brand']).T"),
    img("00_target_distributions.png"), img("01_missing_values.png"), img("02_brand_performance.png"),
    md("The three brands are almost identical on every metric, at around 76% profitable campaigns and a "
       "portfolio ROI of about 1.94."),
    code("pd.DataFrame(s['campaign_type']).T"),
    img("03_campaign_type_brand.png"),
    md("### Top and bottom campaigns"),
    code("pd.DataFrame(s['top10_roi'])"),
    code("pd.DataFrame(s['bottom10_roi'])"),
    code("pd.DataFrame(s['top_vs_bottom_decile_medians'])"),
    md("### Spend, clicks, revenue and ROI"),
    img("04_correlation.png"), img("05_spend_clicks_revenue_roi.png"),
    code("pd.DataFrame({'pearson': s['corr_with_roi'], 'spearman': s['spearman_with_roi']})"),
    md("### The key driver: acquisition cost per conversion\nOrder value per conversion is spread between ₹200 "
       "and ₹800, so a campaign is profitable exactly when the order value is higher than the acquisition cost."),
    code("pd.DataFrame(s['profit_by_cost_band']).T"),
    img("06_profit_by_acquisition_cost.png"),
    md("### Channel effectiveness"),
    code("pd.DataFrame(s['channel']).T"),
    code("pd.DataFrame(s['num_channels']).T"),
    img("07_channel_effectiveness.png"), img("08_audience_segment_language.png"), img("09_monthly_trend.png"),

    md("## 5. Model building\n"
       "* **Split:** stratified 80/20 train/test. 3-fold CV on the training data picks the algorithm, the "
       "best one is tuned with `RandomizedSearchCV`, and the test set is used once.\n"
       "* **Pipeline:** median imputation → log1p for skewed money/count columns → scaling; one-hot "
       "encoding for categories; channel flags passed through. Revenue is modelled on a log scale.\n"
       "* **Leakage rule:** ROI is never a classification input (asserted in code).\n"
       "* **Two feature sets:** *spec* follows the brief (the revenue model may use ROI; the classifier "
       "excludes ROI but may use Revenue). *forecast* removes both ROI and Revenue. *(~6 minutes)*"),
    code("res = train.run(feat)"),
    code("comp = pd.read_csv(config.REPORTS_DIR / 'model_comparison.csv')\n"
         "comp[comp.task == 'regression'][['variant', 'model', 'cv_r2', 'test_R2', 'test_RMSE', 'test_MAE', "
         "'test_MSE', 'tuned']]"),
    code("comp[comp.task == 'classification'][['variant', 'model', 'cv_accuracy', 'test_Accuracy', "
         "'test_Precision', 'test_Recall', 'test_F1', 'test_ROC_AUC', 'tuned']]"),

    md("## 6. Model evaluation"),
    code("pd.DataFrame({k: v['test_metrics'] for k, v in res.items()}).round(4)"),
    img("10_model_comparison.png"), img("11_regression_actual_vs_predicted.png"),
    img("12_classification_confusion_roc.png"), img("13_feature_importance.png"),
    md("### Why the spec models are near-perfect, and how far the forecast models can go\n"
       "* `Revenue = Spend × (1 + ROI)` exactly, so with ROI available a linear model on log-features "
       "reproduces revenue (R² = 1.0).\n"
       "* `Profit ⇔ Revenue > Spend`, so with Revenue available (and ROI excluded) the classifier is ~99.9% "
       "accurate.\n"
       "* Without ROI or Revenue, the missing piece is order value per conversion, which is spread evenly "
       "over ₹200–800 and recorded nowhere else. The cells below compute the best accuracy and R² any model "
       "could reach under that condition."),
    code("c = feat.Acquisition_Cost; p = ((800 - c) / 600).clip(0, 1)\n"
         "print(f'Theoretical best accuracy without Revenue/ROI: {np.maximum(p, 1 - p).mean():.4f}')\n"
         "print(f\"Forecast classifier accuracy:                 {res['classification_forecast']['test_metrics']['Accuracy']:.4f}\")\n"
         "r2_ceiling = 1 - ((feat.Revenue - 500 * feat.Conversions) ** 2).sum() / "
         "((feat.Revenue - feat.Revenue.mean()) ** 2).sum()\n"
         "print(f'Revenue R² if order value were always its ₹500 mean: {r2_ceiling:.4f}')\n"
         "print(f\"Forecast regression R²:                            {res['regression_forecast']['test_metrics']['R2']:.4f}\")"),

    md("## 7. Prediction on a new campaign (same preprocessing as training)"),
    code("demo = dict(Brand='Nykaa', Campaign_Type='Paid Ads', Target_Audience='Youth', Language='Hindi',\n"
         "            Customer_Segment='Youth', Channel_Used=['Instagram', 'YouTube'], Duration=15,\n"
         "            Impressions=60000, Clicks=5000, Leads=2000, Conversions=1100, Acquisition_Cost=150,\n"
         "            Engagement_Score=13.5, Date='2025-03-10', ROI=2.3, Revenue=None)\n"
         "rows = []\nfor cost in [150, 450, 1200]:\n    for v in ['spec', 'forecast']:\n"
         "        d = dict(demo, Acquisition_Cost=cost, ROI=(500 - cost) / cost)\n"
         "        r = predict_campaign(d, v)\n"
         "        rows.append({'acq_cost': cost, 'model_set': v, 'pred_revenue': round(r['predicted_revenue']),\n"
         "                     'spend': r['total_spend'], 'prediction': r['prediction'],\n"
         "                     'p_profit': round(r['profit_probability'], 3)})\npd.DataFrame(rows)"),
    md("## 8. Summary\n"
       "* 166,665 raw campaigns → **164,712** after cleaning. ROI was validated, 28,000 values were "
       "recovered exactly from the ROI identity, and the remaining gaps were imputed.\n"
       "* **Acquisition cost per conversion decides profitability.** Channel mix, campaign type, audience "
       "and brand hardly matter.\n"
       "* Spec models: revenue **R² 1.000**, profit/loss **accuracy 99.9%**, both above the brief's 0.95 "
       "targets, with ROI kept out of the classifier.\n"
       "* Forecast models (no ROI/Revenue): R² 0.76 and accuracy 90.9%, close to the theoretical ceiling.\n"
       "* See `reports/EDA_Report.md` and `reports/Model_Evaluation_Report.md` for the full write-up."),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
print("Executing notebook (about 7 minutes)…")
NotebookClient(nb, timeout=3600, kernel_name="python3",
               resources={"metadata": {"path": str(OUT.parent)}}).execute()
nbf.write(nb, OUT)
print("Wrote", OUT)
