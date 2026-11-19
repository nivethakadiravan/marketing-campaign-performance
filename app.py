"""Streamlit app: Multi-Brand Marketing Campaign Performance & Prediction.

Run:  streamlit run app.py
"""
import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.predict import predict_campaign, predict_frame

st.set_page_config(page_title="Marketing Campaign Intelligence", page_icon="📈", layout="wide")
BRAND_COLORS = {"Nykaa": "#e5007e", "Purplle": "#6a2c91", "Tira": "#1d3557"}
OPTIONS = {
    "Campaign_Type": ["Email", "Influencer", "Paid Ads", "SEO", "Social Media"],
    "Target_Audience": ["College Students", "Premium Shoppers", "Tier 2 City Customers",
                        "Working Women", "Youth"],
    "Customer_Segment": ["College Students", "Premium Shoppers", "Tier 2 City Customers",
                         "Working Women", "Youth"],
    "Language": ["Bengali", "English", "Hindi", "Tamil"],
}


def inr(x: float) -> str:
    if abs(x) >= 1e10:
        return f"₹{x / 1e7:,.0f} Cr"
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


@st.cache_data(show_spinner="Loading campaign data…")
def load_data() -> pd.DataFrame:
    if not config.FEATURES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(config.FEATURES_CSV, parse_dates=["Date"])
    return df


@st.cache_data
def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


df = load_data()
metrics = load_json(config.REPORTS_DIR / "metrics.json")
eda = load_json(config.REPORTS_DIR / "eda_summary.json")
crep = load_json(config.CLEANING_REPORT)
models_ready = (config.MODELS_DIR / "revenue_model_spec.joblib").exists()

st.sidebar.title("📈 Campaign Intelligence")
page = st.sidebar.radio("Navigate", ["Predict a campaign", "Performance dashboard", "Batch prediction",
                                     "Model performance", "Data cleaning", "Insights"])
if metrics:
    st.sidebar.markdown("---")
    rs, cs = metrics["regression_spec"], metrics["classification_spec"]
    st.sidebar.metric("Revenue model R²", f"{rs['test_metrics']['R2']:.4f}", help=rs["best_model"])
    st.sidebar.metric("Profit/Loss accuracy", f"{cs['test_metrics']['Accuracy']:.2%}", help=cs["best_model"])
    st.sidebar.caption("Brief-spec models. The stricter forecast models are on the Model performance page.")
if not models_ready:
    st.error("Models not found. Run `python run_pipeline.py` first.")
    st.stop()

# ======================================================================= PREDICT
if page == "Predict a campaign":
    st.title("Predict revenue and profit / loss for a campaign")
    mode = st.radio(
        "Model set",
        ["Brief specification (Revenue model uses ROI · Profit model excludes ROI)",
         "Pre-launch forecast (no ROI, no Revenue)"], horizontal=False)
    variant = "spec" if mode.startswith("Brief") else "forecast"

    with st.form("campaign"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Campaign**")
            brand = st.selectbox("Brand", list(BRAND_COLORS))
            ctype = st.selectbox("Campaign type", OPTIONS["Campaign_Type"], index=2)
            channels = st.multiselect("Channels used (multi-select)", config.CHANNELS,
                                      default=["Instagram", "YouTube"])
            duration = st.slider("Duration (days)", 5, 30, 15)
            run_date = st.date_input("Date", date(2025, 3, 10))
        with c2:
            st.markdown("**Audience**")
            audience = st.selectbox("Target audience", OPTIONS["Target_Audience"], index=4)
            segment = st.selectbox("Customer segment", OPTIONS["Customer_Segment"], index=4)
            language = st.selectbox("Language", OPTIONS["Language"], index=2)
            engagement = st.slider("Engagement score", 2.0, 31.0, 13.5, 0.1)
        with c3:
            st.markdown("**Funnel & cost**")
            impressions = st.number_input("Impressions", 1_000, 1_000_000, 60_000, 1_000)
            clicks = st.number_input("Clicks", 1, 1_000_000, 5_000, 100)
            leads = st.number_input("Leads", 1, 1_000_000, 2_000, 50)
            conversions = st.number_input("Conversions", 1, 1_000_000, 1_100, 50)
            acq = st.number_input("Acquisition cost per conversion (₹)", 1.0, 20_000.0, 150.0, 10.0)
            roi = st.number_input("ROI (required by the brief-spec revenue model)", -0.99, 100.0, 2.3, 0.1,
                                  disabled=variant == "forecast")
        submitted = st.form_submit_button("Predict", type="primary")

    if clicks > impressions or leads > clicks or conversions > leads:
        st.warning("Funnel looks inconsistent (expected impressions ≥ clicks ≥ leads ≥ conversions).")
    if submitted:
        vals = dict(Brand=brand, Campaign_Type=ctype, Target_Audience=audience, Language=language,
                    Customer_Segment=segment, Channel_Used=channels or [], Duration=duration,
                    Impressions=impressions, Clicks=clicks, Leads=leads, Conversions=conversions,
                    Acquisition_Cost=acq, Engagement_Score=engagement, Date=str(run_date),
                    ROI=roi if variant == "spec" else np.nan, Revenue=np.nan)
        r = predict_campaign(vals, variant)
        k = st.columns(4)
        k[0].metric("Predicted revenue", inr(r["predicted_revenue"]))
        k[1].metric("Total spend (cost × conversions)", inr(r["total_spend"]))
        k[2].metric("Implied ROI", f"{r['implied_roi']:.2f}")
        k[3].metric("Prediction", r["prediction"], f"{r['profit_probability']:.1%} profit probability",
                    delta_color="normal" if r["prediction"] == "Profit" else "inverse")
        st.caption(f"Revenue model: {r['revenue_model']} · Profit/Loss model: {r['profit_model']} "
                   f"({'spec' if variant == 'spec' else 'forecast'} feature set)")

        g1, g2, g3 = st.columns(3)
        with g1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=r["profit_probability"] * 100,
                                         number={"suffix": "%"}, title={"text": "Profit probability"},
                                         gauge={"axis": {"range": [0, 100]},
                                                "bar": {"color": "#2a9d8f" if r["prediction"] == "Profit"
                                                        else "#e76f51"},
                                                "threshold": {"line": {"color": "black"}, "value": 50}}))
            fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig, width="stretch")
        with g2:
            fig = go.Figure(go.Funnel(y=["Impressions", "Clicks", "Leads", "Conversions"],
                                      x=[impressions, clicks, leads, conversions],
                                      textinfo="value+percent previous", marker_color="#1f4e79"))
            fig.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), title="Input funnel")
            st.plotly_chart(fig, width="stretch")
        with g3:
            fig = px.bar(x=["Spend", "Predicted revenue"], y=[r["total_spend"], r["predicted_revenue"]],
                         color=["Spend", "Revenue"], color_discrete_sequence=["#e76f51", "#2a9d8f"],
                         labels={"x": "", "y": "₹"}, title="Spend vs predicted revenue")
            fig.update_layout(height=280, showlegend=False, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, width="stretch")

        aov = r["predicted_revenue"] / conversions
        if acq > 800:
            st.error(f"Acquisition cost ₹{acq:,.0f} is above the highest order value seen in the data "
                     "(₹800). Campaigns like this were loss-making in 100% of cases.")
        elif acq < 200:
            st.success(f"Acquisition cost ₹{acq:,.0f} is below the lowest order value (₹200). "
                       "Every historical campaign in this cost range was profitable.")
        else:
            st.info(f"Acquisition cost ₹{acq:,.0f} is inside the ₹200–800 order-value range, so the "
                    f"outcome depends on order value (predicted ≈ ₹{aov:,.0f} per conversion).")
        with st.expander("Features the model received (after the same feature engineering as training)"):
            eng = pd.Series(r["engineered"]).rename("value")
            st.dataframe(eng.to_frame(), width="stretch")

# ======================================================================= DASHBOARD
elif page == "Performance dashboard":
    st.title("Campaign performance dashboard")
    brands = st.multiselect("Brands", list(BRAND_COLORS), default=list(BRAND_COLORS))
    d = df[df.Brand.isin(brands)]
    if d.empty:
        st.stop()
    spend, rev = d.Total_Spend.sum(), d.Revenue.sum()
    k = st.columns(5)
    k[0].metric("Campaigns", f"{len(d):,}")
    k[1].metric("Revenue", inr(rev))
    k[2].metric("Spend", inr(spend))
    k[3].metric("Portfolio ROI", f"{(rev - spend) / spend:.2f}")
    k[4].metric("Profitable campaigns", f"{d.Profit_Flag.mean():.1%}")

    b = d.groupby("Brand").agg(revenue=("Revenue", "sum"), spend=("Total_Spend", "sum"),
                               median_roi=("ROI", "median"), profit_rate=("Profit_Flag", "mean")).reset_index()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(b.melt(id_vars="Brand", value_vars=["revenue", "spend"]), x="Brand", y="value",
                           color="variable", barmode="group", title="Revenue vs spend by brand",
                           color_discrete_sequence=["#2a9d8f", "#e76f51"], labels={"value": "₹"}),
                    width="stretch")
    c2.plotly_chart(px.bar(b, x="Brand", y="profit_rate", color="Brand", color_discrete_map=BRAND_COLORS,
                           title="Share of profitable campaigns", text_auto=".1%")
                    .update_yaxes(tickformat=".0%"), width="stretch")

    ct = d[d.Campaign_Type != "Unknown"].groupby(["Campaign_Type", "Brand"]).agg(
        median_roi=("ROI", "median")).reset_index()
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(ct, x="Campaign_Type", y="median_roi", color="Brand", barmode="group",
                           color_discrete_map=BRAND_COLORS, title="Median ROI by campaign type"),
                    width="stretch")
    rows = []
    for ch, col in zip(config.CHANNELS, config.CHANNEL_COLS):
        s = d[d[col] == 1]
        rows.append({"Channel": ch, "median_roi": s.ROI.median(), "profit_rate": s.Profit_Flag.mean(),
                     "campaigns": len(s)})
    chd = pd.DataFrame(rows).sort_values("median_roi")
    c4.plotly_chart(px.bar(chd, y="Channel", x="median_roi", orientation="h", text_auto=".3f",
                           title="Median ROI by channel (multi-label: a campaign counts for each channel)",
                           color_discrete_sequence=["#1f4e79"]), width="stretch")

    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, np.inf]
    labels = ["<100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700", "700-800",
              "800-1k", "1k-2k", ">2k"]
    band = d.assign(band=pd.cut(d.Acquisition_Cost, bins, labels=labels)).groupby("band", observed=True) \
        .Profit_Flag.mean().reset_index()
    st.plotly_chart(px.bar(band, x="band", y="Profit_Flag", text_auto=".0%",
                           title="Profitable campaigns by acquisition cost per conversion (₹)",
                           labels={"band": "Acquisition cost band", "Profit_Flag": "Profitable"},
                           color_discrete_sequence=["#2a9d8f"]).update_yaxes(tickformat=".0%"),
                    width="stretch")

    smp = d.sample(min(5000, len(d)), random_state=1)
    c5, c6 = st.columns(2)
    fig = px.scatter(smp, x="Total_Spend", y="Revenue", color="Profit_Label", log_x=True, log_y=True,
                     opacity=0.4, color_discrete_map={"Profit": "#2a9d8f", "Loss": "#e76f51"},
                     title="Spend vs revenue (sample of 5,000)", hover_data=["Campaign_ID", "Brand", "ROI"])
    c5.plotly_chart(fig, width="stretch")
    c6.plotly_chart(px.scatter(smp, x="Clicks", y="Revenue", color="Brand", opacity=0.35,
                               color_discrete_map=BRAND_COLORS, title="Clicks vs revenue"),
                    width="stretch")
    m = d.dropna(subset=["Date"]).assign(Month=lambda x: x.Date.dt.to_period("M").dt.to_timestamp()) \
        .groupby(["Month", "Brand"]).agg(revenue=("Revenue", "sum")).reset_index()
    st.plotly_chart(px.line(m, x="Month", y="revenue", color="Brand", markers=True,
                            color_discrete_map=BRAND_COLORS, title="Monthly revenue"), width="stretch")

    st.subheader("Top and bottom campaigns")
    metric = st.radio("Rank by", ["ROI", "Revenue"], horizontal=True)
    cols = ["Campaign_ID", "Brand", "Campaign_Type", "Channel_Used", "Conversions", "Acquisition_Cost",
            "Total_Spend", "Revenue", "ROI"]
    t1, t2 = st.columns(2)
    t1.markdown(f"**Top 10 by {metric}**")
    t1.dataframe(d.nlargest(10, metric)[cols], hide_index=True, width="stretch")
    t2.markdown(f"**Bottom 10 by {metric}**")
    t2.dataframe(d.nsmallest(10, metric)[cols], hide_index=True, width="stretch")

# ======================================================================= BATCH
elif page == "Batch prediction":
    st.title("Batch prediction")
    st.write("Upload a CSV with the raw campaign columns (same as the dataset). Every row gets a "
             "predicted revenue and profit / loss.")
    variant = st.radio("Model set", ["forecast", "spec"], horizontal=True,
                       format_func=lambda v: "Pre-launch forecast (no ROI/Revenue)" if v == "forecast"
                       else "Brief specification (needs ROI column)")
    up = st.file_uploader("CSV file", type="csv")
    sample = df.sample(200, random_state=7)[["Campaign_ID", "Brand", "Campaign_Type", "Target_Audience",
                                             "Duration", "Channel_Used", "Impressions", "Clicks", "Leads",
                                             "Conversions", "Acquisition_Cost", "ROI", "Language",
                                             "Engagement_Score", "Customer_Segment", "Date"]]
    st.download_button("Download a 200-row sample to try", sample.to_csv(index=False), "sample_campaigns.csv")
    if up:
        bdf = pd.read_csv(up)
        if variant == "spec" and "ROI" not in bdf:
            st.error("The brief-spec revenue model needs an ROI column.")
            st.stop()
        out = predict_frame(bdf, variant)
        st.success(f"Scored {len(out):,} campaigns")
        c1, c2 = st.columns(2)
        c1.metric("Predicted total revenue", inr(out.Predicted_Revenue.sum()))
        c2.metric("Predicted profitable", f"{(out.Predicted_Profit_Loss == 'Profit').mean():.1%}")
        st.dataframe(out.head(300), width="stretch")
        st.download_button("Download predictions", out.to_csv(index=False), "campaign_predictions.csv")

# ======================================================================= MODELS
elif page == "Model performance":
    st.title("Model performance")
    st.markdown(
        "**Two feature sets per task.** The *spec* set follows the brief: Revenue regression may use ROI; "
        "Profit/Loss classification uses everything **except ROI** (leakage rule). Because "
        "`Revenue = Spend × (1 + ROI)` and `Profit ⇔ Revenue > Spend`, the spec models are close to exact. "
        "The *forecast* set removes both ROI and Revenue, showing what can be predicted before revenue is "
        "known.")
    comp = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    for task, cols in [("regression", ["model", "cv_r2", "test_R2", "test_RMSE", "test_MAE", "test_MSE",
                                       "test_MAPE_pct", "tuned"]),
                       ("classification", ["model", "cv_accuracy", "test_Accuracy", "test_Precision",
                                           "test_Recall", "test_F1", "test_ROC_AUC", "tuned"])]:
        st.subheader("Revenue regression" if task == "regression" else "Profit / Loss classification")
        for variant in ("spec", "forecast"):
            sub = comp[(comp.task == task) & (comp.variant == variant)][cols]
            m = metrics[f"{task}_{variant}"]
            st.markdown(f"**{variant}** · best: **{m['best_model']}** · numeric inputs: "
                        f"`{', '.join(m['numeric_features'])}`")
            fmt = {c: "{:,.4f}" for c in cols if c not in ("model", "tuned")}
            if task == "regression":
                fmt.update({"test_RMSE": "{:,.0f}", "test_MAE": "{:,.0f}", "test_MSE": "{:,.3e}",
                            "test_MAPE_pct": "{:.2f}"})
            st.dataframe(sub.style.format(fmt), hide_index=True, width="stretch")
    for f in ["10_model_comparison.png", "11_regression_actual_vs_predicted.png",
              "12_classification_confusion_roc.png", "13_feature_importance.png"]:
        p = config.FIGURES_DIR / f
        if p.exists():
            st.image(str(p), width="stretch")

# ======================================================================= CLEANING
elif page == "Data cleaning":
    st.title("Data cleaning & preprocessing")
    k = st.columns(4)
    k[0].metric("Raw rows (3 CSVs)", f"{crep.get('rows_raw', 0):,}")
    k[1].metric("Rows with any missing value", f"{eda.get('raw_rows_with_any_missing_pct', 0)}%")
    k[2].metric("Rows dropped", f"{crep.get('rows_dropped_target_unrecoverable', 0):,}")
    k[3].metric("Clean rows", f"{crep.get('rows_final', 0):,}")
    miss = pd.DataFrame(eda.get("raw_missing_pct", {}))
    if not miss.empty:
        st.plotly_chart(px.bar(miss.reset_index().melt(id_vars="index"), y="index", x="value",
                               color="variable", barmode="group", orientation="h",
                               color_discrete_map=BRAND_COLORS,
                               labels={"index": "", "value": "% missing", "variable": "Brand"},
                               title="Missing values in the raw files", height=550), width="stretch")
    st.markdown(f"""
| Step | Result |
|---|---|
| Campaign_ID missing | {crep.get('campaign_id_rebuilt', 0):,} rebuilt from the `<prefix>-CMP-<1000+row>` pattern (matches {crep.get('campaign_id_pattern_match_pct', 0)}% of present IDs) |
| Duplicates | {crep.get('duplicate_rows', 0)} duplicate rows, {crep.get('duplicate_campaign_ids', 0)} duplicate IDs |
| ROI validation | {crep.get('roi_rows_checked', 0):,} rows checked against `(Revenue − Cost×Conversions) / (Cost×Conversions)`; max difference {crep.get('roi_max_abs_diff', 0)} (rounding); {crep.get('roi_corrected', 0)} corrected |
| Recovered with the ROI identity | {', '.join(f'{k}: {v:,}' for k, v in crep.get('recovered_by_roi_identity', {}).items())} |
| Funnel-ratio imputation | {', '.join(f'{k}: {v:,}' for k, v in crep.get('imputation', {}).get('funnel_ratio', {}).items())} |
| Engagement from CTR fit | {crep.get('imputation', {}).get('engagement_from_ctr', 0):,} |
| Categorical gaps → "Unknown" | {', '.join(f'{k}: {v:,}' for k, v in crep.get('imputation', {}).get('unknown_category', {}).items())} |
| Target unrecoverable (dropped) | {crep.get('rows_dropped_target_unrecoverable', 0):,} |
""")
    st.subheader("Multi-label encoding of Channel_Used")
    ex = df[["Channel_Used"] + config.CHANNEL_COLS + ["Num_Channels"]].drop_duplicates("Channel_Used").head(8)
    st.dataframe(ex, hide_index=True, width="stretch")

# ======================================================================= INSIGHTS
elif page == "Insights":
    st.title("Insights & recommendations")
    o = eda["overall"]
    band = eda["profit_by_cost_band"]
    ch = pd.DataFrame(eda["channel"]).T
    rs, rf = metrics["regression_spec"], metrics["regression_forecast"]
    cs, cf = metrics["classification_spec"], metrics["classification_forecast"]
    st.markdown(f"""
**Portfolio**
- {eda['n_campaigns']:,} campaigns generated **{inr(o['revenue'])}** revenue on **{inr(o['spend'])}** spend,
  a portfolio ROI of **{o['portfolio_roi']:.2f}**. **{o['profit_rate']:.1%}** of campaigns were profitable.
- The three brands perform almost identically (median ROI {', '.join(f"{b} {v['median_roi']:.2f}" for b, v in eda['brand'].items())}),
  so the differences are driven by *how* campaigns are run, not which brand runs them.

**What drives profitability**
- **Acquisition cost per conversion is the deciding factor.** Order value per conversion ranges ₹200–800.
  Every campaign with cost below ₹200 was profitable ({band['<100']['campaigns'] + band['100-200']['campaigns']:,} campaigns),
  profitability falls to {band['500-600']['profit_rate']:.0%} at ₹500–600 and to ~0% above ₹800.
- Top-decile ROI campaigns have a median cost of ₹{eda['top_vs_bottom_decile_medians']['Top 10% ROI']['Acquisition_Cost']:.0f}
  vs ₹{eda['top_vs_bottom_decile_medians']['Bottom 10% ROI']['Acquisition_Cost']:.0f} in the bottom decile, and
  **{eda['top_vs_bottom_decile_medians']['Top 10% ROI']['CTR']:.1%} vs {eda['top_vs_bottom_decile_medians']['Bottom 10% ROI']['CTR']:.1%} CTR**.
- **Channel choice barely moves ROI** (median ROI {ch.median_roi.min():.2f}–{ch.median_roi.max():.2f} across channels),
  and using 1, 2 or 3 channels gives the same results. Campaign type, audience and language also differ by under 0.06 ROI.

**Models**
- Brief-spec models: revenue **R² {rs['test_metrics']['R2']:.4f}** ({rs['best_model']}), profit/loss
  **accuracy {cs['test_metrics']['Accuracy']:.2%}** ({cs['best_model']}, ROI excluded).
- Forecast models without ROI/Revenue: R² **{rf['test_metrics']['R2']:.3f}**, accuracy **{cf['test_metrics']['Accuracy']:.2%}**.
  The revenue ceiling comes from order value per conversion, which varies randomly between ₹200 and ₹800 and is
  not recorded anywhere else in the data.

**Recommendations**
1. Set a hard **acquisition-cost ceiling of ~₹300 per conversion** (92% of campaigns at ₹200–300 were profitable); pause or rework anything above ₹800.
2. Optimise for **CTR and engagement** (creative and targeting quality): they separate top and bottom performers far more than channel mix.
3. Don't spend effort reshuffling channels or campaign types for ROI; reallocate budget towards campaigns with low cost per conversion.
4. Use the forecast classifier as a **pre-launch screen**: it flags likely loss-making plans from cost and funnel estimates with ~{cf['test_metrics']['Accuracy']:.0%} accuracy.
""")
    figs = sorted(config.FIGURES_DIR.glob("0*.png"))
    if figs:
        pick = st.selectbox("EDA figure", [f.name for f in figs])
        st.image(str(config.FIGURES_DIR / pick), width="stretch")
