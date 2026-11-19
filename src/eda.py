"""Exploratory data analysis: brand, campaign, channel and funnel performance.

Saves figures to reports/figures and numbers to reports/eda_summary.json.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .cleaning import load_raw  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
BRAND_COLORS = {"Nykaa": "#e5007e", "Purplle": "#6a2c91", "Tira": "#1d3557"}


def _fig(name):
    p = config.FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return str(p.relative_to(config.ROOT))


def channel_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (campaign, channel) for channel-level analysis."""
    rows = []
    for ch, col in zip(config.CHANNELS, config.CHANNEL_COLS):
        sub = df[df[col] == 1]
        rows.append(pd.DataFrame({"Channel": ch, "ROI": sub.ROI, "Revenue": sub.Revenue,
                                  "Profit_Flag": sub.Profit_Flag, "Conversions": sub.Conversions,
                                  "CTR": sub.CTR, "Acquisition_Cost": sub.Acquisition_Cost}))
    return pd.concat(rows, ignore_index=True)


def run_eda(df: pd.DataFrame) -> dict:
    s, figs = {}, []
    s["n_campaigns"] = int(len(df))
    s["date_range"] = [str(pd.to_datetime(df.Date).min().date()), str(pd.to_datetime(df.Date).max().date())]

    # 0. Missing values in the raw data
    raw = load_raw()
    rr = raw.drop(columns=["_row", "_prefix"])
    miss = (rr.drop(columns="Brand").isna().groupby(rr["Brand"]).mean() * 100).T
    s["raw_missing_pct"] = miss.round(2).to_dict()
    s["raw_rows_with_any_missing_pct"] = round(float(raw.drop(columns=["_row", "_prefix"])
                                                     .isna().any(axis=1).mean() * 100), 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    miss.plot.barh(ax=ax, color=[BRAND_COLORS[b] for b in miss.columns], width=0.8)
    ax.set(title="Missing values in the raw data (% of rows)", xlabel="% missing")
    figs.append(_fig("01_missing_values.png"))

    # 1. Brand performance
    b = df.groupby("Brand").agg(campaigns=("Campaign_ID", "count"), revenue=("Revenue", "sum"),
                                spend=("Total_Spend", "sum"), avg_roi=("ROI", "mean"),
                                median_roi=("ROI", "median"), profit_rate=("Profit_Flag", "mean"),
                                avg_ctr=("CTR", "mean"), avg_engagement=("Engagement_Score", "mean"))
    b["portfolio_roi"] = (b.revenue - b.spend) / b.spend
    s["brand"] = b.round(4).to_dict(orient="index")
    s["overall"] = {"revenue": float(df.Revenue.sum()), "spend": float(df.Total_Spend.sum()),
                    "portfolio_roi": float((df.Revenue.sum() - df.Total_Spend.sum()) / df.Total_Spend.sum()),
                    "profit_rate": float(df.Profit_Flag.mean()), "median_roi": float(df.ROI.median()),
                    "mean_roi": float(df.ROI.mean())}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (col, title, fmt) in zip(axes, [("revenue", "Total revenue (₹ Cr)", 1e7),
                                            ("spend", "Total spend (₹ Cr)", 1e7),
                                            ("median_roi", "Median ROI", 1),
                                            ("profit_rate", "Share of profitable campaigns", 1)]):
        vals = b[col] / fmt
        ax.bar(b.index, vals, color=[BRAND_COLORS[i] for i in b.index])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.2f}" if fmt == 1 else f"{v:,.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
    figs.append(_fig("02_brand_performance.png"))

    # 2. Campaign type x brand
    ct = df[df.Campaign_Type != "Unknown"].groupby("Campaign_Type").agg(
        campaigns=("ROI", "size"), median_roi=("ROI", "median"), profit_rate=("Profit_Flag", "mean"),
        avg_revenue=("Revenue", "mean"), avg_conversions=("Conversions", "mean"))
    s["campaign_type"] = ct.round(4).to_dict(orient="index")
    pv = df[df.Campaign_Type != "Unknown"].pivot_table(index="Campaign_Type", columns="Brand",
                                                       values="ROI", aggfunc="median")
    pv2 = df[df.Campaign_Type != "Unknown"].pivot_table(index="Campaign_Type", columns="Brand",
                                                        values="Profit_Flag", aggfunc="mean") * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.heatmap(pv, annot=True, fmt=".2f", cmap="crest", ax=axes[0])
    axes[0].set(title="Median ROI by campaign type and brand", xlabel="", ylabel="")
    sns.heatmap(pv2, annot=True, fmt=".1f", cmap="crest", ax=axes[1])
    axes[1].set(title="Profitable campaigns (%) by type and brand", xlabel="", ylabel="")
    figs.append(_fig("03_campaign_type_brand.png"))

    # 3. Top / bottom campaigns
    cols = ["Campaign_ID", "Brand", "Campaign_Type", "Channel_Used", "Revenue", "Total_Spend", "ROI"]
    s["top10_roi"] = df.nlargest(10, "ROI")[cols].round(2).to_dict(orient="records")
    s["bottom10_roi"] = df.nsmallest(10, "ROI")[cols].round(2).to_dict(orient="records")
    s["top10_revenue"] = df.nlargest(10, "Revenue")[cols].round(2).to_dict(orient="records")
    top_q, bot_q = df.ROI.quantile(0.9), df.ROI.quantile(0.1)
    comp = pd.DataFrame({
        "Top 10% ROI": df[df.ROI >= top_q][["Acquisition_Cost", "CTR", "Conversion_Rate",
                                            "Engagement_Score", "Conversions", "Num_Channels"]].median(),
        "Bottom 10% ROI": df[df.ROI <= bot_q][["Acquisition_Cost", "CTR", "Conversion_Rate",
                                               "Engagement_Score", "Conversions", "Num_Channels"]].median()})
    s["top_vs_bottom_decile_medians"] = comp.round(4).to_dict()

    # 4. Relationships: spend / clicks / revenue / ROI
    num = ["Duration", "Impressions", "Clicks", "Leads", "Conversions", "Revenue", "Acquisition_Cost",
           "Total_Spend", "ROI", "Engagement_Score", "CTR", "Lead_Rate", "Conversion_Rate", "Num_Channels"]
    corr = df[num].corr()
    s["corr_with_revenue"] = corr["Revenue"].drop("Revenue").round(3).sort_values(ascending=False).to_dict()
    s["corr_with_roi"] = corr["ROI"].drop("ROI").round(3).sort_values(ascending=False).to_dict()
    s["spearman_with_roi"] = df[num].corr(method="spearman")["ROI"].drop("ROI").round(3) \
        .sort_values(ascending=False).to_dict()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation between campaign metrics")
    figs.append(_fig("04_correlation.png"))

    smp = df.sample(6000, random_state=1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for brand, g in smp.groupby("Brand"):
        axes[0].scatter(g.Total_Spend, g.Revenue, s=5, alpha=0.35, c=BRAND_COLORS[brand], label=brand)
    lim = [smp[["Total_Spend", "Revenue"]].min().min(), smp[["Total_Spend", "Revenue"]].max().max()]
    axes[0].plot(lim, lim, c="black", lw=1, ls="--", label="break-even")
    axes[0].set(xscale="log", yscale="log", title="Spend vs revenue (above the line = profit)",
                xlabel="Total spend (₹, log)", ylabel="Revenue (₹, log)")
    axes[0].legend(markerscale=3)
    axes[1].scatter(smp.Clicks, smp.Revenue / 1e5, s=5, alpha=0.3, c="#2a9d8f")
    axes[1].set(title=f"Clicks vs revenue (r = {corr.loc['Clicks', 'Revenue']:.2f})",
                xlabel="Clicks", ylabel="Revenue (₹ lakh)")
    axes[2].scatter(smp.Acquisition_Cost, smp.ROI, s=5, alpha=0.3, c="#e76f51")
    axes[2].axhline(0, c="black", lw=1)
    axes[2].set(xscale="log", title="Acquisition cost vs ROI", xlabel="Acquisition cost per conversion (₹, log)",
                ylabel="ROI")
    figs.append(_fig("05_spend_clicks_revenue_roi.png"))

    # 5. Acquisition-cost threshold: the main profitability driver
    aov = df.Revenue / df.Conversions
    s["avg_order_value"] = {"min": float(aov.min()), "median": float(aov.median()), "max": float(aov.max())}
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, np.inf]
    labels = ["<100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700",
              "700-800", "800-1k", "1k-2k", ">2k"]
    df = df.assign(Cost_Band=pd.cut(df.Acquisition_Cost, bins=bins, labels=labels))
    band = df.groupby("Cost_Band", observed=True).agg(campaigns=("ROI", "size"),
                                                      profit_rate=("Profit_Flag", "mean"),
                                                      median_roi=("ROI", "median"))
    s["profit_by_cost_band"] = band.round(4).to_dict(orient="index")
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(band.index.astype(str), band.profit_rate * 100, color="#2a9d8f")
    ax1.set(ylabel="Profitable campaigns (%)", xlabel="Acquisition cost per conversion (₹)",
            title="Profitability collapses once acquisition cost exceeds the order value (₹200–800)")
    for i, v in enumerate(band.profit_rate * 100):
        ax1.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(band.index.astype(str), band.campaigns, c="#1d3557", marker="o")
    ax2.set_ylabel("Campaigns", color="#1d3557")
    ax2.grid(False)
    figs.append(_fig("06_profit_by_acquisition_cost.png"))

    # 6. Channel effectiveness (multi-label)
    cl = channel_long(df)
    ch = cl.groupby("Channel").agg(campaigns=("ROI", "size"), median_roi=("ROI", "median"),
                                   mean_roi=("ROI", "mean"), profit_rate=("Profit_Flag", "mean"),
                                   avg_revenue=("Revenue", "mean"), avg_ctr=("CTR", "mean"))
    s["channel"] = ch.round(4).to_dict(orient="index")
    known = df[df.Channel_Unknown == 0]
    nch = known.groupby("Num_Channels").agg(campaigns=("ROI", "size"), median_roi=("ROI", "median"),
                                            profit_rate=("Profit_Flag", "mean"),
                                            avg_revenue=("Revenue", "mean"))
    s["num_channels"] = nch.round(4).to_dict(orient="index")
    combo = known.groupby("Channel_Used").agg(campaigns=("ROI", "size"), median_roi=("ROI", "median"),
                                              profit_rate=("Profit_Flag", "mean"))
    combo = combo[combo.campaigns >= 300].sort_values("median_roi", ascending=False)
    s["best_channel_combos"] = combo.head(5).round(4).to_dict(orient="index")
    s["worst_channel_combos"] = combo.tail(5).round(4).to_dict(orient="index")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    chs = ch.sort_values("median_roi")
    axes[0].barh(chs.index, chs.median_roi, color="#1f4e79")
    axes[0].set(title="Median ROI by channel", xlabel="Median ROI")
    axes[1].barh(chs.index, chs.profit_rate * 100, color="#2a9d8f")
    axes[1].set(title="Profitable campaigns (%) by channel", xlabel="%",
                xlim=(chs.profit_rate.min() * 100 - 3, chs.profit_rate.max() * 100 + 2))
    axes[2].bar(nch.index.astype(str), nch.median_roi, color="#e9c46a")
    axes[2].set(title="Median ROI by number of channels", xlabel="Channels used", ylabel="Median ROI")
    figs.append(_fig("07_channel_effectiveness.png"))

    # 7. Audience, segment, language
    seg = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for ax, col in zip(axes, ["Target_Audience", "Customer_Segment", "Language"]):
        g = df[df[col] != "Unknown"].groupby(col).agg(median_roi=("ROI", "median"),
                                                      profit_rate=("Profit_Flag", "mean"),
                                                      campaigns=("ROI", "size")).sort_values("median_roi")
        seg[col] = g.round(4).to_dict(orient="index")
        ax.barh(g.index, g.median_roi, color="#6a2c91")
        ax.set(title=f"Median ROI by {col.replace('_', ' ').lower()}", xlabel="Median ROI")
    s["segments"] = seg
    figs.append(_fig("08_audience_segment_language.png"))

    # 8. Monthly trend
    d = df.dropna(subset=["Date"]).copy()
    d["Month_Start"] = pd.to_datetime(d.Date).dt.to_period("M").dt.to_timestamp()
    mt = d.groupby(["Month_Start", "Brand"]).agg(revenue=("Revenue", "sum"),
                                                 median_roi=("ROI", "median")).reset_index()
    s["monthly_revenue_cv_pct"] = round(float(d.groupby("Month_Start").Revenue.sum().std()
                                              / d.groupby("Month_Start").Revenue.sum().mean() * 100), 2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))
    for brand, g in mt.groupby("Brand"):
        axes[0].plot(g.Month_Start, g.revenue / 1e7, marker="o", c=BRAND_COLORS[brand], label=brand)
        axes[1].plot(g.Month_Start, g.median_roi, marker="o", c=BRAND_COLORS[brand], label=brand)
    axes[0].set(title="Monthly revenue by brand (₹ Cr)", ylabel="₹ Cr")
    axes[1].set(title="Monthly median ROI by brand", ylabel="Median ROI")
    for ax in axes:
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
    figs.append(_fig("09_monthly_trend.png"))

    # 9. Target distributions
    s["profit_flag_counts"] = df.Profit_Label.value_counts().to_dict()
    s["roi_quantiles"] = df.ROI.quantile([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(3).to_dict()
    s["revenue_quantiles"] = df.Revenue.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).round(0).to_dict()
    s["funnel_medians"] = {"CTR": float(df.CTR.median()), "Lead_Rate": float(df.Lead_Rate.median()),
                           "Conversion_Rate": float(df.Conversion_Rate.median())}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    sns.histplot(df.Revenue / 1e5, bins=80, ax=axes[0], color="#1f4e79")
    axes[0].set(title="Revenue distribution (right-skewed)", xlabel="Revenue (₹ lakh)")
    sns.histplot(df.ROI.clip(upper=df.ROI.quantile(0.99)), bins=80, ax=axes[1], color="#2a9d8f")
    axes[1].axvline(0, c="#e76f51")
    axes[1].set(title="ROI distribution (clipped at 99th pct; red = break-even)", xlabel="ROI")
    pc = df.groupby("Brand").Profit_Label.value_counts(normalize=True).unstack() * 100
    pc[["Profit", "Loss"]].plot.bar(stacked=True, ax=axes[2], color=["#2a9d8f", "#e76f51"], rot=0)
    axes[2].set(title="Profit_Flag by brand (%)", ylabel="%", xlabel="")
    figs.append(_fig("00_target_distributions.png"))

    s["figures"] = sorted(figs)
    with open(config.REPORTS_DIR / "eda_summary.json", "w") as fh:
        json.dump(s, fh, indent=2, default=str)
    return s


if __name__ == "__main__":
    out = run_eda(pd.read_csv(config.FEATURES_CSV))
    print(json.dumps({k: out[k] for k in ("n_campaigns", "overall", "brand")}, indent=2, default=str))
