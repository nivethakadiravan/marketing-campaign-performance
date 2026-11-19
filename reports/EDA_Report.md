# EDA Report: Multi-Brand Marketing Campaigns (Nykaa · Purplle · Tira)

**Source:** `nykaa_` / `purplle_` / `tira_campaign_data_with_nulls.csv` (3 × 55,555 rows × 16 columns)
**Period:** 1 Jul 2024 – 24 Jun 2025
**Notebook:** [`notebooks/01_cleaning_eda_and_modeling.ipynb`](../notebooks/01_cleaning_eda_and_modeling.ipynb). Every number below comes from it.
**Model results:** [Model Evaluation Report](Model_Evaluation_Report.md)

---

## 1. The dataset

| Item | Value |
|---|---|
| Files | 3 brand CSVs, identical schema, 55,555 campaigns each → **166,665 rows** |
| Columns | Campaign_ID, Campaign_Type, Target_Audience, Duration, Channel_Used, Impressions, Clicks, Leads, Conversions, Revenue, Acquisition_Cost, ROI, Language, Engagement_Score, Customer_Segment, Date |
| Categories | 5 campaign types (Email, Influencer, Paid Ads, SEO, Social Media), 5 audiences / segments, 4 languages, 6 channels |
| `Channel_Used` | 1–3 channels per campaign in one comma-separated string (e.g. `"WhatsApp, Google, YouTube"`), so it needs multi-label encoding |
| `Acquisition_Cost` | Cost **per conversion**. Total spend = Acquisition_Cost × Conversions |

## 2. Data quality and cleaning

| Issue found | Action taken |
|---|---|
| **Every column is ~5% empty** (`figures/01_missing_values.png`), and **54.7% of rows have at least one gap** | Dropping incomplete rows would lose half the data, so gaps were recovered or imputed instead (below) |
| `Campaign_ID` missing in 5,417 Purplle/Tira rows | IDs follow `<NY/PU/TI>-CMP-<1000 + row number>` (100% of present IDs match), so they were **rebuilt exactly** |
| Duplicates | 0 duplicate rows and 0 duplicate IDs after rebuilding (checked anyway) |
| Value ranges / funnel order | No negatives. Impressions ≥ Clicks ≥ Leads ≥ Conversions holds in every row |
| **ROI validation** | ROI = (Revenue − Spend) / Spend with Spend = Acquisition_Cost × Conversions. Checked on 136,312 complete rows, the largest difference is **0.028** (rounding), so **0 corrections** were needed |
| Missing ROI / Revenue / Acquisition_Cost / Conversions | **Recovered exactly** from that identity when the other three values were present: ROI 7,064 · Revenue 7,137 · Acquisition_Cost 6,991 · Conversions 6,846 (**28,038 values**) |
| Revenue or ROI still unknown | 1,953 rows (1.2%) **dropped**, because target values should not be invented |
| Impressions / Clicks / Leads missing | Funnel-ratio imputation from neighbouring funnel steps using brand-median CTR, lead rate and conversion rate (~8,000 each), clipped to the observed range, with funnel order preserved |
| Conversions **and** cost both missing (362) | Spend is still known (Revenue / (1 + ROI)), so Conversions was estimated from Leads and the cost backed out. ROI stays unchanged |
| Engagement_Score missing (7,978) | Engagement tracks CTR almost perfectly (r = 0.96), so it was estimated with a linear fit on CTR |
| Duration missing (8,236) | Brand median |
| Categorical gaps (~8,100 per column) | Set to `"Unknown"`: an honest separate category, not a guessed value |
| Date missing (7,989) | Left empty. Month/weekday features are median-imputed inside the model pipeline |

**Result: 164,712 clean campaigns** (Nykaa 54,878 · Purplle 54,908 · Tira 54,926), with no missing values except Date.

## 3. Feature engineering

| Feature | How it is built |
|---|---|
| `Ch_Email … Ch_YouTube` | **Multi-label (multi-hot) encoding** of `Channel_Used` with `MultiLabelBinarizer`. `"Google, YouTube"` → Ch_Google = 1, Ch_YouTube = 1 |
| `Num_Channels`, `Channel_Unknown` | Number of channels used; flag for a missing channel list |
| **`Profit_Flag`** | **1 if ROI > 0 (Profit), else 0 (Loss / break-even)**: 125,401 profit (76.1%) and 39,311 loss |
| `Total_Spend` | Acquisition_Cost × Conversions |
| `CTR`, `Lead_Rate`, `Conversion_Rate` | Clicks/Impressions, Leads/Clicks, Conversions/Leads |
| `Month`, `Weekday` | From Date |

## 4. Exploratory data analysis

### 4.1 Overall and brand performance (`figures/02_brand_performance.png`)

| | Nykaa | Purplle | Tira | **All** |
|---|---|---|---|---|
| Campaigns | 54,878 | 54,908 | 54,926 | **164,712** |
| Revenue | ₹2,832.6 Cr | ₹2,818.7 Cr | ₹2,813.2 Cr | **₹8,464.5 Cr** |
| Spend | ₹958.0 Cr | ₹960.7 Cr | ₹960.2 Cr | **₹2,878.9 Cr** |
| Portfolio ROI | 1.96 | 1.93 | 1.93 | **1.94** |
| Median ROI | 1.24 | 1.22 | 1.24 | 1.23 |
| Profitable campaigns | 76.1% | 75.9% | 76.4% | **76.1%** |

- Every ₹1 of spend returned ₹2.94 of revenue overall, but **24% of campaigns lost money**.
- **The brands are practically indistinguishable** on every metric (CTR 8.5%, engagement 13.8, ROI within 0.04). Brand is not what separates good campaigns from bad ones.
- ROI is heavily right-skewed: median 1.23, mean 2.69, 99th percentile 21.1, max 79.3. Revenue is also skewed (median ₹3.6 L, 99th percentile ₹23.3 L), which is why the models use log scales.

### 4.2 Campaign types (`figures/03_campaign_type_brand.png`)

| Type | Median ROI | Profitable | Avg revenue |
|---|---|---|---|
| Paid Ads | **1.27** | 76.5% | ₹5.19 L |
| Social Media | 1.24 | **76.8%** | ₹5.16 L |
| SEO | 1.22 | 75.9% | ₹5.11 L |
| Email | 1.22 | 75.8% | ₹5.14 L |
| Influencer | 1.21 | 75.9% | ₹5.10 L |

Paid Ads and Social Media are marginally ahead, but the whole spread is only 0.06 ROI and 1 percentage point of profitability.

### 4.3 Top and bottom campaigns
- **Best ROI:** TI-CMP-46406 (Tira, Social Media, Google + Instagram), ROI 79.3: ₹40.4 L revenue on ₹50 K spend. NY-CMP-13007 (Nykaa, Paid Ads) has the highest revenue in the dataset (₹45.8 L) at ROI 74.4.
- **Worst ROI:** TI-CMP-29039 (Tira, Email), ROI −0.99: ₹3,895 revenue on ₹2.86 L spend.
- **What separates the top 10% of campaigns from the bottom 10%** (medians):

| | Top 10% ROI | Bottom 10% ROI |
|---|---|---|
| Acquisition cost / conversion | **₹47** | **₹1,208** |
| CTR | 11.7% | 4.4% |
| Conversion rate (lead → sale) | 62.7% | 47.0% |
| Engagement score | 20.1 | 6.5 |
| Conversions | 2,185 | 173 |
| Channels used | 2 | 2 |

### 4.4 Relationships between spend, clicks, revenue and ROI (`figures/04_correlation.png`, `05_spend_clicks_revenue_roi.png`)
- **Revenue** rises with the funnel: Conversions (r = 0.88), Leads (0.81) and Clicks (0.72). Bigger, better-converting campaigns earn more.
- **ROI** is driven by cost. Its Spearman correlation with Acquisition_Cost is **−0.93**, much stronger than its correlation with conversions (+0.81) or engagement (+0.54).
- Total spend is essentially **uncorrelated with revenue** (r ≈ 0). Spending more does not buy more revenue. On the spend-vs-revenue chart, profit and loss campaigns sit either side of the break-even line at every spend level.
- Engagement_Score tracks CTR almost one-for-one (r = 0.96).

### 4.5 The profitability driver: acquisition cost vs order value (`figures/06_profit_by_acquisition_cost.png`)
Revenue per conversion (average order value) is spread between **₹200 and ₹800** (median ₹499). It is unrelated to brand, type, channel or audience. A campaign is therefore profitable exactly when its order value is higher than its acquisition cost:

| Acquisition cost per conversion | Campaigns | Profitable | Median ROI |
|---|---|---|---|
| < ₹100 | 37,771 | **100%** | 6.79 |
| ₹100–200 | 41,999 | **100%** | 2.41 |
| ₹200–300 | 25,007 | 92% | 1.03 |
| ₹300–400 | 15,419 | 76% | 0.45 |
| ₹400–500 | 10,325 | 59% | 0.13 |
| ₹500–600 | 7,263 | 41% | −0.09 |
| ₹600–700 | 5,411 | 26% | −0.22 |
| ₹700–800 | 3,952 | 9% | −0.34 |
| > ₹800 | 17,565 | **~0%** | −0.45 to −0.83 |

**Takeaway:** below ₹200 per conversion a campaign cannot lose money, and above ₹800 it cannot make money. Almost all the uncertainty sits in the ₹200–800 band.

### 4.6 Channel effectiveness (`figures/07_channel_effectiveness.png`)
Each channel appears in ~52,000 campaigns (multi-label counting):

| Channel | Median ROI | Profitable | Avg revenue |
|---|---|---|---|
| Instagram | **1.26** | **76.4%** | ₹5.17 L |
| Email | 1.25 | 76.3% | ₹5.17 L |
| WhatsApp | 1.23 | 76.3% | ₹5.13 L |
| Facebook | 1.23 | 76.2% | ₹5.14 L |
| Google | 1.23 | 75.9% | ₹5.12 L |
| YouTube | 1.22 | 76.0% | ₹5.12 L |

- Using **1, 2 or 3 channels gives the same median ROI** (1.23 / 1.24 / 1.24).
- Best combinations (≥ 300 campaigns): Facebook + Google + Instagram (1.34), Instagram + WhatsApp (1.33). The weakest is Google + WhatsApp (1.16). These are small differences that could be partly noise.
- **Channel mix is a weak lever compared with acquisition cost.**

### 4.7 Audience, segment, language and time (`figures/08_audience_segment_language.png`, `09_monthly_trend.png`)
- Premium Shoppers (target audience) and Hindi (language) are marginally the best (median ROI 1.25 and 1.26), but the spread across every audience, segment and language is under 0.04 ROI.
- Monthly revenue is flat at about ₹220–245 Cr per brand, and all three brands move together, with no brand pulling ahead over time. There is no seasonality beyond calendar effects: February dips because it has 28 days, and June 2025 drops because the data ends on the 24th. Monthly median ROI moves randomly between 1.15 and 1.34.

## 5. Key insights

1. **Acquisition cost per conversion is the main profitability factor.** Order values are ₹200–800, so campaigns under ₹200 per conversion were all profitable and campaigns over ₹800 were almost all loss-making.
2. **Quality of engagement separates winners from losers.** Top-decile campaigns have 2.7× the CTR and 3× the engagement of bottom-decile ones, and convert leads 16 points better.
3. **Brand, channel, campaign type, audience and language barely matter for ROI.** All differences are ≤ 0.06 median ROI. Budget reallocation between channels will not fix unprofitable campaigns.
4. **Spending more does not raise revenue.** Total spend and revenue are uncorrelated. Efficiency (cost per conversion) is what counts, not scale.
5. **Data quality:** missing values were widespread, but the deterministic ROI formula let 28,038 key values be recovered exactly rather than guessed.

## 6. Recommendations

1. Set a **cost-per-conversion ceiling of about ₹300** (92% of campaigns at ₹200–300 were profitable). Review anything above ₹500 and stop campaigns above ₹800.
2. Invest in **creative and targeting quality**, which lift CTR and engagement, rather than in adding channels.
3. Treat channel and campaign-type choices as **secondary optimisations**. The few slightly better combinations (Instagram + WhatsApp, Facebook + Google + Instagram) are worth A/B testing, not wholesale budget shifts.
4. Use the **pre-launch forecast model** in the app to screen campaign plans. It flags likely loss-makers from planned cost and funnel numbers with ~91% accuracy.
