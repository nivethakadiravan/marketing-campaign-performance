# Model Evaluation Report: Revenue Regression & Profit/Loss Classification

**Data:** 164,712 cleaned campaigns (see the [EDA Report](EDA_Report.md))
**Notebook:** [`notebooks/01_cleaning_eda_and_modeling.ipynb`](../notebooks/01_cleaning_eda_and_modeling.ipynb)
**Outputs:** [`model_comparison.csv`](model_comparison.csv) · [`metrics.json`](metrics.json) · models in [`models/`](../models)

---

## 1. Set-up

| Item | Choice |
|---|---|
| Split | Stratified 80 / 20 on Profit_Flag: **131,769 train / 32,943 test**, `random_state = 42` |
| Model selection | 3-fold cross-validation on the training data only. The best algorithm is tuned with `RandomizedSearchCV`, and tuned settings are kept only if they beat the defaults |
| Final score | The tuned model is scored **once** on the untouched test set |
| Preprocessing (inside the `Pipeline`, so it is learned from training folds only) | median imputation → `log1p` for skewed money/count columns (Impressions, Clicks, Leads, Conversions, Acquisition_Cost, Total_Spend, Revenue) → `StandardScaler`; `log(1 + ROI)` for ROI; `OneHotEncoder` for Brand, Campaign_Type, Target_Audience, Language, Customer_Segment; the 6 multi-label channel flags and `Channel_Unknown` passed through |
| Regression target | Revenue, learned on a log scale (`TransformedTargetRegressor`) and predicted back in ₹ |
| Classification target | `Profit_Flag` (1 = ROI > 0): 76.1% Profit / 23.9% Loss |
| Algorithms | Linear / Logistic Regression, Decision Tree, Random Forest, Gradient Boosting (HistGradientBoosting) |

### Two feature sets: the brief's spec, plus a stricter forecast

| | Numeric inputs (plus categorical, channel and date features) | Why |
|---|---|---|
| **spec** (regression) | Duration, Impressions, Clicks, Leads, Conversions, Acquisition_Cost, Engagement_Score, Total_Spend, CTR, Lead_Rate, Conversion_Rate, Month, Weekday **+ ROI** | Follows the brief: "predict Revenue based on campaign features" |
| **spec** (classification) | same base features **+ Revenue, NO ROI** | Follows the brief's rule: **exclude ROI from classification features** (enforced with an `assert` in `src/train.py`) |
| **forecast** (both) | base features only: **no ROI, no Revenue** | What a marketer knows *before* revenue is booked. This is the honest test of predictive power |

**Why both?** Revenue, Spend and ROI are tied by the identity **Revenue = Spend × (1 + ROI)**, and **Profit ⇔ Revenue > Spend**. ROI is calculated *from* Revenue, so a revenue model that uses ROI is closer to reconstructing a known value than to forecasting. Likewise, a classifier that sees Revenue and Spend can work out profit even without ROI. The spec models meet the brief (and its R² ≥ 0.95 / accuracy ≥ 0.95 targets), and the forecast models show what is learnable without those inputs.

## 2. Revenue regression

### 2.1 Model comparison (test set, 32,943 campaigns)

| Feature set | Model | CV R² | **Test R²** | RMSE (₹) | MAE (₹) | MSE | MAPE |
|---|---|---|---|---|---|---|---|
| spec | **Linear Regression** ✔ | 1.0000 | **1.0000** | 6 | 5 | 39.0 | 0.002% |
| spec | Random Forest | 0.9982 | 0.9988 | 16,653 | 3,894 | 2.77e8 | 0.60% |
| spec | Gradient Boosting | 0.9960 | 0.9966 | 28,105 | 10,430 | 7.90e8 | 2.02% |
| spec | Decision Tree | 0.9956 | 0.9958 | 31,389 | 11,538 | 9.85e8 | 1.90% |
| forecast | **Linear Regression** ✔ | 0.7650 | **0.7621** | 235,265 | 156,439 | 5.53e10 | 35.3% |
| forecast | Gradient Boosting | 0.7618 | 0.7596 | 236,475 | 156,780 | 5.59e10 | 35.4% |
| forecast | Random Forest | 0.7579 | 0.7546 | 238,929 | 157,865 | 5.71e10 | 35.4% |
| forecast | Decision Tree | 0.7070 | 0.7085 | 260,393 | 166,120 | 6.78e10 | 37.7% |

(`figures/10_model_comparison.png`, `figures/11_regression_actual_vs_predicted.png`)

- **Spec: R² = 1.000, well above the 0.95 target.** Linear Regression wins because, on log scales, log Revenue = log Spend + log(1 + ROI) is exactly linear. The ₹6 RMSE is just rounding in the source data. The tree models get 0.996–0.999 because they approximate the same relationship with steps.
- **Forecast: R² = 0.762.** Without ROI, the missing piece is order value per conversion. It is spread evenly over ₹200–800 and not related to any other column. If every conversion were worth the average ₹500, R² would be 0.772, so the forecast model is within 0.01 of that limit. All four algorithms agree (0.71–0.76), which confirms the limit is in the data, not in the model.

### 2.2 What drives revenue (`figures/13_feature_importance.png`, permutation importance)
- **Spec:** ROI and Total_Spend carry the whole prediction, as the identity says they should.
- **Forecast:** **Conversions** dominates, far ahead of anything else; Acquisition_Cost and Leads add a little. Campaign type, channel, audience and language contribute essentially nothing.

## 3. Profit / Loss classification (ROI excluded)

### 3.1 Model comparison (test set)

| Feature set | Model | CV accuracy | **Test accuracy** | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| spec | **Logistic Regression** ✔ (tuned C = 100) | 0.9986 | **0.9986** | 0.9992 | 0.9990 | 0.9991 | 1.0000 |
| spec | Gradient Boosting | 0.9980 | 0.9981 | 0.9985 | 0.9990 | 0.9987 | 1.0000 |
| spec | Decision Tree | 0.9955 | 0.9962 | 0.9973 | 0.9977 | 0.9975 | 0.9966 |
| spec | Random Forest | 0.9803 | 0.9815 | 0.9840 | 0.9918 | 0.9879 | 0.9982 |
| forecast | **Gradient Boosting** ✔ (tuned) | 0.9118 | **0.9088** | 0.9231 | 0.9601 | 0.9413 | 0.9645 |
| forecast | Random Forest | 0.9114 | 0.9089 | 0.9239 | 0.9594 | 0.9413 | 0.9636 |
| forecast | Logistic Regression | 0.9112 | 0.9087 | 0.9306 | 0.9510 | 0.9407 | 0.9651 |
| forecast | Decision Tree | 0.9020 | 0.9015 | 0.9211 | 0.9523 | 0.9364 | 0.9484 |

Precision, recall and F1 are for the Profit class. (`figures/12_classification_confusion_roc.png`)

**Confusion matrices (test set):**

| spec (Logistic Regression) | Pred. Loss | Pred. Profit |
|---|---|---|
| Actual Loss | 7,841 | 21 |
| Actual Profit | 24 | 25,057 |

| forecast (Gradient Boosting) | Pred. Loss | Pred. Profit |
|---|---|---|
| Actual Loss | 5,857 | 2,005 |
| Actual Profit | 1,001 | 24,080 |

- **Spec: 99.86% accuracy without ROI, above the 0.95 target.** Only 45 of 32,943 campaigns are misclassified, all of them within 0.016 ROI of break-even. Logistic Regression wins because log Revenue − log Spend > 0 is exactly the profit rule, and that is linear in the log-features.
- **Forecast: 90.9% accuracy, ROC-AUC 0.96.** Profit then comes down to whether the order value (₹200–800, unknown) beats the acquisition cost. The best accuracy any model could reach with that information is **91.08%**, computed in the notebook, and the model reaches 90.9%. 84% of its errors are campaigns with acquisition cost of ₹300–700, where the outcome is close to a coin toss. It makes no errors below ₹200 or above ₹800.
- The forecast model is better at confirming profits (recall 96%) than at catching losses (74.5% of losses caught). If missing a loss is costly, lowering the decision threshold would trade some precision for more losses caught.

### 3.2 Key factors affecting profitability (permutation importance)
- **Spec:** Revenue and Total_Spend, i.e. the profit rule itself.
- **Forecast:** **Acquisition_Cost is about 60× more important than any other input** (accuracy drop 0.26 vs ≤ 0.005). Engagement, funnel rates, channels, campaign type and audience add almost nothing. This matches the EDA: cost per conversion relative to the ₹200–800 order value decides profit.

## 4. Data leakage checks
- ROI is **never** a classification input: it is excluded by the feature lists and enforced by an `assert`.
- All imputation, scaling and encoding is fitted inside the scikit-learn `Pipeline`, so cross-validation folds and the test set never influence preprocessing.
- `Total_Spend` (= Acquisition_Cost × Conversions) is a cost input known before revenue, so it is not leakage. `Profit_Flag` is derived only from ROI and is used only as a target.
- The forecast feature set removes Revenue as well, to show performance without any post-campaign financial information.

## 5. Model selection & deployment

| Use | Saved model | Test performance |
|---|---|---|
| Revenue, brief spec | `models/revenue_model_spec.joblib` (Linear Regression) | R² 1.000 |
| Revenue, pre-launch forecast | `models/revenue_model_forecast.joblib` (Linear Regression) | R² 0.762 |
| Profit/Loss, brief spec (no ROI) | `models/profit_model_spec.joblib` (Logistic Regression) | Accuracy 99.86% |
| Profit/Loss, pre-launch forecast | `models/profit_model_forecast.joblib` (Gradient Boosting) | Accuracy 90.9% |

The Streamlit app (`app.py`) lets the user enter campaign details (brand, type, audience, multi-select channels, funnel numbers, cost, and ROI for the spec model). It applies **the same feature engineering** (`src/features.build_input_row`): multi-label channel encoding, Total_Spend and funnel ratios. It then shows predicted revenue, total spend, implied ROI, a Profit/Loss verdict with probability gauge, the input funnel and a cost-band explanation. A batch page scores whole CSVs.

## 6. Limitations
- The data looks synthetically generated: order value is spread evenly and independent of everything else, and channel, type and audience have no effect. That is why the forecast models stop at about 0.76 R² and 91% accuracy. With real data, features such as product mix, basket size or past customer value would be needed to predict order value.
- The spec models' near-perfect scores reflect the accounting identity between Revenue, Spend and ROI, not forecasting skill. They are useful for validation and for what-if checks ("what revenue does a 2.5 ROI imply?"), while the forecast models are the ones to use for planning.
