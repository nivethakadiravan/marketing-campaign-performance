"""Regression (Revenue) and classification (Profit/Loss) models.

Two feature sets per task (see config.FEATURE_SETS):
  * "spec"     – the brief's specification: Revenue regression may use ROI;
                 Profit/Loss classification uses every feature EXCEPT ROI.
  * "forecast" – stricter, deployable before revenue is known: no ROI and no Revenue.

For every task/feature-set: 4 algorithms are compared with 3-fold CV on the training
split, the best is tuned with RandomizedSearchCV, and the tuned model is evaluated
once on the held-out 20% test split.
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

from . import config  # noqa: E402
from .transforms import log1p_clip, log_roi  # noqa: E402

sns.set_theme(style="whitegrid")
RS = config.RANDOM_STATE


# ------------------------------------------------------------------ preprocessing
_log1p, _log_roi = log1p_clip, log_roi


def make_preprocessor(numeric: list[str]) -> ColumnTransformer:
    logs = [c for c in numeric if c in config.LOG_COLS]
    roi = [c for c in numeric if c == "ROI"]
    plain = [c for c in numeric if c not in logs + roi]
    steps = [
        ("log", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("log", FunctionTransformer(_log1p, feature_names_out="one-to-one")),
                          ("sc", StandardScaler())]), logs),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), plain),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), config.CATEGORICAL),
        ("chan", "passthrough", config.CHANNEL_COLS + ["Channel_Unknown"]),
    ]
    if roi:
        steps.insert(1, ("roi", Pipeline([("imp", SimpleImputer(strategy="median")),
                                          ("log", FunctionTransformer(_log_roi,
                                                                      feature_names_out="one-to-one")),
                                          ("sc", StandardScaler())]), roi))
    return ColumnTransformer(steps, verbose_feature_names_out=False)


def regressors() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=14, min_samples_leaf=5, random_state=RS),
        "Random Forest": RandomForestRegressor(n_estimators=120, min_samples_leaf=2, max_samples=0.4,
                                               n_jobs=-1, random_state=RS),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                                           random_state=RS),
    }


def classifiers() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, max_samples=0.4,
                                                n_jobs=-1, random_state=RS),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                            random_state=RS),
    }


SEARCH_SPACES = {
    "Linear Regression": {},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"model__max_depth": [8, 12, 16, 20, None],
                      "model__min_samples_leaf": [1, 2, 5, 10, 20]},
    "Random Forest": {"model__n_estimators": [100, 200], "model__max_depth": [None, 20, 30],
                      "model__min_samples_leaf": [1, 2, 4], "model__max_features": [1.0, 0.5, "sqrt"]},
    "Gradient Boosting": {"model__learning_rate": [0.05, 0.1, 0.2], "model__max_iter": [300, 500],
                          "model__max_leaf_nodes": [31, 63], "model__l2_regularization": [0, 0.1, 1.0]},
}


def build_pipeline(task: str, model, numeric: list[str]) -> Pipeline:
    pre = make_preprocessor(numeric)
    if task == "regression":
        # Revenue is heavily right-skewed: learn log1p(Revenue), predict back on the original scale
        if not isinstance(model, TransformedTargetRegressor):
            model = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
        # keep the 'model__' prefix pointing at the inner estimator for tuning
        return Pipeline([("pre", pre), ("ttr", model)])
    return Pipeline([("pre", pre), ("model", model)])


def _space(task, name):
    sp = SEARCH_SPACES[name]
    if task == "regression":
        return {k.replace("model__", "ttr__regressor__"): v for k, v in sp.items()}
    return sp


# ------------------------------------------------------------------ metrics
def reg_metrics(y, p) -> dict:
    mse = mean_squared_error(y, p)
    return {"R2": r2_score(y, p), "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y, p),
            "MSE": float(mse), "MAPE_pct": 100 * mean_absolute_percentage_error(y, p)}


def clf_metrics(y, p, proba) -> dict:
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p),
            "Recall": recall_score(y, p), "F1": f1_score(y, p), "ROC_AUC": roc_auc_score(y, proba)}


# ------------------------------------------------------------------ training
def model_columns(numeric: list[str]) -> list[str]:
    return numeric + config.CATEGORICAL + config.CHANNEL_COLS + ["Channel_Unknown"]


def train_task(df: pd.DataFrame, task: str, variant: str, idx_tr, idx_te) -> dict:
    numeric = config.FEATURE_SETS[task][variant]
    assert not (task == "classification" and "ROI" in numeric), "ROI must not be a classification input"
    cols = model_columns(numeric)
    target = "Revenue" if task == "regression" else "Profit_Flag"
    X_tr, X_te = df.loc[idx_tr, cols], df.loc[idx_te, cols]
    y_tr, y_te = df.loc[idx_tr, target], df.loc[idx_te, target]

    if task == "regression":
        models, cv = regressors(), KFold(config.CV_FOLDS, shuffle=True, random_state=RS)
        scoring, key = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}, "r2"
    else:
        models, cv = classifiers(), StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=RS)
        scoring, key = {"accuracy": "accuracy", "f1": "f1"}, "accuracy"

    print(f"\n--- {task} / {variant}  ({len(numeric)} numeric features"
          f"{' incl. ROI' if 'ROI' in numeric else ''}{' incl. Revenue' if 'Revenue' in numeric else ''})")
    rows = []
    for name, m in models.items():
        t0 = time.time()
        res = cross_validate(build_pipeline(task, m, numeric), X_tr, y_tr, cv=cv, scoring=scoring,
                             n_jobs=1)
        row = {"model": name, **{f"cv_{k}": float(np.mean(res[f"test_{k}"])) for k in scoring},
               "cv_seconds": round(time.time() - t0, 1)}
        if task == "regression":
            row["cv_rmse"] = -row["cv_rmse"]
        rows.append(row)
        print(f"  {name:<20} CV {key}={row[f'cv_{key}']:.4f}  ({row['cv_seconds']}s)")

    comp = pd.DataFrame(rows).sort_values(f"cv_{key}", ascending=False).reset_index(drop=True)
    best_name = comp.loc[0, "model"]

    # Tune the best algorithm (search on a sample for speed, then refit on all training data)
    space = _space(task, best_name)
    best_params = {}
    pipe = build_pipeline(task, models[best_name], numeric)
    if space:
        n = min(len(X_tr), 40_000)
        samp = X_tr.sample(n, random_state=RS).index
        search = RandomizedSearchCV(pipe, space, n_iter=min(8, int(np.prod([len(v) for v in space.values()]))),
                                    cv=cv, scoring=scoring[key], random_state=RS, n_jobs=1)
        search.fit(X_tr.loc[samp], y_tr.loc[samp])
        default = cross_validate(build_pipeline(task, models[best_name], numeric), X_tr.loc[samp],
                                 y_tr.loc[samp], cv=cv, scoring=scoring[key])["test_score"].mean()
        if search.best_score_ > default:  # only keep tuned settings if they beat the defaults
            best_params = search.best_params_
            pipe.set_params(**best_params)
            print(f"  tuned {best_name}: {best_params} ({default:.4f} -> {search.best_score_:.4f})")
        else:
            print(f"  tuning did not beat defaults for {best_name} ({search.best_score_:.4f} <= "
                  f"{default:.4f}); keeping defaults")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    fit_s = round(time.time() - t0, 1)

    # Every model also gets a test score so the comparison table is complete
    test_rows = []
    for name, m in models.items():
        p = pipe if name == best_name else build_pipeline(task, m, numeric).fit(X_tr, y_tr)
        if task == "regression":
            test_rows.append({"model": name, **{f"test_{k}": v for k, v in
                                                reg_metrics(y_te, p.predict(X_te)).items()}})
        else:
            pr = p.predict_proba(X_te)[:, 1]
            test_rows.append({"model": name, **{f"test_{k}": v for k, v in
                                                clf_metrics(y_te, (pr >= 0.5).astype(int), pr).items()}})
    comp = comp.merge(pd.DataFrame(test_rows), on="model")
    comp["tuned"] = comp["model"].eq(best_name)

    if task == "regression":
        pred = pipe.predict(X_te)
        metrics = reg_metrics(y_te, pred)
        extra = {"y_test": y_te.to_numpy(), "pred": pred}
    else:
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = clf_metrics(y_te, pred, proba)
        extra = {"y_test": y_te.to_numpy(), "pred": pred, "proba": proba,
                 "confusion_matrix": confusion_matrix(y_te, pred).tolist()}
    print(f"  BEST {best_name}: " + ", ".join(f"{k}={v:,.4f}" for k, v in metrics.items()))

    # Permutation importance on a test sample: which inputs drive the prediction
    samp = X_te.sample(min(4000, len(X_te)), random_state=RS).index
    pi = permutation_importance(pipe, X_te.loc[samp], y_te.loc[samp], n_repeats=5,
                                random_state=RS, n_jobs=1,
                                scoring="r2" if task == "regression" else "accuracy")
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)

    out = config.MODELS_DIR / f"{'revenue' if task == 'regression' else 'profit'}_model_{variant}.joblib"
    joblib.dump({"pipeline": pipe, "features": cols, "numeric": numeric, "model_name": best_name,
                 "task": task, "variant": variant}, out)
    return {"task": task, "variant": variant, "best_model": best_name, "best_params": best_params,
            "fit_seconds": fit_s, "features": cols, "test_metrics": metrics,
            "comparison": comp, "importance": imp, **extra}


def run(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_csv(config.FEATURES_CSV)
    idx_tr, idx_te = train_test_split(df.index, test_size=config.TEST_SIZE, random_state=RS,
                                      stratify=df["Profit_Flag"])
    print(f"Train {len(idx_tr):,} | Test {len(idx_te):,} | Profit share {df.Profit_Flag.mean():.3f}")
    results = {}
    for task in ("regression", "classification"):
        for variant in ("spec", "forecast"):
            results[f"{task}_{variant}"] = train_task(df, task, variant, idx_tr, idx_te)
    _save(results, df.loc[idx_te])
    return results


# ------------------------------------------------------------------ outputs
def _fig(name):
    p = config.FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return str(p.relative_to(config.ROOT))


def _save(results: dict, test_df: pd.DataFrame) -> None:
    summary, figs, tables = {}, [], []
    for key, r in results.items():
        comp = r["comparison"].copy()
        comp.insert(0, "variant", r["variant"])
        comp.insert(0, "task", r["task"])
        tables.append(comp)
        summary[key] = {k: r[k] for k in ("task", "variant", "best_model", "best_params",
                                          "test_metrics", "fit_seconds")}
        summary[key]["n_features"] = len(r["features"])
        summary[key]["numeric_features"] = config.FEATURE_SETS[r["task"]][r["variant"]]
        summary[key]["top_features"] = r["importance"].head(10).round(4).to_dict()
        if "confusion_matrix" in r:
            summary[key]["confusion_matrix"] = r["confusion_matrix"]
    pd.concat(tables).round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    with open(config.REPORTS_DIR / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # 1. comparison bars
    allc = pd.concat(tables)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    reg = allc[allc.task == "regression"]
    sns.barplot(data=reg, y="model", x="test_R2", hue="variant", palette=["#1f4e79", "#e9c46a"], ax=axes[0])
    axes[0].axvline(0.95, ls="--", c="#e76f51", lw=1)
    axes[0].set(title="Revenue regression – test R²  (dashed = 0.95 target)", xlabel="R²", ylabel="",
                xlim=(min(0.4, reg.test_R2.min() - 0.05), 1.0))
    clf = allc[allc.task == "classification"]
    sns.barplot(data=clf, y="model", x="test_Accuracy", hue="variant", palette=["#1f4e79", "#e9c46a"],
                ax=axes[1])
    axes[1].axvline(0.95, ls="--", c="#e76f51", lw=1)
    axes[1].set(title="Profit/Loss classification – test accuracy", xlabel="Accuracy", ylabel="",
                xlim=(min(0.7, clf.test_Accuracy.min() - 0.03), 1.0))
    figs.append(_fig("10_model_comparison.png"))

    # 2. actual vs predicted + residuals (spec + forecast regression)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, key, col in [(axes[0], "regression_spec", "#1f4e79"), (axes[1], "regression_forecast", "#e9c46a")]:
        r = results[key]
        s = np.random.default_rng(RS).choice(len(r["y_test"]), 4000, replace=False)
        ax.scatter(r["y_test"][s] / 1e5, r["pred"][s] / 1e5, s=6, alpha=0.35, c=col)
        lim = max(r["y_test"][s].max(), r["pred"][s].max()) / 1e5
        ax.plot([0, lim], [0, lim], c="#e76f51", lw=1)
        ax.set(title=f"{r['variant']}: {r['best_model']} (R² {r['test_metrics']['R2']:.3f})",
               xlabel="Actual revenue (₹ lakh)", ylabel="Predicted revenue (₹ lakh)")
    r = results["regression_spec"]
    res = (r["y_test"] - r["pred"]) / 1e5
    sns.histplot(res, bins=80, ax=axes[2], color="#2a9d8f")
    axes[2].set(title="Residuals – spec model", xlabel="Actual − predicted (₹ lakh)",
                xlim=np.percentile(res, [0.5, 99.5]))
    figs.append(_fig("11_regression_actual_vs_predicted.png"))

    # 3. confusion matrices + ROC
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, key in [(axes[0], "classification_spec"), (axes[1], "classification_forecast")]:
        r = results[key]
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Loss", "Profit"], yticklabels=["Loss", "Profit"])
        ax.set(title=f"{r['variant']}: {r['best_model']} (acc {r['test_metrics']['Accuracy']:.3f})",
               xlabel="Predicted", ylabel="Actual")
    for key, col in [("classification_spec", "#1f4e79"), ("classification_forecast", "#e9c46a")]:
        r = results[key]
        fpr, tpr, _ = roc_curve(r["y_test"], r["proba"])
        axes[2].plot(fpr, tpr, c=col, label=f"{r['variant']} (AUC {r['test_metrics']['ROC_AUC']:.3f})")
    axes[2].plot([0, 1], [0, 1], ls="--", c="grey")
    axes[2].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    axes[2].legend()
    figs.append(_fig("12_classification_confusion_roc.png"))

    # 4. feature importance
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, key in zip(axes.ravel(), ["regression_spec", "regression_forecast",
                                      "classification_spec", "classification_forecast"]):
        imp = results[key]["importance"].head(10)[::-1]
        ax.barh(imp.index, imp.values, color="#2a9d8f")
        ax.set(title=f"{key.replace('_', ' / ')} – permutation importance",
               xlabel="Drop in R²" if "regression" in key else "Drop in accuracy")
    figs.append(_fig("13_feature_importance.png"))
    summary["_figures"] = figs
    with open(config.REPORTS_DIR / "metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)


if __name__ == "__main__":
    run()
