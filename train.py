"""Clean the data, run the time-based backtest, and fit the final model.

Run: python train.py            (full run, a few minutes)
     python train.py --skip-backtest   (only fit and save the final model)
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from freight_rate.config import (
    ARTIFACT_DIR,
    BACKTEST_FOLDS,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from freight_rate.data import build_market_table, city_coordinates, clean_features, load_loads
from freight_rate.metrics import regression_metrics
from freight_rate.model import LaneRatePerMileBaseline, LogLinearBaseline, RateModel
from freight_rate.validation import backtest, find_label_outliers

CANDIDATES = {
    "lane_rate_per_mile": LaneRatePerMileBaseline,
    "log_linear_ols": LogLinearBaseline,
    "lgbm_no_time": lambda: RateModel(trend="none"),
    "lgbm_linear_drift": lambda: RateModel(trend="linear"),
    "lgbm_date_feature": lambda: RateModel(trend="feature"),
}
FINAL_MODEL = "lgbm_date_feature"


def unseen_city_check(train: pd.DataFrame, outliers: pd.Series, n_cities: int = 8) -> dict:
    """Validation has 8 cities that never appear in training (~12% of rows).

    Simulate that on the last backtest fold: hide 8 random cities from the
    training window and compare error on test loads that touch a hidden city
    with loads that do not. For simplicity this uses the full-data label
    flags; it compares two groups of the same test window, so that choice
    affects both groups alike.
    """
    start, end = BACKTEST_FOLDS[-1]
    rng = np.random.default_rng(RANDOM_STATE)
    cities = sorted(set(train.pickup) | set(train.delivery))
    hidden = sorted(rng.choice(cities, size=n_cities, replace=False))
    touches = train.pickup.isin(hidden) | train.delivery.isin(hidden)

    fit_rows = train[(train.date < start) & ~touches & ~outliers]
    test = train[(train.date >= start) & (train.date <= end) & ~outliers]
    prediction = pd.Series(RateModel().fit(fit_rows).predict(test), index=test.index)
    result = {"hidden_cities": hidden, "fold": f"{start}..{end}"}
    test_touches = touches.loc[test.index]
    for name, mask in [("touches_hidden_city", test_touches), ("seen_cities_only", ~test_touches)]:
        result[name] = regression_metrics(test.loc[mask, TARGET], prediction[mask])
    return result


def random_split_comparison(train: pd.DataFrame, outliers: pd.Series) -> dict:
    """Score the same model with a shuffled 5-fold split, for contrast only.

    A random split lets the model see loads from the same days as the test
    loads, so it measures interpolation, not forecasting the next months.
    """
    rows = train[~outliers]
    oof = np.zeros(len(rows))
    for fit_idx, test_idx in KFold(5, shuffle=True, random_state=RANDOM_STATE).split(rows):
        oof[test_idx] = RateModel().fit(rows.iloc[fit_idx]).predict(rows.iloc[test_idx])
    return regression_metrics(rows[TARGET], oof)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    train_raw = load_loads(TRAIN_PATH)
    validation_raw = load_loads(VALIDATION_PATH)
    market = build_market_table(train_raw, validation_raw)
    train = clean_features(train_raw, market)

    outliers = find_label_outliers(train)
    print(f"label outliers: {outliers.sum()} of {len(train)} ({outliers.mean():.2%})")

    metrics: dict = {
        "data_quality": {
            "train_rows": len(train),
            "negative_weight": int(train.weight_was_negative.sum()),
            "missing_weight": int(train.weight_missing.sum()),
            "capped_weight": int(train.weight_capped.sum()),
            "missing_market_index": int(train.market_index_missing.sum()),
            "label_outliers": int(outliers.sum()),
            "label_outlier_share": float(outliers.mean()),
        }
    }

    if not args.skip_backtest:
        results = backtest(train, outliers, CANDIDATES)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(REPORT_DIR / "backtest_results.csv", index=False)
        cols = ["mae", "rmse", "mape_pct", "median_ape_pct", "bias_pct"]
        summary = results.groupby(["subset", "model"])[cols].mean().round(3)
        print("\nBacktest, mean over folds:\n", summary.to_string())
        print("\nMAPE % by fold (clean labels):\n",
              results[results.subset == "clean_labels"].pivot_table(index="model", columns="fold", values="mape_pct").round(3).to_string())
        metrics["backtest_mean"] = {
            f"{subset}/{model}": row.to_dict() for (subset, model), row in summary.iterrows()
        }
        metrics["backtest_folds"] = results.round(4).to_dict(orient="records")

        city = unseen_city_check(train, outliers)
        print("\nunseen-city check:", json.dumps(city, indent=1, default=float))
        metrics["unseen_city_check"] = city

        shuffled = random_split_comparison(train, outliers)
        print("\nrandom 5-fold split (clean labels):", json.dumps(shuffled, default=float))
        metrics["random_kfold_clean_labels"] = shuffled

    final_rows = train[~outliers]
    model = RateModel(trend="feature").fit(final_rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "market": market,
            "cities": city_coordinates(train_raw, validation_raw),
        },
        MODEL_PATH,
    )
    metrics["final_model"] = {
        "name": FINAL_MODEL,
        "training_rows": len(final_rows),
        "date_range": [str(final_rows.date.min().date()), str(final_rows.date.max().date())],
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nsaved model -> {MODEL_PATH}\nsaved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
