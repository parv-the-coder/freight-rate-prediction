"""Predict the 12,000 validation loads and the 31 fixed December chart rows.

Run after train.py:  python predict.py
Writes validation_predictions.csv and fills predicted_rate in
data/december_chart_inputs.csv (the file score.py reads).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from freight_rate.config import DECEMBER_PATH, MODEL_PATH, PREDICTIONS_PATH, TEMPLATE_PATH, VALIDATION_PATH
from freight_rate.data import clean_features, load_loads


def predict_validation(bundle: dict) -> pd.DataFrame:
    validation = clean_features(load_loads(VALIDATION_PATH), bundle["market"])
    predicted = pd.Series(bundle["model"].predict(validation), index=validation["load_id"])

    template = pd.read_csv(TEMPLATE_PATH)
    if set(template["load_id"]) != set(predicted.index) or len(template) != len(predicted):
        raise ValueError("template load_ids do not match validation.csv")
    template["predicted_rate"] = template["load_id"].map(predicted).round(2)
    return template


def predict_december(bundle: dict) -> pd.DataFrame:
    """Price the fixed lane for every December day.

    These rows only carry lane, equipment, weight and date, so coordinates
    come from the city table and the market signals from that day's loads in
    validation.csv (daily median market_index, mean quote_signal).
    """
    chart = pd.read_csv(DECEMBER_PATH)
    columns = list(chart.columns)
    rows = chart.drop(columns="predicted_rate").copy()
    rows["date"] = pd.to_datetime(rows["date"])

    cities = bundle["cities"]
    for side in ["pickup", "delivery"]:
        rows[f"{side}_lat"] = rows[side].map(cities["lat"])
        rows[f"{side}_lon"] = rows[side].map(cities["lon"])
    coord_columns = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    if rows[coord_columns].isna().any().any():
        raise ValueError("unknown city in December inputs")
    rows["market_index"] = np.nan
    rows["quote_signal"] = np.nan

    cleaned = clean_features(rows, bundle["market"])
    chart["predicted_rate"] = np.round(bundle["model"].predict(cleaned), 2)
    return chart[columns]


def main() -> None:
    bundle = joblib.load(MODEL_PATH)

    predictions = predict_validation(bundle)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    rate = predictions["predicted_rate"]
    print(f"wrote {len(predictions):,} predictions -> {PREDICTIONS_PATH}")
    print(f"  min {rate.min():,.2f} | median {rate.median():,.2f} | max {rate.max():,.2f}")

    december = predict_december(bundle)
    december.to_csv(DECEMBER_PATH, index=False)
    print(f"wrote December chart rows -> {DECEMBER_PATH}")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
