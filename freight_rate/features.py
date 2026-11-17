"""Model features built from a cleaned load frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

FEATURES = [
    "log_distance",
    "equipment",
    "weight",
    "weight_missing",
    "weight_capped",
    "market_index",
    "market_index_day",
    "quote_signal",
    "quote_signal_day",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "day_of_week",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    unknown = set(frame["equipment"]) - set(EQUIPMENT)
    if unknown:
        raise ValueError(f"unknown equipment types: {sorted(unknown)}")

    x = pd.DataFrame(index=frame.index)
    x["log_distance"] = np.log(frame["distance"])
    x["equipment"] = pd.Categorical(frame["equipment"], categories=EQUIPMENT)
    for column in [
        "weight",
        "weight_missing",
        "weight_capped",
        "market_index",
        "market_index_day",
        "quote_signal",
        "quote_signal_day",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
    ]:
        x[column] = frame[column]
    x["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
    x["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]
    x["day_of_week"] = frame["date"].dt.dayofweek
    return x[FEATURES]
