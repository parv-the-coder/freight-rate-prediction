"""Loading and cleaning of the raw load files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import START_DATE, WEIGHT_CAP


def load_loads(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def build_market_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Daily market level from the load features (no labels involved).

    `market_index` moves with a weekly cycle and a seasonal swing, and every
    load on a given day carries a noisy reading of it (within-day std ~0.025).
    The same holds for the daily average of `quote_signal`. The per-day
    median/mean is a much cleaner signal than any single row, and it is also
    what lets us price the December chart rows, which only come with a date.
    """
    loads = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    table = loads.groupby("date").agg(
        market_index_day=("market_index", "median"),
        quote_signal_day=("quote_signal", "mean"),
    )
    if table.index.duplicated().any():
        raise ValueError("market table has duplicated dates")
    return table


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One (lat, lon) per city, taken from pickup and delivery columns."""
    parts = []
    for frame in frames:
        for side in ["pickup", "delivery"]:
            part = frame[[side, f"{side}_lat", f"{side}_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    coords = pd.concat(parts).drop_duplicates()
    if coords["city"].duplicated().any():
        raise ValueError("a city has more than one coordinate pair")
    return coords.set_index("city")


def clean_features(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Fix input-feature quality issues. Never drops rows."""
    out = frame.copy()

    # Negative weights are sign errors: their absolute values follow the same
    # distribution as the positive weights.
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight_capped"] = (out["weight"] >= WEIGHT_CAP).astype(int)

    out = out.join(market, on="date")
    missing_days = out["market_index_day"].isna()
    if missing_days.any():
        dates = sorted(out.loc[missing_days, "date"].dt.date.unique())
        raise ValueError(f"no market data for dates: {dates[:5]}")

    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["market_index_day"])
    out["quote_signal"] = out["quote_signal"].fillna(out["quote_signal_day"])

    out["days_since_start"] = (out["date"] - pd.Timestamp(START_DATE)).dt.days
    return out


def flag_label_outliers(log_target: pd.Series, log_oof_pred: np.ndarray, factor: float) -> pd.Series:
    """True where the posted rate is more than `factor`x off the OOF prediction."""
    return (log_target - log_oof_pred).abs() > np.log(factor)
