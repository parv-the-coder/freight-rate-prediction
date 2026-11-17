"""Rate models: two simple baselines and the LightGBM model used for submission."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import LGBM_PARAMS, TARGET
from .features import EQUIPMENT, FEATURES, build_features


def _trend_design(frame: pd.DataFrame) -> pd.DataFrame:
    """Small log-linear design used only to estimate the time drift."""
    x = pd.DataFrame(index=frame.index)
    x["log_distance"] = np.log(frame["distance"])
    for equipment in EQUIPMENT[1:]:
        x[equipment] = (frame["equipment"] == equipment).astype(float)
    x["weight"] = frame["weight"].fillna(frame["weight"].median()) / 1000
    x["log_market_index_day"] = np.log(frame["market_index_day"])
    x["log_quote_signal_day"] = np.log(frame["quote_signal_day"])
    x["days_since_start"] = frame["days_since_start"].astype(float)
    return x


class RateModel:
    """LightGBM on log(rate).

    `trend` controls how calendar time enters the model:
      - "feature": days since 2025-01-01 is a feature. Trees cannot
        extrapolate, so future dates get the level of the latest training
        period. This is the submitted model (best in the backtest).
      - "none": no calendar time; time effects only through the market signals.
      - "linear": a log-linear drift per day is estimated with OLS (controlling
        for distance, equipment, weight and daily market level), removed from
        the target before boosting and added back at prediction time.
    """

    def __init__(self, trend: str = "feature", params: dict | None = None):
        if trend not in {"linear", "none", "feature"}:
            raise ValueError(f"unknown trend option: {trend}")
        self.trend = trend
        self.params = {**LGBM_PARAMS, **(params or {})}
        self.drift_per_day = 0.0
        self.features = FEATURES + (["days_since_start"] if trend == "feature" else [])

    def _x(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = build_features(frame)
        if self.trend == "feature":
            x["days_since_start"] = frame["days_since_start"]
        return x[self.features]

    def fit(self, frame: pd.DataFrame) -> "RateModel":
        y = np.log(frame[TARGET].to_numpy())
        if self.trend == "linear":
            design = _trend_design(frame)
            ols = LinearRegression().fit(design, y)
            self.drift_per_day = float(ols.coef_[list(design.columns).index("days_since_start")])
        y = y - self.drift_per_day * frame["days_since_start"].to_numpy()
        self.booster = lgb.LGBMRegressor(**self.params).fit(self._x(frame), y)
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        drift = self.drift_per_day * frame["days_since_start"].to_numpy()
        return self.booster.predict(self._x(frame)) + drift

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))


class LaneRatePerMileBaseline:
    """Median rate-per-mile of the lane and equipment, times distance.

    Falls back to pickup-city, then equipment-wide medians for unseen lanes.
    """

    def fit(self, frame: pd.DataFrame) -> "LaneRatePerMileBaseline":
        rpm = frame[TARGET] / frame["distance"]
        data = frame.assign(rpm=rpm)
        self.lane = data.groupby(["pickup", "delivery", "equipment"])["rpm"].median()
        self.origin = data.groupby(["pickup", "equipment"])["rpm"].median()
        self.equipment = data.groupby("equipment")["rpm"].median()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        lane_key = pd.MultiIndex.from_frame(frame[["pickup", "delivery", "equipment"]])
        origin_key = pd.MultiIndex.from_frame(frame[["pickup", "equipment"]])
        rpm = pd.Series(self.lane.reindex(lane_key).to_numpy(), index=frame.index)
        rpm = rpm.fillna(pd.Series(self.origin.reindex(origin_key).to_numpy(), index=frame.index))
        rpm = rpm.fillna(frame["equipment"].map(self.equipment))
        return (rpm * frame["distance"]).to_numpy()


class LogLinearBaseline:
    """OLS on log(rate) with the same drift design plus per-row signals."""

    def _x(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = _trend_design(frame)
        x["log_market_index"] = np.log(frame["market_index"])
        x["log_quote_signal"] = np.log(frame["quote_signal"])
        return x

    def fit(self, frame: pd.DataFrame) -> "LogLinearBaseline":
        self.ols = LinearRegression().fit(self._x(frame), np.log(frame[TARGET]))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.ols.predict(self._x(frame)))
