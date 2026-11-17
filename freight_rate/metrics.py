from __future__ import annotations

import numpy as np


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    ape = np.abs(error) / actual
    return {
        "n": int(actual.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(100 * np.mean(ape)),
        "median_ape_pct": float(100 * np.median(ape)),
        "bias_pct": float(100 * (predicted.sum() / actual.sum() - 1)),
    }
