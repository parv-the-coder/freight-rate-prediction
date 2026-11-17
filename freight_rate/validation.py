"""Label-outlier detection and the time-based backtest."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import BACKTEST_FOLDS, LABEL_OUTLIER_FACTOR, RANDOM_STATE, TARGET
from .data import flag_label_outliers
from .metrics import regression_metrics
from .model import RateModel


def find_label_outliers(frame: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    """Flag corrupted labels using out-of-fold predictions of a robust model.

    The Huber loss keeps the corrupted rows from pulling the fit, and
    out-of-fold predictions keep each row from being judged by a model that
    has already seen it.
    """
    oof = np.zeros(len(frame))
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    robust = {"objective": "huber", "alpha": 0.1, "n_estimators": 600, "learning_rate": 0.05}
    for train_idx, test_idx in folds.split(frame):
        model = RateModel(trend="feature", params=robust).fit(frame.iloc[train_idx])
        oof[test_idx] = model.predict_log(frame.iloc[test_idx])
    flags = flag_label_outliers(np.log(frame[TARGET]), oof, LABEL_OUTLIER_FACTOR)
    return flags.rename("label_outlier")


def backtest(
    frame: pd.DataFrame,
    eval_outlier: pd.Series,
    models: dict[str, Callable[[], object]],
    folds=BACKTEST_FOLDS,
) -> pd.DataFrame:
    """Rolling-origin backtest.

    For every fold the training window is everything strictly before the test
    window. Label outliers are re-detected inside the training window only,
    so no information from the test months is used for cleaning. Test months
    are scored twice: on every row, and on rows whose label is not flagged as
    corrupted (`eval_outlier`, from the full-data pass, used only for scoring).
    """
    rows = []
    for start, end in folds:
        train = frame[frame["date"] < start]
        test = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        train_clean = train[~find_label_outliers(train)]
        for name, factory in models.items():
            pred = factory().fit(train_clean).predict(test)
            subsets = {
                "all_labels": np.ones(len(test), dtype=bool),
                "clean_labels": ~eval_outlier.loc[test.index].to_numpy(),
            }
            for subset, mask in subsets.items():
                rows.append(
                    {
                        "fold": f"{start[:7]}..{end[:7]}",
                        "train_rows": len(train_clean),
                        "model": name,
                        "subset": subset,
                        **regression_metrics(test.loc[mask, TARGET], pred[mask]),
                    }
                )
    return pd.DataFrame(rows)
