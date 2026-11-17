from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

TARGET = "posted_rate"
START_DATE = "2025-01-01"
RANDOM_STATE = 42

# Weights pile up at exactly this value in both files, i.e. the field is capped.
WEIGHT_CAP = 47_500.0

# A label is treated as corrupted when it is more than 1.5x away from the
# out-of-fold prediction. Clean labels sit within -14%/+18% of the prediction
# and corrupted ones are 2.2x-5.1x too high or 0.17x-0.45x too low, so any
# threshold between 1.3x and 2.0x flags exactly the same rows.
LABEL_OUTLIER_FACTOR = 1.5

# Rolling-origin backtest: each fold trains on everything before `test_start`
# and scores the following two months, mirroring the Nov-Dec horizon that
# follows the Jan-Oct training data.
BACKTEST_FOLDS = [
    ("2025-05-01", "2025-06-30"),
    ("2025-07-01", "2025-08-31"),
    ("2025-09-01", "2025-10-31"),
]

LGBM_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    n_estimators=1500,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
