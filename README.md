# Freight Rate Prediction

Predicts the posted rate of a truckload from lane, distance, equipment, weight, date and market signals.
The submitted model is a LightGBM regressor on `log(posted_rate)`, trained on Jan–Oct 2025 after
cleaning feature errors and removing corrupted labels, and validated with a rolling time-based backtest.

## Run

Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

python eda.py        # data-quality summary + figures in reports/figures/
python train.py      # cleaning, backtest, final fit -> artifacts/model.joblib, reports/metrics.json
python predict.py    # -> validation_predictions.csv, fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`python train.py --skip-backtest` only fits and saves the final model.
The full run with the backtest took about 3.5 minutes on a 16-thread CPU.

## Repository layout

```
data/                     provided input files (train_test, validation, template, December inputs)
freight_rate/
  config.py               paths, constants, backtest folds, LightGBM parameters
  data.py                 loading, feature cleaning, daily market table, label-outlier rule
  features.py             model feature matrix
  model.py                RateModel (LightGBM) and the two baselines
  validation.py           out-of-fold label-outlier detection, rolling-origin backtest
  metrics.py              MAE / RMSE / MAPE / bias
eda.py                    exploration summary and figures
train.py                  end-to-end training and validation
predict.py                validation predictions and December chart rows
score.py                  provided scorer (unchanged)
validation_predictions.csv
scorer_results/candidate_december.png
reports/                  report, backtest results, metrics, figures
```

## Data quality and cleaning

| Issue | Train | Validation | Handling |
|---|---|---|---|
| Negative `weight` | 292 | 145 | Sign error (absolute values follow the positive distribution) → `abs()` |
| Missing `weight` | 300 | 165 | Left as missing (LightGBM routes it) + `weight_missing` flag |
| `weight` capped at 47,500 lb | 1,204 | 301 | Kept, `weight_capped` flag |
| Missing `market_index` | 374 | 249 | Filled with that day's median `market_index` |
| Corrupted `posted_rate` (×0.17–0.45 or ×2.2–5.1) | 677 (1.4%) | n/a | Detected out-of-fold and dropped from training |
| 8 cities only in validation | – | 12.1% of rows | City name is not a feature; coordinates are |

Coordinates are not real-world positions (e.g. Boston and Providence share longitude −69.5), but each
city has exactly one coordinate pair, so they still work as a location encoding. `distance` is consistent
within each lane (±2%) and is used as the mileage.

## Validation approach

The loads to predict (Nov–Dec 2025) all come after the labelled data (Jan–Oct 2025), so the model is
validated as a forecast, not with a random split:

- **Rolling-origin backtest.** Three folds, each training on every load before a cut-off and testing on
  the next two months (May–Jun, Jul–Aug, Sep–Oct): the same 2-month horizon as Nov–Dec after October.
- **No leakage from cleaning.** Corrupted labels are re-detected inside each fold's training window only.
- **Two scores per fold.** All labels, and clean labels only (corrupted labels are random multiples of
  the true rate, so no model can predict them).
- **Unseen-city check.** 8 cities are hidden from training in the last fold, to mimic the 8 new
  cities in validation.
- **Random split, for contrast only.** A shuffled 5-fold split scores 1.25% MAPE against 2.09% in the
  time-based backtest; it overstates accuracy because train and test share the same days.

## Results

Backtest, mean of the 3 folds, clean labels (`reports/backtest_results.csv`):

| Model | MAPE % | MAE $ | RMSE $ | Bias % |
|---|---|---|---|---|
| Lane median rate per mile × distance | 4.80 | 112 | 160 | −2.08 |
| Log-linear regression | 3.14 | 68 | 92 | +0.47 |
| LightGBM, no calendar time | 2.44 | 54 | 77 | −1.68 |
| LightGBM + linear time drift | 2.30 | 55 | 82 | +1.56 |
| **LightGBM + date feature (submitted)** | **2.09** | **50** | **76** | **−0.33** |

Unseen-city check (Sep–Oct): MAPE 2.20% on loads touching a hidden city vs 1.93% on the rest.

The submitted model is refit on all 47,323 clean Jan–Oct loads. Because trees do not extrapolate, Nov–Dec
dates get the latest (October) level, while the daily market signals drive the day-to-day movement.
The December chart (`scorer_results/candidate_december.png`) shows a weekly cycle following
`market_index`, from about $818 (Sundays) to $831 (Thursdays), for the fixed 360-mile Dry Van load.
