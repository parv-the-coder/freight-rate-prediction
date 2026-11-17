"""Exploratory checks and figures used in the report.

Run: python eda.py   (prints a data-quality summary, writes reports/figures/*.png)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_rate.config import FIGURE_DIR, TRAIN_PATH, VALIDATION_PATH, WEIGHT_CAP
from freight_rate.data import load_loads


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def summary(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    print("rows: train", len(train), "| validation", len(validation))
    print("dates: train", train.date.min().date(), "->", train.date.max().date(),
          "| validation", validation.date.min().date(), "->", validation.date.max().date())
    for name, frame in [("train", train), ("validation", validation)]:
        print(f"\n[{name}]")
        print("  missing weight:", frame.weight.isna().sum(), "| missing market_index:", frame.market_index.isna().sum())
        weight = frame.weight.abs()
        print("  negative weight:", (frame.weight < 0).sum(),
              f"| weight == {WEIGHT_CAP:,.0f}:", (weight == WEIGHT_CAP).sum(),
              "| weight == 5,000 (minimum):", (weight == 5_000).sum())
        print("  duplicate load_id:", frame.load_id.duplicated().sum())

    train_cities = set(train.pickup) | set(train.delivery)
    new = sorted((set(validation.pickup) | set(validation.delivery)) - train_cities)
    touching = (validation.pickup.isin(new) | validation.delivery.isin(new)).sum()
    print(f"\ncities only in validation ({len(new)}): {', '.join(new)}")
    print(f"validation rows touching them: {touching} ({touching / len(validation):.1%})")
    lanes = set(zip(train.pickup, train.delivery))
    seen = np.mean([lane in lanes for lane in zip(validation.pickup, validation.delivery)])
    print(f"validation rows on a lane seen in training: {seen:.1%}")

    lane_median = train.groupby(["pickup", "delivery"]).distance.transform("median")
    print("\ndistance / lane median distance, 0.1%-99.9%:",
          np.round((train.distance / lane_median).quantile([0.001, 0.999]).to_numpy(), 3))
    rpm = train.posted_rate / train.distance
    print("rate per mile quantiles:", rpm.quantile([0, 0.005, 0.01, 0.5, 0.99, 0.995, 1]).round(2).to_dict())


def figures(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

    # 1. Rate per mile vs distance: the corrupted labels sit well away from the main band.
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    colors = {"Dry Van": "#2E6F95", "Reefer": "#C8553D", "Flatbed": "#7A9E3B"}
    for equipment, group in train.groupby("equipment"):
        ax.scatter(group.distance, group.posted_rate / group.distance, s=2, alpha=0.25,
                   color=colors[equipment], label=equipment, rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Posted rate per mile ($, log scale)")
    ax.set_title("Rate per mile vs distance (train)", loc="left", fontweight="bold")
    ax.legend(markerscale=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rate_per_mile_vs_distance.png")
    plt.close(fig)

    # 2. Daily market index across train and validation.
    fig, ax = plt.subplots(figsize=(7, 3.0), dpi=160)
    for name, frame, color in [("train", train, "#2E6F95"), ("validation", validation, "#C8553D")]:
        daily = frame.groupby("date").market_index.median()
        ax.plot(daily.index, daily.to_numpy(), color=color, linewidth=1, label=name)
    ax.set_ylabel("Daily median market_index")
    ax.set_title("Market index: weekly cycle on top of a seasonal swing", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "market_index_daily.png")
    plt.close(fig)

    # 3. Weight distribution with the data-quality issues marked.
    fig, ax = plt.subplots(figsize=(7, 2.8), dpi=160)
    ax.hist(train.weight.dropna(), bins=np.arange(-48_000, 48_001, 500), color="#2E6F95")
    ax.set_yscale("log")
    ax.axvline(WEIGHT_CAP, color="#C8553D", linestyle="--", linewidth=1)
    ax.annotate(f"cap {WEIGHT_CAP:,.0f} lb", (WEIGHT_CAP, 1500), xytext=(-8, 0), textcoords="offset points",
                color="#C8553D", ha="right", va="center")
    ax.annotate("negative weights (sign errors)", (-30_000, 40), color="#C8553D", ha="center")
    ax.set_xlabel("Weight (lb) as given")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title("Raw weight: negative sign errors and a pile-up at the cap", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "weight_distribution.png")
    plt.close(fig)


def main() -> None:
    train = load_loads(TRAIN_PATH)
    validation = load_loads(VALIDATION_PATH)
    summary(train, validation)
    figures(train, validation)
    print(f"\nfigures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
