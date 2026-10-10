
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
OUTPUT = ROOT / "data/processed/qpm/historical_stress_2022_summary.csv"

BASE_DATE = pd.Timestamp("2021-12-31")
START = pd.Timestamp("2022-01-03")
END = pd.Timestamp("2022-12-30")

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime"
]


data = pd.read_csv(INPUT, parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found")

if BASE_DATE not in data["Date"].values:
    raise ValueError("Baseline date missing")

missing = [s for s in STRATEGIES if s not in data.columns]

if missing:
    raise ValueError(f"Missing strategies: {missing}")

period = data[data["Date"].between(START, END)].copy()
period = period.reset_index(drop=True)

if period.empty:
    raise ValueError("Empty analysis window")

if period["Date"].iloc[0] != START:
    raise ValueError("Missing start date")

if period["Date"].iloc[-1] != END:
    raise ValueError("Missing end date")

if len(period) != 251:
    raise ValueError(f"Unexpected trading days: {len(period)}")


results = []

for strategy in STRATEGIES:

    returns = period[strategy].to_numpy(dtype=float)

    if not np.isfinite(returns).all():
        raise ValueError(f"Invalid returns: {strategy}")

    if (returns < -1).any():
        raise ValueError(f"Return below -100%: {strategy}")

    wealth = np.concatenate([
        [1.0],
        np.cumprod(1.0 + returns)
    ])

    dates = pd.DatetimeIndex(
        [BASE_DATE] + period["Date"].tolist()
    )

    running_peak = np.maximum.accumulate(wealth)
    drawdown = wealth / running_peak - 1.0

    trough_idx = int(np.argmin(drawdown))
    peak_idx = int(np.argmax(wealth[:trough_idx + 1]))

    worst_idx = int(np.argmin(returns))

    results.append({
        "strategy": strategy,
        "cumulative_return": wealth[-1] - 1.0,
        "maximum_drawdown": drawdown[trough_idx],
        "worst_daily_return": returns[worst_idx],
        "peak_date": dates[peak_idx].date(),
        "trough_date": dates[trough_idx].date(),
        "worst_day_date": period["Date"].iloc[worst_idx].date()
    })


results = pd.DataFrame(results)

results = results.sort_values(
    "maximum_drawdown",
    ascending=False
).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)


print("===== 2022 INFLATION & RATE-HIKE STRESS =====")
print(f"Window: {START.date()} -> {END.date()}")
print(f"Trading days: {len(period)}")

display = results[
    [
        "strategy",
        "cumulative_return",
        "maximum_drawdown",
        "worst_daily_return"
    ]
].copy()

for col in [
    "cumulative_return",
    "maximum_drawdown",
    "worst_daily_return"
]:
    display[col] *= 100

print("\nPortfolio Risk Metrics (%):")

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:+.3f}"
    )
)

print("\nMaximum Drawdown Timing:")

print(
    results[
        ["strategy", "peak_date", "trough_date"]
    ].to_string(index=False)
)

print("\nWorst Daily Return Timing:")

print(
    results[
        ["strategy", "worst_day_date"]
    ].to_string(index=False)
)

print(f"\nSaved to: {OUTPUT}")
