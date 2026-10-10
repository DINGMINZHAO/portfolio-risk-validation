
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
OUTPUT = ROOT / "data/processed/qpm/historical_stress_2020_crash_returns.csv"

START = pd.Timestamp("2020-02-20")
END = pd.Timestamp("2020-03-23")
BASE_DATE = pd.Timestamp("2020-02-19")

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]


data = pd.read_csv(INPUT, parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

missing = [s for s in STRATEGIES if s not in data.columns]

if missing:
    raise ValueError(f"Missing strategies: {missing}")

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found")

if BASE_DATE not in data["Date"].values:
    raise ValueError("Baseline date is missing")

crash = data[
    (data["Date"] >= START)
    & (data["Date"] <= END)
].copy()

if crash.empty:
    raise ValueError("No observations in crash window")

if crash["Date"].iloc[0] != START or crash["Date"].iloc[-1] != END:
    raise ValueError("Crash window boundaries are missing")

returns = crash[STRATEGIES].astype(float)

if not np.isfinite(returns.to_numpy()).all():
    raise ValueError("Invalid portfolio returns found")

if (returns < -1).any().any():
    raise ValueError("Portfolio return below -100%")

cumulative_returns = (1 + returns).prod(axis=0) - 1

results = pd.DataFrame({
    "strategy": STRATEGIES,
    "crash_return": cumulative_returns.reindex(STRATEGIES).values,
})

results = results.sort_values(
    "crash_return",
    ascending=False
).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)


print("===== COVID-19 CRASH PERFORMANCE =====")
print(f"Window: {START.date()} -> {END.date()}")
print(f"Trading days: {len(crash)}")

print("\nCumulative Returns:")

for _, row in results.iterrows():
    print(
        f"{row['strategy']:<20}"
        f"{row['crash_return']:>10.2%}"
    )

print(f"\nSaved to: {OUTPUT}")
