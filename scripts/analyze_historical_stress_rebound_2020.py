
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/interim/qpm/portfolio_returns_main.csv"

OUTPUT = (
    ROOT
    / "data/processed/qpm/historical_stress_2020_rebound_summary.csv"
)

BASE_DATE = pd.Timestamp("2020-02-19")

CRASH_START = pd.Timestamp("2020-02-20")
CRASH_END = pd.Timestamp("2020-03-23")

REBOUND_START = pd.Timestamp("2020-03-24")
REBOUND_END = pd.Timestamp("2020-04-30")

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime"
]


def cumulative_return(returns):
    return np.prod(1.0 + returns) - 1.0


data = pd.read_csv(INPUT, parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found")

missing = [s for s in STRATEGIES if s not in data.columns]

if missing:
    raise ValueError(f"Missing strategies: {missing}")

if BASE_DATE not in data["Date"].values:
    raise ValueError("Baseline date is missing")


crash = data[
    data["Date"].between(CRASH_START, CRASH_END)
].copy()

rebound = data[
    data["Date"].between(REBOUND_START, REBOUND_END)
].copy()

full = data[
    data["Date"].between(CRASH_START, REBOUND_END)
].copy()


for name, frame, start, end in [
    ("Crash", crash, CRASH_START, CRASH_END),
    ("Rebound", rebound, REBOUND_START, REBOUND_END),
    ("Full", full, CRASH_START, REBOUND_END)
]:

    if frame.empty:
        raise ValueError(f"Empty window: {name}")

    if frame["Date"].iloc[0] != start:
        raise ValueError(f"Missing start date: {name}")

    if frame["Date"].iloc[-1] != end:
        raise ValueError(f"Missing end date: {name}")

    values = frame[STRATEGIES].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"Invalid returns: {name}")

    if (values < -1).any():
        raise ValueError(f"Return below -100%: {name}")


if len(full) != len(crash) + len(rebound):
    raise ValueError("Window partition mismatch")

combined_dates = pd.concat([
    crash["Date"],
    rebound["Date"]
]).reset_index(drop=True)

if not combined_dates.equals(full["Date"].reset_index(drop=True)):
    raise ValueError("Window dates do not align")


results = []

for strategy in STRATEGIES:

    crash_returns = crash[strategy].to_numpy(dtype=float)
    rebound_returns = rebound[strategy].to_numpy(dtype=float)
    full_returns = full[strategy].to_numpy(dtype=float)

    crash_total = cumulative_return(crash_returns)
    rebound_total = cumulative_return(rebound_returns)
    full_total = cumulative_return(full_returns)

    reconstructed_full = (
        (1.0 + crash_total)
        * (1.0 + rebound_total)
        - 1.0
    )

    if not np.isclose(
        full_total,
        reconstructed_full,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError(
            f"Cumulative return reconciliation failed: {strategy}"
        )

    results.append({
        "strategy": strategy,
        "crash_return": crash_total,
        "rebound_return": rebound_total,
        "full_return": full_total
    })


results = pd.DataFrame(results)

results = results.sort_values(
    "full_return",
    ascending=False
).reset_index(drop=True)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT, index=False)


print("===== COVID-19 CRASH AND REBOUND PERFORMANCE =====")

print(
    f"Crash: {CRASH_START.date()} -> {CRASH_END.date()} "
    f"({len(crash)} days)"
)

print(
    f"Rebound: {REBOUND_START.date()} -> {REBOUND_END.date()} "
    f"({len(rebound)} days)"
)

print(
    f"Full: {CRASH_START.date()} -> {REBOUND_END.date()} "
    f"({len(full)} days)"
)


display = results.copy()

for col in [
    "crash_return",
    "rebound_return",
    "full_return"
]:
    display[col] = display[col] * 100


print("\nCumulative Returns (%):")

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:+.3f}"
    )
)


print("\n===== PERFORMANCE RANKINGS =====")

for metric in [
    "crash_return",
    "rebound_return",
    "full_return"
]:

    ranking = results.sort_values(
        metric,
        ascending=False
    )

    print(f"\n{metric}:")

    for rank, (_, row) in enumerate(
        ranking.iterrows(),
        start=1
    ):
        print(
            f"{rank}. {row['strategy']:<20} "
            f"{row[metric]:+.2%}"
        )


print("\nCumulative return reconciliation: PASS")

print(f"\nSaved to: {OUTPUT}")
