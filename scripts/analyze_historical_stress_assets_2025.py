
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/interim/qpm/asset_returns_arithmetic.csv"

OUTPUT = (
    ROOT
    / "data/processed/qpm/historical_stress_2025_asset_returns.csv"
)

BASE_DATE = pd.Timestamp("2025-04-02")

WINDOWS = {
    "crash": (
        pd.Timestamp("2025-04-03"),
        pd.Timestamp("2025-04-08")
    ),
    "rebound": (
        pd.Timestamp("2025-04-09"),
        pd.Timestamp("2025-04-30")
    ),
    "full": (
        pd.Timestamp("2025-04-03"),
        pd.Timestamp("2025-04-30")
    )
}

EXPECTED_DAYS = {
    "crash": 4,
    "rebound": 15,
    "full": 19
}

ASSETS = [
    "SPY", "EFA", "EEM", "TLT",
    "IEF", "GLD", "VNQ", "DBC"
]


data = pd.read_csv(INPUT, parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found")

missing = [a for a in ASSETS if a not in data.columns]

if missing:
    raise ValueError(f"Missing assets: {missing}")

if BASE_DATE not in data["Date"].values:
    raise ValueError("Baseline date missing")


window_returns = {}

for name, (start, end) in WINDOWS.items():

    period = data[data["Date"].between(start, end)].copy()

    if len(period) != EXPECTED_DAYS[name]:
        raise ValueError(
            f"{name}: unexpected trading days: {len(period)}"
        )

    if period["Date"].iloc[0] != start:
        raise ValueError(f"{name}: missing start date")

    if period["Date"].iloc[-1] != end:
        raise ValueError(f"{name}: missing end date")

    values = period[ASSETS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"{name}: invalid asset returns")

    if (values < -1).any():
        raise ValueError(f"{name}: return below -100%")

    cumulative = (1.0 + period[ASSETS]).prod(axis=0) - 1.0

    window_returns[name] = cumulative


crash = window_returns["crash"]
rebound = window_returns["rebound"]
full = window_returns["full"]

reconstructed_full = (1.0 + crash) * (1.0 + rebound) - 1.0

if not np.allclose(
    reconstructed_full.to_numpy(),
    full.to_numpy(),
    atol=1e-10
):
    raise ValueError("Window reconciliation failed")


results = pd.DataFrame({
    "asset": ASSETS,
    "crash_return": crash.reindex(ASSETS).values,
    "rebound_return": rebound.reindex(ASSETS).values,
    "full_return": full.reindex(ASSETS).values
})

results = results.sort_values(
    "crash_return",
    ascending=False
).reset_index(drop=True)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT, index=False)


print("===== 2025 APRIL TRADE POLICY SHOCK =====")

print(f"Baseline: {BASE_DATE.date()}")

for name, (start, end) in WINDOWS.items():
    print(
        f"{name.upper():<10} "
        f"{start.date()} -> {end.date()} "
        f"({EXPECTED_DAYS[name]} trading days)"
    )


display = results.copy()

for col in [
    "crash_return",
    "rebound_return",
    "full_return"
]:
    display[col] *= 100

print("\n===== ASSET CUMULATIVE RETURNS (%) =====")

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:+.3f}"
    )
)


print("\n===== CRASH RANKING =====")

for _, row in results.iterrows():
    print(
        f"{row['asset']:<6} "
        f"{row['crash_return']:+.3%}"
    )


print("\nWindow reconciliation: PASS")
print(f"\nSaved to: {OUTPUT}")
