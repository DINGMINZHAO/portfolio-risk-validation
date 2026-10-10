
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/interim/qpm/asset_returns_arithmetic.csv"

OUTPUT = (
    ROOT
    / "data/processed/qpm/historical_stress_2022_asset_returns.csv"
)

BASE_DATE = pd.Timestamp("2021-12-31")
START = pd.Timestamp("2022-01-03")
END = pd.Timestamp("2022-12-30")

ASSETS = [
    "SPY", "EFA", "EEM", "TLT",
    "IEF", "GLD", "VNQ", "DBC"
]

ASSET_CLASSES = {
    "SPY": "US Equity",
    "EFA": "Developed Equity",
    "EEM": "Emerging Equity",
    "TLT": "Long-Term Treasury",
    "IEF": "Intermediate Treasury",
    "GLD": "Gold",
    "VNQ": "Real Estate",
    "DBC": "Commodities"
}


data = pd.read_csv(INPUT, parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found")

missing = [asset for asset in ASSETS if asset not in data.columns]

if missing:
    raise ValueError(f"Missing assets: {missing}")

if BASE_DATE not in data["Date"].values:
    raise ValueError("Baseline date is missing")

period = data[
    data["Date"].between(START, END)
].copy()

if period.empty:
    raise ValueError("Empty analysis window")

if period["Date"].iloc[0] != START:
    raise ValueError("Missing start date")

if period["Date"].iloc[-1] != END:
    raise ValueError("Missing end date")

values = period[ASSETS].to_numpy(dtype=float)

if not np.isfinite(values).all():
    raise ValueError("Invalid asset returns")

if (values < -1).any():
    raise ValueError("Asset return below -100%")


cumulative_returns = (
    (1.0 + period[ASSETS]).prod(axis=0) - 1.0
)

results = pd.DataFrame({
    "asset": ASSETS,
    "asset_class": [ASSET_CLASSES[a] for a in ASSETS],
    "cumulative_return": cumulative_returns.reindex(ASSETS).values
})

results = results.sort_values(
    "cumulative_return",
    ascending=False
).reset_index(drop=True)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)


print("===== 2022 ASSET-LEVEL STRESS PERFORMANCE =====")

print(f"Window: {START.date()} -> {END.date()}")
print(f"Trading days: {len(period)}")

print("\nAnnual Cumulative Returns (%):")

display = results.copy()
display["cumulative_return"] *= 100

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:+.3f}"
    )
)


print("\n===== STOCK-BOND STRESS CHECK =====")

for asset in ["SPY", "EFA", "EEM", "TLT", "IEF"]:
    value = cumulative_returns[asset]

    print(f"{asset:<6}{value:+.2%}")

print(f"\nSaved to: {OUTPUT}")
