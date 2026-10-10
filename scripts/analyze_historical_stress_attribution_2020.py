
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ASSET_FILE = ROOT / "data/interim/qpm/asset_returns_arithmetic.csv"
WEIGHT_FILE = ROOT / "data/interim/qpm/daily_effective_weights_main.csv"
PORTFOLIO_FILE = ROOT / "data/interim/qpm/portfolio_returns_main.csv"

OUTPUT = ROOT / "data/processed/qpm/historical_stress_2020_crash_attribution.csv"

START = pd.Timestamp("2020-02-20")
END = pd.Timestamp("2020-03-23")

ASSETS = [
    "SPY", "EFA", "EEM", "TLT",
    "IEF", "GLD", "VNQ", "DBC"
]

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime"
]

WEIGHT_COLS = [f"{asset}_weight" for asset in ASSETS]


assets = pd.read_csv(ASSET_FILE, parse_dates=["Date"])
weights = pd.read_csv(
    WEIGHT_FILE,
    parse_dates=["Date", "weight_date"]
)
portfolios = pd.read_csv(PORTFOLIO_FILE, parse_dates=["Date"])

assets = assets.sort_values("Date")
weights = weights.sort_values(["strategy", "Date"])
portfolios = portfolios.sort_values("Date")

for name, frame, keys in [
    ("Asset returns", assets, ["Date"]),
    ("Portfolio returns", portfolios, ["Date"]),
    ("Effective weights", weights, ["strategy", "Date"])
]:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate observations: {name}")


crash = portfolios[
    portfolios["Date"].between(START, END)
].copy()

if crash.empty:
    raise ValueError("No portfolio returns in crash window")

if crash["Date"].iloc[0] != START or crash["Date"].iloc[-1] != END:
    raise ValueError("Crash window boundaries are missing")

dates = pd.DatetimeIndex(crash["Date"])

crash = crash.set_index("Date")

asset_returns = (
    assets.set_index("Date")
    .reindex(dates)[ASSETS]
)

if asset_returns.isna().any().any():
    raise ValueError("Missing asset returns")

asset_values = asset_returns.to_numpy(dtype=float)

if not np.isfinite(asset_values).all():
    raise ValueError("Invalid asset returns")


records = []

for strategy in STRATEGIES:

    selected = weights[
        (weights["strategy"] == strategy)
        & (weights["Date"].between(START, END))
    ].copy()

    selected = selected.set_index("Date").reindex(dates)

    if selected[WEIGHT_COLS + ["weight_date"]].isna().any().any():
        raise ValueError(f"Missing effective weights: {strategy}")

    if not (selected["weight_date"].to_numpy() < dates.to_numpy()).all():
        raise ValueError(f"Invalid weight timing: {strategy}")

    weight_values = selected[WEIGHT_COLS].to_numpy(dtype=float)

    if not np.isfinite(weight_values).all():
        raise ValueError(f"Invalid weights: {strategy}")

    if not np.allclose(weight_values.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"Weights do not sum to one: {strategy}")

    daily_returns = (weight_values * asset_values).sum(axis=1)

    reference_returns = crash[strategy].to_numpy(dtype=float)

    if not np.allclose(
        daily_returns,
        reference_returns,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError(f"Portfolio return mismatch: {strategy}")

    wealth_before = np.concatenate([
        [1.0],
        np.cumprod(1.0 + daily_returns[:-1])
    ])

    daily_contributions = (
        wealth_before[:, None]
        * weight_values
        * asset_values
    )

    contributions = daily_contributions.sum(axis=0)

    cumulative_return = np.prod(1.0 + daily_returns) - 1.0

    if not np.isclose(
        contributions.sum(),
        cumulative_return,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError(f"Attribution mismatch: {strategy}")

    record = {
        "strategy": strategy,
        **dict(zip(ASSETS, contributions)),
        "total_return": cumulative_return
    }

    records.append(record)


results = pd.DataFrame(records).set_index("strategy")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.reset_index().to_csv(OUTPUT, index=False)


print("===== COVID-19 CRASH RETURN ATTRIBUTION =====")
print(f"Window: {START.date()} -> {END.date()}")
print(f"Trading days: {len(dates)}")

print("\nETF Contributions (percentage points):")

contribution_table = results[ASSETS] * 100

print(contribution_table.round(3).to_string())

print("\nTotal Portfolio Returns:")

for strategy in STRATEGIES:
    print(
        f"{strategy:<20}"
        f"{results.loc[strategy, 'total_return']:>10.2%}"
    )


PAIRS = [
    ("minvar_static", "meanvar_static"),
    ("minvar_regime", "meanvar_regime")
]

for minvar, meanvar in PAIRS:

    difference = (
        results.loc[meanvar, ASSETS]
        - results.loc[minvar, ASSETS]
    ) * 100

    comparison = pd.DataFrame({
        "ETF": ASSETS,
        "MeanVar_Advantage_pp": difference.values
    })

    comparison = comparison.sort_values(
        "MeanVar_Advantage_pp",
        ascending=False
    )

    print(f"\n===== {meanvar} vs {minvar} =====")
    print(comparison.round(3).to_string(index=False))

    print(
        f"Total advantage: "
        f"{difference.sum():+.3f} percentage points"
    )


print(f"\nSaved to: {OUTPUT}")
