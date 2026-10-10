
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ASSET_FILE = ROOT / "data/interim/qpm/asset_returns_arithmetic.csv"
PORTFOLIO_FILE = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
WEIGHTS_FILE = ROOT / "data/interim/qpm/daily_effective_weights_main.csv"

OUTPUT_DIR = ROOT / "data/processed/qpm"

START = pd.Timestamp("2022-01-03")
END = pd.Timestamp("2022-12-30")

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
portfolios = pd.read_csv(PORTFOLIO_FILE, parse_dates=["Date"])
weights = pd.read_csv(WEIGHTS_FILE, parse_dates=["Date"])

assets = assets.sort_values("Date")
portfolios = portfolios.sort_values("Date")
weights = weights.sort_values(["Date", "strategy"])

for name, data in [
    ("assets", assets),
    ("portfolios", portfolios)
]:
    if data["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {name}")

if weights.duplicated(["Date", "strategy"]).any():
    raise ValueError("Duplicate daily strategy weights")

for columns, data, name in [
    (ASSETS, assets, "assets"),
    (STRATEGIES, portfolios, "portfolios"),
    (WEIGHT_COLS, weights, "weights")
]:
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


assets = assets[assets["Date"].between(START, END)]
portfolios = portfolios[portfolios["Date"].between(START, END)]
weights = weights[weights["Date"].between(START, END)]

dates = pd.DatetimeIndex(portfolios["Date"])

if len(dates) != 251:
    raise ValueError(f"Unexpected trading days: {len(dates)}")

if dates[0] != START or dates[-1] != END:
    raise ValueError("Incorrect analysis window")


asset_returns = (
    assets.set_index("Date")
    .reindex(dates)[ASSETS]
    .astype(float)
)

portfolio_returns = (
    portfolios.set_index("Date")
    .reindex(dates)[STRATEGIES]
    .astype(float)
)

if not np.isfinite(asset_returns.to_numpy()).all():
    raise ValueError("Invalid asset returns")

if not np.isfinite(portfolio_returns.to_numpy()).all():
    raise ValueError("Invalid portfolio returns")


attribution_rows = []
weight_rows = []

for strategy in STRATEGIES:

    w = (
        weights[weights["strategy"] == strategy]
        .set_index("Date")
        .reindex(dates)[WEIGHT_COLS]
        .astype(float)
    )

    if not np.isfinite(w.to_numpy()).all():
        raise ValueError(f"Invalid weights: {strategy}")

    weight_values = w.to_numpy()
    asset_values = asset_returns.to_numpy()

    weight_sums = weight_values.sum(axis=1)

    if not np.allclose(weight_sums, 1.0, atol=1e-6):
        raise ValueError(f"Weights do not sum to 1: {strategy}")

    daily_asset_contributions = weight_values * asset_values

    reconstructed_returns = daily_asset_contributions.sum(axis=1)

    actual_returns = portfolio_returns[strategy].to_numpy()

    max_error = np.max(
        np.abs(reconstructed_returns - actual_returns)
    )

    if max_error > 1e-8:
        raise ValueError(
            f"Return reconstruction failed: "
            f"{strategy}, error={max_error:.10f}"
        )

    wealth = np.cumprod(1.0 + actual_returns)

    wealth_before = np.concatenate([
        [1.0],
        wealth[:-1]
    ])

    compounded_contributions = (
        daily_asset_contributions
        * wealth_before[:, None]
    )

    asset_contributions = compounded_contributions.sum(axis=0)

    cumulative_return = wealth[-1] - 1.0
    contribution_sum = asset_contributions.sum()

    if not np.isclose(
        contribution_sum,
        cumulative_return,
        atol=1e-8
    ):
        raise ValueError(
            f"Attribution reconciliation failed: {strategy}"
        )

    attribution_row = {
        "strategy": strategy,
        "cumulative_return_pp": cumulative_return * 100
    }

    weight_row = {"strategy": strategy}

    for i, asset in enumerate(ASSETS):
        attribution_row[asset] = asset_contributions[i] * 100
        weight_row[asset] = weight_values[:, i].mean() * 100

    attribution_rows.append(attribution_row)
    weight_rows.append(weight_row)


attribution = pd.DataFrame(attribution_rows).set_index("strategy")
average_weights = pd.DataFrame(weight_rows).set_index("strategy")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

attribution.to_csv(
    OUTPUT_DIR / "historical_stress_2022_attribution.csv"
)

average_weights.to_csv(
    OUTPUT_DIR / "historical_stress_2022_average_weights.csv"
)


print("===== 2022 RETURN ATTRIBUTION =====")
print(f"Window: {START.date()} -> {END.date()}")
print(f"Trading days: {len(dates)}")

print("\nAsset Contributions (percentage points):")

print(
    attribution.round(3).to_string()
)

print("\n===== AVERAGE DAILY PORTFOLIO WEIGHTS (%) =====")

print(
    average_weights.round(2).to_string()
)


print("\n===== MEANVAR MINUS MINVAR (pp) =====")

for suffix in ["static", "regime"]:

    difference = (
        attribution.loc[f"meanvar_{suffix}", ASSETS]
        - attribution.loc[f"minvar_{suffix}", ASSETS]
    )

    total = difference.sum()

    print(f"\n{suffix.upper()}")

    print(
        difference.sort_values(ascending=False)
        .round(3)
        .to_string()
    )

    print(f"Total advantage: {total:+.3f} pp")


print("\n===== REGIME MINUS STATIC (pp) =====")

for method in ["minvar", "meanvar"]:

    difference = (
        attribution.loc[f"{method}_regime", ASSETS]
        - attribution.loc[f"{method}_static", ASSETS]
    )

    print(f"\n{method.upper()}")

    print(
        difference.sort_values(ascending=False)
        .round(3)
        .to_string()
    )

    print(
        f"Total Regime effect: {difference.sum():+.3f} pp"
    )


print("\nAttribution reconciliation: PASS")

print("\nSaved:")
print(OUTPUT_DIR / "historical_stress_2022_attribution.csv")
print(OUTPUT_DIR / "historical_stress_2022_average_weights.csv")
