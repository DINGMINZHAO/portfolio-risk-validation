
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

WEIGHT_FILE = ROOT / "data/interim/qpm/daily_effective_weights_main.csv"
REGIME_FILE = ROOT / "data/external/qpm/upstream/regime_labels.csv"

OUTPUT = ROOT / "data/processed/qpm/historical_stress_2020_weight_snapshots.csv"

START = pd.Timestamp("2020-02-20")
END = pd.Timestamp("2020-03-23")

SNAPSHOT_DATES = pd.to_datetime([
    "2020-02-20",
    "2020-03-02",
    "2020-03-23"
])

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

RISK_ASSETS = ["SPY", "EFA", "EEM", "VNQ", "DBC"]
DEFENSIVE_ASSETS = ["TLT", "IEF", "GLD"]

WEIGHT_COLS = [f"{asset}_weight" for asset in ASSETS]


weights = pd.read_csv(
    WEIGHT_FILE,
    parse_dates=["Date", "weight_date"]
)

regimes = pd.read_csv(REGIME_FILE)

regimes = regimes.rename(
    columns={regimes.columns[0]: "Date"}
)

regimes["Date"] = pd.to_datetime(regimes["Date"])

weights = weights[
    weights["Date"].between(START, END)
    & weights["strategy"].isin(STRATEGIES)
].copy()

weights = weights.sort_values(
    ["strategy", "Date"]
).reset_index(drop=True)

if weights.duplicated(["strategy", "Date"]).any():
    raise ValueError("Duplicate strategy-date observations")

if weights[WEIGHT_COLS].isna().any().any():
    raise ValueError("Missing portfolio weights")

if not np.isfinite(weights[WEIGHT_COLS].to_numpy()).all():
    raise ValueError("Invalid portfolio weights")

if not np.allclose(
    weights[WEIGHT_COLS].sum(axis=1),
    1.0,
    atol=1e-8
):
    raise ValueError("Weights do not sum to one")

if not (weights["weight_date"] < weights["Date"]).all():
    raise ValueError("Weight timing violation")

if weights.groupby("Date")["strategy"].nunique().min() != len(STRATEGIES):
    raise ValueError("Incomplete strategy coverage")

weights["risk_weight"] = weights[
    [f"{asset}_weight" for asset in RISK_ASSETS]
].sum(axis=1)

weights["defensive_weight"] = weights[
    [f"{asset}_weight" for asset in DEFENSIVE_ASSETS]
].sum(axis=1)


print("===== REGIME DETECTION =====")

crisis_regimes = regimes[
    regimes["Date"].between(START, END)
].copy()

stressed_dates = crisis_regimes.loc[
    crisis_regimes["regime"] == "stressed",
    "Date"
]

if stressed_dates.empty:
    raise ValueError("No stressed regime detected")

first_stressed = stressed_dates.min()

print(f"First stressed signal: {first_stressed.date()}")


print("\n===== EFFECTIVE WEIGHT PERIODS =====")

timing = (
    weights.groupby(
        ["strategy", "weight_date", "regime"],
        as_index=False
    )
    .agg(
        first_effective_date=("Date", "min"),
        last_effective_date=("Date", "max"),
        trading_days=("Date", "size")
    )
)

print(timing.to_string(index=False))

new_weights = timing[
    timing["weight_date"] >= first_stressed
]

if not new_weights.empty:
    first_applied = new_weights["first_effective_date"].min()
    print(f"\nFirst post-signal allocation: {first_applied.date()}")


print("\n===== PORTFOLIO WEIGHT SNAPSHOTS =====")

snapshots = weights[
    weights["Date"].isin(SNAPSHOT_DATES)
].copy()

if len(snapshots) != len(STRATEGIES) * len(SNAPSHOT_DATES):
    raise ValueError("Missing snapshot observations")

for date in SNAPSHOT_DATES:

    sub = snapshots[
        snapshots["Date"] == date
    ].set_index("strategy").reindex(STRATEGIES)

    print(f"\nDate: {date.date()}")

    print(
        "Effective weight date:",
        sub["weight_date"].iloc[0].date()
    )

    print(
        "Regime:",
        sub["regime"].iloc[0]
    )

    table = sub[WEIGHT_COLS] * 100
    table.columns = ASSETS

    print("\nETF Weights (%):")
    print(table.round(2).to_string())

    exposure = sub[
        ["risk_weight", "defensive_weight"]
    ] * 100

    exposure.columns = [
        "Risk_Assets",
        "Defensive_Assets"
    ]

    print("\nAsset Group Exposure (%):")
    print(exposure.round(2).to_string())


print("\n===== REGIME VS STATIC DIFFERENCES =====")

for date in [
    pd.Timestamp("2020-02-20"),
    pd.Timestamp("2020-03-02")
]:

    sub = snapshots[
        snapshots["Date"] == date
    ].set_index("strategy")

    print(f"\nDate: {date.date()}")

    for strategy in ["minvar", "meanvar"]:

        static = sub.loc[
            f"{strategy}_static", WEIGHT_COLS
        ].astype(float)

        regime = sub.loc[
            f"{strategy}_regime", WEIGHT_COLS
        ].astype(float)

        difference = (regime - static) * 100

        print(f"\n{strategy.upper()} Regime - Static (pp):")

        for asset, value in zip(ASSETS, difference):
            print(f"{asset:<6}{value:>+9.3f}")

        print(
            f"Total absolute weight difference: "
            f"{difference.abs().sum():.3f} pp"
        )


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

snapshots.to_csv(OUTPUT, index=False)

print(f"\nSaved to: {OUTPUT}")
