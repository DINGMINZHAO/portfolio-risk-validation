from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/interim/qpm"
OUTPUT_PATH = ROOT / "data/processed/qpm/historical_stress_2025_etf_attribution.csv"

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]

STRATEGIES = [
    "meanvar_static",
    "meanvar_regime",
    "minvar_static",
    "minvar_regime",
    "ew_static",
]

WINDOWS = {
    "Crash": ("2025-04-03", "2025-04-08", 4),
    "Rebound": ("2025-04-09", "2025-04-30", 15),
    "Full Event": ("2025-04-03", "2025-04-30", 19),
}

TOLERANCE = 1e-10


def load_inputs():
    assets = pd.read_csv(
        DATA_DIR / "asset_returns_arithmetic.csv",
        parse_dates=["Date"],
    )

    portfolios = pd.read_csv(
        DATA_DIR / "portfolio_returns_main.csv",
        parse_dates=["Date"],
    )

    weights = pd.read_csv(
        DATA_DIR / "daily_effective_weights_main.csv",
        parse_dates=["Date", "weight_date"],
    )

    for name, df in [
        ("asset returns", assets),
        ("portfolio returns", portfolios),
    ]:
        if df["Date"].isna().any() or df["Date"].duplicated().any():
            raise ValueError(f"Invalid or duplicate dates in {name}.")

    if weights.duplicated(["Date", "strategy"]).any():
        raise ValueError("Duplicate (Date, strategy) records in weights.")

    asset_returns = assets.set_index("Date").sort_index()
    portfolio_returns = portfolios.set_index("Date").sort_index()
    effective_weights = weights.set_index(
        ["Date", "strategy"]
    ).sort_index()

    required_weight_columns = [
        f"{asset}_weight" for asset in ASSETS
    ]

    for name, df, columns in [
        ("asset returns", asset_returns, ASSETS),
        ("portfolio returns", portfolio_returns, STRATEGIES),
        ("effective weights", effective_weights, required_weight_columns),
    ]:
        missing = [
            column for column in columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {missing}"
            )

    return asset_returns, portfolio_returns, effective_weights


def calculate_contributions(
    asset_returns,
    portfolio_returns,
    weights,
    window,
    strategy,
):
    start, end, expected_days = WINDOWS[window]

    dates = portfolio_returns.loc[start:end].index

    if len(dates) != expected_days:
        raise ValueError(
            f"{window}: expected {expected_days} trading days, "
            f"found {len(dates)}."
        )

    asset_daily = asset_returns.loc[
        dates, ASSETS
    ].to_numpy(dtype=float)

    lookup = pd.MultiIndex.from_product(
        [dates, [strategy]],
        names=["Date", "strategy"],
    )

    weight_columns = [
        f"{asset}_weight" for asset in ASSETS
    ]

    daily_weights = weights.loc[
        lookup, weight_columns
    ].to_numpy(dtype=float)

    portfolio_daily = portfolio_returns.loc[
        dates, strategy
    ].to_numpy(dtype=float)

    if not (
        np.isfinite(asset_daily).all()
        and np.isfinite(daily_weights).all()
        and np.isfinite(portfolio_daily).all()
    ):
        raise ValueError(
            f"Missing or non-finite data: {window}, {strategy}"
        )

    if not np.allclose(
        daily_weights.sum(axis=1),
        1.0,
        atol=TOLERANCE,
        rtol=0,
    ):
        raise ValueError(
            f"Portfolio weights do not sum to one: "
            f"{window}, {strategy}"
        )

    reconstructed_daily = (
        daily_weights * asset_daily
    ).sum(axis=1)

    daily_error = np.max(
        np.abs(reconstructed_daily - portfolio_daily)
    )

    if daily_error > TOLERANCE:
        raise ValueError(
            f"Daily return reconciliation failed for "
            f"{window}, {strategy}: "
            f"maximum difference = {daily_error:.3e}"
        )

    wealth_before = np.concatenate((
        [1.0],
        np.cumprod(1.0 + portfolio_daily)[:-1],
    ))

    asset_contributions = (
        wealth_before[:, None]
        * daily_weights
        * asset_daily
    ).sum(axis=0)

    full_return = float(
        np.prod(1.0 + portfolio_daily) - 1.0
    )

    attribution_error = abs(
        asset_contributions.sum() - full_return
    )

    if attribution_error > TOLERANCE:
        raise ValueError(
            f"Attribution reconciliation failed for "
            f"{window}, {strategy}: "
            f"difference = {attribution_error:.3e}"
        )

    records = [
        {
            "window": window,
            "strategy": strategy,
            "asset": asset,
            "contribution": float(contribution),
        }
        for asset, contribution in zip(
            ASSETS, asset_contributions
        )
    ]

    return (
        records,
        full_return,
        daily_error,
        attribution_error,
    )


def display_window(result, window):
    subset = result[result["window"] == window]

    table = subset.pivot(
        index="asset",
        columns="strategy",
        values="contribution",
    )

    table = table.reindex(ASSETS)

    table = 100 * table[
        ["meanvar_static", "minvar_static", "ew_static"]
    ]

    table.columns = [
        "MeanVar",
        "MinVar",
        "Equal Weight",
    ]

    table["MeanVar - MinVar"] = (
        table["MeanVar"] - table["MinVar"]
    )

    table.loc["TOTAL"] = table.sum(axis=0)

    print(
        f"\n{window} — ETF contribution "
        f"(percentage points):"
    )

    print(
        table.to_string(
            float_format=lambda value: f"{value:+.3f}"
        )
    )


def main():
    asset_returns, portfolio_returns, weights = load_inputs()

    records = []
    returns = {}

    max_daily_error = 0.0
    max_attribution_error = 0.0

    for window in WINDOWS:
        for strategy in STRATEGIES:
            (
                rows,
                total_return,
                daily_error,
                attribution_error,
            ) = calculate_contributions(
                asset_returns,
                portfolio_returns,
                weights,
                window,
                strategy,
            )

            records.extend(rows)

            returns[(window, strategy)] = total_return

            max_daily_error = max(
                max_daily_error,
                daily_error,
            )

            max_attribution_error = max(
                max_attribution_error,
                attribution_error,
            )

    result = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        float_format="%.12f",
    )

    print(
        "2025 April Trade Policy Shock "
        "— ETF Return Attribution"
    )

    print(
        "Attribution uses daily effective weights "
        "and opening portfolio wealth."
    )

    print(
        "Tables show Static and Equal Weight; "
        "all five strategies are saved in the CSV."
    )

    for window in [
        "Crash",
        "Rebound",
        "Full Event",
    ]:
        display_window(result, window)

    print("\nFull Event — total portfolio return:")

    for strategy in STRATEGIES:
        print(
            f"  {strategy:17s}: "
            f"{returns[('Full Event', strategy)]:+.3%}"
        )

    print("\nDaily return reconciliation: PASS")

    print(
        f"  Maximum absolute difference: "
        f"{max_daily_error:.3e}"
    )

    print("ETF contribution reconciliation: PASS")

    print(
        f"  Maximum absolute difference: "
        f"{max_attribution_error:.3e}"
    )

    print(f"Output saved: {OUTPUT_PATH}")

    print(
        "CSV contribution values are decimals; "
        "terminal tables use percentage points."
    )


if __name__ == "__main__":
    main()