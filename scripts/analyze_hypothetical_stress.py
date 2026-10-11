
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/qpm/daily_effective_weights_main.csv"
OUTPUT_DIR = ROOT / "data/processed/qpm"

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]

SNAPSHOTS = {
    "2025-04-01": {
        "weight_date": "2025-03-31",
        "regime": "calm",
    },
    "2025-05-01": {
        "weight_date": "2025-04-30",
        "regime": "stressed",
    },
}

# 20-trading-day cumulative asset shocks, in percent
SCENARIOS = {
    "A_equity_crash": [
        -20.00, -21.00, -23.00, 2.00,
        1.00, 0.00, -30.00, -19.00,
    ],
    "B_inflation_rate_shock": [
        -8.00, -8.00, -8.00, -7.00,
        -4.00, -2.00, -12.00, 3.00,
    ],
    "C_joint_historical_worst": [
        -30.99, -33.74, -44.62, -14.62,
        -6.57, -19.67, -42.59, -31.01,
    ],
}


def load_snapshot_weights():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(
        INPUT,
        parse_dates=["Date", "weight_date"],
    )

    weight_cols = [f"{asset}_weight" for asset in ASSETS]

    required = [
        "Date", "strategy", "weight_date", "regime",
        *weight_cols,
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    snapshots = {}

    for date, metadata in SNAPSHOTS.items():
        for strategy in STRATEGIES:

            match = df.loc[
                (df["Date"] == pd.Timestamp(date))
                & (df["strategy"] == strategy)
            ]

            if len(match) != 1:
                raise ValueError(
                    f"Expected one weight row for "
                    f"{date}, {strategy}; found {len(match)}"
                )

            row = match.iloc[0]

            if row["weight_date"] != pd.Timestamp(metadata["weight_date"]):
                raise ValueError(
                    f"Unexpected weight_date for {date}, {strategy}"
                )

            if str(row["regime"]).strip().lower() != metadata["regime"]:
                raise ValueError(
                    f"Unexpected regime for {date}, {strategy}"
                )

            if row["weight_date"] >= pd.Timestamp(date):
                raise ValueError(
                    f"Look-ahead weight detected for {date}, {strategy}"
                )

            weights = row[weight_cols].to_numpy(dtype=float)

            if not np.isfinite(weights).all():
                raise ValueError(
                    f"Invalid weights for {date}, {strategy}"
                )

            if (weights < -1e-8).any():
                raise ValueError(
                    f"Unexpected short position for {date}, {strategy}"
                )

            if not np.isclose(
                weights.sum(), 1.0, atol=1e-8, rtol=0
            ):
                raise ValueError(
                    f"Weights do not sum to 1 for {date}, {strategy}"
                )

            snapshots[date, strategy] = weights

    for method in ("minvar", "meanvar"):
        calm = "2025-04-01"
        stress = "2025-05-01"

        if not np.allclose(
            snapshots[calm, f"{method}_static"],
            snapshots[calm, f"{method}_regime"],
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                f"Calm weights unexpectedly differ for {method}"
            )

        if np.allclose(
            snapshots[stress, f"{method}_static"],
            snapshots[stress, f"{method}_regime"],
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                f"Stressed weights unexpectedly match for {method}"
            )

    return snapshots


def main():
    snapshots = load_snapshot_weights()

    summary_rows = []
    contribution_rows = []
    difference_rows = []

    for date, metadata in SNAPSHOTS.items():
        for scenario, shock_pct in SCENARIOS.items():

            shocks = np.asarray(shock_pct, dtype=float) / 100.0

            if len(shocks) != len(ASSETS) or not np.isfinite(shocks).all():
                raise ValueError(
                    f"Invalid shock vector: {scenario}"
                )

            if (shocks < -1).any():
                raise ValueError(
                    f"Shock below -100%: {scenario}"
                )

            scenario_returns = {}

            for strategy in STRATEGIES:
                weights = snapshots[date, strategy]

                contributions = weights * shocks
                portfolio_return = float(np.dot(weights, shocks))

                if not np.isclose(
                    contributions.sum(),
                    portfolio_return,
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError(
                        f"Contribution mismatch: "
                        f"{date}, {scenario}, {strategy}"
                    )

                scenario_returns[strategy] = portfolio_return

                common = {
                    "snapshot_date": date,
                    "weight_date": metadata["weight_date"],
                    "regime": metadata["regime"],
                    "scenario": scenario,
                    "strategy": strategy,
                }

                summary_rows.append({
                    **common,
                    "portfolio_return": portfolio_return,
                })

                for asset, weight, shock, contribution in zip(
                    ASSETS, weights, shocks, contributions
                ):
                    contribution_rows.append({
                        **common,
                        "asset": asset,
                        "weight": float(weight),
                        "asset_shock": float(shock),
                        "contribution": float(contribution),
                    })

            for method in ("minvar", "meanvar"):
                static = f"{method}_static"
                regime = f"{method}_regime"

                diff = (
                    scenario_returns[regime]
                    - scenario_returns[static]
                )

                exposure_diff = (
                    snapshots[date, regime]
                    - snapshots[date, static]
                )

                if not np.isclose(
                    np.dot(exposure_diff, shocks),
                    diff,
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError(
                        f"Regime difference mismatch: "
                        f"{date}, {scenario}, {method}"
                    )

                difference_rows.append({
                    "snapshot_date": date,
                    "regime": metadata["regime"],
                    "scenario": scenario,
                    "method": method,
                    "static_return": scenario_returns[static],
                    "regime_return": scenario_returns[regime],
                    "regime_minus_static": diff,
                })

    summary = pd.DataFrame(summary_rows)
    contributions = pd.DataFrame(contribution_rows)
    differences = pd.DataFrame(difference_rows)

    if (
        len(summary) != 30
        or len(contributions) != 240
        or len(differences) != 12
    ):
        raise ValueError(
            "Unexpected number of scenario results"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR / "hypothetical_stress_summary.csv",
        index=False,
        float_format="%.12f",
    )

    contributions.to_csv(
        OUTPUT_DIR / "hypothetical_stress_contributions.csv",
        index=False,
        float_format="%.12f",
    )

    differences.to_csv(
        OUTPUT_DIR / "hypothetical_stress_regime_differences.csv",
        index=False,
        float_format="%.12f",
    )

    print("Hypothetical Scenario Stress Testing")
    print(
        "Assumption: fixed initial holdings, "
        "one 20-trading-day cumulative shock"
    )

    for date, metadata in SNAPSHOTS.items():
        print(
            f"\nSnapshot: {date} | "
            f"regime: {metadata['regime']} | "
            f"weights: {metadata['weight_date']}"
        )

        view = summary.loc[
            summary["snapshot_date"] == date
        ].pivot(
            index="scenario",
            columns="strategy",
            values="portfolio_return",
        )

        view = (
            view.reindex(
                index=SCENARIOS,
                columns=STRATEGIES,
            ) * 100
        )

        print("Portfolio scenario returns (%)")
        print(
            view.to_string(
                float_format=lambda x: f"{x:+.2f}"
            )
        )

    print(
        "\nRegime minus Static differences "
        "(percentage points)"
    )

    for row in differences.itertuples(index=False):
        print(
            f"{row.snapshot_date} | "
            f"{row.scenario} | "
            f"{row.method}: "
            f"{row.regime_minus_static * 100:+.3f} pp"
        )

    print(
        "\nValidation: PASS "
        "(weights, attribution, and strategy differences)"
    )

    print(f"Outputs saved in: {OUTPUT_DIR}")
    print(
        "CSV return and contribution values use decimals; "
        "terminal values use percent / pp."
    )


if __name__ == "__main__":
    main()
