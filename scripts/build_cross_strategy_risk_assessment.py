
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/interim/qpm"
RESULTS = ROOT / "data/processed/qpm"

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]

SCENARIOS = [
    "A_equity_crash",
    "B_inflation_rate_shock",
    "C_joint_historical_worst",
]

SNAPSHOTS = ["2025-04-01", "2025-05-01"]


def load_table(path, columns, keys):
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")

    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    if df.duplicated(keys).any():
        raise ValueError(f"{path.name}: duplicate {keys}")

    return df


def strategy_values(df, column):
    subset = df.set_index("strategy").reindex(STRATEGIES)

    if subset[column].isna().any():
        raise ValueError(f"Missing strategy values for {column}")

    return subset[column].astype(float) * 100


def main():
    profile = load_table(
        BASE / "baseline_risk_profile.csv",
        ["strategy", "annualized_return", "annualized_vol",
         "max_drawdown", "worst_day"],
        ["strategy"],
    )

    crash_2020 = load_table(
        RESULTS / "historical_stress_2020_crash_risk_metrics.csv",
        ["strategy", "cumulative_return", "maximum_drawdown"],
        ["strategy"],
    )

    year_2022 = load_table(
        RESULTS / "historical_stress_2022_summary.csv",
        ["strategy", "cumulative_return", "maximum_drawdown"],
        ["strategy"],
    )

    crash_2025 = load_table(
        RESULTS / "historical_stress_2025_summary.csv",
        ["strategy", "crash_return", "crash_mdd"],
        ["strategy"],
    )

    hypo = load_table(
        RESULTS / "hypothetical_stress_summary.csv",
        ["snapshot_date", "scenario", "strategy", "portfolio_return"],
        ["snapshot_date", "scenario", "strategy"],
    )

    overview = pd.DataFrame(index=STRATEGIES)
    overview.index.name = "strategy"

    sources = [
        ("annualized_return_pct", profile, "annualized_return"),
        ("annualized_vol_pct", profile, "annualized_vol"),
        ("full_sample_mdd_pct", profile, "max_drawdown"),
        ("worst_day_pct", profile, "worst_day"),
        ("crash_2020_return_pct", crash_2020, "cumulative_return"),
        ("crash_2020_mdd_pct", crash_2020, "maximum_drawdown"),
        ("year_2022_return_pct", year_2022, "cumulative_return"),
        ("year_2022_mdd_pct", year_2022, "maximum_drawdown"),
        ("crash_2025_return_pct", crash_2025, "crash_return"),
        ("crash_2025_mdd_pct", crash_2025, "crash_mdd"),
    ]

    for output_name, table, source_name in sources:
        overview[output_name] = strategy_values(table, source_name)

    if (
        len(hypo) != len(SNAPSHOTS) * len(SCENARIOS) * len(STRATEGIES)
        or not set(hypo["strategy"]).issubset(STRATEGIES)
    ):
        raise ValueError("Hypothetical scenario dataset is incomplete")

    hypo_matrix = hypo.pivot(
        index=["snapshot_date", "scenario"],
        columns="strategy",
        values="portfolio_return",
    ).reindex(
        index=pd.MultiIndex.from_product(
            [SNAPSHOTS, SCENARIOS],
            names=["snapshot_date", "scenario"],
        ),
        columns=STRATEGIES,
    )

    if hypo_matrix.isna().any().any():
        raise ValueError("Incomplete hypothetical scenario grid")

    hypo_matrix = hypo_matrix.astype(float) * 100

    for scenario in SCENARIOS:
        calm = hypo_matrix.loc[("2025-04-01", scenario)]

        for method in ("minvar", "meanvar"):
            if not np.isclose(
                calm[f"{method}_regime"],
                calm[f"{method}_static"],
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError(
                    f"Calm strategy mismatch: {scenario}, {method}"
                )

    differences = []

    for method in ("minvar", "meanvar"):
        static = f"{method}_static"
        regime = f"{method}_regime"

        for metric in overview.columns:
            differences.append({
                "family": method,
                "comparison": "full_sample_or_historical",
                "metric": metric,
                "static_value_pct": overview.loc[static, metric],
                "regime_value_pct": overview.loc[regime, metric],
                "regime_minus_static_pp": (
                    overview.loc[regime, metric]
                    - overview.loc[static, metric]
                ),
            })

        for (date, scenario), row in hypo_matrix.iterrows():
            differences.append({
                "family": method,
                "comparison": date,
                "metric": scenario,
                "static_value_pct": row[static],
                "regime_value_pct": row[regime],
                "regime_minus_static_pp": row[regime] - row[static],
            })

    differences = pd.DataFrame(differences)

    var = load_table(
        RESULTS / "var_backtest_summary.csv",
        ["model", "strategy", "confidence_level", "observed_rate"],
        ["model", "strategy", "confidence_level"],
    )

    tests = load_table(
        RESULTS / "var_backtest_statistical_tests.csv",
        ["model", "strategy", "confidence_level",
         "kupiec_p_value", "independence_p_value",
         "conditional_coverage_p_value"],
        ["model", "strategy", "confidence_level"],
    )

    es = load_table(
        RESULTS / "es_all_models_summary.csv",
        ["model", "strategy", "confidence_level",
         "mean_es_residual_bp", "mean_fz0_score"],
        ["model", "strategy", "confidence_level"],
    )

    def fhs99(df):
        return df.loc[
            (df["model"] == "garch_fhs")
            & np.isclose(df["confidence_level"], 0.99)
        ].set_index("strategy").reindex(STRATEGIES)

    vr = fhs99(var)
    ts = fhs99(tests)
    er = fhs99(es)

    context = pd.DataFrame({
        "strategy": STRATEGIES,
        "fhs99_exceedance_rate_pct": (
            vr["observed_rate"].to_numpy() * 100
        ),
        "kupiec_p_value": ts["kupiec_p_value"].to_numpy(),
        "independence_p_value": ts["independence_p_value"].to_numpy(),
        "conditional_coverage_p_value": (
            ts["conditional_coverage_p_value"].to_numpy()
        ),
        "mean_es_residual_bp": (
            er["mean_es_residual_bp"].to_numpy()
        ),
        "mean_fz0_score": er["mean_fz0_score"].to_numpy(),
    })

    if context.isna().any().any():
        raise ValueError("Incomplete GARCH-FHS 99% validation context")

    RESULTS.mkdir(parents=True, exist_ok=True)

    overview.reset_index().to_csv(
        RESULTS / "cross_strategy_risk_overview.csv",
        index=False,
    )

    hypo_matrix.reset_index().to_csv(
        RESULTS / "cross_strategy_hypothetical_matrix.csv",
        index=False,
    )

    differences.to_csv(
        RESULTS / "cross_strategy_regime_effects.csv",
        index=False,
    )

    context.to_csv(
        RESULTS / "cross_strategy_fhs99_validation.csv",
        index=False,
    )

    pd.set_option("display.width", 160)

    print("\nFULL-SAMPLE RISK PROFILE (%)")
    print(
        overview[
            [
                "annualized_return_pct",
                "annualized_vol_pct",
                "full_sample_mdd_pct",
            ]
        ].round(2).to_string()
    )

    print("\nHISTORICAL EVENT RETURNS (%)")
    print(
        overview[
            [
                "crash_2020_return_pct",
                "year_2022_return_pct",
                "crash_2025_return_pct",
            ]
        ].round(2).to_string()
    )

    print("\nHYPOTHETICAL SCENARIOS: STRESSED SNAPSHOT (%)")
    print(
        hypo_matrix.loc["2025-05-01"].round(2).to_string()
    )

    print("\nREGIME MINUS STATIC: STRESSED SNAPSHOT (pp)")

    stressed = differences[
        differences["comparison"] == "2025-05-01"
    ]

    print(
        stressed.pivot(
            index="metric",
            columns="family",
            values="regime_minus_static_pp",
        ).reindex(SCENARIOS).round(3).to_string()
    )

    print("\nGARCH-FHS: 99% VAR VALIDATION CONTEXT")
    print(
        context[
            [
                "strategy",
                "fhs99_exceedance_rate_pct",
                "independence_p_value",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nSaved four evidence tables in: {RESULTS}")
    print(
        "Note: FZ0 scores must not be used to rank portfolio risk "
        "across different strategies."
    )


if __name__ == "__main__":
    main()
