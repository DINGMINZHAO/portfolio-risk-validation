
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
OUTPUT_PATH = ROOT / "data/processed/qpm/historical_stress_2025_summary.csv"

STRATEGIES = {
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime",
    "ew_static": "Equal Weight",
}

WINDOWS = {
    "crash": ("2025-04-03", "2025-04-08", 4),
    "rebound": ("2025-04-09", "2025-04-30", 15),
    "full": ("2025-04-03", "2025-04-30", 19),
}


def cumulative_return(returns):
    return float(np.prod(1.0 + returns) - 1.0)


def max_drawdown(returns):
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def main():
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    required = ["Date", *STRATEGIES.keys()]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["Date"].isna().any() or df["Date"].duplicated().any():
        raise ValueError("Date column contains missing or duplicate dates.")

    df = df.sort_values("Date").set_index("Date")
    if pd.Timestamp("2025-04-02") not in df.index:
        raise ValueError("Baseline trading date 2025-04-02 is missing.")

    windows = {}
    for name, (start, end, expected_days) in WINDOWS.items():
        part = df.loc[start:end, list(STRATEGIES)]
        if len(part) != expected_days:
            raise ValueError(
                f"{name}: expected {expected_days} trading days, found {len(part)}."
            )
        values = part.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: invalid or missing portfolio returns.")
        if (values < -1.0).any():
            raise ValueError(f"{name}: found a daily arithmetic return below -100%.")

        windows[name] = part

    rows = []
    for strategy, label in STRATEGIES.items():
        crash = windows["crash"][strategy].to_numpy(dtype=float)
        rebound = windows["rebound"][strategy].to_numpy(dtype=float)
        full = windows["full"][strategy].to_numpy(dtype=float)

        crash_return = cumulative_return(crash)
        rebound_return = cumulative_return(rebound)
        full_return = cumulative_return(full)

        reconstructed_full = (
            (1.0 + crash_return) * (1.0 + rebound_return) - 1.0
        )
        error = reconstructed_full - full_return

        rows.append({
            "strategy": strategy,
            "crash_return": crash_return,
            "rebound_return": rebound_return,
            "full_return": full_return,
            "crash_mdd": max_drawdown(crash),
            "crash_worst_day": float(np.min(crash)),
            "full_mdd": max_drawdown(full),
            "full_worst_day": float(np.min(full)),
            "reconciliation_error": error,
        })

    summary = pd.DataFrame(rows)

    tolerance = 1e-10
    if (summary["reconciliation_error"].abs() > tolerance).any():
        raise ValueError(
            "Crash/Rebound compounding does not match Full Event return."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False, float_format="%.12f")

    print("2025 April Trade Policy Shock — Portfolio Stress Testing")
    print("Baseline: 2025-04-02 close (initial wealth = 1.0)")

    for name, (start, end, _) in WINDOWS.items():
        print(
            f"{name.capitalize():8s}: {start} to {end} | "
            f"{len(windows[name])} trading days"
        )

    display = (
        summary.set_index("strategy")
        .drop(columns="reconciliation_error") * 100.0
    )
    display.index = [STRATEGIES[name] for name in display.index]

    print("\nPortfolio results (all numbers in %):")
    print(display.to_string(float_format=lambda value: f"{value:+.3f}"))

    by_strategy = summary.set_index("strategy")

    print("\nRegime minus Static return differences (percentage points):")

    for family in ("meanvar", "minvar"):
        regime = by_strategy.loc[f"{family}_regime"]
        static = by_strategy.loc[f"{family}_static"]

        differences = {
            window: 100.0 * (
                regime[f"{window}_return"] - static[f"{window}_return"]
            )
            for window in WINDOWS
        }

        print(
            f"{family:7s}: "
            + " | ".join(
                f"{window} {difference:+.3f} pp"
                for window, difference in differences.items()
            )
        )

    max_error = summary["reconciliation_error"].abs().max()

    print(
        f"\nWindow reconciliation: PASS "
        f"(max absolute error: {max_error:.3e})"
    )
    print(f"Output saved: {OUTPUT_PATH}")
    print(
        "Note: CSV return/risk values are decimals; "
        "terminal results are percentages."
    )


if __name__ == "__main__":
    main()
