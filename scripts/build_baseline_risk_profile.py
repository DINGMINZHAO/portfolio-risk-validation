from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS = 252

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "qpm"
    / "portfolio_returns_main.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "qpm"
    / "baseline_risk_profile.csv"
)


STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "Minimum Variance Static",
    "minvar_regime": "Minimum Variance Regime-aware",
    "meanvar_static": "Mean-Variance Static",
    "meanvar_regime": "Mean-Variance Regime-aware",
}


def geometric_annualized_return(returns):
    total_growth = (1.0 + returns).prod()
    n_days = len(returns)

    return total_growth ** (TRADING_DAYS / n_days) - 1.0


def calculate_max_drawdown(returns):
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0

    return drawdown.min()


def validate_returns(data):
    if "Date" not in data.columns:
        raise ValueError("Date column is missing.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    missing_strategies = [
        strategy
        for strategy in STRATEGIES
        if strategy not in data.columns
    ]

    if missing_strategies:
        raise ValueError(
            f"Missing strategy columns: {missing_strategies}"
        )

    strategy_data = data[list(STRATEGIES.keys())]

    if strategy_data.isna().any().any():
        raise ValueError("Missing portfolio returns found.")

    if not np.isfinite(strategy_data.to_numpy()).all():
        raise ValueError("Non-finite portfolio returns found.")

    if (strategy_data <= -1.0).any().any():
        raise ValueError(
            "Return <= -100% found. "
            "Compounded wealth cannot be calculated safely."
        )


def build_baseline_profile(data):
    records = []

    for strategy, strategy_name in STRATEGIES.items():
        returns = data[strategy]

        record = {
            "strategy": strategy,
            "strategy_name": strategy_name,
            "daily_mean": returns.mean(),
            "daily_vol": returns.std(ddof=1),
            "annualized_return": geometric_annualized_return(returns),
            "annualized_vol": returns.std(ddof=1) * np.sqrt(TRADING_DAYS),
            "skewness": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "worst_day": returns.min(),
            "best_day": returns.max(),
            "historical_5pct": returns.quantile(0.05),
            "historical_1pct": returns.quantile(0.01),
            "max_drawdown": calculate_max_drawdown(returns),
        }

        records.append(record)

    return pd.DataFrame(records)


def main():
    data = pd.read_csv(INPUT_PATH, parse_dates=["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    validate_returns(data)

    profile = build_baseline_profile(data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUTPUT_PATH, index=False)

    print("\nBASELINE RISK PROFILE")
    print("=" * 100)

    display = profile.copy()

    percentage_columns = [
        "daily_mean",
        "daily_vol",
        "annualized_return",
        "annualized_vol",
        "worst_day",
        "best_day",
        "historical_5pct",
        "historical_1pct",
        "max_drawdown",
    ]

    for column in percentage_columns:
        display[column] = display[column].map(
            lambda x: f"{x:.4%}"
        )

    display["skewness"] = display["skewness"].map(
        lambda x: f"{x:.4f}"
    )

    display["excess_kurtosis"] = display[
        "excess_kurtosis"
    ].map(lambda x: f"{x:.4f}")

    print(display.to_string(index=False))

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()