
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


WINDOW = 252
CONFIDENCE_LEVELS = [0.95, 0.99]

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
    / "processed"
    / "qpm"
    / "rolling_normal_var_es_forecasts.csv"
)

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "Minimum Variance Static",
    "minvar_regime": "Minimum Variance Regime-aware",
    "meanvar_static": "Mean-Variance Static",
    "meanvar_regime": "Mean-Variance Regime-aware",
}


def validate_input(data):
    if "Date" not in data.columns:
        raise ValueError("Date column is missing.")

    if data["Date"].isna().any():
        raise ValueError("Invalid dates found.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    missing = [
        strategy
        for strategy in STRATEGIES
        if strategy not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing strategies: {missing}")

    if len(data) <= WINDOW:
        raise ValueError("Insufficient observations.")

    for strategy in STRATEGIES:
        values = pd.to_numeric(
            data[strategy],
            errors="raise",
        )

        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(
                f"Invalid returns: {strategy}"
            )


def calculate_forecasts(data):
    records = []

    for strategy, strategy_name in STRATEGIES.items():
        returns = data[strategy].astype(float)

        mu_hat = (
            returns
            .rolling(WINDOW, min_periods=WINDOW)
            .mean()
            .shift(1)
        )

        sigma_hat = (
            returns
            .rolling(WINDOW, min_periods=WINDOW)
            .std(ddof=1)
            .shift(1)
        )

        for confidence in CONFIDENCE_LEVELS:
            z = norm.ppf(confidence)

            var_forecast = (
                -mu_hat + z * sigma_hat
            )

            es_forecast = (
                -mu_hat
                + sigma_hat
                * norm.pdf(z)
                / (1.0 - confidence)
            )

            frame = pd.DataFrame({
                "Date": data["Date"],
                "strategy": strategy,
                "strategy_name": strategy_name,
                "confidence_level": confidence,
                "window_days": WINDOW,
                "estimation_start_date":
                    data["Date"].shift(WINDOW),
                "estimation_end_date":
                    data["Date"].shift(1),
                "mu_hat": mu_hat,
                "sigma_hat": sigma_hat,
                "var_forecast": var_forecast,
                "es_forecast": es_forecast,
            })

            frame = frame.dropna(
                subset=["mu_hat", "sigma_hat"]
            ).copy()

            if not (
                frame["estimation_end_date"] < frame["Date"]
            ).all():
                raise ValueError(
                    "Look-ahead detected."
                )

            if not (
                frame["es_forecast"] >= frame["var_forecast"]
            ).all():
                raise ValueError(
                    "ES must be greater than or equal to VaR."
                )

            records.append(frame)

    forecasts = pd.concat(
        records,
        ignore_index=True,
    )

    forecasts = forecasts.sort_values(
        ["Date", "strategy", "confidence_level"]
    ).reset_index(drop=True)

    return forecasts


def main():
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    validate_input(data)

    forecasts = calculate_forecasts(data)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nROLLING NORMAL VAR / ES")
    print("=" * 80)

    print(f"Window: {WINDOW} trading days")
    print(f"Portfolios: {len(STRATEGIES)}")
    print(f"Confidence levels: {CONFIDENCE_LEVELS}")
    print(f"Forecast rows: {len(forecasts):,}")

    print(
        f"Forecast period: "
        f"{forecasts['Date'].min().date()} -> "
        f"{forecasts['Date'].max().date()}"
    )

    print("\nLATEST FORECASTS")
    print("=" * 80)

    latest = forecasts[
        forecasts["Date"] == forecasts["Date"].max()
    ].copy()

    columns = [
        "mu_hat",
        "sigma_hat",
        "var_forecast",
        "es_forecast",
    ]

    for column in columns:
        latest[column] = latest[column].map(
            lambda x: f"{x:.4%}"
        )

    print(
        latest[
            [
                "strategy",
                "confidence_level",
                "mu_hat",
                "sigma_hat",
                "var_forecast",
                "es_forecast",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
