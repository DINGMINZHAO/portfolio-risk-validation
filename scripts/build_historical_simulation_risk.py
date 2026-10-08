from pathlib import Path

import numpy as np
import pandas as pd


WINDOW = 1008
CONFIDENCE_LEVELS = (0.95, 0.99)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "interim" / "qpm" / "portfolio_returns_main.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "qpm"
    / "historical_simulation_var_es_forecasts.csv"
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
    if data["Date"].isna().any() or data["Date"].duplicated().any():
        raise ValueError("Invalid or duplicate dates found.")
    missing = [strategy for strategy in STRATEGIES if strategy not in data.columns]
    if missing:
        raise ValueError(f"Missing strategies: {missing}")
    if len(data) <= WINDOW:
        raise ValueError(f"More than {WINDOW} daily returns are required.")
    for strategy in STRATEGIES:
        values = pd.to_numeric(data[strategy], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid returns: {strategy}")
        if (values < -1).any():
            raise ValueError(f"Return below -100%: {strategy}")


def historical_var_es(sorted_returns, confidence):
    n = len(sorted_returns)
    tail_mass = n * (1.0 - confidence)

    nearest_integer = round(tail_mass)
    if np.isclose(tail_mass, nearest_integer, rtol=0.0, atol=1e-10):
        tail_mass = float(nearest_integer)

    full_count = int(np.floor(tail_mass))
    fraction = tail_mass - full_count
    quantile_index = max(int(np.ceil(tail_mass)) - 1, 0)
    var_forecast = -sorted_returns[quantile_index]

    tail_sum = sorted_returns[:full_count].sum()
    if fraction > 0:
        tail_sum += fraction * sorted_returns[full_count]

    es_forecast = -tail_sum / tail_mass
    return float(var_forecast), float(es_forecast)


def build_forecasts(data):
    dates = data["Date"].to_numpy()
    records = []

    for strategy, strategy_name in STRATEGIES.items():
        returns = data[strategy].to_numpy(dtype=float)

        for i in range(WINDOW, len(data)):
            history = np.sort(returns[i - WINDOW:i])

            for confidence in CONFIDENCE_LEVELS:
                var_value, es_value = historical_var_es(history, confidence)
                if es_value + 1e-12 < var_value:
                    raise ValueError("ES below VaR: check tail calculation.")

                records.append({
                    "Date": dates[i],
                    "strategy": strategy,
                    "strategy_name": strategy_name,
                    "model": "historical_simulation",
                    "confidence_level": confidence,
                    "window_days": WINDOW,
                    "estimation_start_date": dates[i - WINDOW],
                    "estimation_end_date": dates[i - 1],
                    "var_forecast": var_value,
                    "es_forecast": es_value,
                })

    forecasts = pd.DataFrame(records).sort_values(
        ["Date", "strategy", "confidence_level"]
    ).reset_index(drop=True)

    if not (forecasts["estimation_end_date"] < forecasts["Date"]).all():
        raise ValueError("Look-ahead detected.")
    return forecasts


def main():
    data = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    validate_input(data)

    forecasts = build_forecasts(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(OUTPUT_PATH, index=False)

    print("\nHISTORICAL SIMULATION VAR / ES")
    print(f"Window: {WINDOW} trading days")
    print(f"Portfolios: {len(STRATEGIES)}")
    print(f"Confidence levels: {list(CONFIDENCE_LEVELS)}")
    print(f"Forecast rows: {len(forecasts):,}")
    print(
        f"Forecast period: {forecasts['Date'].min().date()} "
        f"-> {forecasts['Date'].max().date()}"
    )
    latest = forecasts[forecasts["Date"] == forecasts["Date"].max()].copy()
    for column in ("var_forecast", "es_forecast"):
        latest[column] = latest[column].map(lambda value: f"{value:.4%}")
    print("\nLATEST FORECASTS")
    print(latest[["strategy", "confidence_level", "var_forecast", "es_forecast"]].to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
