from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
OUTPUT_PATH = ROOT / "data/processed/qpm/garch_normal_var_es_forecasts.csv"
PARAMETER_PATH = ROOT / "data/processed/qpm/garch_normal_fit_parameters.csv"

WINDOW = 1008
REFIT_EVERY = 21
CONFIDENCE_LEVELS = (0.95, 0.99)

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "Minimum Variance Static",
    "minvar_regime": "Minimum Variance Regime-aware",
    "meanvar_static": "Mean-Variance Static",
    "meanvar_regime": "Mean-Variance Regime-aware",
}


def read_returns():
    data = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    if data["Date"].isna().any() or data["Date"].duplicated().any():
        raise ValueError("Missing or duplicate dates.")

    if len(data) <= WINDOW:
        raise ValueError(f"Need more than {WINDOW} daily observations.")

    for strategy in STRATEGIES:
        if strategy not in data:
            raise ValueError(f"Missing return series: {strategy}")

        values = pd.to_numeric(
            data[strategy], errors="raise"
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values < -1).any():
            raise ValueError(f"Invalid daily returns: {strategy}")

        data[strategy] = values

    return data


def fit_garch(history_pct, strategy, end_date):
    model = arch_model(
        history_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False,
    )

    result = model.fit(
        disp="off",
        show_warning=False,
        options={"maxiter": 1000},
    )

    if result.convergence_flag != 0:
        result = model.fit(
            disp="off",
            show_warning=False,
            starting_values=result.params.to_numpy(),
            options={"maxiter": 2000},
        )

    if result.convergence_flag != 0:
        raise RuntimeError(
            f"GARCH did not converge: {strategy}, {end_date}"
        )

    params = result.params

    mu = float(params["mu"])
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    last_sigma2 = float(
        result.conditional_volatility[-1] ** 2
    )

    first_variance = (
        omega
        + alpha * (history_pct[-1] - mu) ** 2
        + beta * last_sigma2
    )

    if not np.isfinite(first_variance) or first_variance <= 0:
        raise ValueError(
            f"Invalid GARCH variance: {strategy}, {end_date}"
        )

    return (mu, omega, alpha, beta), first_variance


def main():
    data = read_returns()

    dates = data["Date"].to_numpy()
    records = []
    parameter_records = []

    for strategy, strategy_name in STRATEGIES.items():
        returns_pct = (
            100.0 * data[strategy].to_numpy(dtype=float)
        )

        print(f"Fitting {strategy} ...", flush=True)

        for i in range(WINDOW, len(data)):
            refit = (i - WINDOW) % REFIT_EVERY == 0

            if refit:
                history = returns_pct[i - WINDOW:i]

                (mu, omega, alpha, beta), variance = fit_garch(
                    history,
                    strategy,
                    pd.Timestamp(dates[i - 1]).date(),
                )

                fit_start = dates[i - WINDOW]
                fit_end = dates[i - 1]

                parameter_records.append({
                    "strategy": strategy,
                    "refit_date": dates[i],
                    "estimation_start_date": fit_start,
                    "estimation_end_date": fit_end,
                    "mu_pct": mu,
                    "omega_pct2": omega,
                    "alpha": alpha,
                    "beta": beta,
                    "alpha_plus_beta": alpha + beta,
                })

            else:
                previous_innovation = returns_pct[i - 1] - mu

                variance = (
                    omega
                    + alpha * previous_innovation ** 2
                    + beta * variance
                )

            if not np.isfinite(variance) or variance <= 0:
                raise ValueError(
                    f"Invalid forecast variance: "
                    f"{strategy}, {dates[i]}"
                )

            mu_hat = mu / 100.0
            sigma_hat = np.sqrt(variance) / 100.0

            for confidence in CONFIDENCE_LEVELS:
                z = norm.ppf(confidence)

                var_value = -mu_hat + sigma_hat * z

                es_value = (
                    -mu_hat
                    + sigma_hat * norm.pdf(z) / (1 - confidence)
                )

                if es_value < var_value:
                    raise ValueError(
                        "ES must not be smaller than VaR."
                    )

                records.append({
                    "Date": dates[i],
                    "strategy": strategy,
                    "strategy_name": strategy_name,
                    "model": "garch_normal",
                    "confidence_level": confidence,
                    "window_days": WINDOW,
                    "refit_every_days": REFIT_EVERY,
                    "estimation_start_date": fit_start,
                    "estimation_end_date": fit_end,
                    "information_end_date": dates[i - 1],
                    "mu_hat": mu_hat,
                    "sigma_hat": sigma_hat,
                    "var_forecast": var_value,
                    "es_forecast": es_value,
                })

    forecasts = pd.DataFrame(records).sort_values(
        ["Date", "strategy", "confidence_level"]
    ).reset_index(drop=True)

    parameters = pd.DataFrame(parameter_records)

    if not (
        (
            forecasts["estimation_end_date"]
            < forecasts["Date"]
        ).all()
        and (
            forecasts["information_end_date"]
            < forecasts["Date"]
        ).all()
    ):
        raise ValueError("Look-ahead detected.")

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    forecasts.to_csv(OUTPUT_PATH, index=False)
    parameters.to_csv(PARAMETER_PATH, index=False)

    print("\nGARCH-NORMAL FORECASTS")

    print(f"Forecast rows: {len(forecasts):,}")
    print(f"Parameter refits: {len(parameters):,}")

    print(
        f"Dates: {forecasts['Date'].min().date()} "
        f"to {forecasts['Date'].max().date()}"
    )

    print(f"Saved forecasts: {OUTPUT_PATH}")
    print(f"Saved parameters: {PARAMETER_PATH}")


if __name__ == "__main__":
    main()