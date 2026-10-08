
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from arch import arch_model


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data/interim/qpm/portfolio_returns_main.csv"

OUTPUT_DIR = ROOT / "data/processed/qpm"

FORECAST_PATH = OUTPUT_DIR / "garch_student_t_var_es_forecasts.csv"
PARAMETER_PATH = OUTPUT_DIR / "garch_student_t_fit_parameters.csv"

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
        raise ValueError("Invalid return dates")

    if len(data) <= WINDOW:
        raise ValueError("Insufficient observations")

    for strategy in STRATEGIES:

        if strategy not in data.columns:
            raise ValueError(f"Missing strategy: {strategy}")

        values = data[strategy].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Invalid returns: {strategy}")

        if (values < -1).any():
            raise ValueError(f"Return below -100%: {strategy}")

    return data


def fit_garch_student_t(history_pct, strategy, end_date):

    model = arch_model(
        history_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="t",
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
            f"GARCH-Student-t did not converge: "
            f"{strategy}, {end_date}"
        )

    params = result.params

    mu = float(params["mu"])
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    nu = float(params["nu"])

    if not np.isfinite(
        [mu, omega, alpha, beta, nu]
    ).all():
        raise ValueError("Non-finite fitted parameters")

    if nu <= 2:
        raise ValueError(
            f"Invalid Student-t degrees of freedom: {nu}"
        )

    last_sigma2 = float(
        result.conditional_volatility[-1] ** 2
    )

    first_variance = (
        omega
        + alpha * (history_pct[-1] - mu) ** 2
        + beta * last_sigma2
    )

    if (
        not np.isfinite(first_variance)
        or first_variance <= 0
    ):
        raise ValueError(
            f"Invalid forecast variance: {strategy}, {end_date}"
        )

    fitted = {
        "mu_pct": mu,
        "omega_pct2": omega,
        "alpha": alpha,
        "beta": beta,
        "alpha_plus_beta": alpha + beta,
        "nu": nu,
        "loglikelihood": float(result.loglikelihood),
        "convergence_flag": int(result.convergence_flag),
    }

    return fitted, first_variance


def student_t_risk(mu, sigma, nu, confidence):

    tail_probability = 1.0 - confidence

    raw_quantile = t.ppf(
        tail_probability,
        df=nu,
    )

    scale = np.sqrt((nu - 2.0) / nu)

    standardized_quantile = scale * raw_quantile

    var_forecast = (
        -mu - sigma * standardized_quantile
    )

    tail_conditional_mean = (
        -scale
        * (nu + raw_quantile ** 2)
        / (nu - 1.0)
        * t.pdf(raw_quantile, df=nu)
        / tail_probability
    )

    es_forecast = (
        -mu - sigma * tail_conditional_mean
    )

    if not np.isfinite(
        [var_forecast, es_forecast]
    ).all():
        raise ValueError("Non-finite Student-t VaR/ES")

    if es_forecast + 1e-12 < var_forecast:
        raise ValueError("ES is smaller than VaR")

    return float(var_forecast), float(es_forecast)


def main():

    data = read_returns()

    dates = data["Date"].to_numpy()

    forecast_records = []
    parameter_records = []

    for strategy, strategy_name in STRATEGIES.items():

        returns_pct = (
            data[strategy].to_numpy(dtype=float) * 100.0
        )

        print(f"Fitting {strategy} ...", flush=True)

        for i in range(WINDOW, len(data)):

            refit = (
                (i - WINDOW) % REFIT_EVERY == 0
            )

            if refit:

                history = returns_pct[i - WINDOW:i]

                fitted, variance = fit_garch_student_t(
                    history,
                    strategy,
                    pd.Timestamp(dates[i - 1]).date(),
                )

                mu = fitted["mu_pct"]
                omega = fitted["omega_pct2"]
                alpha = fitted["alpha"]
                beta = fitted["beta"]
                nu = fitted["nu"]

                fit_start = dates[i - WINDOW]
                fit_end = dates[i - 1]

                parameter_records.append({
                    "strategy": strategy,
                    "refit_date": dates[i],
                    "estimation_start_date": fit_start,
                    "estimation_end_date": fit_end,
                    **fitted,
                })

            else:

                previous_innovation = (
                    returns_pct[i - 1] - mu
                )

                variance = (
                    omega
                    + alpha * previous_innovation ** 2
                    + beta * variance
                )

            if (
                not np.isfinite(variance)
                or variance <= 0
            ):
                raise ValueError(
                    f"Invalid variance: {strategy}, {dates[i]}"
                )

            mu_hat = mu / 100.0
            sigma_hat = np.sqrt(variance) / 100.0

            for confidence in CONFIDENCE_LEVELS:

                var_value, es_value = student_t_risk(
                    mu_hat,
                    sigma_hat,
                    nu,
                    confidence,
                )

                forecast_records.append({
                    "Date": dates[i],
                    "strategy": strategy,
                    "strategy_name": strategy_name,
                    "model": "garch_student_t",
                    "confidence_level": confidence,
                    "window_days": WINDOW,
                    "refit_every_days": REFIT_EVERY,
                    "estimation_start_date": fit_start,
                    "estimation_end_date": fit_end,
                    "information_end_date": dates[i - 1],
                    "mu_hat": mu_hat,
                    "sigma_hat": sigma_hat,
                    "nu": nu,
                    "var_forecast": var_value,
                    "es_forecast": es_value,
                })

    forecasts = pd.DataFrame(forecast_records)

    forecasts = forecasts.sort_values(
        ["Date", "strategy", "confidence_level"]
    ).reset_index(drop=True)

    parameters = pd.DataFrame(parameter_records)

    expected_days = len(data) - WINDOW
    expected_forecasts = (
        expected_days
        * len(STRATEGIES)
        * len(CONFIDENCE_LEVELS)
    )
    expected_refits = (
        len(range(WINDOW, len(data), REFIT_EVERY))
        * len(STRATEGIES)
    )

    if len(forecasts) != expected_forecasts:
        raise ValueError("Incorrect forecast record count")

    if len(parameters) != expected_refits:
        raise ValueError("Incorrect refit record count")

    if forecasts.duplicated(
        ["Date", "strategy", "confidence_level"]
    ).any():
        raise ValueError("Duplicate forecasts")

    if forecasts.isna().any().any():
        raise ValueError("Missing forecast values")

    if not (
        forecasts["information_end_date"] < forecasts["Date"]
    ).all():
        raise ValueError("Look-ahead detected")

    if not (
        forecasts["estimation_end_date"] < forecasts["Date"]
    ).all():
        raise ValueError("Invalid estimation cutoff")

    numeric = forecasts.select_dtypes(
        include="number"
    ).to_numpy()

    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite forecast values")

    wide = forecasts.pivot(
        index=["Date", "strategy"],
        columns="confidence_level",
        values=["var_forecast", "es_forecast"],
    )

    if not (
        wide[("var_forecast", 0.99)]
        >= wide[("var_forecast", 0.95)]
    ).all():
        raise ValueError("VaR confidence ordering failed")

    if not (
        wide[("es_forecast", 0.99)]
        >= wide[("es_forecast", 0.95)]
    ).all():
        raise ValueError("ES confidence ordering failed")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        FORECAST_PATH,
        index=False,
    )

    parameters.to_csv(
        PARAMETER_PATH,
        index=False,
    )

    print("\nGARCH-STUDENT-T FORECASTS")

    print(f"Forecast rows: {len(forecasts):,}")
    print(f"Parameter refits: {len(parameters):,}")

    print(
        f"Dates: {forecasts['Date'].min().date()} "
        f"to {forecasts['Date'].max().date()}"
    )

    print("\nPARAMETER SUMMARY")

    print(
        parameters.groupby("strategy")[
            ["alpha", "beta", "alpha_plus_beta", "nu"]
        ].median().round(4).to_string()
    )

    print("\nDEGREES OF FREEDOM")

    print(
        parameters.groupby("strategy")["nu"]
        .agg(["min", "median", "max"])
        .round(4)
        .to_string()
    )

    print("\nMODEL DIAGNOSTICS")

    print(
        "Convergence failures:",
        int((parameters["convergence_flag"] != 0).sum())
    )

    print(
        "Persistence near or above 1:",
        int((parameters["alpha_plus_beta"] >= 1 - 1e-10).sum())
    )

    print(
        "Degrees of freedom <= 4:",
        int((parameters["nu"] <= 4).sum())
    )

    print(
        "Degrees of freedom >= 100:",
        int((parameters["nu"] >= 100).sum())
    )

    print(f"\nSaved forecasts: {FORECAST_PATH}")
    print(f"Saved parameters: {PARAMETER_PATH}")


if __name__ == "__main__":
    main()
