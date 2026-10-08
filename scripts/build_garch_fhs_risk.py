
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


ROOT = Path(__file__).resolve().parents[1]

RETURNS_PATH = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
NORMAL_PATH = ROOT / "data/processed/qpm/garch_normal_var_es_forecasts.csv"
PARAMS_PATH = ROOT / "data/processed/qpm/garch_normal_fit_parameters.csv"
OUTPUT_PATH = ROOT / "data/processed/qpm/garch_fhs_var_es_forecasts.csv"

WINDOW = 1008
REFIT_EVERY = 21
LEVELS = (0.95, 0.99)

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "Minimum Variance Static",
    "minvar_regime": "Minimum Variance Regime-aware",
    "meanvar_static": "Mean-Variance Static",
    "meanvar_regime": "Mean-Variance Regime-aware",
}


def read_inputs():
    returns = pd.read_csv(RETURNS_PATH, parse_dates=["Date"])
    returns = returns.sort_values("Date").reset_index(drop=True)

    if returns["Date"].isna().any() or returns["Date"].duplicated().any():
        raise ValueError("Invalid return dates")

    if len(returns) - WINDOW != 3839:
        raise ValueError("Unexpected number of return observations")

    for strategy in STRATEGIES:
        if strategy not in returns.columns:
            raise ValueError(f"Missing returns: {strategy}")

        x = returns[strategy].to_numpy(dtype=float)

        if not np.isfinite(x).all() or (x < -1).any():
            raise ValueError(f"Invalid returns: {strategy}")

    normal = pd.read_csv(
        NORMAL_PATH,
        parse_dates=[
            "Date",
            "estimation_start_date",
            "estimation_end_date",
            "information_end_date",
        ],
    )

    required = {
        "Date",
        "strategy",
        "model",
        "confidence_level",
        "mu_hat",
        "sigma_hat",
        "window_days",
        "refit_every_days",
        "estimation_start_date",
        "estimation_end_date",
        "information_end_date",
    }

    if not required.issubset(normal.columns):
        raise ValueError("GARCH-Normal forecasts are missing columns")

    keys = ["Date", "strategy", "confidence_level"]

    if normal.duplicated(keys).any() or normal[keys].isna().any().any():
        raise ValueError("Duplicate or missing GARCH-Normal keys")

    if not normal["model"].eq("garch_normal").all():
        raise ValueError("Unexpected source model")

    if set(normal["confidence_level"]) != set(LEVELS):
        raise ValueError("Unexpected source confidence levels")

    if not normal["window_days"].eq(WINDOW).all():
        raise ValueError("Source estimation window differs")

    if not normal["refit_every_days"].eq(REFIT_EVERY).all():
        raise ValueError("Source refit frequency differs")

    if (normal["information_end_date"] >= normal["Date"]).any():
        raise ValueError("Look-ahead in source forecasts")

    normal = normal.loc[
        normal["confidence_level"] == 0.95
    ].copy()

    if len(normal) != 3839 * len(STRATEGIES):
        raise ValueError("Incomplete GARCH-Normal forecasts")

    params = pd.read_csv(
        PARAMS_PATH,
        parse_dates=[
            "refit_date",
            "estimation_start_date",
            "estimation_end_date",
        ],
    )

    if params.duplicated(["strategy", "refit_date"]).any():
        raise ValueError("Duplicate GARCH parameter refits")

    if not np.isfinite(
        params[
            ["mu_pct", "omega_pct2", "alpha", "beta"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid GARCH parameters")

    dates = returns["Date"].to_numpy()
    expected_forecast_dates = dates[WINDOW:]
    expected_refits = dates[WINDOW::REFIT_EVERY]

    for strategy in STRATEGIES:
        f = normal.loc[
            normal["strategy"] == strategy
        ].sort_values("Date")

        p = params.loc[
            params["strategy"] == strategy
        ].sort_values("refit_date")

        if not np.array_equal(
            f["Date"].to_numpy(),
            expected_forecast_dates,
        ):
            raise ValueError(f"Forecast date mismatch: {strategy}")

        if not np.array_equal(
            p["refit_date"].to_numpy(),
            expected_refits,
        ):
            raise ValueError(f"Parameter refit schedule mismatch: {strategy}")

        if not np.isfinite(
            f[["mu_hat", "sigma_hat"]].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Nonfinite source predictions: {strategy}")

        if (f["sigma_hat"] <= 0).any():
            raise ValueError(f"Nonpositive sigma: {strategy}")

        if not np.array_equal(
            f["information_end_date"].to_numpy(),
            dates[WINDOW - 1:-1],
        ):
            raise ValueError(f"Information cutoff mismatch: {strategy}")

    return returns, normal, params


def initial_filtered_residuals(history_pct, first_params, first_sigma):
    params = np.array(
        [
            first_params["mu_pct"],
            first_params["omega_pct2"],
            first_params["alpha"],
            first_params["beta"],
        ],
        dtype=float,
    )

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

    fixed = model.fix(params)

    vol_pct = np.asarray(
        fixed.conditional_volatility,
        dtype=float,
    )

    if not np.isfinite(vol_pct).all() or np.any(vol_pct <= 0):
        raise ValueError("Invalid in-sample conditional volatility")

    expected_variance = (
        params[1]
        + params[2] * (history_pct[-1] - params[0]) ** 2
        + params[3] * vol_pct[-1] ** 2
    )

    if not np.isclose(
        np.sqrt(expected_variance) / 100.0,
        first_sigma,
        rtol=1e-6,
        atol=1e-10,
    ):
        raise ValueError(
            "Initial filtered volatility does not match existing "
            "GARCH-Normal one-step forecast"
        )

    return (history_pct - params[0]) / vol_pct


def empirical_tail(sorted_z, confidence):
    n = len(sorted_z)

    tail_mass = n * (1.0 - confidence)
    n_full = int(np.floor(tail_mass))
    fraction = tail_mass - n_full
    index = max(int(np.ceil(tail_mass)) - 1, 0)

    quantile = float(sorted_z[index])

    tail_sum = float(sorted_z[:n_full].sum())

    if fraction > 1e-12:
        tail_sum += fraction * float(sorted_z[n_full])

    tail_mean = tail_sum / tail_mass

    if tail_mean > quantile + 1e-12:
        raise ValueError("Empirical ES tail mean exceeds tail quantile")

    return quantile, float(tail_mean)


def main():
    returns, normal, params = read_inputs()

    dates = returns["Date"].to_numpy()
    records = []

    for strategy, strategy_name in STRATEGIES.items():
        print(f"Building FHS: {strategy}", flush=True)

        x = returns[strategy].to_numpy(dtype=float)

        source = normal.loc[
            normal["strategy"] == strategy
        ].sort_values("Date").reset_index(drop=True)

        p = params.loc[
            params["strategy"] == strategy
        ].sort_values("refit_date")

        z_initial = initial_filtered_residuals(
            history_pct=100.0 * x[:WINDOW],
            first_params=p.iloc[0],
            first_sigma=float(source.loc[0, "sigma_hat"]),
        )

        if (
            len(z_initial) != WINDOW
            or not np.isfinite(z_initial).all()
        ):
            raise ValueError(
                f"Invalid initial filtered residuals: {strategy}"
            )

        pool = deque(
            z_initial.tolist(),
            maxlen=WINDOW,
        )

        pool_dates = deque(
            dates[:WINDOW],
            maxlen=WINDOW,
        )

        for j, i in enumerate(range(WINDOW, len(returns))):
            row = source.iloc[j]

            mu = float(row["mu_hat"])
            sigma = float(row["sigma_hat"])

            if (
                len(pool) != WINDOW
                or pool_dates[0] != dates[i - WINDOW]
                or pool_dates[-1] != dates[i - 1]
            ):
                raise ValueError(
                    f"Residual history misalignment: {strategy}, {dates[i]}"
                )

            ordered_z = np.sort(
                np.asarray(pool, dtype=float)
            )

            for confidence in LEVELS:
                z_q, z_tail_mean = empirical_tail(
                    ordered_z,
                    confidence,
                )

                var_value = -mu - sigma * z_q
                es_value = -mu - sigma * z_tail_mean

                if not np.isfinite(
                    [var_value, es_value]
                ).all():
                    raise ValueError("Invalid FHS VaR or ES")

                if es_value + 1e-12 < var_value:
                    raise ValueError("ES below VaR")

                records.append({
                    "Date": dates[i],
                    "strategy": strategy,
                    "strategy_name": strategy_name,
                    "model": "garch_fhs",
                    "confidence_level": confidence,
                    "window_days": WINDOW,
                    "refit_every_days": REFIT_EVERY,
                    "estimation_start_date": row["estimation_start_date"],
                    "estimation_end_date": row["estimation_end_date"],
                    "information_end_date": dates[i - 1],
                    "residual_window_days": WINDOW,
                    "residual_start_date": pool_dates[0],
                    "residual_end_date": pool_dates[-1],
                    "initial_in_sample_residuals": max(
                        0,
                        2 * WINDOW - i,
                    ),
                    "mu_hat": mu,
                    "sigma_hat": sigma,
                    "z_quantile": z_q,
                    "z_tail_mean": z_tail_mean,
                    "var_forecast": var_value,
                    "es_forecast": es_value,
                })

            # Today's observation enters the pool only after forecasting.
            realized_z = (x[i] - mu) / sigma

            if not np.isfinite(realized_z):
                raise ValueError(
                    "Invalid realized standardized innovation"
                )

            pool.append(float(realized_z))
            pool_dates.append(dates[i])

    forecasts = pd.DataFrame(records).sort_values(
        ["Date", "strategy", "confidence_level"]
    ).reset_index(drop=True)

    expected_rows = (
        3839 * len(STRATEGIES) * len(LEVELS)
    )

    if len(forecasts) != expected_rows:
        raise ValueError("Incorrect FHS forecast count")

    if forecasts.duplicated(
        ["Date", "strategy", "confidence_level"]
    ).any():
        raise ValueError("Duplicate FHS forecasts")

    if (
        forecasts["residual_end_date"] >= forecasts["Date"]
    ).any():
        raise ValueError("Look-ahead detected in residual pool")

    if (
        forecasts["information_end_date"] >= forecasts["Date"]
    ).any():
        raise ValueError("Look-ahead detected in source volatility")

    if forecasts.isna().any().any():
        raise ValueError("Missing FHS output fields")

    wide = forecasts.pivot(
        index=["Date", "strategy"],
        columns="confidence_level",
        values=["var_forecast", "es_forecast"],
    )

    if not (
        wide[("var_forecast", 0.99)]
        >= wide[("var_forecast", 0.95)]
    ).all():
        raise ValueError("VaR confidence-level ordering failed")

    if not (
        wide[("es_forecast", 0.99)]
        >= wide[("es_forecast", 0.95)]
    ).all():
        raise ValueError("ES confidence-level ordering failed")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecasts.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nGARCH-FHS FORECASTS")
    print(f"Forecast rows: {len(forecasts):,}")
    print(f"Days: {forecasts['Date'].nunique():,}")

    print(
        f"Dates: {forecasts['Date'].min().date()} "
        f"to {forecasts['Date'].max().date()}"
    )

    print("\nMEDIAN DAILY VaR AND ES (%)")

    display = (
        forecasts.groupby(
            ["strategy", "confidence_level"]
        )[["var_forecast", "es_forecast"]]
        .median() * 100
    )

    print(display.round(4).to_string())
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
