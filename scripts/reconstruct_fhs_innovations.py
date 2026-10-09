
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"

FHS_PATH = DATA_DIR / "garch_fhs_var_es_forecasts.csv"
BACKTEST_PATH = DATA_DIR / "var_backtest_daily.csv"

INNOVATIONS_PATH = DATA_DIR / "garch_fhs_oos_innovations.csv"
AUDIT_PATH = DATA_DIR / "garch_fhs_density_window_audit.csv"

WINDOW = 1008
EXPECTED_DAYS = 3839
LEVELS = (0.95, 0.99)
TOL = 1e-9

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]


def check_close(actual, expected, label):
    if not np.isfinite([actual, expected]).all():
        raise ValueError(f"Non-finite value: {label}")

    error = abs(float(actual) - float(expected))

    if error > TOL:
        raise ValueError(
            f"{label}: actual={actual:.12f}, "
            f"expected={expected:.12f}, "
            f"error={error:.3e}"
        )

    return error


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

    return quantile, float(tail_mean)


def read_inputs():
    fhs = pd.read_csv(
        FHS_PATH,
        parse_dates=[
            "Date",
            "information_end_date",
            "residual_start_date",
            "residual_end_date",
        ],
    )

    backtest = pd.read_csv(
        BACKTEST_PATH,
        parse_dates=["Date"],
    )

    fhs = fhs.loc[
        fhs["model"] == "garch_fhs"
    ].copy()

    backtest = backtest.loc[
        (backtest["model"] == "garch_fhs")
        & (backtest["confidence_level"] == 0.95)
    ].copy()

    keys = ["Date", "strategy", "confidence_level"]

    if fhs.duplicated(keys).any():
        raise ValueError("Duplicate FHS forecast keys.")

    if backtest.duplicated(
        ["Date", "strategy"]
    ).any():
        raise ValueError("Duplicate backtest keys.")

    if len(fhs) != EXPECTED_DAYS * 5 * 2:
        raise ValueError("Unexpected FHS forecast count.")

    if len(backtest) != EXPECTED_DAYS * 5:
        raise ValueError("Unexpected backtest count.")

    if set(fhs["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected FHS strategies.")

    if set(backtest["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected backtest strategies.")

    if set(fhs["confidence_level"]) != set(LEVELS):
        raise ValueError("Unexpected confidence levels.")

    if not fhs["residual_window_days"].eq(WINDOW).all():
        raise ValueError("Residual window is not 1008 days.")

    if (
        fhs["information_end_date"] >= fhs["Date"]
    ).any():
        raise ValueError("Look-ahead in information dates.")

    if (
        fhs["residual_end_date"] >= fhs["Date"]
    ).any():
        raise ValueError("Look-ahead in residual dates.")

    if (fhs["sigma_hat"] <= 0).any():
        raise ValueError("Nonpositive conditional volatility.")

    required = [
        "mu_hat",
        "sigma_hat",
        "z_quantile",
        "z_tail_mean",
        "var_forecast",
        "es_forecast",
    ]

    if not np.isfinite(
        fhs[required].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid FHS numerical values.")

    if not np.isfinite(
        backtest["realized_return"].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid realized returns.")

    return fhs, backtest


def process_strategy(strategy, fhs, backtest):
    subset = fhs.loc[
        fhs["strategy"] == strategy
    ]

    low = subset.loc[
        subset["confidence_level"] == 0.95
    ].sort_values("Date").reset_index(drop=True)

    high = subset.loc[
        subset["confidence_level"] == 0.99
    ].sort_values("Date").reset_index(drop=True)

    actual = backtest.loc[
        backtest["strategy"] == strategy,
        ["Date", "realized_return"],
    ].sort_values("Date").reset_index(drop=True)

    if not (
        len(low) == len(high) == len(actual) == EXPECTED_DAYS
    ):
        raise ValueError(f"Incomplete data: {strategy}")

    if not low["Date"].equals(high["Date"]):
        raise ValueError(f"95/99 dates differ: {strategy}")

    if not low["Date"].equals(actual["Date"]):
        raise ValueError(f"Realized return dates differ: {strategy}")

    for column in ["mu_hat", "sigma_hat"]:
        if not np.allclose(
            low[column],
            high[column],
            rtol=0,
            atol=TOL,
        ):
            raise ValueError(
                f"{column} differs across levels: {strategy}"
            )

    for column in [
        "information_end_date",
        "residual_start_date",
        "residual_end_date",
    ]:
        if not low[column].equals(high[column]):
            raise ValueError(
                f"{column} differs across levels: {strategy}"
            )

    if not (
        low["initial_in_sample_residuals"]
        .equals(high["initial_in_sample_residuals"])
    ):
        raise ValueError(
            f"Initial-pool counts differ: {strategy}"
        )

    innovations = low[
        ["Date", "information_end_date", "mu_hat", "sigma_hat"]
    ].copy()

    innovations["strategy"] = strategy

    innovations["realized_return"] = (
        actual["realized_return"].to_numpy()
    )

    innovations["z"] = (
        innovations["realized_return"]
        - innovations["mu_hat"]
    ) / innovations["sigma_hat"]

    if not np.isfinite(
        innovations["z"].to_numpy()
    ).all():
        raise ValueError(f"Invalid innovations: {strategy}")

    dates = innovations["Date"].to_numpy()
    z = innovations["z"].to_numpy(dtype=float)

    for j in range(1, len(innovations)):
        if low.loc[j, "information_end_date"] != low.loc[j - 1, "Date"]:
            raise ValueError(
                f"Information date mismatch: {strategy}, index={j}"
            )

    counts = low[
        "initial_in_sample_residuals"
    ].to_numpy(dtype=int)

    pure_indices = np.flatnonzero(counts == 0)

    if len(pure_indices) == 0:
        raise ValueError(f"No pure OOS window: {strategy}")

    first_pure = int(pure_indices[0])

    if first_pure != WINDOW:
        raise ValueError(
            f"Unexpected first pure OOS index: "
            f"{strategy}, {first_pure}"
        )

    if not np.array_equal(
        counts,
        np.maximum(0, WINDOW - np.arange(EXPECTED_DAYS)),
    ):
        raise ValueError(
            f"Unexpected initial-pool replacement schedule: {strategy}"
        )

    audit_rows = []

    for j in range(first_pure, EXPECTED_DAYS):
        forecast_date = dates[j]

        training_z = z[j - WINDOW:j]
        training_dates = dates[j - WINDOW:j]

        if len(training_z) != WINDOW:
            raise ValueError("Incorrect training window length.")

        sorted_z = np.sort(training_z)

        window_error = 0.0

        for level, source in [
            (0.95, low),
            (0.99, high),
        ]:
            row = source.iloc[j]

            if row["residual_start_date"] != pd.Timestamp(
                training_dates[0]
            ):
                raise ValueError(
                    f"Residual start mismatch: "
                    f"{strategy}, {forecast_date}"
                )

            if row["residual_end_date"] != pd.Timestamp(
                training_dates[-1]
            ):
                raise ValueError(
                    f"Residual end mismatch: "
                    f"{strategy}, {forecast_date}"
                )

            if row["initial_in_sample_residuals"] != 0:
                raise ValueError(
                    f"Unexpected in-sample residuals: "
                    f"{strategy}, {forecast_date}"
                )

            q, tail_mean = empirical_tail(
                sorted_z,
                level,
            )

            mu = float(row["mu_hat"])
            sigma = float(row["sigma_hat"])

            reconstructed_var = -mu - sigma * q
            reconstructed_es = -mu - sigma * tail_mean

            label = (
                f"{strategy}, "
                f"{pd.Timestamp(forecast_date).date()}, "
                f"{level:.0%}"
            )

            errors = [
                check_close(
                    q,
                    row["z_quantile"],
                    f"Quantile: {label}",
                ),
                check_close(
                    tail_mean,
                    row["z_tail_mean"],
                    f"Tail mean: {label}",
                ),
                check_close(
                    reconstructed_var,
                    row["var_forecast"],
                    f"VaR: {label}",
                ),
                check_close(
                    reconstructed_es,
                    row["es_forecast"],
                    f"ES: {label}",
                ),
            ]

            window_error = max(
                window_error,
                *errors,
            )

        audit_rows.append({
            "Date": pd.Timestamp(forecast_date),
            "strategy": strategy,
            "training_start_date": pd.Timestamp(
                training_dates[0]
            ),
            "training_end_date": pd.Timestamp(
                training_dates[-1]
            ),
            "training_observations": WINDOW,
            "max_absolute_reconciliation_error": window_error,
        })

    print(
        f"{strategy:16s} | "
        f"OOS innovations: {len(z):4d} | "
        f"Pure OOS windows: {len(audit_rows):4d} | "
        f"First: {low.loc[first_pure, 'Date'].date()}"
    )

    return innovations, pd.DataFrame(audit_rows)


def main():
    fhs, backtest = read_inputs()

    innovation_tables = []
    audit_tables = []

    print("\nGARCH-FHS INNOVATION RECONSTRUCTION")
    print("=" * 90)

    for strategy in STRATEGIES:
        innovations, audit = process_strategy(
            strategy,
            fhs,
            backtest,
        )

        innovation_tables.append(innovations)
        audit_tables.append(audit)

    all_innovations = pd.concat(
        innovation_tables,
        ignore_index=True,
    ).sort_values(["Date", "strategy"])

    all_audits = pd.concat(
        audit_tables,
        ignore_index=True,
    ).sort_values(["Date", "strategy"])

    if all_innovations.duplicated(
        ["Date", "strategy"]
    ).any():
        raise ValueError("Duplicate innovations.")

    if all_audits.duplicated(
        ["Date", "strategy"]
    ).any():
        raise ValueError("Duplicate training windows.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_innovations.to_csv(
        INNOVATIONS_PATH,
        index=False,
    )

    all_audits.to_csv(
        AUDIT_PATH,
        index=False,
    )

    print("\nRECONCILIATION SUMMARY")
    print(
        f"Total innovations: {len(all_innovations):,}"
    )
    print(
        f"Pure OOS training windows: {len(all_audits):,}"
    )
    print(
        f"VaR/ES forecast comparisons: {2 * len(all_audits):,}"
    )
    print(
        "Maximum absolute numerical error: "
        f"{all_audits['max_absolute_reconciliation_error'].max():.3e}"
    )

    print("\nSAVED")
    print(INNOVATIONS_PATH)
    print(AUDIT_PATH)

    print("\nAll reconciliation checks passed.")


if __name__ == "__main__":
    main()
