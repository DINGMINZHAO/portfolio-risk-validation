
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import brentq
from scipy.special import ndtr, logsumexp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/density_estimation"

INNOVATIONS_PATH = DATA_DIR / "garch_fhs_oos_innovations.csv"
AUDIT_PATH = DATA_DIR / "garch_fhs_density_window_audit.csv"
FHS_PATH = DATA_DIR / "garch_fhs_var_es_forecasts.csv"

OUTPUT_PATH = DATA_DIR / "garch_kde_minvar_static_daily.csv"
FIGURE_PATH = FIGURE_DIR / "garch_kde_density_comparison.png"

STRATEGY = "minvar_static"
WINDOW = 1008
LEVELS = (0.95, 0.99)

SQRT_2PI = np.sqrt(2.0 * np.pi)


def gaussian_pdf(x):
    return np.exp(-0.5 * x * x) / SQRT_2PI


def silverman_bandwidth(z):
    n = len(z)

    std = np.std(z, ddof=1)
    q25, q75 = np.quantile(z, [0.25, 0.75])
    robust_scale = (q75 - q25) / 1.34

    scale = min(std, robust_scale)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid bandwidth scale.")

    h = 0.9 * scale * n ** (-0.2)

    if h <= 0 or not np.isfinite(h):
        raise ValueError("Invalid KDE bandwidth.")

    return float(h)


def kde_cdf(x, z, h):
    return float(
        np.mean(ndtr((x - z) / h))
    )


def kde_log_density(x, z, h):
    standardized = (x - z) / h

    return float(
        logsumexp(-0.5 * standardized ** 2)
        - np.log(len(z) * h * SQRT_2PI)
    )


def kde_left_tail(z, h, alpha):
    left = float(np.min(z) - 12 * h)
    right = float(np.max(z) + 12 * h)

    quantile = brentq(
        lambda x: kde_cdf(x, z, h) - alpha,
        left,
        right,
        xtol=1e-10,
    )

    u = (quantile - z) / h

    partial_expectation = (
        z * ndtr(u) - h * gaussian_pdf(u)
    )

    tail_mean = float(
        np.mean(partial_expectation) / alpha
    )

    if tail_mean > quantile + 1e-9:
        raise ValueError("KDE ES exceeds its quantile.")

    return float(quantile), tail_mean


def read_data():
    innovations = pd.read_csv(
        INNOVATIONS_PATH,
        parse_dates=["Date", "information_end_date"],
    )

    audit = pd.read_csv(
        AUDIT_PATH,
        parse_dates=[
            "Date",
            "training_start_date",
            "training_end_date",
        ],
    )

    fhs = pd.read_csv(
        FHS_PATH,
        parse_dates=["Date"],
    )

    innovations = (
        innovations.loc[
            innovations["strategy"] == STRATEGY
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    audit = (
        audit.loc[
            audit["strategy"] == STRATEGY
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    fhs = fhs.loc[
        (fhs["strategy"] == STRATEGY)
        & (fhs["model"] == "garch_fhs")
    ].copy()

    if len(innovations) != 3839:
        raise ValueError("Unexpected innovation count.")

    if len(audit) != 2831:
        raise ValueError("Unexpected audit count.")

    if len(fhs) != 3839 * 2:
        raise ValueError("Unexpected FHS forecast count.")

    if innovations["Date"].duplicated().any():
        raise ValueError("Duplicate innovation dates.")

    if audit["Date"].duplicated().any():
        raise ValueError("Duplicate audit dates.")

    if fhs.duplicated(
        ["Date", "confidence_level"]
    ).any():
        raise ValueError("Duplicate FHS forecast keys.")

    if not audit["training_observations"].eq(WINDOW).all():
        raise ValueError("Unexpected training window size.")

    if (
        audit["max_absolute_reconciliation_error"]
        > 1e-9
    ).any():
        raise ValueError("Previous FHS audit did not pass.")

    fhs_by_level = {}

    for level in LEVELS:
        part = (
            fhs.loc[
                fhs["confidence_level"] == level
            ]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if len(part) != len(innovations):
            raise ValueError("Incomplete FHS forecasts.")

        if not part["Date"].equals(innovations["Date"]):
            raise ValueError("FHS dates do not match.")

        if not np.allclose(
            part["mu_hat"],
            innovations["mu_hat"],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Conditional means differ.")

        if not np.allclose(
            part["sigma_hat"],
            innovations["sigma_hat"],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Conditional volatilities differ.")

        fhs_by_level[level] = part

    expected_dates = innovations["Date"].iloc[
        WINDOW:
    ].reset_index(drop=True)

    if not audit["Date"].equals(expected_dates):
        raise ValueError("Audit window dates do not match.")

    return innovations, audit, fhs_by_level


def run_backtest(innovations, audit, fhs_by_level):
    z = innovations["z"].to_numpy(dtype=float)
    dates = innovations["Date"].to_numpy()

    records = []

    for j in range(WINDOW, len(innovations)):
        training = z[j - WINDOW:j]
        actual_z = float(z[j])

        row = innovations.iloc[j]
        audit_row = audit.iloc[j - WINDOW]

        date = pd.Timestamp(dates[j])

        if audit_row["Date"] != date:
            raise ValueError("Forecast date mismatch.")

        if audit_row["training_start_date"] != pd.Timestamp(
            dates[j - WINDOW]
        ):
            raise ValueError("Training start mismatch.")

        if audit_row["training_end_date"] != pd.Timestamp(
            dates[j - 1]
        ):
            raise ValueError("Training end mismatch.")

        if row["information_end_date"] != pd.Timestamp(
            dates[j - 1]
        ):
            raise ValueError("Information cutoff mismatch.")

        h = silverman_bandwidth(training)

        mu = float(row["mu_hat"])
        sigma = float(row["sigma_hat"])
        actual_return = float(row["realized_return"])
        actual_loss = -actual_return

        kde_pit = kde_cdf(actual_z, training, h)

        fhs_pit = float(
            np.mean(training <= actual_z)
        )

        record = {
            "Date": date,
            "strategy": STRATEGY,
            "realized_return": actual_return,
            "realized_z": actual_z,
            "mu_hat": mu,
            "sigma_hat": sigma,
            "kde_bandwidth": h,
            "kde_pit": kde_pit,
            "fhs_pit": fhs_pit,
            "kde_log_score": -kde_log_density(
                actual_z, training, h
            ),
        }

        for level in LEVELS:
            alpha = 1.0 - level
            suffix = int(round(level * 100))

            quantile, tail_mean = kde_left_tail(
                training, h, alpha
            )

            kde_var = -mu - sigma * quantile
            kde_es = -mu - sigma * tail_mean

            source = fhs_by_level[level].iloc[j]

            fhs_var = float(source["var_forecast"])
            fhs_es = float(source["es_forecast"])

            record[f"kde_var_{suffix}"] = kde_var
            record[f"kde_es_{suffix}"] = kde_es
            record[f"fhs_var_{suffix}"] = fhs_var
            record[f"fhs_es_{suffix}"] = fhs_es

            record[f"kde_hit_{suffix}"] = int(
                actual_loss > kde_var
            )
            record[f"fhs_hit_{suffix}"] = int(
                actual_loss > fhs_var
            )

        records.append(record)

    return pd.DataFrame(records)


def plot_last_window(innovations, results):
    last_date = results["Date"].iloc[-1]

    dates = innovations["Date"]
    position = int(
        np.flatnonzero(
            dates.to_numpy() == np.datetime64(last_date)
        )[0]
    )

    training = innovations["z"].to_numpy()[
        position - WINDOW:position
    ]

    h = float(results["kde_bandwidth"].iloc[-1])

    grid = np.linspace(
        np.min(training) - 3 * h,
        np.max(training) + 3 * h,
        500,
    )

    u = (
        grid[:, None] - training[None, :]
    ) / h

    density = gaussian_pdf(u).mean(axis=1) / h
    kde_cumulative = ndtr(u).mean(axis=1)

    sorted_training = np.sort(training)

    empirical_cumulative = (
        np.searchsorted(
            sorted_training,
            grid,
            side="right",
        ) / WINDOW
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=True,
    )

    axes[0].hist(
        training,
        bins=50,
        density=True,
        alpha=0.35,
        color="#7c8fa1",
        label="Historical innovations",
    )

    axes[0].plot(
        grid,
        density,
        color="#247ba0",
        linewidth=2,
        label="Gaussian KDE",
    )

    axes[0].set_ylabel("Probability density")
    axes[0].set_title(
        "Innovation Density Estimation",
        loc="left",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        grid,
        empirical_cumulative,
        color="#7c8fa1",
        linewidth=1.5,
        label="Empirical FHS CDF",
    )

    axes[1].plot(
        grid,
        kde_cumulative,
        color="#247ba0",
        linewidth=2,
        label="Gaussian KDE CDF",
    )

    axes[1].set_xlabel("Standardized innovation")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"GARCH Innovation Distribution: {STRATEGY}\n"
        f"Information available before {last_date.date()}",
        fontsize=14,
    )

    fig.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_summary(results):
    print("\nGARCH-KDE DENSITY BACKTEST")
    print("=" * 65)
    print(f"Portfolio: {STRATEGY}")
    print(f"Forecast days: {len(results)}")
    print(
        f"Period: {results['Date'].min().date()} "
        f"to {results['Date'].max().date()}"
    )

    print("\nVaR EXCEEDANCE COMPARISON")

    for level in LEVELS:
        suffix = int(round(level * 100))

        kde_hits = int(
            results[f"kde_hit_{suffix}"].sum()
        )

        fhs_hits = int(
            results[f"fhs_hit_{suffix}"].sum()
        )

        n = len(results)

        print(
            f"{suffix}% VaR | "
            f"KDE: {kde_hits:3d} ({100*kde_hits/n:.2f}%) | "
            f"FHS: {fhs_hits:3d} ({100*fhs_hits/n:.2f}%)"
        )

    print("\nPIT DIAGNOSTICS")

    for model in ["kde", "fhs"]:
        pit = results[f"{model}_pit"]

        print(
            f"{model.upper():4s} | "
            f"Mean: {pit.mean():.4f} | "
            f"Std: {pit.std(ddof=0):.4f} | "
            f"Below 0.01: {(pit < 0.01).mean():.4f}"
        )

    print("\nKDE DENSITY DIAGNOSTICS")
    print(
        "Mean negative log density: "
        f"{results['kde_log_score'].mean():.4f}"
    )
    print(
        "Median bandwidth: "
        f"{results['kde_bandwidth'].median():.4f}"
    )

    print("\nNote: KDE log score is not directly "
          "comparable with discrete FHS.")

    print("\nSAVED OUTPUTS")
    print(OUTPUT_PATH)
    print(FIGURE_PATH)


def main():
    innovations, audit, fhs = read_data()

    results = run_backtest(
        innovations,
        audit,
        fhs,
    )

    if len(results) != 2831:
        raise ValueError("Unexpected backtest length.")

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    plot_last_window(innovations, results)
    print_summary(results)

    print("\nKDE backtest completed.")


if __name__ == "__main__":
    main()
