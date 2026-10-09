
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/density_estimation"

INNOVATIONS_PATH = DATA_DIR / "garch_fhs_oos_innovations.csv"
AUDIT_PATH = DATA_DIR / "garch_fhs_density_window_audit.csv"
KDE_PATH = DATA_DIR / "garch_kde_minvar_static_daily.csv"

OUTPUT_PATH = DATA_DIR / "garch_fhs_kde_crps_comparison.csv"
FIGURE_PATH = FIGURE_DIR / "garch_fhs_kde_crps_comparison.png"

STRATEGY = "minvar_static"
WINDOW = 1008
EXPECTED_DAYS = 2831

ROLLING_DAYS = 252

QUADRATURE_NODES = 64
CHECK_NODES = 96

SQRT_2PI = np.sqrt(2.0 * np.pi)

GH_X, GH_W = hermgauss(QUADRATURE_NODES)
GH_W = GH_W / np.sqrt(np.pi)

CHECK_X, CHECK_W = hermgauss(CHECK_NODES)
CHECK_W = CHECK_W / np.sqrt(np.pi)


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

    kde = pd.read_csv(
        KDE_PATH,
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

    kde = (
        kde.loc[
            kde["strategy"] == STRATEGY
        ]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(innovations) != 3839:
        raise ValueError("Unexpected innovation count.")

    if len(audit) != EXPECTED_DAYS:
        raise ValueError("Unexpected audit count.")

    if len(kde) != EXPECTED_DAYS:
        raise ValueError("Unexpected KDE count.")

    for name, df in [
        ("innovations", innovations),
        ("audit", audit),
        ("kde", kde),
    ]:
        if df["Date"].isna().any():
            raise ValueError(f"Missing dates: {name}")

        if df["Date"].duplicated().any():
            raise ValueError(f"Duplicate dates: {name}")

    expected_dates = (
        innovations["Date"]
        .iloc[WINDOW:]
        .reset_index(drop=True)
    )

    if not audit["Date"].equals(expected_dates):
        raise ValueError("Audit dates do not match.")

    if not kde["Date"].equals(expected_dates):
        raise ValueError("KDE dates do not match.")

    if not audit["training_observations"].eq(WINDOW).all():
        raise ValueError("Incorrect training window size.")

    if (
        audit["max_absolute_reconciliation_error"]
        > 1e-9
    ).any():
        raise ValueError("Previous FHS audit failed.")

    if not np.allclose(
        kde["realized_z"],
        innovations["z"].iloc[WINDOW:],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Realized innovations differ.")

    for column in ["mu_hat", "sigma_hat"]:
        if not np.allclose(
            kde[column],
            innovations[column].iloc[WINDOW:],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(
                f"GARCH forecasts differ: {column}"
            )

    if (
        not np.isfinite(
            innovations[
                ["z", "sigma_hat", "mu_hat", "realized_return"]
            ].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("Invalid innovation data.")

    if (innovations["sigma_hat"] <= 0).any():
        raise ValueError("Nonpositive volatility.")

    if (
        not np.isfinite(
            kde["kde_bandwidth"].to_numpy(dtype=float)
        ).all()
        or (kde["kde_bandwidth"] <= 0).any()
    ):
        raise ValueError("Invalid KDE bandwidth.")

    return innovations, audit, kde


def silverman_bandwidth(z):
    n = len(z)

    std = np.std(z, ddof=1)

    q25, q75 = np.quantile(
        z,
        [0.25, 0.75],
    )

    robust_scale = (q75 - q25) / 1.34

    scale = min(std, robust_scale)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid bandwidth scale.")

    return float(
        0.9 * scale * n ** (-0.2)
    )


def crps_empirical(sorted_z, observed):
    n = len(sorted_z)

    first_term = np.mean(
        np.abs(sorted_z - observed)
    )

    indices = np.arange(1, n + 1)

    coefficients = 2 * indices - n - 1

    second_term = (
        np.dot(coefficients, sorted_z)
        / (n * n)
    )

    result = float(first_term - second_term)

    if not np.isfinite(result) or result < -1e-10:
        raise ValueError("Invalid empirical CRPS.")

    return max(0.0, result)


def normal_pdf(x):
    return np.exp(-0.5 * x * x) / SQRT_2PI


def kde_crps(sorted_z, observed, bandwidth, nodes, weights):
    n = len(sorted_z)

    h = bandwidth

    difference = observed - sorted_z
    u = difference / h

    first_term = np.mean(
        difference * (2 * ndtr(u) - 1)
        + 2 * h * normal_pdf(u)
    )

    prefix = np.concatenate(
        [
            np.array([0.0]),
            np.cumsum(sorted_z),
        ]
    )

    total = prefix[-1]

    shifts = 2 * h * nodes

    shifted_values = (
        sorted_z[None, :]
        + shifts[:, None]
    )

    positions = np.searchsorted(
        sorted_z,
        shifted_values,
        side="right",
    )

    absolute_sums = (
        (2 * positions - n) * shifted_values
        + total
        - 2 * prefix[positions]
    )

    mean_pair_distances = (
        absolute_sums.sum(axis=1)
        / (n * n)
    )

    second_term = 0.5 * np.dot(
        weights,
        mean_pair_distances,
    )

    result = float(first_term - second_term)

    if not np.isfinite(result) or result < -1e-10:
        raise ValueError("Invalid KDE CRPS.")

    return max(0.0, result)


def run_comparison(innovations, audit, kde):
    z = innovations["z"].to_numpy(dtype=float)
    dates = innovations["Date"].to_numpy()

    records = []
    quadrature_differences = []

    for j in range(WINDOW, len(innovations)):
        index = j - WINDOW

        row = innovations.iloc[j]
        audit_row = audit.iloc[index]
        kde_row = kde.iloc[index]

        forecast_date = pd.Timestamp(dates[j])

        if audit_row["Date"] != forecast_date:
            raise ValueError("Audit alignment failed.")

        if audit_row["training_start_date"] != pd.Timestamp(
            dates[j - WINDOW]
        ):
            raise ValueError("Training start date mismatch.")

        if audit_row["training_end_date"] != pd.Timestamp(
            dates[j - 1]
        ):
            raise ValueError("Training end date mismatch.")

        if row["information_end_date"] != pd.Timestamp(
            dates[j - 1]
        ):
            raise ValueError("Information cutoff mismatch.")

        training = np.sort(
            z[j - WINDOW:j]
        )

        observed_z = float(z[j])

        sigma = float(row["sigma_hat"])
        bandwidth = float(kde_row["kde_bandwidth"])

        reconstructed_bandwidth = silverman_bandwidth(
            training
        )

        if not np.isclose(
            bandwidth,
            reconstructed_bandwidth,
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(
                f"KDE bandwidth mismatch: {forecast_date}"
            )

        fhs_crps_z = crps_empirical(
            training,
            observed_z,
        )

        kde_crps_z = kde_crps(
            training,
            observed_z,
            bandwidth,
            GH_X,
            GH_W,
        )

        if (
            index % 250 == 0
            or index == EXPECTED_DAYS - 1
        ):
            check_crps = kde_crps(
                training,
                observed_z,
                bandwidth,
                CHECK_X,
                CHECK_W,
            )

            quadrature_differences.append(
                abs(check_crps - kde_crps_z)
            )

        fhs_crps_return = sigma * fhs_crps_z
        kde_crps_return = sigma * kde_crps_z

        delta_return = (
            kde_crps_return - fhs_crps_return
        )

        records.append({
            "Date": forecast_date,
            "strategy": STRATEGY,
            "realized_return": float(
                row["realized_return"]
            ),
            "realized_z": observed_z,
            "sigma_hat": sigma,
            "kde_bandwidth": bandwidth,
            "fhs_crps_z": fhs_crps_z,
            "kde_crps_z": kde_crps_z,
            "delta_crps_z": (
                kde_crps_z - fhs_crps_z
            ),
            "fhs_crps_return": fhs_crps_return,
            "kde_crps_return": kde_crps_return,
            "delta_crps_return": delta_return,
            "fhs_crps_bp": (
                10000 * fhs_crps_return
            ),
            "kde_crps_bp": (
                10000 * kde_crps_return
            ),
            "delta_crps_bp": (
                10000 * delta_return
            ),
        })

    result = pd.DataFrame(records)

    if len(result) != EXPECTED_DAYS:
        raise ValueError("Unexpected result count.")

    if not np.isfinite(
        result.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all():
        raise ValueError("Non-finite CRPS results.")

    max_difference = max(quadrature_differences)

    if max_difference > 1e-4:
        raise ValueError(
            "Quadrature discrepancy is too large. "
            f"Maximum: {max_difference:.6e}"
        )

    return result, max_difference


def print_summary(result, quadrature_error):
    fhs_mean = result["fhs_crps_return"].mean()
    kde_mean = result["kde_crps_return"].mean()

    mean_delta = result["delta_crps_return"].mean()

    relative_improvement = (
        100 * (fhs_mean - kde_mean) / fhs_mean
    )

    kde_wins = (
        result["delta_crps_return"] < 0
    ).mean()

    print("\nPAIRED CRPS COMPARISON")
    print("=" * 75)

    print(f"Strategy: {STRATEGY}")
    print(f"Forecast days: {len(result)}")
    print(
        f"Period: {result['Date'].min().date()} "
        f"to {result['Date'].max().date()}"
    )

    print("\nRETURN-SCALE CRPS (BASIS POINTS)")
    print(f"FHS mean CRPS:    {10000*fhs_mean:.6f}")
    print(f"KDE mean CRPS:    {10000*kde_mean:.6f}")
    print(f"KDE minus FHS:    {10000*mean_delta:+.6f}")

    print("\nCOMPARATIVE DIAGNOSTICS")
    print(
        f"KDE improvement relative to FHS: "
        f"{relative_improvement:+.4f}%"
    )
    print(
        f"Days KDE has lower CRPS: "
        f"{100*kde_wins:.2f}%"
    )

    print("\nSTANDARDIZED-INNOVATION CRPS")
    print(
        f"FHS: {result['fhs_crps_z'].mean():.6f}"
    )
    print(
        f"KDE: {result['kde_crps_z'].mean():.6f}"
    )

    print("\nNUMERICAL CHECK")
    print(
        "Maximum 64-vs-96-node quadrature difference: "
        f"{quadrature_error:.3e}"
    )

    annual = (
        result.assign(year=result["Date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            days=("Date", "size"),
            fhs_bp=("fhs_crps_bp", "mean"),
            kde_bp=("kde_crps_bp", "mean"),
            delta_bp=("delta_crps_bp", "mean"),
        )
    )

    print("\nANNUAL PAIRED CRPS (BASIS POINTS)")
    print(
        annual.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nNegative delta means KDE has lower CRPS."
    )


def plot_comparison(result):
    data = result.copy()

    data["rolling_delta_bp"] = (
        data["delta_crps_bp"]
        .rolling(
            ROLLING_DAYS,
            min_periods=ROLLING_DAYS,
        )
        .mean()
    )

    data["cumulative_delta_bp"] = (
        data["delta_crps_bp"].cumsum()
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
    )

    axes[0].plot(
        data["Date"],
        data["rolling_delta_bp"],
        color="#2563a6",
        linewidth=1.6,
    )

    axes[0].axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_title(
        "252-Day Rolling Mean CRPS Difference",
        loc="left",
        fontsize=12,
    )

    axes[0].set_ylabel(
        "KDE minus FHS (bps)"
    )

    axes[1].plot(
        data["Date"],
        data["cumulative_delta_bp"],
        color="#7c3aed",
        linewidth=1.6,
    )

    axes[1].axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    axes[1].set_title(
        "Cumulative Paired CRPS Difference",
        loc="left",
        fontsize=12,
    )

    axes[1].set_ylabel(
        "Cumulative difference (bps)"
    )

    axes[1].set_xlabel("Forecast date")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(
        "GARCH-FHS vs. GARCH-KDE: "
        "Probabilistic Forecast Comparison\n"
        f"Portfolio: {STRATEGY}",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

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


def main():
    innovations, audit, kde = read_data()

    print("Input validation passed.")

    result, quadrature_error = run_comparison(
        innovations,
        audit,
        kde,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    plot_comparison(result)

    print_summary(result, quadrature_error)

    print("\nSAVED OUTPUTS")
    print(OUTPUT_PATH)
    print(FIGURE_PATH)

    print("\nCRPS comparison completed.")


if __name__ == "__main__":
    main()
