
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm

from fit_gmm_snapshot import (
    BOUNDS,
    fit_gmm,
    gmm_nll,
    gmm_cdf,
    gmm_tail,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/density_estimation"

INNOVATIONS_PATH = DATA_DIR / "garch_fhs_oos_innovations.csv"
AUDIT_PATH = DATA_DIR / "garch_fhs_density_window_audit.csv"
KDE_PATH = DATA_DIR / "garch_kde_minvar_static_daily.csv"
CRPS_PATH = DATA_DIR / "garch_fhs_kde_crps_comparison.csv"

OUTPUT_PATH = DATA_DIR / "garch_gmm_minvar_static_daily.csv"
FIGURE_PATH = FIGURE_DIR / "garch_gmm_minvar_static_backtest.png"

STRATEGY = "minvar_static"
WINDOW = 1008
EXPECTED_DAYS = 2831
MULTISTART_EVERY = 21
ROLLING_DAYS = 252

LOWER = np.array([b[0] for b in BOUNDS])
UPPER = np.array([b[1] for b in BOUNDS])


def canonicalize(p):
    p = np.asarray(p, dtype=float)

    if p[2] > p[4]:
        w, m1, s1, m2, s2 = p
        p = np.array([1 - w, m2, s2, m1, s1])

    return p


def initial_parameters(z):
    center = float(np.median(z))
    spread = float(np.std(z, ddof=1))

    return np.array([
        0.90,
        center,
        0.85 * spread,
        center,
        1.90 * spread,
    ])


def fit_rolling_gmm(z, previous, index):
    if previous is None or index % MULTISTART_EVERY == 0:
        initial = (
            initial_parameters(z)
            if previous is None
            else previous
        )

        p, nll, converged = fit_gmm(
            z, initial
        )

        return (
            canonicalize(p),
            float(nll),
            "multistart",
            int(converged),
        )

    x0 = np.clip(
        previous,
        LOWER + 1e-5,
        UPPER - 1e-5,
    )

    result = minimize(
        gmm_nll,
        x0,
        args=(z,),
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={
            "maxiter": 600,
            "ftol": 1e-10,
            "maxls": 40,
        },
    )

    if result.success and np.isfinite(result.fun):
        return (
            canonicalize(result.x),
            float(result.fun),
            "warm_start",
            1,
        )

    p, nll, converged = fit_gmm(
        z, previous
    )

    return (
        canonicalize(p),
        float(nll),
        "fallback_multistart",
        int(converged),
    )


def normal_absolute_moment(a, s):
    u = a / s

    return (
        2 * s * norm.pdf(u)
        + a * (2 * norm.cdf(u) - 1)
    )


def gmm_crps(p, observed):
    w, m1, s1, m2, s2 = p

    weights = np.array([w, 1 - w])
    means = np.array([m1, m2])
    scales = np.array([s1, s2])

    first = np.sum(
        weights * normal_absolute_moment(
            observed - means,
            scales,
        )
    )

    second = 0.0

    for i in range(2):
        for j in range(2):
            combined_scale = np.sqrt(
                scales[i] ** 2
                + scales[j] ** 2
            )

            second += (
                weights[i]
                * weights[j]
                * normal_absolute_moment(
                    means[i] - means[j],
                    combined_scale,
                )
            )

    score = float(first - 0.5 * second)

    if not np.isfinite(score) or score < -1e-10:
        raise ValueError("Invalid GMM CRPS.")

    return max(0.0, score)


def gmm_moments(p):
    w, m1, s1, m2, s2 = p

    mean = w * m1 + (1 - w) * m2

    second = (
        w * (m1 ** 2 + s1 ** 2)
        + (1 - w) * (m2 ** 2 + s2 ** 2)
    )

    std = np.sqrt(second - mean ** 2)

    return float(mean), float(std)


def read_inputs():
    innovations = pd.read_csv(
        INNOVATIONS_PATH,
        parse_dates=[
            "Date",
            "information_end_date",
        ],
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

    crps = pd.read_csv(
        CRPS_PATH,
        parse_dates=["Date"],
    )

    tables = []

    for df in [innovations, audit, kde, crps]:
        part = (
            df.loc[df["strategy"] == STRATEGY]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        if part["Date"].isna().any():
            raise ValueError("Missing dates.")

        if part["Date"].duplicated().any():
            raise ValueError("Duplicate dates.")

        tables.append(part)

    innovations, audit, kde, crps = tables

    if len(innovations) != 3839:
        raise ValueError("Unexpected innovation count.")

    for name, df in [
        ("audit", audit),
        ("KDE", kde),
        ("CRPS", crps),
    ]:
        if len(df) != EXPECTED_DAYS:
            raise ValueError(
                f"Unexpected {name} count."
            )

    dates = (
        innovations["Date"]
        .iloc[WINDOW:]
        .reset_index(drop=True)
    )

    for name, df in [
        ("audit", audit),
        ("KDE", kde),
        ("CRPS", crps),
    ]:
        if not df["Date"].equals(dates):
            raise ValueError(
                f"{name} date alignment failed."
            )

    if not audit["training_observations"].eq(WINDOW).all():
        raise ValueError(
            "Invalid training window lengths."
        )

    if (
        audit["max_absolute_reconciliation_error"]
        > 1e-9
    ).any():
        raise ValueError(
            "FHS reconstruction audit failed."
        )

    if not np.allclose(
        kde["realized_z"],
        innovations["z"].iloc[WINDOW:],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            "Innovation reconciliation failed."
        )

    for column in ["mu_hat", "sigma_hat"]:
        if not np.allclose(
            kde[column],
            innovations[column].iloc[WINDOW:],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(
                f"GARCH reconciliation failed: {column}"
            )

    for column in ["fhs_crps_return", "kde_crps_return"]:
        if not np.isfinite(
            crps[column].to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"Invalid reference CRPS: {column}"
            )

    if not np.isfinite(
        innovations[
            ["z", "realized_return", "mu_hat", "sigma_hat"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid innovation data.")

    if (innovations["sigma_hat"] <= 0).any():
        raise ValueError("Invalid GARCH volatility.")

    return innovations, audit, kde, crps


def run_backtest(innovations, audit, kde, crps):
    z_values = innovations["z"].to_numpy(dtype=float)
    dates = innovations["Date"].to_numpy()

    previous = None
    records = []

    for j in range(WINDOW, len(innovations)):
        index = j - WINDOW

        row = innovations.iloc[j]
        audit_row = audit.iloc[index]
        kde_row = kde.iloc[index]
        crps_row = crps.iloc[index]

        date = pd.Timestamp(dates[j])

        if audit_row["Date"] != date:
            raise ValueError(
                "Training audit date mismatch."
            )

        if audit_row["training_start_date"] != pd.Timestamp(
            dates[j - WINDOW]
        ):
            raise ValueError(
                "Training start date mismatch."
            )

        if audit_row["training_end_date"] != pd.Timestamp(
            dates[j - 1]
        ):
            raise ValueError(
                "Training end date mismatch."
            )

        if row["information_end_date"] != pd.Timestamp(
            dates[j - 1]
        ):
            raise ValueError(
                "Information cutoff mismatch."
            )

        training = z_values[j - WINDOW:j]

        p, nll, mode, successful = fit_rolling_gmm(
            training,
            previous,
            index,
        )

        previous = p.copy()

        w, m1, s1, m2, s2 = p

        mixture_mean, mixture_std = gmm_moments(p)

        actual_z = float(z_values[j])
        actual_return = float(row["realized_return"])

        mu = float(row["mu_hat"])
        sigma = float(row["sigma_hat"])

        actual_loss = -actual_return

        gmm_pit = float(
            gmm_cdf(actual_z, p)
        )

        gmm_crps_z = gmm_crps(
            p, actual_z
        )

        gmm_crps_return = (
            sigma * gmm_crps_z
        )

        fhs_crps_return = float(
            crps_row["fhs_crps_return"]
        )

        kde_crps_return = float(
            crps_row["kde_crps_return"]
        )

        record = {
            "Date": date,
            "strategy": STRATEGY,
            "realized_return": actual_return,
            "realized_loss": actual_loss,
            "realized_z": actual_z,
            "mu_hat": mu,
            "sigma_hat": sigma,
            "fit_mode": mode,
            "successful_starts": successful,
            "negative_log_likelihood": nll,
            "gmm_weight1": w,
            "gmm_location1": m1,
            "gmm_scale1": s1,
            "gmm_weight2": 1 - w,
            "gmm_location2": m2,
            "gmm_scale2": s2,
            "gmm_mean": mixture_mean,
            "gmm_std": mixture_std,
            "gmm_pit": gmm_pit,
            "fhs_pit": float(kde_row["fhs_pit"]),
            "kde_pit": float(kde_row["kde_pit"]),
            "gmm_crps_z": gmm_crps_z,
            "gmm_crps_return": gmm_crps_return,
            "fhs_crps_return": fhs_crps_return,
            "kde_crps_return": kde_crps_return,
            "gmm_crps_bp": 10000 * gmm_crps_return,
            "fhs_crps_bp": 10000 * fhs_crps_return,
            "kde_crps_bp": 10000 * kde_crps_return,
            "delta_gmm_fhs_bp": 10000 * (
                gmm_crps_return - fhs_crps_return
            ),
            "delta_gmm_kde_bp": 10000 * (
                gmm_crps_return - kde_crps_return
            ),
            "weight_at_boundary": (
                w <= 0.031 or w >= 0.969
            ),
            "scale_at_boundary": (
                min(s1, s2) <= 0.121
                or max(s1, s2) >= 3.999
            ),
        }

        for confidence in (0.95, 0.99):
            suffix = int(round(100 * confidence))

            q, tail_mean = gmm_tail(
                p, confidence
            )

            gmm_var = -mu - sigma * q
            gmm_es = -mu - sigma * tail_mean

            if gmm_es < gmm_var - 1e-10:
                raise ValueError(
                    "GMM ES below VaR."
                )

            record[f"gmm_z_quantile_{suffix}"] = q
            record[f"gmm_z_tail_mean_{suffix}"] = tail_mean
            record[f"gmm_var_{suffix}"] = gmm_var
            record[f"gmm_es_{suffix}"] = gmm_es
            record[f"gmm_hit_{suffix}"] = int(
                actual_loss > gmm_var
            )

            for model in ["fhs", "kde"]:
                record[f"{model}_var_{suffix}"] = float(
                    kde_row[f"{model}_var_{suffix}"]
                )
                record[f"{model}_es_{suffix}"] = float(
                    kde_row[f"{model}_es_{suffix}"]
                )
                record[f"{model}_hit_{suffix}"] = int(
                    kde_row[f"{model}_hit_{suffix}"]
                )

        records.append(record)

        if (
            (index + 1) % 100 == 0
            or index + 1 == EXPECTED_DAYS
        ):
            print(
                f"Completed {index + 1:4d}/"
                f"{EXPECTED_DAYS} | "
                f"{date.date()} | "
                f"mode={mode}",
                flush=True,
            )

    result = pd.DataFrame(records)

    if len(result) != EXPECTED_DAYS:
        raise ValueError(
            "Unexpected forecast count."
        )

    if not np.isfinite(
        result.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all():
        raise ValueError(
            "Non-finite output values."
        )

    if not result["gmm_pit"].between(0, 1).all():
        raise ValueError("Invalid GMM PIT.")

    return result


def print_summary(result):
    print("\nGARCH-GMM ROLLING BACKTEST")
    print("=" * 75)

    print(f"Portfolio: {STRATEGY}")
    print(f"Forecast days: {len(result)}")
    print(
        f"Period: {result['Date'].min().date()} "
        f"to {result['Date'].max().date()}"
    )

    print("\nRETURN-SCALE MEAN CRPS (BPS)")

    means = {}

    for model in ["fhs", "kde", "gmm"]:
        value = result[f"{model}_crps_bp"].mean()
        means[model] = value

        print(
            f"{model.upper():4s}: {value:.6f}"
        )

    improvement = (
        100
        * (means["fhs"] - means["gmm"])
        / means["fhs"]
    )

    print(
        f"\nGMM improvement vs. FHS: "
        f"{improvement:+.4f}%"
    )

    print(
        "GMM better than FHS (% days): "
        f"{100 * (result['delta_gmm_fhs_bp'] < 0).mean():.2f}%"
    )

    print("\nVaR EXCEEDANCE RATES")

    for level in (95, 99):
        print(f"\n{level}% VaR")

        for model in ["fhs", "kde", "gmm"]:
            hits = int(
                result[f"{model}_hit_{level}"].sum()
            )

            rate = 100 * hits / len(result)

            print(
                f"  {model.upper():4s}: "
                f"{hits:3d} hits | {rate:.3f}%"
            )

    print("\nPIT DIAGNOSTICS")

    for model in ["fhs", "kde", "gmm"]:
        pit = result[f"{model}_pit"]

        print(
            f"{model.upper():4s}: "
            f"mean={pit.mean():.4f} | "
            f"std={pit.std(ddof=0):.4f} | "
            f"below 0.01={(pit < 0.01).mean():.4f}"
        )

    print("\nGMM PARAMETER STABILITY")

    print(
        "Multistart fits: "
        f"{(result['fit_mode'] == 'multistart').sum()}"
    )

    print(
        "Fallback multistart fits: "
        f"{(result['fit_mode'] == 'fallback_multistart').sum()}"
    )

    print(
        "Weight at boundary: "
        f"{result['weight_at_boundary'].sum()}"
    )

    print(
        "Scale at boundary: "
        f"{result['scale_at_boundary'].sum()}"
    )

    print(
        "Median weight of broader component: "
        f"{result['gmm_weight2'].median():.4f}"
    )

    print(
        "Median scale ratio: "
        f"{(result['gmm_scale2'] / result['gmm_scale1']).median():.4f}"
    )

    annual = (
        result.assign(year=result["Date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            days=("Date", "size"),
            fhs_bp=("fhs_crps_bp", "mean"),
            kde_bp=("kde_crps_bp", "mean"),
            gmm_bp=("gmm_crps_bp", "mean"),
            delta_gmm_fhs_bp=(
                "delta_gmm_fhs_bp", "mean"
            ),
        )
    )

    print("\nANNUAL CRPS COMPARISON")

    print(
        annual.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def plot_results(result):
    data = result.copy()

    data["rolling_delta"] = (
        data["delta_gmm_fhs_bp"]
        .rolling(
            ROLLING_DAYS,
            min_periods=ROLLING_DAYS,
        )
        .mean()
    )

    data["cumulative_delta"] = (
        data["delta_gmm_fhs_bp"].cumsum()
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(13, 13),
        sharex=True,
    )

    dates = data["Date"]

    axes[0].plot(
        dates,
        data["rolling_delta"],
        linewidth=1.4,
        color="#2563a6",
    )

    axes[0].axhline(
        0,
        color="gray",
        linestyle="--",
    )

    axes[0].set_title(
        "252-Day Rolling Mean CRPS Difference: GMM - FHS",
        loc="left",
    )
    axes[0].set_ylabel("Basis points")

    axes[1].plot(
        dates,
        data["cumulative_delta"],
        linewidth=1.5,
        color="#7c3aed",
    )

    axes[1].axhline(
        0,
        color="gray",
        linestyle="--",
    )

    axes[1].set_title(
        "Cumulative CRPS Difference: GMM - FHS",
        loc="left",
    )
    axes[1].set_ylabel("Cumulative bps")

    axes[2].plot(
        dates,
        data["gmm_weight2"],
        linewidth=1.1,
        color="#15803d",
    )

    axes[2].set_title(
        "Weight of Broader Gaussian Component",
        loc="left",
    )
    axes[2].set_ylabel("Mixture weight")

    axes[3].plot(
        dates,
        data["gmm_scale1"],
        linewidth=1.0,
        label="Narrower component",
    )

    axes[3].plot(
        dates,
        data["gmm_scale2"],
        linewidth=1.0,
        label="Broader component",
    )

    axes[3].set_title(
        "GMM Component Scales",
        loc="left",
    )
    axes[3].set_ylabel("Scale")
    axes[3].set_xlabel("Forecast date")
    axes[3].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(
        "GARCH-GMM: Predictive Performance and Stability\n"
        f"{STRATEGY} | 2014-2025",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
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
    innovations, audit, kde, crps = read_inputs()

    print("Input validation passed.")
    print(
        f"Starting {EXPECTED_DAYS} rolling GMM fits...",
        flush=True,
    )

    result = run_backtest(
        innovations,
        audit,
        kde,
        crps,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    plot_results(result)
    print_summary(result)

    print("\nSAVED OUTPUTS")
    print(OUTPUT_PATH)
    print(FIGURE_PATH)

    print("\nGARCH-GMM backtest completed.")


if __name__ == "__main__":
    main()
