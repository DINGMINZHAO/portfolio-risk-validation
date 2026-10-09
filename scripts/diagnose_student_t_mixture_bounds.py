
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fit_student_t_mixture_snapshot import (
    load_snapshot,
    fit_mixture,
    negative_log_likelihood,
    mixture_pdf,
    mixture_cdf,
    mixture_tail,
    mixture_moments,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/density_estimation"

OUTPUT_CSV = (
    DATA_DIR / "student_t_mixture_bound_sensitivity.csv"
)

OUTPUT_FIGURE = (
    FIGURE_DIR / "student_t_mixture_bound_sensitivity.png"
)

DF_CAPS = [40.0, 100.0, 300.0]

STRATEGY = "minvar_static"


def make_bounds(df_cap):
    return [
        (0.03, 0.97),
        (-3.0, 3.0),
        (0.12, 4.0),
        (2.5, df_cap),
        (-3.0, 3.0),
        (0.12, 4.0),
        (2.5, df_cap),
    ]


def canonicalize(p):
    p = np.asarray(p, dtype=float)

    if p[2] > p[5]:
        w, m1, s1, v1, m2, s2, v2 = p

        p = np.array([
            1 - w,
            m2,
            s2,
            v2,
            m1,
            s1,
            v1,
        ])

    return p


def fit_with_cap(z, df_cap, previous):
    bounds = make_bounds(df_cap)

    center = float(np.median(z))
    spread = float(np.std(z, ddof=1))

    q25, q75 = np.quantile(z, [0.25, 0.75])

    starts = [
        previous.copy(),
        np.array([
            0.80, center, 0.75 * spread, 9.0,
            center, 1.60 * spread, 5.0,
        ]),
        np.array([
            0.50, q25, 0.80 * spread, 10.0,
            q75, 0.80 * spread, 10.0,
        ]),
        np.array([
            0.90, center, 0.85 * spread,
            0.80 * df_cap,
            center, 1.90 * spread,
            0.80 * df_cap,
        ]),
        np.array([
            0.65, center, 0.65 * spread, 15.0,
            center, 1.90 * spread, 4.0,
        ]),
    ]

    # The previous solution remains feasible after
    # increasing the degrees-of-freedom bounds.
    best_p = previous.copy()

    best_nll = negative_log_likelihood(
        best_p, z
    )

    converged = 0

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    for start in starts:
        x0 = np.clip(
            start,
            lower + 1e-5,
            upper - 1e-5,
        )

        result = minimize(
            negative_log_likelihood,
            x0,
            args=(z,),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 800,
                "ftol": 1e-10,
                "maxls": 40,
            },
        )

        if (
            result.success
            and np.isfinite(result.fun)
        ):
            converged += 1

            if result.fun < best_nll:
                best_nll = float(result.fun)
                best_p = result.x.copy()

    return (
        canonicalize(best_p),
        float(best_nll),
        converged,
    )


def summarize_fit(
    p,
    cap,
    nll,
    converged,
    date,
    row,
):
    w, m1, s1, v1, m2, s2, v2 = p

    mean, std = mixture_moments(p)

    mu = float(row["mu_hat"])
    sigma = float(row["sigma_hat"])

    q95, es95 = mixture_tail(p, 0.95)
    q99, es99 = mixture_tail(p, 0.99)

    return {
        "Date": date,
        "strategy": STRATEGY,
        "df_upper_bound": cap,
        "negative_log_likelihood": nll,
        "converged_starts": converged,
        "weight1": w,
        "location1": m1,
        "scale1": s1,
        "df1": v1,
        "weight2": 1 - w,
        "location2": m2,
        "scale2": s2,
        "df2": v2,
        "df1_at_upper_bound": (
            v1 >= cap - 0.01
        ),
        "df2_at_upper_bound": (
            v2 >= cap - 0.01
        ),
        "mixture_mean": mean,
        "mixture_std": std,
        "q95": q95,
        "q99": q99,
        "tail_mean95": es95,
        "tail_mean99": es99,
        "var95_pct": 100 * (-mu - sigma * q95),
        "var99_pct": 100 * (-mu - sigma * q99),
        "es95_pct": 100 * (-mu - sigma * es95),
        "es99_pct": 100 * (-mu - sigma * es99),
    }


def plot_distributions(z, fitted, date):
    grid = np.linspace(-6.0, 3.0, 1400)

    empirical_cdf = (
        np.searchsorted(
            np.sort(z),
            grid,
            side="right",
        ) / len(z)
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
    )

    axes[0].hist(
        z,
        bins=60,
        density=True,
        alpha=0.25,
        color="gray",
        label="Historical innovations",
    )

    for cap, p in fitted.items():
        axes[0].plot(
            grid,
            mixture_pdf(grid, p),
            linewidth=1.8,
            label=f"Student-t Mixture: df cap {int(cap)}",
        )

    axes[0].set_xlim(-5.5, 3.0)
    axes[0].set_title(
        "Fitted Probability Density",
        loc="left",
    )
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        grid,
        empirical_cdf,
        color="gray",
        linewidth=1.4,
        label="Empirical FHS",
    )

    for cap, p in fitted.items():
        axes[1].plot(
            grid,
            mixture_cdf(grid, p),
            linewidth=1.8,
            label=f"df cap {int(cap)}",
        )

    axes[1].set_xlim(-5.0, -1.0)
    axes[1].set_ylim(0, 0.12)

    axes[1].axhline(
        0.01,
        linestyle="--",
        linewidth=1,
        color="gray",
        alpha=0.7,
    )

    axes[1].axhline(
        0.05,
        linestyle=":",
        linewidth=1,
        color="gray",
        alpha=0.7,
    )

    axes[1].set_title(
        "Left-Tail Cumulative Distribution",
        loc="left",
    )

    axes[1].set_xlabel(
        "Standardized innovation"
    )

    axes[1].set_ylabel(
        "Cumulative probability"
    )

    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"Student-t Mixture: Degrees-of-Freedom Sensitivity\n"
        f"{STRATEGY} | Forecast: {date.date()}",
        fontsize=14,
    )

    fig.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    row, training, references = load_snapshot()

    date = row["Date"]

    print("\nSTUDENT-T MIXTURE BOUND SENSITIVITY")
    print("=" * 82)

    print(f"Strategy: {STRATEGY}")
    print(f"Forecast date: {date.date()}")
    print(f"Training observations: {len(training)}")

    print("\nFitting baseline: df cap = 40")

    baseline_p, successful, baseline_nll = (
        fit_mixture(training)
    )

    fitted = {
        40.0: baseline_p,
    }

    records = [
        summarize_fit(
            baseline_p,
            40.0,
            baseline_nll,
            successful,
            date,
            row,
        )
    ]

    previous = baseline_p

    for cap in DF_CAPS[1:]:
        print(
            f"Fitting sensitivity model: "
            f"df cap = {int(cap)}",
            flush=True,
        )

        p, nll, converged = fit_with_cap(
            training,
            cap,
            previous,
        )

        fitted[cap] = p

        records.append(
            summarize_fit(
                p,
                cap,
                nll,
                converged,
                date,
                row,
            )
        )

        previous = p

    summary = pd.DataFrame(records)

    summary["log_likelihood_gain_vs_40"] = (
        baseline_nll
        - summary["negative_log_likelihood"]
    )

    summary["gain_per_observation"] = (
        summary["log_likelihood_gain_vs_40"]
        / len(training)
    )

    print("\nLIKELIHOOD AND PARAMETER DIAGNOSTICS")

    columns = [
        "df_upper_bound",
        "negative_log_likelihood",
        "log_likelihood_gain_vs_40",
        "converged_starts",
        "weight2",
        "scale1",
        "scale2",
        "df1",
        "df2",
        "df1_at_upper_bound",
        "df2_at_upper_bound",
    ]

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print("\n99% TAIL FORECAST SENSITIVITY")

    columns = [
        "df_upper_bound",
        "q99",
        "tail_mean99",
        "var99_pct",
        "es99_pct",
        "mixture_mean",
        "mixture_std",
    ]

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    ref = references[0.99]

    print("\nFHS REFERENCE")
    print(
        f"99% VaR: "
        f"{100 * float(ref['var_forecast']):.5f}%"
    )
    print(
        f"99% ES:  "
        f"{100 * float(ref['es_forecast']):.5f}%"
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    plot_distributions(
        training,
        fitted,
        date,
    )

    print("\nSAVED OUTPUTS")
    print(OUTPUT_CSV)
    print(OUTPUT_FIGURE)

    print("\nBound sensitivity analysis completed.")


if __name__ == "__main__":
    main()
