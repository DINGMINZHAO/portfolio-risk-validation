
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize, brentq
from scipy.special import logsumexp
from scipy.stats import norm

from fit_student_t_mixture_snapshot import (
    load_snapshot,
    mixture_pdf,
    mixture_cdf,
    mixture_tail,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/density_estimation"

T_PATH = DATA_DIR / "student_t_mixture_bound_sensitivity.csv"

OUTPUT_CSV = DATA_DIR / "garch_gmm_snapshot_minvar_static.csv"
OUTPUT_FIG = FIGURE_DIR / "garch_gmm_vs_t_mixture_snapshot.png"

STRATEGY = "minvar_static"

BOUNDS = [
    (0.03, 0.97),
    (-3.0, 3.0),
    (0.12, 4.0),
    (-3.0, 3.0),
    (0.12, 4.0),
]


def gmm_logpdf(z, p):
    w, m1, s1, m2, s2 = p

    left = np.log(w) + norm.logpdf(
        z, loc=m1, scale=s1
    )
    right = np.log1p(-w) + norm.logpdf(
        z, loc=m2, scale=s2
    )

    return np.logaddexp(left, right)


def gmm_pdf(z, p):
    return np.exp(gmm_logpdf(z, p))


def gmm_cdf(z, p):
    w, m1, s1, m2, s2 = p

    return (
        w * norm.cdf(z, loc=m1, scale=s1)
        + (1 - w) * norm.cdf(z, loc=m2, scale=s2)
    )


def gmm_nll(p, z):
    return float(-gmm_logpdf(z, p).sum())


def fit_gmm(z, initial_from_t):
    center = float(np.median(z))
    spread = float(np.std(z, ddof=1))

    q25, q75 = np.quantile(z, [0.25, 0.75])

    starts = [
        initial_from_t,
        [0.90, center, 0.85 * spread,
         center, 1.90 * spread],
        [0.80, center, 0.75 * spread,
         center, 1.60 * spread],
        [0.50, q25, 0.80 * spread,
         q75, 0.80 * spread],
        [0.65, center, 0.65 * spread,
         center, 2.00 * spread],
        [0.30, q25, 1.40 * spread,
         q75, 0.75 * spread],
    ]

    lower = np.array([b[0] for b in BOUNDS])
    upper = np.array([b[1] for b in BOUNDS])

    solutions = []

    for start in starts:
        x0 = np.clip(
            np.asarray(start, dtype=float),
            lower + 1e-5,
            upper - 1e-5,
        )

        result = minimize(
            gmm_nll,
            x0,
            args=(z,),
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={
                "maxiter": 1200,
                "ftol": 1e-11,
                "maxls": 40,
            },
        )

        if result.success and np.isfinite(result.fun):
            solutions.append(result)

    if not solutions:
        raise RuntimeError("All GMM optimizations failed.")

    best = min(solutions, key=lambda x: x.fun)
    p = np.asarray(best.x, dtype=float)

    if p[2] > p[4]:
        w, m1, s1, m2, s2 = p
        p = np.array([1 - w, m2, s2, m1, s1])

    return p, float(best.fun), len(solutions)


def gmm_tail(p, confidence):
    alpha = 1.0 - confidence

    q = brentq(
        lambda x: gmm_cdf(x, p) - alpha,
        -100,
        100,
        xtol=1e-11,
    )

    w, m1, s1, m2, s2 = p

    a1 = (q - m1) / s1
    a2 = (q - m2) / s2

    partial1 = m1 * norm.cdf(a1) - s1 * norm.pdf(a1)
    partial2 = m2 * norm.cdf(a2) - s2 * norm.pdf(a2)

    tail_mean = (
        w * partial1 + (1 - w) * partial2
    ) / alpha

    if (
        not np.isfinite([q, tail_mean]).all()
        or tail_mean > q + 1e-9
    ):
        raise ValueError("Invalid Gaussian mixture tail.")

    return float(q), float(tail_mean)


def read_t_reference(date):
    df = pd.read_csv(T_PATH, parse_dates=["Date"])

    selected = df.loc[
        (df["strategy"] == STRATEGY)
        & (df["Date"] == date)
        & np.isclose(df["df_upper_bound"], 300.0)
    ]

    if len(selected) != 1:
        raise ValueError("Expected one df-cap-300 reference.")

    row = selected.iloc[0]

    if not np.isclose(
        float(row["weight1"]) + float(row["weight2"]),
        1.0,
        atol=1e-8,
    ):
        raise ValueError("Invalid mixture weights.")

    p = np.array([
        row["weight1"],
        row["location1"],
        row["scale1"],
        row["df1"],
        row["location2"],
        row["scale2"],
        row["df2"],
    ], dtype=float)

    if not np.isfinite(p).all():
        raise ValueError("Invalid Student-t parameters.")

    return p, float(row["negative_log_likelihood"])


def plot_comparison(z, p_gmm, p_t, date):
    grid = np.linspace(-6, 4, 1400)

    empirical_cdf = (
        np.searchsorted(
            np.sort(z), grid, side="right"
        ) / len(z)
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 9)
    )

    axes[0].hist(
        z,
        bins=60,
        density=True,
        color="gray",
        alpha=0.25,
        label="Historical innovations",
    )

    axes[0].plot(
        grid, gmm_pdf(grid, p_gmm),
        linewidth=2,
        label="Two-component GMM",
    )

    axes[0].plot(
        grid, mixture_pdf(grid, p_t),
        linewidth=2,
        linestyle="--",
        label="Two-component Student-t Mixture",
    )

    axes[0].set_xlim(-5.5, 3)
    axes[0].set_ylabel("Density")
    axes[0].set_title(
        "Probability Density",
        loc="left",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        grid, empirical_cdf,
        color="gray",
        linewidth=1.5,
        label="Empirical FHS",
    )

    axes[1].plot(
        grid, gmm_cdf(grid, p_gmm),
        linewidth=2,
        label="GMM",
    )

    axes[1].plot(
        grid, mixture_cdf(grid, p_t),
        linewidth=2,
        linestyle="--",
        label="Student-t Mixture",
    )

    axes[1].axhline(
        0.01, color="gray",
        linewidth=1, linestyle=":"
    )

    axes[1].axhline(
        0.05, color="gray",
        linewidth=1, linestyle=":"
    )

    axes[1].set_xlim(-5, -1)
    axes[1].set_ylim(0, 0.12)
    axes[1].set_xlabel("Standardized innovation")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_title(
        "Left-Tail CDF",
        loc="left",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        "GMM vs. Student-t Mixture\n"
        f"{STRATEGY} | Forecast: {date.date()}",
        fontsize=14,
    )

    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    row, z, fhs_refs = load_snapshot()

    date = row["Date"]
    mu = float(row["mu_hat"])
    sigma = float(row["sigma_hat"])

    p_t, nll_t = read_t_reference(date)

    initial = [
        p_t[0], p_t[1], p_t[2],
        p_t[4], p_t[5],
    ]

    p_gmm, nll_gmm, converged = fit_gmm(
        z, initial
    )

    w, m1, s1, m2, s2 = p_gmm

    print("\nGAUSSIAN MIXTURE SNAPSHOT")
    print("=" * 75)
    print(f"Strategy: {STRATEGY}")
    print(f"Forecast date: {date.date()}")
    print(f"Training observations: {len(z)}")
    print(f"Converged starts: {converged}/6")

    print("\nFITTED GMM COMPONENTS")
    print(
        f"Component 1: weight={w:.5f}, "
        f"location={m1:.5f}, scale={s1:.5f}"
    )
    print(
        f"Component 2: weight={1-w:.5f}, "
        f"location={m2:.5f}, scale={s2:.5f}"
    )

    print("\nIN-SAMPLE LIKELIHOOD")
    print(f"GMM NLL:               {nll_gmm:.6f}")
    print(f"Student-t Mixture NLL: {nll_t:.6f}")
    print(f"GMM minus t NLL:       {nll_gmm-nll_t:+.6f}")

    print("\nTAIL FORECAST COMPARISON")

    records = []

    for level in (0.95, 0.99):
        q_gmm, tail_gmm = gmm_tail(p_gmm, level)
        q_t, tail_t = mixture_tail(p_t, level)

        fhs = fhs_refs[level]

        var_gmm = -mu - sigma * q_gmm
        es_gmm = -mu - sigma * tail_gmm

        var_t = -mu - sigma * q_t
        es_t = -mu - sigma * tail_t

        var_fhs = float(fhs["var_forecast"])
        es_fhs = float(fhs["es_forecast"])

        print(
            f"\n{int(level*100)}% confidence"
            f"\n  VaR (%): GMM={100*var_gmm:.5f} | "
            f"t-Mixture={100*var_t:.5f} | "
            f"FHS={100*var_fhs:.5f}"
            f"\n  ES  (%): GMM={100*es_gmm:.5f} | "
            f"t-Mixture={100*es_t:.5f} | "
            f"FHS={100*es_fhs:.5f}"
        )

        records.append({
            "Date": date,
            "strategy": STRATEGY,
            "confidence_level": level,
            "gmm_weight1": w,
            "gmm_location1": m1,
            "gmm_scale1": s1,
            "gmm_weight2": 1-w,
            "gmm_location2": m2,
            "gmm_scale2": s2,
            "gmm_nll": nll_gmm,
            "t_mixture_nll": nll_t,
            "gmm_z_quantile": q_gmm,
            "t_z_quantile": q_t,
            "gmm_z_tail_mean": tail_gmm,
            "t_z_tail_mean": tail_t,
            "gmm_var": var_gmm,
            "t_var": var_t,
            "fhs_var": var_fhs,
            "gmm_es": es_gmm,
            "t_es": es_t,
            "fhs_es": es_fhs,
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(
        OUTPUT_CSV, index=False
    )

    plot_comparison(z, p_gmm, p_t, date)

    print("\nSAVED OUTPUTS")
    print(OUTPUT_CSV)
    print(OUTPUT_FIG)
    print("\nGMM snapshot completed.")


if __name__ == "__main__":
    main()
