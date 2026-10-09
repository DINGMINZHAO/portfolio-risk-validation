
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/density_estimation"

INNOVATIONS_PATH = DATA_DIR / "garch_fhs_oos_innovations.csv"
AUDIT_PATH = DATA_DIR / "garch_fhs_density_window_audit.csv"
FHS_PATH = DATA_DIR / "garch_fhs_var_es_forecasts.csv"

OUTPUT_PATH = DATA_DIR / "student_t_mixture_snapshot_minvar_static.csv"
FIGURE_PATH = FIGURE_DIR / "student_t_mixture_snapshot_minvar_static.png"

STRATEGY = "minvar_static"
WINDOW = 1008
LEVELS = (0.95, 0.99)

BOUNDS = [
    (0.03, 0.97),
    (-3.0, 3.0),
    (0.12, 4.0),
    (2.5, 40.0),
    (-3.0, 3.0),
    (0.12, 4.0),
    (2.5, 40.0),
]


def mixture_pdf(x, p):
    w, m1, s1, v1, m2, s2, v2 = p

    return (
        w * t.pdf(x, v1, loc=m1, scale=s1)
        + (1 - w) * t.pdf(x, v2, loc=m2, scale=s2)
    )


def mixture_cdf(x, p):
    w, m1, s1, v1, m2, s2, v2 = p

    return (
        w * t.cdf(x, v1, loc=m1, scale=s1)
        + (1 - w) * t.cdf(x, v2, loc=m2, scale=s2)
    )


def negative_log_likelihood(p, z):
    w, m1, s1, v1, m2, s2, v2 = p

    left = (
        np.log(w)
        + t.logpdf(z, v1, loc=m1, scale=s1)
    )

    right = (
        np.log1p(-w)
        + t.logpdf(z, v2, loc=m2, scale=s2)
    )

    return float(
        -np.logaddexp(left, right).sum()
    )


def fit_mixture(z):
    center = float(np.median(z))
    spread = float(np.std(z, ddof=1))

    q25, q75 = np.quantile(
        z, [0.25, 0.75]
    )

    starts = [
        [
            0.80, center, 0.75 * spread, 9.0,
            center, 1.6 * spread, 5.0,
        ],
        [
            0.50, q25, 0.80 * spread, 10.0,
            q75, 0.80 * spread, 10.0,
        ],
        [
            0.65, center, 0.65 * spread, 15.0,
            center, 1.9 * spread, 4.0,
        ],
    ]

    solutions = []

    for start in starts:
        x0 = np.clip(
            start,
            [v[0] + 1e-4 for v in BOUNDS],
            [v[1] - 1e-4 for v in BOUNDS],
        )

        result = minimize(
            negative_log_likelihood,
            x0,
            args=(z,),
            method="L-BFGS-B",
            bounds=BOUNDS,
            options={
                "maxiter": 800,
                "ftol": 1e-10,
                "maxls": 40,
            },
        )

        if result.success and np.isfinite(result.fun):
            solutions.append(result)

    if not solutions:
        raise RuntimeError(
            "All Student-t mixture optimizations failed."
        )

    best = min(
        solutions,
        key=lambda result: result.fun,
    )

    p = np.asarray(best.x, dtype=float)

    if p[2] > p[5]:
        w, m1, s1, v1, m2, s2, v2 = p

        p = np.array([
            1 - w, m2, s2, v2,
            m1, s1, v1,
        ])

    return p, len(solutions), float(best.fun)


def mixture_moments(p):
    w, m1, s1, v1, m2, s2, v2 = p

    mean = w * m1 + (1 - w) * m2

    second = (
        w * (
            m1 * m1
            + s1 * s1 * v1 / (v1 - 2)
        )
        + (1 - w) * (
            m2 * m2
            + s2 * s2 * v2 / (v2 - 2)
        )
    )

    return (
        float(mean),
        float(np.sqrt(second - mean * mean)),
    )


def component_partial_moment(q, loc, scale, df):
    a = (q - loc) / scale

    return (
        loc * t.cdf(a, df)
        - scale
        * (df + a * a)
        / (df - 1)
        * t.pdf(a, df)
    )


def mixture_tail(p, confidence):
    w, m1, s1, v1, m2, s2, v2 = p

    alpha = 1 - confidence

    q = brentq(
        lambda x: mixture_cdf(x, p) - alpha,
        -100.0,
        100.0,
        xtol=1e-11,
    )

    partial = (
        w * component_partial_moment(
            q, m1, s1, v1
        )
        + (1 - w) * component_partial_moment(
            q, m2, s2, v2
        )
    )

    tail_mean = float(partial / alpha)

    if (
        not np.isfinite([q, tail_mean]).all()
        or tail_mean > q + 1e-9
    ):
        raise ValueError(
            "Invalid mixture VaR/ES tail."
        )

    return float(q), tail_mean


def empirical_tail(z, confidence):
    ordered = np.sort(z)

    mass = len(z) * (1 - confidence)
    n_full = int(np.floor(mass))
    frac = mass - n_full

    index = max(
        int(np.ceil(mass)) - 1,
        0,
    )

    q = float(ordered[index])

    tail_sum = float(
        ordered[:n_full].sum()
    )

    if frac > 1e-12:
        tail_sum += frac * float(
            ordered[n_full]
        )

    return q, tail_sum / mass


def load_snapshot():
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
    ]

    if (
        len(innovations) != 3839
        or len(audit) != 2831
    ):
        raise ValueError(
            "Unexpected innovation/audit length."
        )

    if (
        innovations["Date"].duplicated().any()
        or audit["Date"].duplicated().any()
    ):
        raise ValueError("Duplicate dates.")

    j = len(innovations) - 1

    row = innovations.iloc[j]
    date = row["Date"]

    audit_row = audit.iloc[-1]

    training = (
        innovations["z"]
        .iloc[j - WINDOW:j]
        .to_numpy(dtype=float)
    )

    training_dates = (
        innovations["Date"]
        .iloc[j - WINDOW:j]
    )

    if (
        not np.isfinite(training).all()
        or len(training) != WINDOW
    ):
        raise ValueError(
            "Invalid rolling innovation window."
        )

    if (
        audit_row["Date"] != date
        or audit_row["training_start_date"]
        != training_dates.iloc[0]
        or audit_row["training_end_date"]
        != training_dates.iloc[-1]
    ):
        raise ValueError(
            "Residual window date mismatch."
        )

    if (
        row["information_end_date"]
        != training_dates.iloc[-1]
    ):
        raise ValueError(
            "Information cutoff mismatch."
        )

    if (
        int(audit_row["training_observations"])
        != WINDOW
        or audit_row[
            "max_absolute_reconciliation_error"
        ] > 1e-9
    ):
        raise ValueError(
            "Previous FHS window reconciliation failed."
        )

    if fhs.duplicated(
        ["Date", "confidence_level"]
    ).any():
        raise ValueError("Duplicate FHS keys.")

    references = {}

    for level in LEVELS:
        match = fhs.loc[
            fhs["Date"].eq(date)
            & np.isclose(
                fhs["confidence_level"],
                level,
            )
        ]

        if len(match) != 1:
            raise ValueError(
                f"Missing FHS snapshot: {level}"
            )

        ref = match.iloc[0]

        if not np.allclose(
            [row["mu_hat"], row["sigma_hat"]],
            [ref["mu_hat"], ref["sigma_hat"]],
            atol=1e-11,
            rtol=0,
        ):
            raise ValueError(
                "GARCH mean/sigma mismatch."
            )

        q, es = empirical_tail(
            training, level
        )

        if not np.allclose(
            [q, es],
            [
                ref["z_quantile"],
                ref["z_tail_mean"],
            ],
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError(
                "FHS empirical tail reconciliation failed."
            )

        references[level] = ref

    return row, training, references


def plot_result(z, p, date):
    grid = np.linspace(
        -6, 6, 1200
    )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(11, 8),
        sharex=True,
    )

    axes[0].hist(
        z,
        bins=np.linspace(-6, 6, 70),
        weights=(
            np.ones(len(z))
            / (len(z) * (12 / 69))
        ),
        alpha=0.32,
        label="Historical innovations",
    )

    axes[0].plot(
        grid,
        mixture_pdf(grid, p),
        linewidth=2,
        label="Two-component Student-t mixture",
    )

    axes[0].set_ylabel(
        "Probability density"
    )

    axes[0].legend(
        frameon=False
    )

    axes[1].step(
        grid,
        np.searchsorted(
            np.sort(z),
            grid,
            side="right",
        ) / len(z),
        where="post",
        linewidth=1.1,
        label="Empirical FHS CDF",
    )

    axes[1].plot(
        grid,
        mixture_cdf(grid, p),
        linewidth=2,
        label="Student-t mixture CDF",
    )

    axes[1].set_xlim(-6, 2.5)
    axes[1].set_ylim(0, 0.95)

    axes[1].set_ylabel(
        "Cumulative probability"
    )

    axes[1].set_xlabel(
        "Standardized innovation"
    )

    axes[1].legend(
        frameon=False
    )

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"Student-t Mixture Snapshot - {STRATEGY}\n"
        f"Forecast date: {date.date()}",
        fontsize=14,
    )

    fig.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    row, training, references = load_snapshot()

    p, successful, nll = fit_mixture(training)

    mean, std = mixture_moments(p)

    date = row["Date"]

    mu = float(row["mu_hat"])
    sigma = float(row["sigma_hat"])

    print("\nSTUDENT-T MIXTURE SNAPSHOT")
    print("=" * 67)

    print(
        f"Strategy: {STRATEGY} | "
        f"Forecast date: {date.date()}"
    )

    print(
        f"Training observations: {WINDOW} | "
        f"Converged starts: {successful}/3"
    )

    print(
        "In-sample negative log likelihood: "
        f"{nll:.4f}"
    )

    print("\nFITTED COMPONENTS")

    print(
        f"Component 1: weight={p[0]:.4f}, "
        f"location={p[1]:.4f}, "
        f"scale={p[2]:.4f}, "
        f"df={p[3]:.4f}"
    )

    print(
        f"Component 2: weight={1-p[0]:.4f}, "
        f"location={p[4]:.4f}, "
        f"scale={p[5]:.4f}, "
        f"df={p[6]:.4f}"
    )

    print(
        f"Mixture mean={mean:.4f} | "
        f"mixture std={std:.4f}"
    )

    print("\nTAIL FORECASTS: MIXTURE vs. FHS")

    records = []

    for level in LEVELS:
        q_mix, es_mix = mixture_tail(
            p, level
        )

        ref = references[level]

        q_fhs = float(
            ref["z_quantile"]
        )

        es_fhs = float(
            ref["z_tail_mean"]
        )

        var_mix = -mu - sigma * q_mix
        var_fhs = float(
            ref["var_forecast"]
        )

        risk_es_mix = -mu - sigma * es_mix
        risk_es_fhs = float(
            ref["es_forecast"]
        )

        print(
            f"{int(level*100)}% | "
            f"q: {q_mix:+.4f} vs {q_fhs:+.4f} | "
            f"ES z: {es_mix:+.4f} vs {es_fhs:+.4f} | "
            f"VaR %: {100*var_mix:.4f} "
            f"vs {100*var_fhs:.4f} | "
            f"ES %: {100*risk_es_mix:.4f} "
            f"vs {100*risk_es_fhs:.4f}"
        )

        records.append({
            "Date": date,
            "strategy": STRATEGY,
            "confidence_level": level,
            "mixture_weight1": p[0],
            "mixture_mu1": p[1],
            "mixture_scale1": p[2],
            "mixture_df1": p[3],
            "mixture_weight2": 1 - p[0],
            "mixture_mu2": p[4],
            "mixture_scale2": p[5],
            "mixture_df2": p[6],
            "mixture_mean": mean,
            "mixture_std": std,
            "mixture_z_quantile": q_mix,
            "fhs_z_quantile": q_fhs,
            "mixture_z_tail_mean": es_mix,
            "fhs_z_tail_mean": es_fhs,
            "mixture_var": var_mix,
            "fhs_var": var_fhs,
            "mixture_es": risk_es_mix,
            "fhs_es": risk_es_fhs,
        })

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(records).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    plot_result(
        training,
        p,
        date,
    )

    print("\nSAVED OUTPUTS")
    print(OUTPUT_PATH)
    print(FIGURE_PATH)

    print("\nSnapshot fitting completed.")


if __name__ == "__main__":
    main()
