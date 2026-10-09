
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/es_backtesting"

INPUT_PATH = DATA_DIR / "garch_gmm_minvar_static_daily.csv"

DAILY_PATH = DATA_DIR / "es_joint_fz0_minvar_static_daily.csv"
SUMMARY_PATH = DATA_DIR / "es_joint_fz0_minvar_static_summary.csv"
FIGURE_PATH = FIGURE_DIR / "es_joint_fz0_minvar_static.png"

STRATEGY = "minvar_static"
MODELS = ("fhs", "gmm")
LEVELS = (0.95, 0.99)

EXPECTED_DAYS = 2831
ROLLING_DAYS = 252

BLOCK_LENGTH = 21
BOOTSTRAP_REPLICATIONS = 2000
RANDOM_SEED = 42


def load_data():
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    data = (
        data.loc[data["strategy"] == STRATEGY]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if len(data) != EXPECTED_DAYS:
        raise ValueError("Unexpected backtesting length.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates.")

    if (
        data["Date"].min() != pd.Timestamp("2014-09-08")
        or data["Date"].max() != pd.Timestamp("2025-12-08")
    ):
        raise ValueError("Unexpected date range.")

    required = [
        "realized_return",
        "realized_loss",
    ]

    for level in LEVELS:
        suffix = int(round(100 * level))

        for model in MODELS:
            required.extend([
                f"{model}_var_{suffix}",
                f"{model}_es_{suffix}",
                f"{model}_hit_{suffix}",
            ])

    missing = sorted(
        set(required) - set(data.columns)
    )

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if not np.isfinite(
        data[required].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite input values.")

    if not np.allclose(
        data["realized_loss"],
        -data["realized_return"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Loss and return mismatch.")

    for level in LEVELS:
        suffix = int(round(100 * level))

        for model in MODELS:
            var = data[f"{model}_var_{suffix}"]
            es = data[f"{model}_es_{suffix}"]

            if (es <= 0).any():
                raise ValueError(
                    f"FZ0 requires positive ES: {model}, {suffix}"
                )

            if (es < var - 1e-10).any():
                raise ValueError(
                    f"ES below VaR: {model}, {suffix}"
                )

            expected_hits = (
                data["realized_loss"] > var
            ).astype(int)

            if not np.array_equal(
                expected_hits.to_numpy(),
                data[f"{model}_hit_{suffix}"].to_numpy(dtype=int),
            ):
                raise ValueError(
                    f"Exceedance mismatch: {model}, {suffix}"
                )

    return data


def fz0_score(loss, var, es, alpha):
    excess = np.maximum(
        loss - var,
        0.0,
    )

    score = (
        excess / (alpha * es)
        + var / es
        + np.log(es)
        - 1.0
    )

    if not np.isfinite(score).all():
        raise ValueError("Invalid FZ0 score.")

    return score


def calculate_daily_scores(data):
    result = data[
        [
            "Date",
            "strategy",
            "realized_return",
            "realized_loss",
        ]
    ].copy()

    loss = data[
        "realized_loss"
    ].to_numpy(dtype=float)

    for level in LEVELS:
        suffix = int(round(100 * level))
        alpha = 1 - level

        for model in MODELS:
            var = data[
                f"{model}_var_{suffix}"
            ].to_numpy(dtype=float)

            es = data[
                f"{model}_es_{suffix}"
            ].to_numpy(dtype=float)

            result[f"{model}_score_{suffix}"] = (
                fz0_score(
                    loss,
                    var,
                    es,
                    alpha,
                )
            )

        result[f"delta_score_{suffix}"] = (
            result[f"gmm_score_{suffix}"]
            - result[f"fhs_score_{suffix}"]
        )

    return result


def bootstrap_indices(n):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    blocks_needed = int(
        np.ceil(n / BLOCK_LENGTH)
    )

    starts = rng.integers(
        0,
        n - BLOCK_LENGTH + 1,
        size=(
            BOOTSTRAP_REPLICATIONS,
            blocks_needed,
        ),
    )

    offsets = np.arange(
        BLOCK_LENGTH
    )

    indices = (
        starts[:, :, None]
        + offsets[None, None, :]
    ).reshape(
        BOOTSTRAP_REPLICATIONS,
        -1,
    )

    return indices[:, :n]


def build_summary(result):
    indices = bootstrap_indices(
        len(result)
    )

    rows = []

    for level in LEVELS:
        suffix = int(round(100 * level))

        fhs = result[
            f"fhs_score_{suffix}"
        ].to_numpy(dtype=float)

        gmm = result[
            f"gmm_score_{suffix}"
        ].to_numpy(dtype=float)

        delta = gmm - fhs

        bootstrap_means = (
            delta[indices].mean(axis=1)
        )

        ci_lower, ci_upper = np.quantile(
            bootstrap_means,
            [0.025, 0.975],
        )

        rows.append({
            "strategy": STRATEGY,
            "confidence_level": level,
            "forecast_days": len(result),
            "fhs_mean_score": fhs.mean(),
            "gmm_mean_score": gmm.mean(),
            "mean_delta_gmm_minus_fhs": delta.mean(),
            "median_delta": np.median(delta),
            "gmm_better_fraction": (
                delta < 0
            ).mean(),
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
            "ci_contains_zero": (
                ci_lower <= 0 <= ci_upper
            ),
        })

    return pd.DataFrame(rows)


def print_results(result, summary):
    print("\nJOINT VaR-ES FZ0 COMPARISON")
    print("=" * 80)

    print(f"Strategy: {STRATEGY}")
    print(f"Forecast days: {len(result)}")

    print(
        f"Period: {result['Date'].min().date()} "
        f"to {result['Date'].max().date()}"
    )

    print(
        f"Bootstrap: {BOOTSTRAP_REPLICATIONS} replications, "
        f"{BLOCK_LENGTH}-day blocks"
    )

    print("\nMEAN JOINT SCORES")
    print(
        summary[
            [
                "confidence_level",
                "fhs_mean_score",
                "gmm_mean_score",
                "mean_delta_gmm_minus_fhs",
                "gmm_better_fraction",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "ci_contains_zero",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    annual_rows = []

    for level in LEVELS:
        suffix = int(round(100 * level))

        annual = (
            result.assign(
                year=result["Date"].dt.year
            )
            .groupby("year", as_index=False)
            .agg(
                mean_delta=(
                    f"delta_score_{suffix}",
                    "mean",
                )
            )
        )

        annual["confidence_level"] = level

        annual_rows.append(annual)

    annual = pd.concat(
        annual_rows,
        ignore_index=True,
    )

    pivot = annual.pivot(
        index="year",
        columns="confidence_level",
        values="mean_delta",
    )

    pivot.columns = [
        f"delta_{int(round(100 * c))}"
        for c in pivot.columns
    ]

    print("\nANNUAL MEAN SCORE DIFFERENCES")
    print("Negative values favor GMM.")

    print(
        pivot.to_string(
            float_format=lambda x: f"{x:+.7f}",
        )
    )


def plot_results(result):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
    )

    for ax, suffix in zip(
        axes,
        [95, 99],
    ):
        delta = result[
            f"delta_score_{suffix}"
        ]

        rolling = delta.rolling(
            ROLLING_DAYS,
            min_periods=ROLLING_DAYS,
        ).mean()

        ax.plot(
            result["Date"],
            rolling,
            linewidth=1.5,
            color="#2563a6",
        )

        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1,
        )

        ax.set_title(
            f"{suffix}% VaR-ES: "
            "252-Day Rolling Mean FZ0 Difference",
            loc="left",
        )

        ax.set_ylabel(
            "GMM - FHS"
        )

        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(
        "Forecast date"
    )

    fig.suptitle(
        "Joint VaR-ES Forecast Comparison\n"
        f"{STRATEGY} | GARCH-GMM vs. GARCH-FHS",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
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
    data = load_data()

    print("Input validation passed.")

    daily = calculate_daily_scores(data)
    summary = build_summary(daily)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        DAILY_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    plot_results(daily)
    print_results(daily, summary)

    print("\nSAVED OUTPUTS")
    print(DAILY_PATH)
    print(SUMMARY_PATH)
    print(FIGURE_PATH)

    print("\nJoint VaR-ES comparison completed.")


if __name__ == "__main__":
    main()
