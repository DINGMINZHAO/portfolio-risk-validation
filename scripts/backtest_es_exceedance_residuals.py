
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/es_backtesting"

INPUT_PATH = DATA_DIR / "garch_gmm_minvar_static_daily.csv"

DAILY_OUTPUT = DATA_DIR / "es_exceedance_residuals_daily.csv"
SUMMARY_OUTPUT = DATA_DIR / "es_exceedance_residuals_summary.csv"
FIGURE_OUTPUT = FIGURE_DIR / "es_exceedance_residuals.png"

STRATEGY = "minvar_static"

MODELS = ["fhs", "gmm"]
LEVELS = [0.95, 0.99]

EXPECTED_DAYS = 2831

BLOCK_LENGTH = 21
BOOTSTRAP_REPLICATIONS = 2000
RANDOM_SEED = 42

ROLLING_WINDOW = 252


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
        raise ValueError("Unexpected sample length.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates.")

    if (
        data["Date"].min() != pd.Timestamp("2014-09-08")
        or data["Date"].max() != pd.Timestamp("2025-12-08")
    ):
        raise ValueError("Unexpected backtesting dates.")

    required = [
        "Date",
        "realized_return",
        "realized_loss",
        "sigma_hat",
    ]

    for model in MODELS:
        for level in LEVELS:
            suffix = int(round(level * 100))

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

    numeric_columns = [
        col for col in required if col != "Date"
    ]

    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite numerical data.")

    if not np.allclose(
        data["realized_loss"],
        -data["realized_return"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Return/loss mismatch.")

    if (data["sigma_hat"] <= 0).any():
        raise ValueError("Invalid conditional volatility.")

    for model in MODELS:
        for level in LEVELS:
            suffix = int(round(level * 100))

            var = data[f"{model}_var_{suffix}"]
            es = data[f"{model}_es_{suffix}"]

            if (es < var - 1e-10).any():
                raise ValueError(
                    f"ES below VaR: {model}, {suffix}"
                )

            expected_hit = (
                data["realized_loss"] > var
            ).astype(int)

            if not np.array_equal(
                expected_hit.to_numpy(),
                data[f"{model}_hit_{suffix}"].to_numpy(dtype=int),
            ):
                raise ValueError(
                    f"Exceedance mismatch: {model}, {suffix}"
                )

    return data


def construct_residuals(data):
    result = data.copy()

    loss = result["realized_loss"].to_numpy(
        dtype=float
    )

    for model in MODELS:
        for level in LEVELS:
            suffix = int(round(level * 100))
            alpha = 1.0 - level

            var = result[
                f"{model}_var_{suffix}"
            ].to_numpy(dtype=float)

            es = result[
                f"{model}_es_{suffix}"
            ].to_numpy(dtype=float)

            realized_excess = np.maximum(
                loss - var,
                0.0,
            )

            expected_excess = alpha * (
                es - var
            )

            residual = (
                realized_excess - expected_excess
            )

            result[
                f"{model}_realized_excess_{suffix}"
            ] = realized_excess

            result[
                f"{model}_expected_excess_{suffix}"
            ] = expected_excess

            result[
                f"{model}_es_residual_{suffix}"
            ] = residual

            result[
                f"{model}_es_residual_bp_{suffix}"
            ] = 10000 * residual

    return result


def generate_bootstrap_indices(n):
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
    )

    indices = indices.reshape(
        BOOTSTRAP_REPLICATIONS,
        -1,
    )

    return indices[:, :n]


def bootstrap_mean_ci(values, indices):
    samples = values[indices]

    bootstrap_means = samples.mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def build_summary(data):
    indices = generate_bootstrap_indices(
        len(data)
    )

    rows = []

    for level in LEVELS:
        suffix = int(round(level * 100))
        alpha = 1.0 - level

        for model in MODELS:
            realized = data[
                f"{model}_realized_excess_{suffix}"
            ].to_numpy(dtype=float)

            expected = data[
                f"{model}_expected_excess_{suffix}"
            ].to_numpy(dtype=float)

            residual = data[
                f"{model}_es_residual_{suffix}"
            ].to_numpy(dtype=float)

            hits = data[
                f"{model}_hit_{suffix}"
            ].to_numpy(dtype=int)

            var = data[
                f"{model}_var_{suffix}"
            ].to_numpy(dtype=float)

            es = data[
                f"{model}_es_{suffix}"
            ].to_numpy(dtype=float)

            loss = data[
                "realized_loss"
            ].to_numpy(dtype=float)

            lower, upper = bootstrap_mean_ci(
                residual,
                indices,
            )

            hit_mask = hits == 1

            rows.append({
                "strategy": STRATEGY,
                "model": model,
                "confidence_level": level,
                "forecast_days": len(data),
                "exceedances": int(hits.sum()),
                "observed_exceedance_rate": float(
                    hits.mean()
                ),
                "nominal_exceedance_rate": alpha,
                "mean_realized_excess_bp": (
                    10000 * realized.mean()
                ),
                "mean_expected_excess_bp": (
                    10000 * expected.mean()
                ),
                "mean_es_residual_bp": (
                    10000 * residual.mean()
                ),
                "bootstrap_ci_lower_bp": (
                    10000 * lower
                ),
                "bootstrap_ci_upper_bp": (
                    10000 * upper
                ),
                "ci_contains_zero": (
                    lower <= 0 <= upper
                ),
                "mean_loss_on_hit_days_pct": (
                    100 * loss[hit_mask].mean()
                    if hit_mask.any()
                    else np.nan
                ),
                "mean_es_on_hit_days_pct": (
                    100 * es[hit_mask].mean()
                    if hit_mask.any()
                    else np.nan
                ),
                "mean_var_on_hit_days_pct": (
                    100 * var[hit_mask].mean()
                    if hit_mask.any()
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def print_summary(summary):
    print("\nES EXCEEDANCE RESIDUAL BACKTEST")
    print("=" * 82)

    print(f"Portfolio: {STRATEGY}")
    print(f"Forecast days: {EXPECTED_DAYS}")
    print("Period: 2014-09-08 to 2025-12-08")
    print(
        f"Block bootstrap: "
        f"{BOOTSTRAP_REPLICATIONS} replications, "
        f"{BLOCK_LENGTH}-day blocks"
    )

    for level in LEVELS:
        suffix = int(round(level * 100))

        sample = summary.loc[
            summary["confidence_level"] == level
        ]

        print(f"\n{suffix}% EXPECTED SHORTFALL")
        print("-" * 82)

        columns = [
            "model",
            "exceedances",
            "mean_realized_excess_bp",
            "mean_expected_excess_bp",
            "mean_es_residual_bp",
            "bootstrap_ci_lower_bp",
            "bootstrap_ci_upper_bp",
            "ci_contains_zero",
        ]

        print(
            sample[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.5f}",
            )
        )

        print("\nLOSS SEVERITY ON EXCEEDANCE DAYS")

        columns = [
            "model",
            "mean_loss_on_hit_days_pct",
            "mean_var_on_hit_days_pct",
            "mean_es_on_hit_days_pct",
        ]

        print(
            sample[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )


def plot_results(data):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        sharex=True,
    )

    dates = data["Date"]

    for model, label in [
        ("fhs", "GARCH-FHS"),
        ("gmm", "GARCH-GMM"),
    ]:
        realized = (
            10000 * data[
                f"{model}_realized_excess_99"
            ]
        )

        expected = (
            10000 * data[
                f"{model}_expected_excess_99"
            ]
        )

        realized_rolling = realized.rolling(
            ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        ).mean()

        expected_rolling = expected.rolling(
            ROLLING_WINDOW,
            min_periods=ROLLING_WINDOW,
        ).mean()

        axes[0].plot(
            dates,
            realized_rolling,
            linewidth=1.5,
            label=f"{label}: realized",
        )

        axes[0].plot(
            dates,
            expected_rolling,
            linewidth=1.2,
            linestyle="--",
            label=f"{label}: forecast",
        )

        residual = (
            data[f"{model}_es_residual_bp_99"]
        )

        axes[1].plot(
            dates,
            residual.cumsum(),
            linewidth=1.6,
            label=label,
        )

    axes[0].set_title(
        "99% ES: Rolling 252-Day Mean Excess Loss",
        loc="left",
    )

    axes[0].set_ylabel(
        "Mean excess loss (bps/day)"
    )

    axes[0].legend(
        frameon=False,
        ncol=2,
        fontsize=9,
    )

    axes[1].axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    axes[1].set_title(
        "Cumulative ES Exceedance Residual",
        loc="left",
    )

    axes[1].set_ylabel(
        "Cumulative residual (bps)"
    )

    axes[1].set_xlabel(
        "Forecast date"
    )

    axes[1].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.suptitle(
        "Expected Shortfall Backtesting\n"
        "MinVar Static: GARCH-FHS vs. GARCH-GMM",
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
        FIGURE_OUTPUT,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    data = load_data()

    print("Input validation passed.")

    result = construct_residuals(data)

    summary = build_summary(result)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        DAILY_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    plot_results(result)
    print_summary(summary)

    print("\nSAVED OUTPUTS")
    print(DAILY_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(FIGURE_OUTPUT)

    print("\nES residual backtest completed.")


if __name__ == "__main__":
    main()
