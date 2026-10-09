
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/es_backtesting"

INPUT_PATH = DATA_DIR / "var_backtest_daily.csv"

DAILY_PATH = DATA_DIR / "es_all_models_daily.csv"
SUMMARY_PATH = DATA_DIR / "es_all_models_summary.csv"
FIGURE_PATH = FIGURE_DIR / "es_all_models_fz0_comparison.png"

MODELS = [
    "rolling_normal",
    "historical_simulation",
    "garch_normal",
    "garch_student_t",
    "garch_fhs",
]

STRATEGIES = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]

LEVELS = [0.95, 0.99]
EXPECTED_DAYS = 3839

MODEL_LABELS = [
    "Rolling Normal",
    "Historical Sim.",
    "GARCH-Normal",
    "GARCH-t",
    "GARCH-FHS",
]

STRATEGY_LABELS = [
    "Equal Weight",
    "MinVar Static",
    "MinVar Regime",
    "MeanVar Static",
    "MeanVar Regime",
]


def load_data():
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    keys = [
        "Date",
        "strategy",
        "model",
        "confidence_level",
    ]

    required = keys + [
        "realized_return",
        "realized_loss",
        "var_forecast",
        "es_forecast",
        "exceedance",
    ]

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[keys].isna().any().any():
        raise ValueError("Missing forecast keys.")

    if df.duplicated(keys).any():
        raise ValueError("Duplicate forecast keys.")

    if set(df["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected strategies.")

    if set(df["model"]) != set(MODELS):
        raise ValueError("Unexpected models.")

    if set(df["confidence_level"]) != set(LEVELS):
        raise ValueError("Unexpected confidence levels.")

    expected_rows = (
        EXPECTED_DAYS
        * len(MODELS)
        * len(STRATEGIES)
        * len(LEVELS)
    )

    if len(df) != expected_rows:
        raise ValueError("Unexpected total row count.")

    numeric = [
        "realized_return",
        "realized_loss",
        "var_forecast",
        "es_forecast",
        "exceedance",
    ]

    if not np.isfinite(
        df[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite inputs.")

    if not np.allclose(
        df["realized_loss"],
        -df["realized_return"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Loss/return mismatch.")

    if (df["es_forecast"] <= 0).any():
        raise ValueError(
            "FZ0 requires positive ES forecasts."
        )

    if (
        df["es_forecast"]
        < df["var_forecast"] - 1e-10
    ).any():
        raise ValueError("ES below VaR.")

    expected_hits = (
        df["realized_loss"] > df["var_forecast"]
    ).astype(int)

    if not np.array_equal(
        expected_hits.to_numpy(),
        df["exceedance"].to_numpy(dtype=int),
    ):
        raise ValueError("Exceedance mismatch.")

    dates = np.sort(df["Date"].unique())

    if len(dates) != EXPECTED_DAYS:
        raise ValueError("Unexpected date count.")

    for _, group in df.groupby(
        ["strategy", "model", "confidence_level"]
    ):
        group_dates = np.sort(
            group["Date"].to_numpy()
        )

        if not np.array_equal(group_dates, dates):
            raise ValueError("Inconsistent forecast dates.")

    loss_consistency = (
        df.groupby(["Date", "strategy"])["realized_loss"]
        .agg(["min", "max"])
    )

    if (
        loss_consistency["max"]
        - loss_consistency["min"]
        > 1e-12
    ).any():
        raise ValueError(
            "Realized losses differ across models."
        )

    return df.sort_values(keys).reset_index(drop=True)


def calculate_scores(data):
    df = data.copy()

    loss = df["realized_loss"].to_numpy(dtype=float)
    var = df["var_forecast"].to_numpy(dtype=float)
    es = df["es_forecast"].to_numpy(dtype=float)

    alpha = (
        1 - df["confidence_level"].to_numpy(dtype=float)
    )

    realized_excess = np.maximum(
        loss - var,
        0.0,
    )

    expected_excess = alpha * (es - var)

    df["realized_excess_bp"] = (
        10000 * realized_excess
    )

    df["expected_excess_bp"] = (
        10000 * expected_excess
    )

    df["es_residual_bp"] = (
        10000 * (realized_excess - expected_excess)
    )

    df["fz0_score"] = (
        realized_excess / (alpha * es)
        + var / es
        + np.log(es)
        - 1.0
    )

    if not np.isfinite(
        df[
            [
                "es_residual_bp",
                "fz0_score",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite score results.")

    return df


def add_fhs_reference(data):
    keys = [
        "Date",
        "strategy",
        "confidence_level",
    ]

    reference = data.loc[
        data["model"] == "garch_fhs",
        keys + ["fz0_score"],
    ].rename(
        columns={
            "fz0_score": "fhs_fz0_score",
        }
    )

    result = data.merge(
        reference,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    if result["fhs_fz0_score"].isna().any():
        raise ValueError("Missing FHS reference.")

    result["fz0_delta_vs_fhs"] = (
        result["fz0_score"]
        - result["fhs_fz0_score"]
    )

    return result


def build_summary(data):
    summary = (
        data.groupby(
            [
                "strategy",
                "model",
                "confidence_level",
            ],
            as_index=False,
        )
        .agg(
            trading_days=("Date", "size"),
            exceedances=("exceedance", "sum"),
            mean_realized_excess_bp=(
                "realized_excess_bp", "mean"
            ),
            mean_expected_excess_bp=(
                "expected_excess_bp", "mean"
            ),
            mean_es_residual_bp=(
                "es_residual_bp", "mean"
            ),
            mean_fz0_score=(
                "fz0_score", "mean"
            ),
            mean_fz0_delta_vs_fhs=(
                "fz0_delta_vs_fhs", "mean"
            ),
        )
    )

    if len(summary) != 50:
        raise ValueError("Expected 50 summary groups.")

    if not summary["trading_days"].eq(EXPECTED_DAYS).all():
        raise ValueError("Inconsistent sample lengths.")

    reference = summary.loc[
        summary["model"] == "garch_fhs"
    ]

    if not np.allclose(
        reference["mean_fz0_delta_vs_fhs"],
        0,
        atol=1e-12,
    ):
        raise ValueError("FHS reference scores not zero.")

    summary["exceedance_rate_pct"] = (
        100 * summary["exceedances"]
        / summary["trading_days"]
    )

    return summary


def print_results(summary):
    print("\nES BACKTESTING: FIVE MODELS, FIVE PORTFOLIOS")
    print("=" * 80)

    print("Trading days per group:", EXPECTED_DAYS)
    print("Models:", len(MODELS))
    print("Portfolios:", len(STRATEGIES))
    print("Summary groups:", len(summary))

    for level in LEVELS:
        suffix = int(round(level * 100))

        part = summary.loc[
            summary["confidence_level"] == level
        ].copy()

        print(f"\n{suffix}% ES: MEAN EXCEEDANCE RESIDUAL (BPS)")
        print("Positive means realized excess exceeds forecast.")

        pivot = part.pivot(
            index="strategy",
            columns="model",
            values="mean_es_residual_bp",
        ).reindex(
            index=STRATEGIES,
            columns=MODELS,
        )

        print(
            pivot.to_string(
                float_format=lambda x: f"{x:+.4f}",
            )
        )

        print(f"\n{suffix}% VaR-ES: MEAN FZ0 DIFFERENCE VS FHS")
        print("Negative means lower joint score than FHS.")

        pivot = part.pivot(
            index="strategy",
            columns="model",
            values="mean_fz0_delta_vs_fhs",
        ).reindex(
            index=STRATEGIES,
            columns=MODELS,
        )

        print(
            pivot.to_string(
                float_format=lambda x: f"{x:+.5f}",
            )
        )


def plot_results(summary):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    values_by_level = []

    for level in LEVELS:
        part = summary.loc[
            summary["confidence_level"] == level
        ]

        matrix = part.pivot(
            index="strategy",
            columns="model",
            values="mean_fz0_delta_vs_fhs",
        ).reindex(
            index=STRATEGIES,
            columns=MODELS,
        ).to_numpy(dtype=float)

        values_by_level.append(matrix)

    vmax = max(
        np.max(np.abs(matrix))
        for matrix in values_by_level
    )

    vmax = max(vmax, 1e-8)

    for ax, level, matrix in zip(
        axes,
        LEVELS,
        values_by_level,
    ):
        image = ax.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_xticks(range(len(MODELS)))
        ax.set_yticks(range(len(STRATEGIES)))

        ax.set_xticklabels(
            MODEL_LABELS,
            rotation=35,
            ha="right",
            fontsize=9,
        )

        ax.set_yticklabels(
            STRATEGY_LABELS,
            fontsize=10,
        )

        ax.set_title(
            f"{int(round(level * 100))}% Joint VaR-ES"
        )

        for i in range(len(STRATEGIES)):
            for j in range(len(MODELS)):
                value = matrix[i, j]

                ax.text(
                    j,
                    i,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white"
                        if abs(value) > 0.55 * vmax
                        else "black"
                    ),
                )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        shrink=0.82,
        pad=0.03,
    )

    colorbar.set_label(
        "Mean FZ0 score difference vs. GARCH-FHS"
    )

    fig.suptitle(
        "Joint VaR-ES Forecast Comparison\n"
        "Five Portfolios and Five Risk Models",
        fontsize=15,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.89,
        top=0.81,
        bottom=0.25,
        wspace=0.25,
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

    daily = calculate_scores(data)
    daily = add_fhs_reference(daily)

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

    plot_results(summary)
    print_results(summary)

    print("\nSAVED OUTPUTS")
    print(DAILY_PATH)
    print(SUMMARY_PATH)
    print(FIGURE_PATH)

    print("\nFive-model ES analysis completed.")


if __name__ == "__main__":
    main()
