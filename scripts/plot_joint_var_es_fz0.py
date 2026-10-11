
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/processed/qpm/es_all_models_summary.csv"
)

OUTPUT = ROOT / "reports/figures/qpm"

MODELS = [
    ("garch_fhs", "GARCH\nFHS"),
    ("garch_student_t", "GARCH\nStudent-t"),
    ("garch_normal", "GARCH\nNormal"),
    ("historical_simulation", "Historical\nSimulation"),
    ("rolling_normal", "Rolling\nNormal"),
]

STRATEGIES = [
    ("ew_static", "Equal Weight", "#8E97A6", "o"),
    ("minvar_static", "MinVar Static", "#2878D8", "o"),
    ("minvar_regime", "MinVar Regime", "#8754D6", "D"),
    ("meanvar_static", "MeanVar Static", "#F47A36", "o"),
    ("meanvar_regime", "MeanVar Regime", "#20AD78", "D"),
]

LEVELS = [0.95, 0.99]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "DejaVu Sans"
    ],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    required = [
        "strategy",
        "model",
        "confidence_level",
        "trading_days",
        "mean_fz0_score",
        "mean_fz0_delta_vs_fhs",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    keys = [
        "strategy",
        "model",
        "confidence_level",
    ]

    if df[keys].duplicated().any():
        raise ValueError(
            "Duplicate model/strategy/level combinations"
        )

    expected = set(product(
        [s[0] for s in STRATEGIES],
        [m[0] for m in MODELS],
        LEVELS,
    ))

    found = set(
        df[keys].itertuples(index=False, name=None)
    )

    if found != expected:
        raise ValueError(
            "Expected 50 complete combinations "
            "of models, strategies and confidence levels"
        )

    if not df["trading_days"].eq(3839).all():
        raise ValueError(
            "Unexpected backtest sample length"
        )

    numeric = df[
        ["mean_fz0_score", "mean_fz0_delta_vs_fhs"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError(
            "Nonfinite FZ0 scores detected"
        )

    baseline = df.loc[
        df["model"] == "garch_fhs",
        [
            "strategy",
            "confidence_level",
            "mean_fz0_score",
        ],
    ].rename(
        columns={"mean_fz0_score": "baseline_score"}
    )

    checked = df.merge(
        baseline,
        on=["strategy", "confidence_level"],
        how="left",
        validate="many_to_one",
    )

    calculated = (
        checked["mean_fz0_score"]
        - checked["baseline_score"]
    )

    if not np.allclose(
        calculated,
        checked["mean_fz0_delta_vs_fhs"],
        atol=1e-8,
        rtol=0,
    ):
        raise ValueError(
            "FZ0 differences do not match source scores"
        )

    return df.set_index(keys).sort_index()


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color("#B7C1CE")

    ax.tick_params(
        axis="both",
        colors="#526278",
        width=0.7,
        length=0,
        pad=7,
    )

    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        color="#E8EDF3",
        linewidth=0.85,
    )


def main():
    data = load_data()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.5),
        dpi=180,
    )

    fig.subplots_adjust(
        left=0.064,
        right=0.985,
        top=0.85,
        bottom=0.255,
        wspace=0.22,
    )

    strategy_keys = [s[0] for s in STRATEGIES]
    offsets = np.linspace(-0.23, 0.23, 5)
    x_positions = np.arange(len(MODELS))

    for panel, (ax, level) in enumerate(
        zip(axes, LEVELS)
    ):

        ax.axhspan(
            -0.02,
            0.012,
            color="#F0F3F7",
            zorder=0,
        )

        ax.axhline(
            0,
            color="#8794A5",
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=1,
        )

        all_values = []

        for x, (model, label) in enumerate(MODELS):

            values = np.array([
                data.loc[
                    (strategy, model, level),
                    "mean_fz0_delta_vs_fhs",
                ]
                for strategy in strategy_keys
            ], dtype=float)

            all_values.extend(values.tolist())

            # Range across the five portfolios
            ax.plot(
                [x, x],
                [values.min(), values.max()],
                color="#ADB6C2",
                linewidth=3.4,
                solid_capstyle="round",
                zorder=2,
            )

            # Median across portfolios
            median = float(np.median(values))

            ax.plot(
                [x - 0.12, x + 0.12],
                [median, median],
                color="#5A687A",
                linewidth=1.2,
                zorder=2,
            )

            # Individual portfolio observations
            for j, (
                strategy, name, color, marker
            ) in enumerate(STRATEGIES):

                ax.scatter(
                    x + offsets[j],
                    values[j],
                    s=62,
                    color=color,
                    marker=marker,
                    edgecolors="white",
                    linewidth=0.85,
                    zorder=4,
                )

        highest = max(all_values)

        if level == 0.95:
            tick_step = 0.05
            lower = -0.014
        else:
            tick_step = 0.15
            lower = -0.020

        upper = (
            np.ceil(
                (highest + tick_step * 0.25)
                / tick_step
            ) * tick_step
        )

        ax.set_ylim(
            lower,
            upper * 1.03,
        )

        ax.set_xlim(
            -0.5,
            len(MODELS) - 0.5,
        )

        ax.set_xticks(x_positions)

        ax.set_xticklabels([
            name for _, name in MODELS
        ])

        ax.yaxis.set_major_locator(
            MultipleLocator(tick_step)
        )

        letter = "A" if panel == 0 else "B"

        ax.set_title(
            f"{letter}   {int(level * 100)}% JOINT VaR–ES",
            loc="left",
            fontweight="semibold",
            color="#233345",
            fontsize=11,
            pad=17,
        )

        style_axis(ax)

    axes[0].set_ylabel(
        "Mean FZ0 score difference vs GARCH-FHS",
        color="#34465B",
        labelpad=11,
    )

    handles = [
        Line2D(
            [0],
            [0],
            linestyle="",
            marker=marker,
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=name,
        )
        for strategy, name, color, marker in STRATEGIES
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.065),
        ncol=5,
        frameon=False,
        fontsize=9.5,
        handletextpad=0.5,
        columnspacing=1.7,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for ext in ["png", "svg"]:
        path = (
            OUTPUT / f"joint_var_es_fz0_comparison.{ext}"
        )

        fig.savefig(
            path,
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.17,
        )

        print(f"Saved: {path}")

    plt.close(fig)

    print()
    print("Joint VaR-ES comparison completed.")
    print("Models:", len(MODELS))
    print("Portfolios:", len(STRATEGIES))
    print("Confidence levels:", LEVELS)
    print("Backtest days per comparison: 3839")


if __name__ == "__main__":
    main()
