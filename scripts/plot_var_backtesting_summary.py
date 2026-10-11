
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/qpm"
OUT = ROOT / "reports/figures/qpm"

MODELS = [
    ("rolling_normal", "Rolling Normal"),
    ("historical_simulation", "Historical Simulation"),
    ("garch_normal", "GARCH-Normal"),
    ("garch_student_t", "GARCH-Student-t"),
    ("garch_fhs", "GARCH-FHS"),
]

STRATEGIES = [
    ("ew_static", "Equal Weight", "#8E97A6", "o"),
    ("minvar_static", "MinVar Static", "#2878D8", "o"),
    ("minvar_regime", "MinVar Regime", "#8754D6", "D"),
    ("meanvar_static", "MeanVar Static", "#F47A36", "o"),
    ("meanvar_regime", "MeanVar Regime", "#20AD78", "D"),
]

LEVELS = (0.95, 0.99)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def load_and_check(path, extra):
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")

    data = pd.read_csv(path)
    keys = ["model", "strategy", "confidence_level"]
    required = keys + extra + ["n_days", "n_exceedances"]

    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing columns {missing}"
        )

    if data[keys].duplicated().any():
        raise ValueError(
            f"{path.name}: duplicate model/strategy/level combinations"
        )

    expected = {
        (m, s, level)
        for m, _ in MODELS
        for s, _, _, _ in STRATEGIES
        for level in LEVELS
    }

    found = set(
        data[keys].itertuples(index=False, name=None)
    )

    if found != expected:
        raise ValueError(
            f"{path.name}: expected 50 complete combinations"
        )

    if not data["n_days"].eq(3839).all():
        raise ValueError(
            f"{path.name}: sample size differs from 3,839 days"
        )

    values = data[extra].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{path.name}: nonfinite plotted values"
        )

    return data.set_index(keys).sort_index()


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#ACB6C4")

    ax.tick_params(
        axis="both",
        colors="#516074",
        width=0.7,
        length=3,
    )

    ax.tick_params(axis="y", length=0)

    ax.set_axisbelow(True)
    ax.xaxis.label.set_color("#304154")


def main():
    coverage = load_and_check(
        DATA / "var_backtest_summary.csv",
        ["observed_rate"],
    )

    tests = load_and_check(
        DATA / "var_backtest_statistical_tests.csv",
        ["independence_p_value"],
    )

    if not np.array_equal(
        coverage["n_exceedances"].values,
        tests["n_exceedances"].values,
    ):
        raise ValueError(
            "Exceedance counts disagree between files"
        )

    fig = plt.figure(figsize=(16.4, 6.5))

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.18),
        left=0.14,
        right=0.975,
        bottom=0.295,
        top=0.85,
        wspace=0.53,
    )

    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    model_ys = np.arange(len(MODELS))[::-1]

    strategy_offsets = np.linspace(
        0.20,
        -0.20,
        len(STRATEGIES),
    )

    # Panels A and B: VaR exceedance rates

    for ax, level, letter in zip(
        axs[:2],
        LEVELS,
        ("A", "B"),
    ):
        nominal = (1 - level) * 100

        ax.axvline(
            nominal,
            color="#6A7686",
            linestyle=(0, (4, 3)),
            linewidth=1.15,
            zorder=1,
        )

        for y in model_ys:
            ax.axhline(
                y,
                color="#EDF0F4",
                linewidth=0.9,
                zorder=0,
            )

        for (model, _), y in zip(MODELS, model_ys):
            for (
                strategy, _, color, marker
            ), offset in zip(STRATEGIES, strategy_offsets):

                rate = coverage.loc[
                    (model, strategy, level),
                    "observed_rate",
                ] * 100

                ax.scatter(
                    rate,
                    y + offset,
                    color=color,
                    marker=marker,
                    s=65,
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=3,
                )

        ax.set_yticks(model_ys)
        ax.set_ylim(-0.57, 4.57)

        if level == 0.95:
            ax.set_xlim(3.65, 6.75)
            ax.set_xticks([
                4, 4.5, 5, 5.5, 6, 6.5
            ])
        else:
            ax.set_xlim(0.65, 2.45)
            ax.set_xticks([
                0.75, 1, 1.25, 1.5,
                1.75, 2, 2.25
            ])

        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x:g}")
        )

        ax.set_xlabel(
            "Observed exceedance rate (%)",
            labelpad=10,
        )

        ax.set_title(
            f"{letter}   {int(level * 100)}% VaR COVERAGE",
            loc="left",
            fontweight="semibold",
            color="#233345",
            pad=18,
        )

        ax.text(
            nominal,
            4.62,
            f"{nominal:g}% nominal",
            ha="center",
            va="bottom",
            color="#657185",
            fontsize=8.5,
            clip_on=False,
        )

        style_ax(ax)

    axs[0].set_yticklabels(
        [label for _, label in MODELS],
        color="#38495D",
    )

    axs[1].set_yticklabels([])

    # Panel C: GARCH-FHS independence tests

    ax = axs[2]
    strategy_ys = np.arange(len(STRATEGIES))[::-1]

    for y in strategy_ys:
        ax.axhline(
            y,
            color="#EDF0F4",
            linewidth=0.9,
            zorder=0,
        )

    ax.axvline(
        0.05,
        color="#6A7686",
        linestyle=(0, (4, 3)),
        linewidth=1.15,
        zorder=1,
    )

    for (
        strategy, _, color, marker
    ), y in zip(STRATEGIES, strategy_ys):

        p95 = tests.loc[
            ("garch_fhs", strategy, 0.95),
            "independence_p_value",
        ]

        p99 = tests.loc[
            ("garch_fhs", strategy, 0.99),
            "independence_p_value",
        ]

        if not (0 < p95 <= 1 and 0 < p99 <= 1):
            raise ValueError(
                "Log p-value axis requires values in (0, 1]"
            )

        ax.plot(
            [p95, p99],
            [y + 0.13, y - 0.13],
            color="#BFC7D2",
            linewidth=1.15,
            zorder=2,
        )

        ax.scatter(
            p95,
            y + 0.13,
            marker=marker,
            s=73,
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
            zorder=4,
        )

        ax.scatter(
            p99,
            y - 0.13,
            marker=marker,
            s=73,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(5e-7, 0.3)

    ax.set_xticks([
        1e-6,
        1e-4,
        1e-2,
        5e-2,
    ])

    ax.set_xticklabels([
        r"$10^{-6}$",
        r"$10^{-4}$",
        r"$10^{-2}$",
        "0.05",
    ])

    ax.minorticks_off()

    ax.set_ylim(-0.57, 4.57)
    ax.set_yticks(strategy_ys)

    ax.set_yticklabels(
        [label for _, label, _, _ in STRATEGIES],
        color="#38495D",
    )

    ax.set_xlabel(
        "Christoffersen independence p-value (log scale)",
        labelpad=10,
    )

    ax.set_title(
        "C   GARCH-FHS INDEPENDENCE",
        loc="left",
        fontweight="semibold",
        color="#233345",
        pad=18,
    )

    ax.text(
        0.05,
        4.62,
        r"$\alpha = 0.05$",
        ha="center",
        va="bottom",
        color="#657185",
        fontsize=8.5,
        clip_on=False,
    )

    style_ax(ax)

    # Legend and sample information

    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=mark,
            markersize=8.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        for _, label, color, mark in STRATEGIES
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.028),
        ncol=5,
        frameon=False,
        fontsize=9.5,
        columnspacing=2.2,
        handletextpad=0.5,
    )

    fig.text(
        0.14,
        0.164,
        "Panel C: open markers = 95% VaR; "
        "filled markers = 99% VaR.",
        fontsize=9,
        color="#677386",
    )

    fig.text(
        0.14,
        0.127,
        "Common backtest: 3,839 trading days  |  "
        "Sep 2010 – Dec 2025",
        fontsize=8.5,
        color="#8190A0",
    )

    # Export

    OUT.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "svg"):
        path = OUT / f"var_backtesting_summary.{ext}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.18,
        )

        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
