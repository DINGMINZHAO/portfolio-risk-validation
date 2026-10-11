from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/qpm/cross_strategy_risk_overview.csv"
OUTPUT = ROOT / "reports/figures/qpm/cross_strategy"

ORDER = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]

LABELS = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime",
}

FAMILY = {
    "ew_static": "ew",
    "minvar_static": "minvar",
    "minvar_regime": "minvar",
    "meanvar_static": "meanvar",
    "meanvar_regime": "meanvar",
}

# editorial / soft information-design palette
BG = "#f4f1eb"
TEXT = "#2f3437"
MUTED = "#6f767c"
GRID = "#d7d2ca"

COLORS = {
    "ew": "#7f8d98",       # neutral blue-grey
    "minvar": "#66b8a6",   # soft teal
    "meanvar": "#e48d6b",  # muted coral
}


def load_data():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing source file: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "strategy",
        "annualized_return_pct",
        "annualized_vol_pct",
        "full_sample_mdd_pct",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["strategy"].duplicated().any():
        raise ValueError("Duplicate strategy rows found.")

    df = df.set_index("strategy").reindex(ORDER)

    vals = df[
        ["annualized_return_pct", "annualized_vol_pct", "full_sample_mdd_pct"]
    ].to_numpy(dtype=float)

    if not np.isfinite(vals).all():
        raise ValueError("Non-finite metric values found.")

    return df


def setup_theme():
    sns.set_theme(
        style="whitegrid",
        rc={
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "grid.color": GRID,
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.edgecolor": BG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def base_axis(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=10, colors=MUTED)
    ax.set_axisbelow(True)


def make_scatter_panel(ax, df):
    base_axis(ax)

    ax.set_xlim(5.9, 13.95)
    ax.set_ylim(4.9, 10.15)

    ax.set_xticks([6, 8, 10, 12, 14])
    ax.set_yticks([5, 6, 7, 8, 9, 10])

    ax.grid(True, color=GRID, linewidth=0.9, alpha=0.70)

    ax.set_xlabel("Annualized volatility (%)", fontsize=11, labelpad=10, color=TEXT)
    ax.set_ylabel("Annualized return (%)", fontsize=11, labelpad=10, color=TEXT)

    # connect static and regime variants
    for family in ["minvar", "meanvar"]:
        static = df.loc[f"{family}_static"]
        regime = df.loc[f"{family}_regime"]

        ax.plot(
            [static["annualized_vol_pct"], regime["annualized_vol_pct"]],
            [static["annualized_return_pct"], regime["annualized_return_pct"]],
            color=COLORS[family],
            linewidth=1.8,
            alpha=0.45,
            zorder=1,
        )

    texts = []

    for strategy in ORDER:
        row = df.loc[strategy]
        family = FAMILY[strategy]
        color = COLORS[family]
        is_regime = strategy.endswith("_regime")

        ax.scatter(
            row["annualized_vol_pct"],
            row["annualized_return_pct"],
            s=185,
            facecolors=BG if is_regime else color,
            edgecolors=color,
            linewidths=2.2 if is_regime else 0.8,
            zorder=4,
        )

        texts.append(
            ax.text(
                row["annualized_vol_pct"],
                row["annualized_return_pct"],
                LABELS[strategy],
                fontsize=10,
                color=TEXT,
                zorder=5,
            )
        )

    adjust_text(
        texts,
        ax=ax,
        expand=(1.15, 1.22),
        force_text=(0.35, 0.45),
        force_static=(0.25, 0.35),
        arrowprops=dict(
            arrowstyle="-",
            color="#aaa49a",
            lw=0.7,
            alpha=0.65,
        ),
    )

    # style legend
    style_legend = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=TEXT,
            markeredgecolor=TEXT,
            markersize=7.5,
            label="Static",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=BG,
            markeredgecolor=TEXT,
            markeredgewidth=1.8,
            markersize=7.5,
            label="Regime-aware",
        ),
    ]

    ax.legend(
        handles=style_legend,
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        handletextpad=0.6,
        labelcolor=TEXT,
    )


def make_drawdown_panel(ax, df):
    base_axis(ax)

    strategies = ORDER
    y_positions = np.arange(len(strategies))[::-1]
    drawdowns = [abs(float(df.loc[s, "full_sample_mdd_pct"])) for s in strategies]

    ax.set_xlim(0, 41)
    ax.set_ylim(-0.5, len(strategies) - 0.5)

    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([LABELS[s] for s in strategies], fontsize=10, color=TEXT)

    ax.grid(True, axis="x", color=GRID, linewidth=0.9, alpha=0.70)
    ax.grid(False, axis="y")

    ax.set_xlabel("Maximum drawdown magnitude (%)", fontsize=11, labelpad=10, color=TEXT)

    for i, strategy in enumerate(strategies):
        y = y_positions[i]
        dd = drawdowns[i]
        family = FAMILY[strategy]
        color = COLORS[family]
        is_regime = strategy.endswith("_regime")

        ax.hlines(
            y=y,
            xmin=0,
            xmax=dd,
            color=color,
            linewidth=2.1,
            alpha=0.34,
            zorder=1,
        )

        ax.scatter(
            dd,
            y,
            s=150,
            facecolors=BG if is_regime else color,
            edgecolors=color,
            linewidths=2.1 if is_regime else 0.8,
            zorder=3,
        )

        ax.text(
            dd + 1.05,
            y,
            f"-{dd:.2f}%",
            va="center",
            ha="left",
            fontsize=10,
            color=TEXT,
        )


def main():
    setup_theme()
    df = load_data()

    fig = plt.figure(figsize=(14.4, 7.8), facecolor=BG)

    # title block
    fig.text(
        0.07, 0.95,
        "Cross-Strategy Risk–Return Profile",
        fontsize=22,
        fontweight="normal",
        color=TEXT,
        ha="left",
        va="top",
    )

    fig.text(
        0.07, 0.905,
        "Full-sample portfolio outcomes across Equal Weight, MinVar and MeanVar strategies",
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="top",
    )

    fig.text(
        0.07, 0.80,
        "A. Annualized Return vs. Volatility",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
        ha="left",
    )

    fig.text(
        0.60, 0.80,
        "B. Full-Sample Maximum Drawdown",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
        ha="left",
    )

    # axes
    ax1 = fig.add_axes([0.07, 0.22, 0.47, 0.50])
    ax2 = fig.add_axes([0.60, 0.22, 0.33, 0.50])

    make_scatter_panel(ax1, df)
    make_drawdown_panel(ax2, df)

    # footer notes
    fig.text(
        0.07, 0.12,
        "Return: geometric annualized   |   Volatility: 252-day annualized   |   Drawdown: full portfolio-return sample",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )

    fig.text(
        0.07, 0.085,
        "Static and Regime-aware variants within the same optimizer are linked in Panel A; comparisons are descriptive rather than causal.",
        fontsize=9.0,
        color=MUTED,
        ha="left",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    png_file = OUTPUT / "cross_strategy_risk_return.png"
    pdf_file = OUTPUT / "cross_strategy_risk_return.pdf"

    fig.savefig(png_file, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(pdf_file, bbox_inches="tight", pad_inches=0.18)

    plt.show()
    plt.close(fig)

    print("Redesigned figure generated with seaborn + adjustText.")
    print(f"PNG: {png_file}")
    print(f"PDF: {pdf_file}")


if __name__ == "__main__":
    main()