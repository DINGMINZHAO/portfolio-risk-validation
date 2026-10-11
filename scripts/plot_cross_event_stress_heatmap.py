
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/qpm/cross_strategy_risk_overview.csv"
OUT = ROOT / "reports/figures/qpm/cross_strategy"

EVENTS = [
    ("2020", "COVID SHOCK", "crash_2020_return_pct"),
    ("2022", "RATES & INFLATION", "year_2022_return_pct"),
    ("2025", "TRADE SHOCK", "crash_2025_return_pct"),
]

ORDER = [
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]

NAMES = [
    "Equal Weight",
    "MinVar · Static",
    "MinVar · Regime",
    "MeanVar · Static",
    "MeanVar · Regime",
]

FAMILY = {
    "ew_static": "equal",
    "minvar_static": "minvar",
    "minvar_regime": "minvar",
    "meanvar_static": "meanvar",
    "meanvar_regime": "meanvar",
}

BG = "#FAFAF7"
INK = "#192733"
MUTED = "#65727A"
GUIDE = "#DCE3E4"

COLORS = {
    "equal": "#5E6B7B",
    "minvar": "#00AFC1",
    "meanvar": "#FF5C75",
}

UP = "#00A98F"
DOWN = "#F04E66"


def read_data():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing input CSV: {SOURCE}")

    df = pd.read_csv(SOURCE)

    needed = {"strategy", *(col for _, _, col in EVENTS)}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["strategy"].duplicated().any():
        raise ValueError("Duplicate strategy rows")

    df = df.set_index("strategy").reindex(ORDER)

    cols = [event[2] for event in EVENTS]
    values = df[cols].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite crisis returns")

    return df


def bare_axis(ax):
    ax.set_facecolor(BG)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        length=0,
        colors=MUTED,
        labelsize=9,
    )


def main():
    df = read_data()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(
        figsize=(15.8, 8.15),
        facecolor=BG,
    )

    fig.text(
        0.075,
        0.945,
        "THREE CRISES  /  FIVE PORTFOLIOS",
        fontsize=20,
        fontweight="bold",
        color=INK,
        va="center",
    )

    fig.text(
        0.075,
        0.893,
        "CUMULATIVE RETURN  ·  %",
        fontsize=9.4,
        fontweight="bold",
        color=MUTED,
    )

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color=COLORS["equal"],
            markerfacecolor=COLORS["equal"],
            markersize=9,
            label="Equal Weight",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color=COLORS["minvar"],
            markerfacecolor=COLORS["minvar"],
            markersize=9,
            label="MinVar",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color=COLORS["meanvar"],
            markerfacecolor=COLORS["meanvar"],
            markersize=9,
            label="MeanVar",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color=INK,
            markerfacecolor=INK,
            markersize=7,
            label="Static",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color=INK,
            markerfacecolor=BG,
            markeredgewidth=1.8,
            markersize=7,
            label="Regime",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.947, 0.902),
        frameon=False,
        ncol=5,
        fontsize=9.5,
        handletextpad=0.45,
        columnspacing=1.4,
    )

    lefts = [0.20, 0.464, 0.728]
    width = 0.226

    for k, (year, subtitle, column) in enumerate(EVENTS):
        left = lefts[k]

        # Main event panel
        ax = fig.add_axes([
            left, 0.389, width, 0.389
        ])

        bare_axis(ax)

        fig.text(
            left,
            0.814,
            year,
            color=INK,
            fontsize=24,
            fontweight="bold",
        )

        fig.text(
            left + 0.088,
            0.824,
            subtitle,
            color=MUTED,
            fontsize=9.2,
            fontweight="bold",
        )

        ax.set_xlim(-22.6, 2.0)
        ax.set_ylim(-0.7, 4.7)

        ax.set_xticks([-20, -15, -10, -5, 0])
        ax.set_xticklabels([
            "−20", "−15", "−10", "−5", "0"
        ])

        ax.set_yticks(range(5))

        for y in range(5):
            ax.hlines(
                y,
                -22.6,
                1.65,
                color=GUIDE,
                linewidth=0.7,
                zorder=0,
            )

        ax.axvline(
            0,
            color=INK,
            linewidth=0.8,
            alpha=0.35,
        )

        if k == 0:
            ax.set_yticklabels(
                NAMES[::-1],
                fontsize=10,
                color=INK,
            )
            ax.tick_params(axis="y", pad=13)
        else:
            ax.set_yticklabels([])

        for i, strategy in enumerate(ORDER):
            y = 4 - i

            ret = float(df.at[strategy, column])
            color = COLORS[FAMILY[strategy]]
            is_regime = strategy.endswith("_regime")

            ax.plot(
                [ret, 0],
                [y, y],
                linewidth=3,
                color=color,
                alpha=0.57,
                solid_capstyle="round",
                zorder=2,
            )

            ax.scatter(
                [ret],
                [y],
                s=160 if is_regime else 135,
                facecolor=BG if is_regime else color,
                edgecolor=color,
                linewidth=2.6 if is_regime else 1,
                zorder=4,
            )

            ax.text(
                ret + 0.68,
                y + 0.01,
                f"{ret:.1f}",
                ha="left",
                va="center",
                fontsize=10.2,
                color=INK,
                fontweight="bold",
                zorder=5,
                bbox={
                    "facecolor": BG,
                    "edgecolor": "none",
                    "pad": 0.35,
                },
            )

        # Regime minus Static panel
        ax_delta = fig.add_axes([
            left, 0.147, width, 0.123
        ])

        bare_axis(ax_delta)

        ax_delta.set_xlim(-0.59, 0.59)
        ax_delta.set_ylim(-0.52, 1.52)

        ax_delta.set_yticks([1, 0])

        if k == 0:
            ax_delta.set_yticklabels(
                ["MinVar", "MeanVar"],
                fontsize=9.5,
            )
            ax_delta.tick_params(axis="y", pad=13)
        else:
            ax_delta.set_yticklabels([])

        ax_delta.set_xticks([-0.5, 0, 0.5])
        ax_delta.set_xticklabels(
            ["−0.5", "0", "+0.5"],
            fontsize=8.5,
        )

        ax_delta.axvline(
            0,
            color=INK,
            linewidth=0.9,
            alpha=0.5,
        )

        for y, family in [(1, "minvar"), (0, "meanvar")]:
            change = float(
                df.at[f"{family}_regime", column]
                - df.at[f"{family}_static", column]
            )

            if change > 0.005:
                color = UP
            elif change < -0.005:
                color = DOWN
            else:
                color = MUTED

            ax_delta.plot(
                [0, change],
                [y, y],
                color=color,
                linewidth=4,
                solid_capstyle="round",
                zorder=2,
            )

            ax_delta.scatter(
                change,
                y,
                s=80,
                color=color,
                edgecolor=BG,
                linewidth=1,
                zorder=3,
            )

            label = (
                "0.00"
                if abs(change) < 0.005
                else f"{change:+.2f}"
            )

            offset = 0.055 if change >= 0 else -0.055

            ax_delta.text(
                change + offset,
                y,
                label,
                va="center",
                ha="left" if change >= 0 else "right",
                fontsize=9.4,
                color=color,
                fontweight="bold",
            )

    fig.text(
        0.075,
        0.302,
        "REGIME  −  STATIC",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )

    fig.text(
        0.075,
        0.280,
        "percentage points",
        fontsize=8.8,
        color=MUTED,
    )

    fig.text(
        0.075,
        0.065,
        "Event windows differ in length  ·  "
        "Compare strategies within each year",
        color=MUTED,
        fontsize=9.3,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = OUT / "cross_event_stress_heatmap.png"
    pdf = OUT / "cross_event_stress_heatmap.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.16,
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.16,
    )

    print(f"PNG: {png}")
    print(f"PDF: {pdf}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
