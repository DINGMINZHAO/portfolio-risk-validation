
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
OUT = ROOT / "reports/figures/qpm"

STRATEGIES = [
    ("ew_static", "Equal Weight", "#8E97A6", "-"),
    ("minvar_static", "MinVar Static", "#2878D8", "-"),
    ("minvar_regime", "MinVar Regime", "#8754D6", (0, (5, 2.5))),
    ("meanvar_static", "MeanVar Static", "#F47A36", "-"),
    ("meanvar_regime", "MeanVar Regime", "#20AD78", (0, (5, 2.5))),
]

EVENTS = [
    {
        "title": "A  COVID CRASH",
        "base": "2020-02-19",
        "start": "2020-02-20",
        "end": "2020-03-23",
        "summary": "historical_stress_2020_crash_returns.csv",
        "column": "crash_return",
        "ticks": [
            "2020-02-19", "2020-02-28", "2020-03-09",
            "2020-03-17", "2020-03-23",
        ],
    },
    {
        "title": "B  INFLATION & RATES",
        "base": "2021-12-31",
        "start": "2022-01-03",
        "end": "2022-12-30",
        "summary": "historical_stress_2022_summary.csv",
        "column": "cumulative_return",
        "ticks": [
            "2022-01-03", "2022-04-01", "2022-07-01",
            "2022-10-03", "2022-12-30",
        ],
    },
    {
        "title": "C  APRIL TRADE SHOCK",
        "base": "2025-04-02",
        "start": "2025-04-03",
        "end": "2025-04-30",
        "summary": "historical_stress_2025_summary.csv",
        "column": "full_return",
        "ticks": [
            "2025-04-02", "2025-04-08", "2025-04-15",
            "2025-04-22", "2025-04-30",
        ],
    },
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def event_paths(data, event):
    base = pd.Timestamp(event["base"])
    start = pd.Timestamp(event["start"])
    end = pd.Timestamp(event["end"])

    strategies = [s[0] for s in STRATEGIES]

    for date in [base, start, end]:
        if date not in data.index:
            raise ValueError(f"Missing event date: {date.date()}")

    frame = data.loc[start:end, strategies]

    if frame.empty or frame.isna().any().any():
        raise ValueError(f"Missing returns: {event['title']}")

    values = frame.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(f"Invalid returns: {event['title']}")

    if (values <= -1).any():
        raise ValueError(f"Return <= -100%: {event['title']}")

    paths = 100 * ((1 + frame).cumprod() - 1)

    initial = pd.DataFrame(
        0.0,
        index=[base],
        columns=strategies,
    )

    paths = pd.concat([initial, paths])

    reference = (
        ROOT / "data/processed/qpm" / event["summary"]
    )

    if reference.exists():
        summary = pd.read_csv(reference).set_index("strategy")

        for strategy in strategies:
            expected = (
                100 * float(summary.loc[strategy, event["column"]])
            )

            actual = float(paths[strategy].iloc[-1])

            if not np.isclose(
                expected, actual, atol=1e-7, rtol=0
            ):
                raise ValueError(
                    f"Stress result mismatch: "
                    f"{event['title']}, {strategy}"
                )

    return paths


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    data = pd.read_csv(
        INPUT,
        parse_dates=["Date"],
    )

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates")

    data = data.sort_values("Date").set_index("Date")

    strategies = [s[0] for s in STRATEGIES]

    if not set(strategies).issubset(data.columns):
        raise ValueError("Missing portfolio return columns")

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.6, 4.95),
        dpi=180,
    )

    fig.subplots_adjust(
        left=0.068,
        right=0.985,
        bottom=0.23,
        top=0.84,
        wspace=0.28,
    )

    for ax, event in zip(axes, EVENTS):

        paths = event_paths(data, event)
        x = np.arange(len(paths))

        values = paths[strategies].to_numpy()

        low = float(np.min(values))
        high = float(np.max(values))

        span = max(high - low, 5.0)

        ax.set_ylim(
            low - 0.10 * span,
            high + 0.12 * span,
        )

        ax.set_xlim(x[0], x[-1])

        ax.axhline(
            0,
            color="#AEB8C4",
            linewidth=0.95,
            linestyle=(0, (3, 3)),
            zorder=0,
        )

        ax.grid(
            axis="y",
            color="#EEF1F5",
            linewidth=0.9,
        )

        ax.set_axisbelow(True)

        # Highlight the initial April 2025 crash

        if event["title"].startswith("C"):
            crash_end = paths.index.get_loc(
                pd.Timestamp("2025-04-08")
            )

            ax.axvspan(
                0,
                crash_end,
                color="#C5CCD6",
                alpha=0.18,
                zorder=0,
            )

        for strategy, label, color, style in STRATEGIES:

            if strategy == "ew_static":
                linewidth = 2.5
                zorder = 2
            else:
                linewidth = 1.9
                zorder = 3

            ax.plot(
                x,
                paths[strategy].to_numpy(),
                color=color,
                linewidth=linewidth,
                linestyle=style,
                alpha=0.97,
                solid_capstyle="round",
                zorder=zorder,
            )

        tick_positions = []
        tick_labels = []

        for tick in event["ticks"]:
            date = pd.Timestamp(tick)

            if date not in paths.index:
                raise ValueError(
                    f"Invalid tick date: {date.date()}"
                )

            tick_positions.append(
                paths.index.get_loc(date)
            )

            if event["title"].startswith("B"):
                tick_labels.append(date.strftime("%b"))
            else:
                tick_labels.append(date.strftime("%b %-d"))

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:g}")
        )

        ax.set_title(
            event["title"],
            loc="left",
            fontweight="semibold",
            color="#223346",
            pad=15,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#B7C1CC")

        ax.tick_params(
            axis="both",
            colors="#53647A",
            labelsize=9,
            length=0,
            pad=7,
        )

    axes[0].set_ylabel(
        "Cumulative return (%)",
        color="#34465B",
        labelpad=10,
    )

    handles = [
        Line2D(
            [0], [0],
            color=color,
            linewidth=2.4,
            linestyle=style,
            label=label,
        )
        for _, label, color, style in STRATEGIES
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.525, 0.045),
        fontsize=9.7,
        columnspacing=2.3,
        handlelength=2.6,
    )

    OUT.mkdir(parents=True, exist_ok=True)

    for ext in ("png", "svg"):
        path = OUT / f"historical_stress_paths.{ext}"

        fig.savefig(
            path,
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.16,
        )

        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
