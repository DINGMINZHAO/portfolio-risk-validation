
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/figures/qpm"

STRATEGIES = [
    ("ew_static", "Equal Weight", "#8E97A6", "o"),
    ("minvar_static", "MinVar Static", "#2878D8", "o"),
    ("minvar_regime", "MinVar Regime", "#8754D6", "D"),
    ("meanvar_static", "MeanVar Static", "#F47A36", "o"),
    ("meanvar_regime", "MeanVar Regime", "#20AD78", "D"),
]

SCENARIOS = [
    ("equity", "A   EQUITY CRASH"),
    ("inflation", "B   INFLATION & RATES"),
    ("joint", "C   JOINT HISTORICAL WORST"),
]

SNAPSHOTS = ["calm", "stressed"]

MAY_REFERENCE = {
    "equity": [-13.75, -4.16, -4.24, -2.12, -2.91],
    "inflation": [-5.75, -3.00, -3.05, -4.06, -4.59],
    "joint": [-27.98, -16.35, -16.89, -15.74, -16.69],
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "DejaVu Sans"
    ],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def simplify(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def lookup_column(columns, names):
    mapping = {simplify(c): c for c in columns}
    return next(
        (mapping[n] for n in names if n in mapping),
        None,
    )


def normalize_strategy(value):
    s = simplify(value)

    if s in {
        "ew", "ew_static", "ew_regime",
        "equal_weight", "equalweight",
        "equal_weight_portfolio",
    }:
        return "ew_static"

    if "minvar" in s or "minimum_variance" in s:
        family = "minvar"
    elif "meanvar" in s or "mean_variance" in s:
        family = "meanvar"
    else:
        return None

    if "regime" in s or "adaptive" in s:
        kind = "regime"
    elif "static" in s:
        kind = "static"
    else:
        return None

    return f"{family}_{kind}"


def normalize_scenario(value):
    s = simplify(value)

    if (
        "joint" in s
        or "historical_worst" in s
        or "worst" in s
        or s == "c"
    ):
        return "joint"

    if (
        "inflation" in s
        or "rate" in s
        or s == "b"
    ):
        return "inflation"

    if (
        "equity" in s
        or "crash" in s
        or s == "a"
    ):
        return "equity"

    return None


def normalize_snapshot(value):
    s = simplify(value)

    if (
        "calm" in s
        or s in {
            "apr", "april", "april_1",
            "2025_04_01", "2025_03_31",
        }
    ):
        return "calm"

    if (
        "stressed" in s
        or "stress" in s
        or s in {
            "may", "may_1",
            "2025_05_01", "2025_04_30",
        }
    ):
        return "stressed"

    try:
        date = pd.Timestamp(value).normalize()

        if date in (
            pd.Timestamp("2025-04-01"),
            pd.Timestamp("2025-03-31"),
        ):
            return "calm"

        if date in (
            pd.Timestamp("2025-05-01"),
            pd.Timestamp("2025-04-30"),
        ):
            return "stressed"

    except (TypeError, ValueError, OverflowError):
        pass

    return None


def parse_result_csv(path):
    try:
        raw = pd.read_csv(path)
    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return None

    strategy = lookup_column(
        raw.columns,
        [
            "strategy", "portfolio",
            "portfolio_name", "strategy_name",
        ],
    )

    scenario = lookup_column(
        raw.columns,
        [
            "scenario", "scenario_name",
            "scenario_id", "shock_scenario",
            "stress_scenario",
        ],
    )

    snapshot = lookup_column(
        raw.columns,
        [
            "snapshot", "snapshot_date",
            "effective_date", "as_of_date",
            "weight_date", "market_state",
            "state", "regime", "date",
        ],
    )

    result = lookup_column(
        raw.columns,
        [
            "portfolio_return",
            "portfolio_stress_return",
            "portfolio_stressed_return",
            "stressed_return",
            "scenario_return",
            "stress_return",
            "scenario_pnl",
            "portfolio_pnl",
            "portfolio_loss",
            "scenario_loss",
            "stress_loss",
            "return_pct",
            "loss_pct",
            "portfolio_return_pct",
            "portfolio_loss_pct",
            "scenario_return_pct",
            "scenario_loss_pct",
            "return",
            "loss",
        ],
    )

    if strategy and scenario and snapshot and result:
        data = raw[
            [strategy, scenario, snapshot, result]
        ].copy()

        data.columns = [
            "strategy", "scenario", "snapshot", "value"
        ]

    elif strategy and snapshot:
        cols = [
            c for c in raw.columns
            if normalize_scenario(c) is not None
        ]

        if not cols:
            return None

        data = raw.melt(
            id_vars=[strategy, snapshot],
            value_vars=cols,
            var_name="scenario",
            value_name="value",
        )

        data = data.rename(columns={
            strategy: "strategy",
            snapshot: "snapshot",
        })

    elif scenario and snapshot:
        cols = [
            c for c in raw.columns
            if normalize_strategy(c) is not None
        ]

        if not cols:
            return None

        data = raw.melt(
            id_vars=[scenario, snapshot],
            value_vars=cols,
            var_name="strategy",
            value_name="value",
        )

        data = data.rename(columns={
            scenario: "scenario",
            snapshot: "snapshot",
        })

    elif strategy and scenario:
        cols = [
            c for c in raw.columns
            if normalize_snapshot(c) is not None
        ]

        if not cols:
            return None

        data = raw.melt(
            id_vars=[strategy, scenario],
            value_vars=cols,
            var_name="snapshot",
            value_name="value",
        )

        data = data.rename(columns={
            strategy: "strategy",
            scenario: "scenario",
        })

    else:
        return None

    data["strategy"] = data["strategy"].map(
        normalize_strategy
    )

    data["scenario"] = data["scenario"].map(
        normalize_scenario
    )

    data["snapshot"] = data["snapshot"].map(
        normalize_snapshot
    )

    data["value"] = pd.to_numeric(
        data["value"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "strategy", "scenario",
            "snapshot", "value",
        ]
    )

    return data[
        ["snapshot", "scenario", "strategy", "value"]
    ]


def read_stress_results():
    roots = [
        ROOT / "data/processed/qpm",
        ROOT / "data/interim/qpm",
        ROOT / "outputs",
        ROOT / "reports",
    ]

    candidates = sorted({
        p
        for folder in roots
        if folder.exists()
        for p in folder.rglob("*.csv")
        if any(
            word in p.stem.lower()
            for word in ("hypothetical", "scenario")
        )
    })

    expected = pd.MultiIndex.from_product(
        [
            SNAPSHOTS,
            [x[0] for x in SCENARIOS],
            [x[0] for x in STRATEGIES],
        ],
        names=["snapshot", "scenario", "strategy"],
    )

    valid = []

    for path in candidates:
        df = parse_result_csv(path)

        if df is None or df.empty:
            continue

        df = df.groupby(
            ["snapshot", "scenario", "strategy"],
            as_index=False,
        )["value"].mean()

        found = set(
            map(
                tuple,
                df[
                    ["snapshot", "scenario", "strategy"]
                ].to_numpy(),
            )
        )

        if len(df) != len(expected) or found != set(expected):
            continue

        keyed = df.set_index(
            ["snapshot", "scenario", "strategy"]
        )["value"].reindex(expected)

        if not np.isfinite(keyed.to_numpy()).all():
            continue

        reference = np.array([
            MAY_REFERENCE[scenario][i]
            for scenario, _ in SCENARIOS
            for i in range(5)
        ])

        observed = keyed.loc["stressed"].to_numpy()

        choices = [
            (
                float(np.max(
                    np.abs(scale * observed - reference)
                )),
                scale,
            )
            for scale in [1.0, 100.0, -1.0, -100.0]
        ]

        error, scale = min(choices)

        if error <= 0.08:
            valid.append((
                error,
                path,
                keyed * scale,
            ))

    if not valid:
        names = "\n".join(
            f"  - {p.relative_to(ROOT)}"
            for p in candidates
        ) or "  (none found)"

        raise FileNotFoundError(
            "No complete hypothetical stress results found.\n"
            "Expected: 5 strategies x 3 scenarios x "
            "2 snapshots = 30 results.\n"
            "CSV files inspected:\n"
            + names
            + "\nRun analyze_hypothetical_stress.py first "
            "if its outputs are missing. If it still fails, "
            "share the CSV filename and column names."
        )

    error, path, series = min(
        valid,
        key=lambda x: x[0],
    )

    print(
        f"Validated source: {path.relative_to(ROOT)}"
    )

    print(
        f"May snapshot maximum discrepancy: "
        f"{error:.3f} percentage points"
    )

    return series


def main():
    values = read_stress_results()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.2, 5.2),
    )

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.83,
        bottom=0.245,
        wspace=0.29,
    )

    for ax, (scenario, title) in zip(
        axes, SCENARIOS
    ):
        subset = values.xs(
            scenario,
            level="scenario",
        )

        numeric = subset.to_numpy()

        low = float(np.min(numeric))
        high = float(np.max(numeric))

        padding = max(
            (high - low) * 0.13,
            0.65,
        )

        ax.set_ylim(
            low - padding,
            high + padding,
        )

        ax.set_xlim(-0.17, 1.17)

        ax.set_xticks(
            [0, 1],
            ["CALM", "STRESSED"],
        )

        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5)
        )

        ax.grid(
            axis="y",
            color="#EDF0F4",
            linewidth=0.9,
        )

        ax.set_axisbelow(True)

        ax.axvspan(
            0.94,
            1.06,
            color="#F1F4F8",
            zorder=0,
        )

        for key, label, color, marker in STRATEGIES:
            y = [
                subset.loc[(snap, key)]
                for snap in SNAPSHOTS
            ]

            style = (
                "--" if key.endswith("_regime")
                else "-"
            )

            ax.plot(
                [0, 1],
                y,
                color=color,
                linewidth=(
                    2.2 if key == "ew_static" else 1.9
                ),
                linestyle=style,
                alpha=0.92,
                zorder=3,
            )

            ax.scatter(
                [0, 1],
                y,
                color=color,
                marker=marker,
                s=55,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        ax.spines["bottom"].set_color("#B8C1CD")

        ax.tick_params(
            axis="both",
            labelsize=9.5,
            colors="#526278",
            length=0,
            pad=7,
        )

        ax.set_title(
            title,
            loc="left",
            fontweight="semibold",
            color="#233345",
            pad=17,
        )

    axes[0].set_ylabel(
        "20-day scenario return (%)",
        color="#384B60",
        labelpad=11,
    )

    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle=(
                "--" if key.endswith("_regime")
                else "-"
            ),
            linewidth=2,
            markersize=6,
            label=label,
        )
        for key, label, color, marker in STRATEGIES
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.042),
        frameon=False,
        ncol=5,
        fontsize=9.4,
        handlelength=2.5,
        columnspacing=2,
    )

    fig.text(
        0.065,
        0.137,
        "Fixed weights  |  Calm: Apr 1, 2025  |  "
        "Stressed: May 1, 2025",
        color="#8794A5",
        fontsize=8.7,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for ext in ["png", "svg"]:
        path = (
            OUT / f"hypothetical_stress_snapshots.{ext}"
        )

        fig.savefig(
            path,
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.15,
        )

        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
