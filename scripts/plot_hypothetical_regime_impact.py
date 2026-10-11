
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FILE = ROOT / "data/processed/qpm/hypothetical_stress_summary.csv"
WEIGHT_FILE = ROOT / "data/interim/qpm/daily_effective_weights_main.csv"

OUTPUT_DIR = ROOT / "reports/figures/qpm/hypothetical_stress"

SNAPSHOT = "2025-05-01"
WEIGHT_DATE = "2025-04-30"

SCENARIOS = {
    "A_equity_crash": "Equity Crash",
    "B_inflation_rate_shock": "Inflation & Rates",
    "C_joint_historical_worst": "Joint Worst-Case",
}

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime",
}

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]

COLORS = {
    "ew_static": "#969CA4",
    "minvar_static": "#326CA8",
    "minvar_regime": "#7CB5DC",
    "meanvar_static": "#C45652",
    "meanvar_regime": "#E5A19B",
}


def load_returns():
    df = pd.read_csv(SUMMARY_FILE)
    df = df[df["snapshot_date"] == SNAPSHOT].copy()

    expected = len(SCENARIOS) * len(STRATEGIES)

    if len(df) != expected:
        raise ValueError(
            f"Expected {expected} rows, found {len(df)}"
        )

    if df.duplicated(["scenario", "strategy"]).any():
        raise ValueError("Duplicate scenario-strategy rows")

    table = df.pivot(
        index="scenario",
        columns="strategy",
        values="portfolio_return",
    )

    table = table.reindex(
        index=list(SCENARIOS),
        columns=list(STRATEGIES),
    )

    if table.isna().any().any():
        raise ValueError("Missing scenario return values")

    return table.astype(float) * 100


def load_weight_changes():
    df = pd.read_csv(WEIGHT_FILE)

    df = df[
        (df["Date"] == SNAPSHOT)
        & (df["weight_date"] == WEIGHT_DATE)
    ].copy()

    weight_cols = [f"{asset}_weight" for asset in ASSETS]

    weights = {}

    for strategy in [
        "minvar_static",
        "minvar_regime",
        "meanvar_static",
        "meanvar_regime",
    ]:
        match = df[df["strategy"] == strategy]

        if len(match) != 1:
            raise ValueError(f"Invalid weights: {strategy}")

        values = match.iloc[0][weight_cols].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite weights: {strategy}")

        if not np.isclose(values.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Weights do not sum to 1: {strategy}")

        weights[strategy] = values

    changes = pd.DataFrame({
        "asset": ASSETS,
        "MinVar": (
            weights["minvar_regime"]
            - weights["minvar_static"]
        ) * 100,
        "MeanVar": (
            weights["meanvar_regime"]
            - weights["meanvar_static"]
        ) * 100,
    })

    changes["magnitude"] = (
        changes[["MinVar", "MeanVar"]].abs().max(axis=1)
    )

    changes = changes[changes["magnitude"] > 0.05]
    changes = changes.sort_values("magnitude")

    return changes


def main():
    returns = load_returns()
    weights = load_weight_changes()

    differences = pd.DataFrame({
        "MinVar": returns["minvar_regime"] - returns["minvar_static"],
        "MeanVar": returns["meanvar_regime"] - returns["meanvar_static"],
    })

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
    })

    fig = plt.figure(figsize=(15, 9.5))

    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.975,
        top=0.82,
        bottom=0.15,
        height_ratios=[1.08, 1],
        hspace=0.50,
        wspace=0.30,
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Unintended Effects of Regime-Dependent Portfolio Constraints",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )

    fig.text(
        0.075,
        0.935,
        "Hypothetical Stress Testing | "
        "2025-05-01 Stressed Snapshot",
        ha="left",
        fontsize=11,
        color="#666666",
    )

    # Panel A: Absolute portfolio scenario returns

    x = np.arange(len(returns))
    width = 0.15
    legend_handles = []

    for i, (strategy, label) in enumerate(STRATEGIES.items()):
        bars = ax1.bar(
            x + (i - 2) * width,
            returns[strategy].values,
            width=width * 0.92,
            color=COLORS[strategy],
            label=label,
        )

        legend_handles.append(bars)

    fig.legend(
        legend_handles,
        list(STRATEGIES.values()),
        loc="upper left",
        bbox_to_anchor=(0.075, 0.900),
        ncol=5,
        frameon=False,
        fontsize=10,
        columnspacing=1.8,
    )

    ax1.axhline(0, color="#333333", linewidth=1)

    ax1.set_xticks(x)
    ax1.set_xticklabels(list(SCENARIOS.values()))

    ax1.set_ylabel("Portfolio Return (%)")

    ax1.set_title(
        "A. Absolute Portfolio Losses",
        loc="left",
        pad=13,
    )

    ax1.set_ylim(
        returns.min().min() - 2.5,
        1.0,
    )

    ax1.grid(axis="y", linestyle="--", alpha=0.22)
    ax1.set_axisbelow(True)

    # Panel B: Regime-aware minus Static

    x2 = np.arange(len(differences))
    width2 = 0.34

    bars_min = ax2.bar(
        x2 - width2 / 2,
        differences["MinVar"],
        width=width2,
        color=COLORS["minvar_static"],
        label="MinVar",
    )

    bars_mean = ax2.bar(
        x2 + width2 / 2,
        differences["MeanVar"],
        width=width2,
        color=COLORS["meanvar_static"],
        label="MeanVar",
    )

    ax2.axhline(0, color="#333333", linewidth=1)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        list(SCENARIOS.values()),
        rotation=10,
    )

    ax2.set_ylabel("Regime - Static (pp)")

    ax2.set_title(
        "B. Incremental Effect of Regime Constraints",
        loc="left",
        pad=13,
    )

    ax2.set_ylim(
        differences.min().min() - 0.22,
        0.12,
    )

    ax2.grid(axis="y", linestyle="--", alpha=0.22)
    ax2.set_axisbelow(True)

    ax2.legend(
        frameon=False,
        loc="lower left",
        ncol=2,
    )

    for bars in [bars_min, bars_mean]:
        for bar in bars:
            value = bar.get_height()

            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                value - 0.035,
                f"{value:+.3f}",
                ha="center",
                va="top",
                fontsize=9,
            )

    # Panel C: Allocation changes

    y = np.arange(len(weights))
    height = 0.34

    bars_w_min = ax3.barh(
        y - height / 2,
        weights["MinVar"],
        height=height,
        color=COLORS["minvar_static"],
        label="MinVar",
    )

    bars_w_mean = ax3.barh(
        y + height / 2,
        weights["MeanVar"],
        height=height,
        color=COLORS["meanvar_static"],
        label="MeanVar",
    )

    ax3.axvline(0, color="#333333", linewidth=1)

    ax3.set_yticks(y)
    ax3.set_yticklabels(weights["asset"])

    ax3.set_xlabel("Weight Change (percentage points)")

    ax3.set_title(
        "C. Allocation Changes: Regime - Static",
        loc="left",
        pad=13,
    )

    ax3.grid(axis="x", linestyle="--", alpha=0.22)
    ax3.set_axisbelow(True)

    max_change = weights[["MinVar", "MeanVar"]].abs().max().max()

    ax3.set_xlim(
        -max_change - 2,
        max_change + 2,
    )

    ax3.legend(
        frameon=False,
        loc="lower right",
        ncol=2,
    )

    for bars in [bars_w_min, bars_w_mean]:
        for bar in bars:
            value = bar.get_width()

            if abs(value) < 0.10:
                continue

            ax3.text(
                value + (0.15 if value > 0 else -0.15),
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.2f}",
                ha="left" if value > 0 else "right",
                va="center",
                fontsize=9,
            )

    # Figure notes

    fig.text(
        0.5,
        0.080,
        "Fixed initial holdings | "
        "20-trading-day cumulative shocks | "
        "Portfolio weights dated 2025-04-30",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.text(
        0.5,
        0.050,
        "Negative differences in Panel B indicate "
        "higher scenario losses under Regime-aware constraints.",
        ha="center",
        fontsize=9,
        color="#777777",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / "hypothetical_regime_comparison.png"
    pdf_path = OUTPUT_DIR / "hypothetical_regime_comparison.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.show()

    print("Final hypothetical stress figure generated.")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
