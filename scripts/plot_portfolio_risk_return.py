
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/interim/qpm/baseline_risk_profile.csv"
OUTPUT = ROOT / "reports/figures/qpm"

STRATEGIES = [
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

COLORS = {
    "ew_static": "#8E97A6",
    "minvar_static": "#2878D8",
    "minvar_regime": "#8754D6",
    "meanvar_static": "#F47A36",
    "meanvar_regime": "#20AD78",
}

MARKERS = {
    "ew_static": "o",
    "minvar_static": "o",
    "minvar_regime": "D",
    "meanvar_static": "o",
    "meanvar_regime": "D",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "Arimo", "DejaVu Sans"
    ],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT).set_index("strategy")

    missing = set(STRATEGIES) - set(df.index)
    if missing:
        raise ValueError(f"Missing strategies: {sorted(missing)}")

    df = df.loc[STRATEGIES].copy()

    for col in [
        "annualized_return",
        "annualized_vol",
        "max_drawdown",
    ]:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid values in {col}")

    df["ret"] = 100 * df["annualized_return"]
    df["vol"] = 100 * df["annualized_vol"]
    df["dd"] = -100 * df["max_drawdown"]

    if (df["vol"] <= 0).any() or (df["dd"] < 0).any():
        raise ValueError("Unexpected volatility or drawdown values")

    fig = plt.figure(figsize=(11.7, 5.1), dpi=150)

    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[1.40, 1.0],
        wspace=0.50,
        left=0.08,
        right=0.97,
        top=0.87,
        bottom=0.18,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel A: Return versus volatility

    for a, b in [
        ("minvar_static", "minvar_regime"),
        ("meanvar_static", "meanvar_regime"),
    ]:
        ax.plot(
            df.loc[[a, b], "vol"],
            df.loc[[a, b], "ret"],
            color="#A9B0BC",
            lw=1.45,
            zorder=2,
        )

    for strat in STRATEGIES:
        row = df.loc[strat]

        ax.scatter(
            row["vol"],
            row["ret"],
            s=102 if "regime" in strat else 98,
            color=COLORS[strat],
            marker=MARKERS[strat],
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )

    ax.annotate(
        "Equal Weight",
        (
            df.loc["ew_static", "vol"],
            df.loc["ew_static", "ret"],
        ),
        xytext=(-4, 13),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#687384",
    )

    ax.text(
        6.73, 5.34, "MinVar",
        fontsize=9.2,
        color="#586473",
    )

    ax.text(
        8.55, 9.48, "MeanVar",
        fontsize=9.2,
        color="#586473",
    )

    ax.set(
        xlim=(6.15, 13.40),
        ylim=(5.10, 9.82),
        xlabel="Annualized volatility (%)",
        ylabel="Annualized return (%)",
    )

    ax.set_xticks([7, 8, 9, 10, 11, 12, 13])
    ax.set_yticks([5.5, 6.5, 7.5, 8.5, 9.5])

    ax.grid(
        True,
        axis="both",
        color="#EDF0F4",
        lw=0.8,
    )
    ax.set_axisbelow(True)

    ax.set_title(
        "A   RETURN–VOLATILITY",
        loc="left",
        pad=19,
        weight="semibold",
        color="#233345",
        fontsize=10.5,
    )

    # Panel B: Maximum drawdown

    y = np.arange(len(STRATEGIES))[::-1]

    for yi, strat in zip(y, STRATEGIES):
        row = df.loc[strat]

        ax2.axhline(
            yi,
            color="#EEF1F5",
            lw=0.8,
            zorder=0,
        )

        ax2.scatter(
            row["dd"],
            yi,
            s=102 if "regime" in strat else 98,
            marker=MARKERS[strat],
            c=COLORS[strat],
            zorder=5,
            edgecolors="white",
            linewidth=1.2,
        )

        ax2.annotate(
            f"{row['dd']:.1f}",
            (row["dd"], yi),
            xytext=(9, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="#536071",
        )

    ax2.set(
        xlim=(15.5, 42),
        ylim=(-0.65, 4.65),
        xlabel="Maximum drawdown magnitude (%)",
    )

    ax2.set_xticks([20, 25, 30, 35, 40])
    ax2.set_yticks(y)
    ax2.set_yticklabels([LABELS[s] for s in STRATEGIES])

    ax2.tick_params(axis="y", length=0, pad=8)
    ax2.tick_params(axis="x", pad=7)

    ax2.set_title(
        "B   MAXIMUM DRAWDOWN",
        loc="left",
        pad=19,
        weight="semibold",
        color="#233345",
        fontsize=10.5,
    )

    ax2.spines["left"].set_visible(False)

    # Shared visual formatting

    for axis in [ax, ax2]:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        axis.spines["bottom"].set_color("#ABB4C1")
        axis.spines["left"].set_color("#ABB4C1")

        axis.tick_params(
            axis="both",
            colors="#536071",
            width=0.7,
            length=3,
        )

        axis.xaxis.label.set_color("#354356")
        axis.yaxis.label.set_color("#354356")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT / "portfolio_risk_return.png",
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.22,
    )

    fig.savefig(
        OUTPUT / "portfolio_risk_return.svg",
        bbox_inches="tight",
        pad_inches=0.22,
    )

    plt.close(fig)

    print("Saved:", OUTPUT / "portfolio_risk_return.png")
    print("Saved:", OUTPUT / "portfolio_risk_return.svg")


if __name__ == "__main__":
    main()
