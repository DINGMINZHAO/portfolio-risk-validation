
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/interim/qpm/portfolio_returns_main.csv"

OUTPUT = (
    ROOT
    / "reports/figures/qpm/historical_stress/"
    "inflation_2022_wealth.png"
)

BASE_DATE = pd.Timestamp("2021-12-31")
START = pd.Timestamp("2022-01-03")
END = pd.Timestamp("2022-12-30")

STRATEGIES = [
    "meanvar_static",
    "meanvar_regime",
    "minvar_static",
    "minvar_regime",
    "ew_static"
]

STYLES = {
    "meanvar_static": ("MeanVar Static", "#C3514D", "-"),
    "meanvar_regime": ("MeanVar Regime", "#C3514D", "--"),
    "minvar_static": ("MinVar Static", "#3478A4", "-"),
    "minvar_regime": ("MinVar Regime", "#3478A4", "--"),
    "ew_static": ("Equal Weight", "#666666", "-")
}


data = pd.read_csv(INPUT, parse_dates=["Date"])
data = data.sort_values("Date").reset_index(drop=True)

if data["Date"].duplicated().any():
    raise ValueError("Duplicate dates found")

if BASE_DATE not in data["Date"].values:
    raise ValueError("Baseline date missing")

missing = [s for s in STRATEGIES if s not in data.columns]

if missing:
    raise ValueError(f"Missing strategies: {missing}")

period = data[data["Date"].between(START, END)].copy()

if len(period) != 251:
    raise ValueError(f"Unexpected trading days: {len(period)}")

if period["Date"].iloc[0] != START:
    raise ValueError("Missing start date")

if period["Date"].iloc[-1] != END:
    raise ValueError("Missing end date")

returns = period.set_index("Date")[STRATEGIES].astype(float)

if not np.isfinite(returns.to_numpy()).all():
    raise ValueError("Invalid returns")

if (returns < -1).any().any():
    raise ValueError("Return below -100%")


wealth = (1.0 + returns).cumprod()

baseline = pd.DataFrame(
    1.0,
    index=pd.DatetimeIndex([BASE_DATE]),
    columns=STRATEGIES
)

wealth = pd.concat([baseline, wealth]).sort_index()

full_returns = wealth.iloc[-1] - 1.0


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300
})

fig, ax = plt.subplots(figsize=(12.5, 6.3))

fig.patch.set_facecolor("white")
ax.set_facecolor("white")


for strategy in STRATEGIES:

    label, color, linestyle = STYLES[strategy]

    ax.plot(
        wealth.index,
        wealth[strategy],
        label=f"{label} ({full_returns[strategy]:+.2%})",
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        alpha=0.95
    )


ax.axhline(
    1.0,
    color="#444444",
    linestyle=":",
    linewidth=1.1,
    alpha=0.7
)

ax.set_title(
    "2022 Inflation & Rate-Hike Stress: Portfolio Wealth Dynamics",
    loc="left",
    fontweight="bold",
    pad=18
)

ax.set_ylabel("Normalized Portfolio Wealth")
ax.set_xlabel("Date")

ax.set_xlim(BASE_DATE, END)

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=2)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#BBBBBB")
ax.spines["bottom"].set_color("#BBBBBB")

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=False,
    title="Portfolio (Full Return)"
)

fig.subplots_adjust(
    left=0.09,
    right=0.73,
    bottom=0.13,
    top=0.88
)


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close(fig)


print("===== 2022 PORTFOLIO WEALTH =====")

print(f"Window: {START.date()} -> {END.date()}")
print(f"Trading days: {len(period)}")

for strategy in STRATEGIES:
    print(
        f"{strategy:<20}"
        f"Wealth: {wealth[strategy].iloc[-1]:.6f}  "
        f"Return: {full_returns[strategy]:+.3%}"
    )

print(f"\nFigure saved to: {OUTPUT}")
