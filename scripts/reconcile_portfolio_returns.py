from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUR_FILE = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
QPM_FILE = ROOT / "data/external/qpm/upstream/backtest_results.csv"

STRATEGIES = [
    "ew_static",
    "ew_regime",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
]


ours = pd.read_csv(OUR_FILE, parse_dates=["Date"])
qpm = pd.read_csv(QPM_FILE, parse_dates=["date"])

ours = ours.rename(columns={"Date": "date"})

qpm = qpm[
    ["date"] + STRATEGIES
].copy()

merged = ours.merge(
    qpm,
    on="date",
    how="inner",
    suffixes=("_ours", "_qpm")
)

print("===== P&L RECONCILIATION =====")
print(f"Matched rows: {len(merged):,}")
print(
    f"Date range: "
    f"{merged['date'].min().date()} -> "
    f"{merged['date'].max().date()}"
)

print("\nMaximum absolute return difference:")

for strategy in STRATEGIES:
    diff = (
        merged[f"{strategy}_ours"]
        - merged[f"{strategy}_qpm"]
    ).abs()

    print(
        f"{strategy}: "
        f"max={diff.max():.12f}, "
        f"mean={diff.mean():.12f}"
    )
    