from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PRICE_FILE = ROOT / "data/raw/qpm/raw_adj_close_prices.csv"
WEIGHT_FILE = ROOT / "data/external/qpm/upstream/portfolio_weights.csv"

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]


prices = pd.read_csv(PRICE_FILE, parse_dates=["Date"])
prices = prices.sort_values("Date").reset_index(drop=True)

print("===== PRICE AUDIT =====")
print(f"Rows: {len(prices):,}")
print(f"Date range: {prices['Date'].min().date()} -> {prices['Date'].max().date()}")
print(f"Duplicate dates: {prices['Date'].duplicated().sum()}")

print("\nMissing values:")
print(prices[ASSETS].isna().sum())

print("\nFirst valid date:")
for asset in ASSETS:
    first_date = prices.loc[prices[asset].notna(), "Date"].min()
    print(f"{asset}: {first_date.date()}")

print("\nNon-positive prices:")
print((prices[ASSETS] <= 0).sum())

common = prices.dropna(subset=ASSETS)

print(
    f"\nCommon 8-asset sample: "
    f"{common['Date'].min().date()} -> {common['Date'].max().date()} "
    f"({len(common):,} rows)"
)


weights = pd.read_csv(WEIGHT_FILE, parse_dates=["date"])
weights = weights.sort_values(["strategy", "date"]).reset_index(drop=True)

print("\n===== WEIGHT AUDIT =====")
print(f"Rows: {len(weights):,}")
print(f"Strategies: {weights['strategy'].nunique()}")

print("\nStrategy counts:")
print(weights.groupby("strategy").size())

print(
    f"\nDuplicate strategy-date rows: "
    f"{weights.duplicated(['strategy', 'date']).sum()}"
)

print("\nMissing weights:")
print(weights[ASSETS].isna().sum())

weight_sums = weights[ASSETS].sum(axis=1)

print(
    f"\nWeight sum range: "
    f"{weight_sums.min():.12f} -> {weight_sums.max():.12f}"
)

print(
    f"Minimum individual weight: "
    f"{weights[ASSETS].min().min():.12f}"
)

print(
    f"Maximum individual weight: "
    f"{weights[ASSETS].max().max():.12f}"
)

print("\nStrategy date ranges:")
print(
    weights.groupby("strategy")["date"]
    .agg(["min", "max"])
)