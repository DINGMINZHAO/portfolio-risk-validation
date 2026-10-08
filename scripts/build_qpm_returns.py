from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/raw/qpm/raw_adj_close_prices.csv"
PRICE_OUTPUT = ROOT / "data/interim/qpm/asset_prices_clean.csv"
RETURN_OUTPUT = ROOT / "data/interim/qpm/asset_returns_arithmetic.csv"

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]


prices = pd.read_csv(INPUT, parse_dates=["Date"])
prices = prices.sort_values("Date").reset_index(drop=True)

prices = prices[["Date"] + ASSETS]

# Keep only the common 8-asset sample.
prices = prices.dropna(subset=ASSETS).reset_index(drop=True)

returns = prices.copy()
returns[ASSETS] = prices[ASSETS].pct_change()

# First row has no previous price, so its return is undefined.
returns = returns.dropna(subset=ASSETS).reset_index(drop=True)

PRICE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

prices.to_csv(PRICE_OUTPUT, index=False)
returns.to_csv(RETURN_OUTPUT, index=False)

print("===== CLEAN PRICE PANEL =====")
print(f"Rows: {len(prices):,}")
print(f"Date range: {prices['Date'].min().date()} -> {prices['Date'].max().date()}")
print(f"Missing values: {prices[ASSETS].isna().sum().sum()}")

print("\n===== ARITHMETIC RETURNS =====")
print(f"Rows: {len(returns):,}")
print(f"Date range: {returns['Date'].min().date()} -> {returns['Date'].max().date()}")
print(f"Missing values: {returns[ASSETS].isna().sum().sum()}")

print("\nReturn ranges:")
for asset in ASSETS:
    print(
        f"{asset}: "
        f"{returns[asset].min():.4%} -> "
        f"{returns[asset].max():.4%}"
    )