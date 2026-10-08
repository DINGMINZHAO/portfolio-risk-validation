from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RETURN_FILE = ROOT / "data/interim/qpm/asset_returns_arithmetic.csv"
WEIGHT_FILE = ROOT / "data/external/qpm/upstream/portfolio_weights.csv"

WEIGHT_OUTPUT = ROOT / "data/interim/qpm/daily_effective_weights_main.csv"
RETURN_OUTPUT = ROOT / "data/interim/qpm/portfolio_returns_main.csv"

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]

ANALYSIS_END = pd.Timestamp("2025-12-08")


returns = pd.read_csv(RETURN_FILE, parse_dates=["Date"])
weights = pd.read_csv(WEIGHT_FILE, parse_dates=["date"])

returns = returns.sort_values("Date").reset_index(drop=True)
weights = weights.sort_values(["strategy", "date"]).reset_index(drop=True)

first_weight_date = weights["date"].min()

returns = returns[
    (returns["Date"] > first_weight_date)
    & (returns["Date"] <= ANALYSIS_END)
].copy()

return_cols = {
    asset: f"{asset}_return"
    for asset in ASSETS
}

daily_returns = returns.rename(columns=return_cols)

all_effective_weights = []
all_portfolio_returns = []

for strategy in sorted(weights["strategy"].unique()):

    strategy_weights = weights[
        weights["strategy"] == strategy
    ].copy()

    weight_cols = {
        asset: f"{asset}_weight"
        for asset in ASSETS
    }

    strategy_weights = strategy_weights[
        ["date", "regime"] + ASSETS
    ].rename(
        columns={
            "date": "weight_date",
            **weight_cols
        }
    )

    aligned = pd.merge_asof(
        daily_returns.sort_values("Date"),
        strategy_weights.sort_values("weight_date"),
        left_on="Date",
        right_on="weight_date",
        direction="backward",
        allow_exact_matches=False
    )

    aligned["strategy"] = strategy

    aligned["portfolio_return"] = sum(
        aligned[f"{asset}_weight"]
        * aligned[f"{asset}_return"]
        for asset in ASSETS
    )

    effective_weights = aligned[
        ["Date", "strategy", "weight_date", "regime"]
        + [f"{asset}_weight" for asset in ASSETS]
    ].copy()

    portfolio_returns = aligned[
        ["Date", "strategy", "portfolio_return"]
    ].copy()

    all_effective_weights.append(effective_weights)
    all_portfolio_returns.append(portfolio_returns)


effective_weights = pd.concat(
    all_effective_weights,
    ignore_index=True
)

portfolio_returns_long = pd.concat(
    all_portfolio_returns,
    ignore_index=True
)

portfolio_returns = (
    portfolio_returns_long
    .pivot(
        index="Date",
        columns="strategy",
        values="portfolio_return"
    )
    .reset_index()
)

effective_weights.to_csv(
    WEIGHT_OUTPUT,
    index=False
)

portfolio_returns.to_csv(
    RETURN_OUTPUT,
    index=False
)


print("===== DAILY EFFECTIVE WEIGHTS =====")
print(f"Rows: {len(effective_weights):,}")
print(
    f"Date range: "
    f"{effective_weights['Date'].min().date()} -> "
    f"{effective_weights['Date'].max().date()}"
)
print(
    f"Strategies: "
    f"{effective_weights['strategy'].nunique()}"
)

timing_ok = (
    effective_weights["weight_date"]
    < effective_weights["Date"]
).all()

print(f"Strict timing rule satisfied: {timing_ok}")

weight_columns = [
    f"{asset}_weight"
    for asset in ASSETS
]

weight_sums = effective_weights[
    weight_columns
].sum(axis=1)

print(
    f"Daily weight sum range: "
    f"{weight_sums.min():.12f} -> "
    f"{weight_sums.max():.12f}"
)


print("\n===== PORTFOLIO RETURNS =====")
print(f"Rows: {len(portfolio_returns):,}")
print(
    f"Date range: "
    f"{portfolio_returns['Date'].min().date()} -> "
    f"{portfolio_returns['Date'].max().date()}"
)

print(
    f"Missing portfolio returns: "
    f"{portfolio_returns.drop(columns='Date').isna().sum().sum()}"
)

print("\nReturn ranges:")

for strategy in portfolio_returns.columns[1:]:
    print(
        f"{strategy}: "
        f"{portfolio_returns[strategy].min():.4%} -> "
        f"{portfolio_returns[strategy].max():.4%}"
    )

if {"ew_static", "ew_regime"}.issubset(portfolio_returns.columns):
    ew_difference = (
        portfolio_returns["ew_static"]
        - portfolio_returns["ew_regime"]
    ).abs().max()

    print(
        f"\nMax |ew_static - ew_regime|: "
        f"{ew_difference:.12f}"
    )