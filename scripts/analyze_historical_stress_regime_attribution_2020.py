
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ASSET_FILE = ROOT / "data/interim/qpm/asset_returns_arithmetic.csv"
WEIGHT_FILE = ROOT / "data/interim/qpm/daily_effective_weights_main.csv"
PORTFOLIO_FILE = ROOT / "data/interim/qpm/portfolio_returns_main.csv"

OUTPUT = ROOT / "data/processed/qpm/historical_stress_2020_regime_attribution.csv"

START = pd.Timestamp("2020-02-20")
ADJUSTMENT = pd.Timestamp("2020-03-02")
END = pd.Timestamp("2020-03-23")

ASSETS = [
    "SPY", "EFA", "EEM", "TLT",
    "IEF", "GLD", "VNQ", "DBC"
]

PAIRS = [
    ("minvar_static", "minvar_regime"),
    ("meanvar_static", "meanvar_regime")
]

STRATEGIES = [s for pair in PAIRS for s in pair]
WEIGHT_COLS = [f"{asset}_weight" for asset in ASSETS]


assets = pd.read_csv(ASSET_FILE, parse_dates=["Date"])
weights = pd.read_csv(
    WEIGHT_FILE,
    parse_dates=["Date", "weight_date"]
)
portfolios = pd.read_csv(PORTFOLIO_FILE, parse_dates=["Date"])

for name, frame in [
    ("Asset returns", assets),
    ("Portfolio returns", portfolios)
]:
    if frame["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates in {name}")

if weights.duplicated(["Date", "strategy"]).any():
    raise ValueError("Duplicate effective weights")

assets = assets.set_index("Date").sort_index()
portfolios = portfolios.set_index("Date").sort_index()

dates = portfolios.loc[START:END].index

if len(dates) != 23 or dates[0] != START or dates[-1] != END:
    raise ValueError("Unexpected crash window")

if not dates.isin(assets.index).all():
    raise ValueError("Missing asset return dates")

asset_matrix = assets.loc[dates, ASSETS].to_numpy(dtype=float)

if not np.isfinite(asset_matrix).all():
    raise ValueError("Invalid asset returns")

effective_weights = {}
portfolio_returns = {}

for strategy in STRATEGIES:

    selected = weights[
        weights["strategy"] == strategy
    ].set_index("Date").reindex(dates)

    if selected[WEIGHT_COLS + ["weight_date"]].isna().any().any():
        raise ValueError(f"Missing weights: {strategy}")

    if not (
        selected["weight_date"].to_numpy()
        < dates.to_numpy()
    ).all():
        raise ValueError(f"Weight timing violation: {strategy}")

    w = selected[WEIGHT_COLS].to_numpy(dtype=float)

    if not np.isfinite(w).all():
        raise ValueError(f"Invalid weights: {strategy}")

    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"Invalid weight sums: {strategy}")

    r = (w * asset_matrix).sum(axis=1)

    reference = portfolios.loc[
        dates, strategy
    ].to_numpy(dtype=float)

    if not np.allclose(r, reference, rtol=0, atol=1e-10):
        raise ValueError(f"Return mismatch: {strategy}")

    effective_weights[strategy] = w
    portfolio_returns[strategy] = r


pre_mask = dates < ADJUSTMENT
post_mask = dates >= ADJUSTMENT

post_assets = asset_matrix[post_mask]

records = []

print("===== COVID-19 REGIME ADJUSTMENT ATTRIBUTION =====")
print(f"Full window: {START.date()} -> {END.date()}")
print(f"Post-adjustment window: {ADJUSTMENT.date()} -> {END.date()}")
print(f"Pre-adjustment days: {pre_mask.sum()}")
print(f"Post-adjustment days: {post_mask.sum()}")


for static, regime in PAIRS:

    static_w = effective_weights[static]
    regime_w = effective_weights[regime]

    static_r = portfolio_returns[static]
    regime_r = portfolio_returns[regime]

    if not np.allclose(
        static_w[pre_mask],
        regime_w[pre_mask],
        rtol=0,
        atol=1e-10
    ):
        raise ValueError(f"Pre-adjustment weights differ: {static}")

    pre_wealth = np.prod(1.0 + static_r[pre_mask])

    s = static_r[post_mask]
    r = regime_r[post_mask]

    delta_w = regime_w[post_mask] - static_w[post_mask]

    daily_active = (delta_w * post_assets).sum(axis=1)

    if not np.allclose(
        daily_active,
        r - s,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError("Daily active return mismatch")

    static_post_return = np.prod(1.0 + s) - 1.0
    regime_post_return = np.prod(1.0 + r) - 1.0

    static_full_return = np.prod(1.0 + static_r) - 1.0
    regime_full_return = np.prod(1.0 + regime_r) - 1.0

    regime_wealth_before = np.concatenate([
        [1.0],
        np.cumprod(1.0 + r[:-1])
    ])

    static_wealth_after = np.cumprod(1.0 + s)

    future_static_growth = (
        static_wealth_after[-1] / static_wealth_after
    )

    compounding_factor = (
        regime_wealth_before * future_static_growth
    )

    active_contributions = (
        delta_w
        * post_assets
        * compounding_factor[:, None]
    ).sum(axis=0)

    full_contributions = active_contributions * pre_wealth

    post_difference = regime_post_return - static_post_return
    full_difference = regime_full_return - static_full_return

    if not np.isclose(
        active_contributions.sum(),
        post_difference,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError("Post-adjustment attribution mismatch")

    if not np.isclose(
        full_contributions.sum(),
        full_difference,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError("Full-window attribution mismatch")

    asset_post_returns = (
        np.prod(1.0 + post_assets, axis=0) - 1.0
    )

    comparison = pd.DataFrame({
        "ETF": ASSETS,
        "Weight_Shift_pp": delta_w[0] * 100,
        "ETF_Return_pct": asset_post_returns * 100,
        "Active_Contribution_pp": full_contributions * 100
    })

    comparison = comparison.sort_values(
        "Active_Contribution_pp",
        ascending=True
    )

    comparison["Comparison"] = f"{regime} - {static}"

    records.append(comparison)

    print(f"\n===== {regime} vs {static} =====")

    print(f"Pre-adjustment wealth: {pre_wealth:.6f}")

    print("\nPost-adjustment cumulative returns:")
    print(f"Static: {static_post_return:+.4%}")
    print(f"Regime: {regime_post_return:+.4%}")
    print(f"Difference: {post_difference * 100:+.4f} pp")

    print("\nFull crash cumulative returns:")
    print(f"Static: {static_full_return:+.4%}")
    print(f"Regime: {regime_full_return:+.4%}")
    print(f"Difference: {full_difference * 100:+.4f} pp")

    print("\nETF Attribution:")
    print(
        comparison[
            [
                "ETF",
                "Weight_Shift_pp",
                "ETF_Return_pct",
                "Active_Contribution_pp"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:+.4f}"
        )
    )

    print(
        f"\nAttribution total: "
        f"{full_contributions.sum() * 100:+.4f} pp"
    )

    print("Reconciliation: PASS")


results = pd.concat(records, ignore_index=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print(f"\nSaved to: {OUTPUT}")
