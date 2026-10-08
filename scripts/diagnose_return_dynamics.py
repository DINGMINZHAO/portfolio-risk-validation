from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


TRADING_DAYS = 252
ROLLING_WINDOW = 63
ACF_LAGS = 20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "qpm"
    / "portfolio_returns_main.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "qpm"
    / "return_diagnostics_summary.csv"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "qpm"
    / "return_diagnostics"
)


STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "Minimum Variance Static",
    "minvar_regime": "Minimum Variance Regime-aware",
    "meanvar_static": "Mean-Variance Static",
    "meanvar_regime": "Mean-Variance Regime-aware",
}


def validate_input(data):
    if "Date" not in data.columns:
        raise ValueError("Date column is missing.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate dates found.")

    missing = [
        strategy
        for strategy in STRATEGIES
        if strategy not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing strategy columns: {missing}")

    strategy_data = data[list(STRATEGIES.keys())]

    if strategy_data.isna().any().any():
        raise ValueError("Missing portfolio returns found.")

    if not np.isfinite(strategy_data.to_numpy()).all():
        raise ValueError("Non-finite portfolio returns found.")


def calculate_diagnostics(returns):
    rolling_vol = (
        returns
        .rolling(ROLLING_WINDOW)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    return {
        "n_obs": len(returns),
        "return_acf_lag1": returns.autocorr(lag=1),
        "return_acf_lag5": returns.autocorr(lag=5),
        "squared_return_acf_lag1": (returns ** 2).autocorr(lag=1),
        "squared_return_acf_lag5": (returns ** 2).autocorr(lag=5),
        "rolling_vol_min": rolling_vol.min(),
        "rolling_vol_median": rolling_vol.median(),
        "rolling_vol_max": rolling_vol.max(),
    }


def make_diagnostic_figure(
    dates,
    returns,
    strategy_name,
    output_path,
):
    rolling_vol = (
        returns
        .rolling(ROLLING_WINDOW)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    mean = returns.mean()
    std = returns.std(ddof=1)

    x_grid = np.linspace(
        returns.min(),
        returns.max(),
        500,
    )

    normal_density = stats.norm.pdf(
        x_grid,
        loc=mean,
        scale=std,
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
    )

    axes[0, 0].plot(
        dates,
        returns,
        linewidth=0.7,
    )
    axes[0, 0].axhline(
        0,
        linewidth=0.8,
    )
    axes[0, 0].set_title(
        "Daily portfolio returns"
    )
    axes[0, 0].set_ylabel(
        "Daily return"
    )

    axes[0, 1].plot(
        dates,
        rolling_vol,
        linewidth=1.0,
    )
    axes[0, 1].set_title(
        f"{ROLLING_WINDOW}-day rolling annualized volatility"
    )
    axes[0, 1].set_ylabel(
        "Annualized volatility"
    )

    axes[1, 0].hist(
        returns,
        bins=60,
        density=True,
        alpha=0.7,
    )
    axes[1, 0].plot(
        x_grid,
        normal_density,
        linewidth=2,
    )
    axes[1, 0].set_title(
        "Empirical distribution vs fitted normal"
    )
    axes[1, 0].set_xlabel(
        "Daily return"
    )
    axes[1, 0].set_ylabel(
        "Density"
    )

    stats.probplot(
        returns,
        dist="norm",
        plot=axes[1, 1],
    )
    axes[1, 1].set_title(
        "Normal Q-Q plot"
    )

    plot_acf(
        returns,
        lags=ACF_LAGS,
        zero=False,
        ax=axes[2, 0],
        title="ACF of daily returns",
    )

    plot_acf(
        returns ** 2,
        lags=ACF_LAGS,
        zero=False,
        ax=axes[2, 1],
        title="ACF of squared daily returns",
    )

    fig.suptitle(
        strategy_name,
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    validate_input(data)

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for strategy, strategy_name in STRATEGIES.items():
        returns = data[strategy].astype(float)

        result = calculate_diagnostics(returns)

        result["strategy"] = strategy
        result["strategy_name"] = strategy_name

        records.append(result)

        figure_path = (
            FIGURE_DIR
            / f"{strategy}_diagnostics.png"
        )

        make_diagnostic_figure(
            dates=data["Date"],
            returns=returns,
            strategy_name=strategy_name,
            output_path=figure_path,
        )

    summary = pd.DataFrame(records)

    column_order = [
        "strategy",
        "strategy_name",
        "n_obs",
        "return_acf_lag1",
        "return_acf_lag5",
        "squared_return_acf_lag1",
        "squared_return_acf_lag5",
        "rolling_vol_min",
        "rolling_vol_median",
        "rolling_vol_max",
    ]

    summary = summary[column_order]

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nRETURN DYNAMICS DIAGNOSTICS")
    print("=" * 110)

    display = summary.copy()

    for column in [
        "rolling_vol_min",
        "rolling_vol_median",
        "rolling_vol_max",
    ]:
        display[column] = display[column].map(
            lambda x: f"{x:.2%}"
        )

    for column in [
        "return_acf_lag1",
        "return_acf_lag5",
        "squared_return_acf_lag1",
        "squared_return_acf_lag5",
    ]:
        display[column] = display[column].map(
            lambda x: f"{x:.4f}"
        )

    print(
        display.to_string(index=False)
    )

    print(
        f"\nSaved summary to: {SUMMARY_PATH}"
    )

    print(
        f"Saved figures to: {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()