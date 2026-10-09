
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/exceedance_dynamics"

BACKTEST_PATH = DATA_DIR / "var_backtest_daily.csv"
FORECAST_PATH = DATA_DIR / "garch_fhs_var_es_forecasts.csv"

OUTPUT_CSV = DATA_DIR / "garch_fhs_april_2025_minvar_static.csv"
OUTPUT_FIGURE = FIGURE_DIR / "garch_fhs_april_2025_minvar_static.png"

MODEL = "garch_fhs"
STRATEGY = "minvar_static"
CONFIDENCE = 0.99

START_DATE = "2025-03-21"
END_DATE = "2025-04-17"

KEYS = ["Date", "strategy", "confidence_level"]


def load_data():
    daily = pd.read_csv(
        BACKTEST_PATH,
        parse_dates=["Date"],
    )

    forecasts = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["Date", "information_end_date"],
    )

    daily = daily.loc[
        (daily["model"] == MODEL)
        & (daily["strategy"] == STRATEGY)
        & np.isclose(
            daily["confidence_level"], CONFIDENCE
        )
    ].copy()

    forecasts = forecasts.loc[
        (forecasts["model"] == MODEL)
        & (forecasts["strategy"] == STRATEGY)
        & np.isclose(
            forecasts["confidence_level"], CONFIDENCE
        )
    ].copy()

    if daily.empty or forecasts.empty:
        raise ValueError("GARCH-FHS forecast data not found.")

    if daily.duplicated(KEYS).any():
        raise ValueError("Duplicate daily backtest records.")

    if forecasts.duplicated(KEYS).any():
        raise ValueError("Duplicate forecast records.")

    daily = daily[
        KEYS + [
            "realized_return",
            "realized_loss",
            "var_forecast",
            "exceedance",
        ]
    ].copy()

    forecasts = forecasts[
        KEYS + [
            "information_end_date",
            "mu_hat",
            "sigma_hat",
            "z_quantile",
            "var_forecast",
        ]
    ].copy()

    data = daily.merge(
        forecasts,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_backtest", "_model"),
    )

    if not data["_merge"].eq("both").all():
        raise ValueError("Forecast and backtest dates do not match.")

    data = data.drop(columns="_merge")

    if data.isna().any().any():
        raise ValueError("Missing values after merging.")

    if (
        data["information_end_date"] >= data["Date"]
    ).any():
        raise ValueError("Look-ahead detected.")

    if (data["sigma_hat"] <= 0).any():
        raise ValueError("Invalid conditional volatility.")

    numeric_columns = [
        "realized_return",
        "realized_loss",
        "var_forecast_backtest",
        "var_forecast_model",
        "exceedance",
        "mu_hat",
        "sigma_hat",
        "z_quantile",
    ]

    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite numerical values.")

    if not np.allclose(
        data["var_forecast_backtest"],
        data["var_forecast_model"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("VaR forecasts do not match.")

    expected_var = -(
        data["mu_hat"]
        + data["sigma_hat"] * data["z_quantile"]
    )

    if not np.allclose(
        expected_var,
        data["var_forecast_model"],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("VaR formula reconciliation failed.")

    if not np.allclose(
        data["realized_loss"],
        -data["realized_return"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Realized loss mismatch.")

    expected_hits = (
        data["realized_loss"]
        > data["var_forecast_backtest"]
    ).astype(int)

    if not np.array_equal(
        expected_hits.to_numpy(),
        data["exceedance"].to_numpy(dtype=int),
    ):
        raise ValueError("Exceedance indicators do not match.")

    data = data.sort_values("Date").reset_index(drop=True)

    if len(data) != 3839:
        raise ValueError("Unexpected backtesting sample length.")

    if int(data["exceedance"].sum()) != 38:
        raise ValueError("Unexpected exceedance count.")

    data["z"] = (
        data["realized_return"] - data["mu_hat"]
    ) / data["sigma_hat"]

    data["z2"] = data["z"] ** 2

    data["loss_pct"] = 100 * data["realized_loss"]
    data["var_pct"] = 100 * data["var_forecast_model"]
    data["sigma_pct"] = 100 * data["sigma_hat"]

    data["sigma_change_pct"] = (
        100 * (
            data["sigma_hat"]
            / data["sigma_hat"].shift(1)
            - 1
        )
    )

    return data


def select_period(data):
    sample = data.loc[
        data["Date"].between(
            START_DATE,
            END_DATE,
        )
    ].copy()

    if sample.empty:
        raise ValueError("No observations in selected period.")

    if sample["Date"].nunique() < 10:
        raise ValueError("Insufficient observations.")

    return sample


def print_diagnostics(sample):
    print("\nGARCH-FHS: APRIL 2025 EVENT STUDY")
    print("=" * 85)

    print(f"Portfolio: {STRATEGY}")
    print(f"Model: {MODEL}")
    print(f"Period: {START_DATE} to {END_DATE}")

    display = sample[
        [
            "Date",
            "loss_pct",
            "var_pct",
            "sigma_pct",
            "sigma_change_pct",
            "z2",
            "exceedance",
        ]
    ].copy()

    display["Date"] = (
        display["Date"].dt.strftime("%Y-%m-%d")
    )

    print("\nDAILY DIAGNOSTICS")
    print(
        display.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nPERIOD SUMMARY")
    print(f"Trading days: {len(sample)}")
    print(
        f"VaR exceedances: "
        f"{int(sample['exceedance'].sum())}"
    )
    print(
        f"Mean z2: {sample['z2'].mean():.3f}"
    )
    print(
        f"Median z2: {sample['z2'].median():.3f}"
    )
    print(
        f"Maximum z2: {sample['z2'].max():.3f}"
    )

    exceedances = sample.loc[
        sample["exceedance"] == 1,
        ["Date", "loss_pct", "var_pct", "sigma_pct", "z2"],
    ]

    if not exceedances.empty:
        print("\nEXCEEDANCE DAYS")
        print(
            exceedances.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )


def plot_diagnostics(sample):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 11),
        sharex=True,
    )

    dates = sample["Date"]

    events = sample.loc[
        sample["exceedance"] == 1
    ]

    axes[0].plot(
        dates,
        sample["loss_pct"],
        color="#34495e",
        linewidth=1.7,
        marker="o",
        markersize=3,
        label="Realized daily loss",
    )

    axes[0].plot(
        dates,
        sample["var_pct"],
        color="#d97706",
        linewidth=2,
        label="99% VaR forecast",
    )

    axes[0].scatter(
        events["Date"],
        events["loss_pct"],
        color="#c0392b",
        marker="x",
        s=65,
        linewidth=1.8,
        label="VaR exceedance",
        zorder=5,
    )

    axes[0].set_title(
        "Realized Loss vs. 99% VaR",
        loc="left",
        fontsize=12,
    )
    axes[0].set_ylabel("Daily loss (%)")
    axes[0].legend(frameon=False)

    axes[1].plot(
        dates,
        sample["sigma_pct"],
        color="#2563a6",
        linewidth=2,
        marker="o",
        markersize=3,
    )

    axes[1].set_title(
        "One-Day-Ahead Conditional Volatility",
        loc="left",
        fontsize=12,
    )
    axes[1].set_ylabel("Forecast sigma (%)")

    axes[2].plot(
        dates,
        sample["z2"],
        color="#7c3aed",
        linewidth=1.5,
        marker="o",
        markersize=4,
    )

    axes[2].axhline(
        1,
        color="#c0392b",
        linestyle="--",
        linewidth=1.2,
        label="Conditional expectation = 1",
    )

    axes[2].set_title(
        "Standardized Squared Forecast Error",
        loc="left",
        fontsize=12,
    )
    axes[2].set_ylabel("Squared error")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.2)

        for event_date in [
            pd.Timestamp("2025-04-03"),
            pd.Timestamp("2025-04-04"),
        ]:
            ax.axvline(
                event_date,
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.6,
            )

    axes[-1].xaxis.set_major_locator(
        mdates.WeekdayLocator(
            byweekday=mdates.MO
        )
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")
    )

    axes[-1].set_xlabel("Trading date")

    fig.suptitle(
        "GARCH-FHS Risk Response: April 2025\n"
        "MinVar Static Portfolio",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    data = load_data()

    print("Data validation passed.")

    sample = select_period(data)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print_diagnostics(sample)
    plot_diagnostics(sample)

    print("\nSAVED OUTPUTS")
    print(OUTPUT_CSV)
    print(OUTPUT_FIGURE)


if __name__ == "__main__":
    main()
