
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/exceedance_dynamics"

BACKTEST_PATH = DATA_DIR / "var_backtest_daily.csv"
FORECAST_PATH = DATA_DIR / "garch_fhs_var_es_forecasts.csv"

EVENTS_PATH = DATA_DIR / "garch_fhs_99_event_time_z2_events.csv"
SUMMARY_PATH = DATA_DIR / "garch_fhs_99_event_time_z2_summary.csv"
FIGURE_PATH = FIGURE_DIR / "garch_fhs_99_event_time_z2.png"

MODEL = "garch_fhs"
CONFIDENCE = 0.99
HORIZON = 5
EXPECTED_DAYS = 3839

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime",
}

EXPECTED_HITS = {
    "ew_static": 46,
    "minvar_static": 38,
    "minvar_regime": 38,
    "meanvar_static": 43,
    "meanvar_regime": 43,
}


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
        & np.isclose(daily["confidence_level"], CONFIDENCE)
    ].copy()

    forecasts = forecasts.loc[
        (forecasts["model"] == MODEL)
        & np.isclose(forecasts["confidence_level"], CONFIDENCE)
    ].copy()

    if daily.empty or forecasts.empty:
        raise ValueError("Missing GARCH-FHS 99% VaR data.")

    keys = ["Date", "strategy", "confidence_level"]

    if daily.duplicated(keys).any():
        raise ValueError("Duplicate daily backtest records.")

    if forecasts.duplicated(keys).any():
        raise ValueError("Duplicate GARCH-FHS forecast records.")

    daily = daily[
        keys + [
            "realized_return",
            "realized_loss",
            "var_forecast",
            "exceedance",
        ]
    ].copy()

    forecasts = forecasts[
        keys + [
            "information_end_date",
            "mu_hat",
            "sigma_hat",
            "z_quantile",
            "var_forecast",
        ]
    ].copy()

    data = daily.merge(
        forecasts,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_backtest", "_model"),
    )

    if not data["_merge"].eq("both").all():
        raise ValueError("Daily backtest and forecast dates do not match.")

    data = data.drop(columns="_merge")

    if data.isna().any().any():
        raise ValueError("Missing values after merging.")

    if (data["information_end_date"] >= data["Date"]).any():
        raise ValueError("Look-ahead detected in forecast information dates.")

    if (data["sigma_hat"] <= 0).any():
        raise ValueError("Nonpositive conditional volatility.")

    numeric_columns = [
        "realized_return",
        "realized_loss",
        "mu_hat",
        "sigma_hat",
        "z_quantile",
        "var_forecast_backtest",
        "var_forecast_model",
        "exceedance",
    ]

    if not np.isfinite(data[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numerical values.")

    if not np.allclose(
        data["var_forecast_backtest"],
        data["var_forecast_model"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("VaR forecasts differ between the two files.")

    if not np.allclose(
        data["realized_loss"],
        -data["realized_return"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Return and loss definitions are inconsistent.")

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
        raise ValueError("GARCH-FHS VaR formula reconciliation failed.")

    calculated_hits = (
        data["realized_loss"] > data["var_forecast_backtest"]
    ).astype(int)

    if not np.array_equal(
        calculated_hits.to_numpy(),
        data["exceedance"].to_numpy(dtype=int),
    ):
        raise ValueError("VaR exceedance indicators do not match.")

    data["z"] = (
        data["realized_return"] - data["mu_hat"]
    ) / data["sigma_hat"]

    data["z2"] = data["z"] ** 2

    data = data.sort_values(
        ["strategy", "Date"]
    ).reset_index(drop=True)

    if set(data["strategy"]) != set(STRATEGIES):
        raise ValueError("Expected five portfolio strategies.")

    for strategy, group in data.groupby("strategy"):
        if len(group) != EXPECTED_DAYS:
            raise ValueError(
                f"Unexpected number of days for {strategy}: {len(group)}"
            )

        actual_hits = int(group["exceedance"].sum())

        if actual_hits != EXPECTED_HITS[strategy]:
            raise ValueError(
                f"Exceedance count mismatch for {strategy}: "
                f"{actual_hits} vs {EXPECTED_HITS[strategy]}"
            )

    print("Data validation passed.")
    print(f"Total observations: {len(data):,}")
    print(f"Date range: {data['Date'].min().date()} "
          f"to {data['Date'].max().date()}")

    return data


def build_event_windows(data):
    records = []
    counts = []

    for strategy in STRATEGIES:
        group = data.loc[
            data["strategy"] == strategy
        ].sort_values("Date").reset_index(drop=True)

        hit_indices = np.flatnonzero(
            group["exceedance"].to_numpy(dtype=int) == 1
        )

        valid_events = 0
        excluded_events = 0

        for event_id, idx in enumerate(hit_indices, start=1):
            if idx + HORIZON >= len(group):
                excluded_events += 1
                continue

            valid_events += 1
            event_date = group.loc[idx, "Date"]

            for k in range(1, HORIZON + 1):
                future = group.iloc[idx + k]

                records.append({
                    "strategy": strategy,
                    "event_id": event_id,
                    "event_date": event_date,
                    "future_date": future["Date"],
                    "event_time": k,
                    "z": future["z"],
                    "z2": future["z2"],
                    "realized_return": future["realized_return"],
                    "mu_hat": future["mu_hat"],
                    "sigma_hat": future["sigma_hat"],
                    "future_exceedance": int(future["exceedance"]),
                })

        counts.append({
            "strategy": strategy,
            "total_exceedances": len(hit_indices),
            "complete_event_windows": valid_events,
            "excluded_near_sample_end": excluded_events,
        })

    events = pd.DataFrame(records)
    counts = pd.DataFrame(counts)

    if events.empty:
        raise ValueError("No complete event windows available.")

    return events, counts


def summarize_events(events):
    summary = (
        events.groupby(
            ["strategy", "event_time"],
            as_index=False,
        )
        .agg(
            mean_z2=("z2", "mean"),
            median_z2=("z2", "median"),
            n_events=("event_id", "nunique"),
        )
    )

    return summary


def plot_event_time(events, summary):
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11, 15),
        sharex=True,
        sharey=True,
    )

    for ax, (strategy, name) in zip(
        axes, STRATEGIES.items()
    ):
        subset = events.loc[
            events["strategy"] == strategy
        ]

        stats = summary.loc[
            summary["strategy"] == strategy
        ].sort_values("event_time")

        for _, event in subset.groupby("event_id"):
            event = event.sort_values("event_time")

            ax.plot(
                event["event_time"],
                event["z2"],
                color="gray",
                linewidth=0.8,
                alpha=0.23,
            )

        ax.plot(
            stats["event_time"],
            stats["mean_z2"],
            color="navy",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label="Mean across events",
            zorder=5,
        )

        ax.axhline(
            1,
            color="crimson",
            linestyle="--",
            linewidth=1.3,
            label="Theoretical mean = 1",
        )

        n_events = subset["event_id"].nunique()

        ax.set_title(
            f"{name} | {n_events} event windows",
            fontsize=11,
            loc="left",
        )

        ax.set_ylabel(
            "Squared standardized error",
            fontsize=9,
        )

        ax.set_xticks(range(1, HORIZON + 1))
        ax.grid(alpha=0.2)

    axes[0].legend(
        loc="upper right",
        fontsize=9,
        frameon=False,
    )

    axes[-1].set_xlabel(
        "Trading days after 99% VaR exceedance",
        fontsize=10,
    )

    fig.suptitle(
        "GARCH-FHS: Post-Exceedance Forecast Errors\n"
        "99% VaR | Event Time +1 to +5",
        fontsize=14,
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.975]
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    print("\nGARCH-FHS EXCEEDANCE EVENT STUDY")
    print("=" * 55)

    data = load_data()

    events, counts = build_event_windows(data)
    summary = summarize_events(events)

    events.to_csv(
        EVENTS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    plot_event_time(events, summary)

    print("\nEVENT COUNTS")
    print(
        counts.to_string(index=False)
    )

    print("\nMEAN STANDARDIZED SQUARED ERRORS")
    display = summary.pivot(
        index="strategy",
        columns="event_time",
        values="mean_z2",
    )

    display = display.reindex(
        list(STRATEGIES)
    )

    print(
        display.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nFIVE-DAY EVENT AVERAGES")

    five_day = (
        events.groupby(
            ["strategy", "event_id"],
            as_index=False,
        )["z2"]
        .mean()
    )

    overall = five_day.groupby(
        "strategy"
    )["z2"].agg(
        ["count", "mean", "median"]
    )

    overall = overall.reindex(
        list(STRATEGIES)
    )

    print(
        overall.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nSAVED OUTPUTS")
    print(EVENTS_PATH)
    print(SUMMARY_PATH)
    print(FIGURE_PATH)


if __name__ == "__main__":
    main()
