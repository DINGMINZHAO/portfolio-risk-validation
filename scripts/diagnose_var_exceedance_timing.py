from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/qpm/var_backtest_daily.csv"

FIGURE_DIR = ROOT / "reports/figures/qpm/var_backtest"
SUMMARY_PATH = ROOT / "data/processed/qpm/var_exceedance_timing_summary_99.csv"

CONFIDENCE = 0.99
ROLLING_DAYS = 63
CRISIS_START = "2020-02-01"
CRISIS_END = "2020-06-30"

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime-aware",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime-aware",
}

MODELS = {
    "rolling_normal": "Rolling Normal (252 days)",
    "historical_simulation": "Historical Simulation (1008 days)",
}


def load_data():
    df = pd.read_csv(INPUT, parse_dates=["Date"])

    required = {
        "Date",
        "model",
        "strategy",
        "confidence_level",
        "var_forecast",
        "realized_return",
        "exceedance",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.loc[
        np.isclose(df["confidence_level"], CONFIDENCE)
    ].copy()

    df = df[
        df["model"].isin(MODELS)
        & df["strategy"].isin(STRATEGIES)
    ].copy()

    keys = ["Date", "model", "strategy"]

    if (
        df.empty
        or df[keys].isna().any().any()
        or df.duplicated(keys).any()
    ):
        raise ValueError("Invalid or duplicate records.")

    df["realized_loss"] = -df["realized_return"]

    expected_hits = (
        df["realized_loss"] > df["var_forecast"]
    ).astype(int)

    if not np.array_equal(
        expected_hits.to_numpy(),
        df["exceedance"].to_numpy()
    ):
        raise ValueError("Exceedance flags are inconsistent.")

    df = df.sort_values(
        ["model", "strategy", "Date"]
    ).reset_index(drop=True)

    df["hits_63"] = (
        df.groupby(["model", "strategy"])["exceedance"]
        .transform(
            lambda x: x.rolling(
                ROLLING_DAYS,
                min_periods=ROLLING_DAYS
            ).sum()
        )
    )

    return df


def build_summary(df):
    records = []

    for (model, strategy), sample in df.groupby(
        ["model", "strategy"]
    ):
        sample = sample.reset_index(drop=True)

        if len(sample) < ROLLING_DAYS:
            raise ValueError("Insufficient observations.")

        hits = sample["exceedance"].to_numpy(dtype=int)

        adjacent_pairs = np.sum(
            hits[1:] * hits[:-1]
        )

        peak_idx = int(sample["hits_63"].idxmax())

        records.append({
            "model": model,
            "strategy": strategy,
            "n_days": len(sample),
            "n_exceedances": int(hits.sum()),
            "adjacent_exceedance_pairs": int(adjacent_pairs),
            "max_exceedances_in_63_days": int(
                sample.loc[peak_idx, "hits_63"]
            ),
            "peak_window_start": sample.loc[
                peak_idx - ROLLING_DAYS + 1, "Date"
            ].date(),
            "peak_window_end": sample.loc[
                peak_idx, "Date"
            ].date(),
        })

    return pd.DataFrame(records)


def plot_rolling_counts(df):
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True
    )

    for ax, (model, title) in zip(
        axes, MODELS.items()
    ):
        model_data = df[df["model"] == model]

        for strategy, name in STRATEGIES.items():
            sample = model_data[
                model_data["strategy"] == strategy
            ]

            ax.plot(
                sample["Date"],
                sample["hits_63"],
                label=name,
                linewidth=1.15
            )

        ax.axhline(
            ROLLING_DAYS * (1 - CONFIDENCE),
            linestyle="--",
            linewidth=1,
            label="Expected count (0.63)"
        )

        ax.set_title(title)
        ax.set_ylabel("Exceedances in 63 trading days")
        ax.grid(alpha=0.2)
        ax.legend(
            loc="upper right",
            fontsize=8,
            ncol=2
        )

    axes[-1].xaxis.set_major_locator(
        mdates.YearLocator(2)
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    fig.suptitle(
        "99% VaR: Rolling 63-Day Exceedance Counts",
        fontsize=14
    )

    fig.tight_layout()

    path = FIGURE_DIR / "var99_rolling_63day_exceedances.png"

    fig.savefig(
        path,
        dpi=170,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


def plot_crisis_comparison(df):
    paths = []

    for strategy, name in STRATEGIES.items():
        sample = df[
            (df["strategy"] == strategy)
            & df["Date"].between(
                CRISIS_START, CRISIS_END
            )
        ]

        if sample.empty:
            raise ValueError(
                "No data in the selected crisis period."
            )

        fig, axes = plt.subplots(
            2, 1,
            figsize=(14, 7),
            sharex=True,
            sharey=True
        )

        for ax, (model, title) in zip(
            axes, MODELS.items()
        ):
            one = sample[
                sample["model"] == model
            ]

            ax.plot(
                one["Date"],
                100 * one["realized_loss"],
                linewidth=0.85,
                alpha=0.65,
                label="Realized daily loss"
            )

            ax.plot(
                one["Date"],
                100 * one["var_forecast"],
                linewidth=1.6,
                label="Predicted 99% VaR"
            )

            breaches = one[
                one["exceedance"] == 1
            ]

            ax.scatter(
                breaches["Date"],
                100 * breaches["realized_loss"],
                marker="x",
                s=40,
                linewidths=1.5,
                zorder=4,
                label=f"Exceedances: {len(breaches)}"
            )

            ax.axhline(
                0,
                linewidth=0.7,
                alpha=0.4
            )

            ax.set_title(title)
            ax.set_ylabel("Daily loss / VaR (%)")
            ax.grid(alpha=0.2)
            ax.legend(
                loc="upper right",
                fontsize=9
            )

        axes[-1].xaxis.set_major_locator(
            mdates.MonthLocator()
        )

        axes[-1].xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m")
        )

        fig.suptitle(
            f"{name}: Actual Losses vs 99% VaR "
            "(February-June 2020)",
            fontsize=13
        )

        fig.tight_layout()

        path = FIGURE_DIR / f"var99_2020_{strategy}.png"

        fig.savefig(
            path,
            dpi=170,
            bbox_inches="tight"
        )

        plt.close(fig)

        paths.append(path)

    return paths


def main():
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = load_data()

    summary = build_summary(data)
    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    overview_path = plot_rolling_counts(data)
    crisis_paths = plot_crisis_comparison(data)

    print("\n99% VAR EXCEEDANCE TIMING SUMMARY")
    print(summary.to_string(index=False))

    print(f"\nSaved summary: {SUMMARY_PATH}")
    print(f"Saved overview: {overview_path}")
    print(
        f"Saved {len(crisis_paths)} crisis charts "
        f"in: {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()