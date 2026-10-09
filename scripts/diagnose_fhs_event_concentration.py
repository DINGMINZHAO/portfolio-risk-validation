
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"

INPUT_PATH = DATA_DIR / "garch_fhs_99_event_time_z2_events.csv"

RANKING_PATH = DATA_DIR / "garch_fhs_99_event_impact_ranking.csv"
SUMMARY_PATH = DATA_DIR / "garch_fhs_99_event_concentration_summary.csv"

TOP_N = 5

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime",
}


def load_data():
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["event_date", "future_date"],
    )

    required = {
        "strategy",
        "event_id",
        "event_date",
        "future_date",
        "event_time",
        "z2",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if set(df["strategy"]) != set(STRATEGIES):
        raise ValueError("Expected five portfolio strategies.")

    keys = ["strategy", "event_id"]

    if df.duplicated(keys + ["event_time"]).any():
        raise ValueError("Duplicate event-time observations.")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing event data.")

    if not np.isfinite(df["z2"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite squared errors.")

    if (df["z2"] < 0).any():
        raise ValueError("Negative squared errors.")

    for _, group in df.groupby(keys):
        if set(group["event_time"]) != {1, 2, 3, 4, 5}:
            raise ValueError("Incomplete five-day event window.")

        if group["event_date"].nunique() != 1:
            raise ValueError("Inconsistent event dates.")

        if not (group["future_date"] > group["event_date"]).all():
            raise ValueError("Invalid event-time ordering.")

    return df


def calculate_event_statistics(df):
    events = (
        df.groupby(
            ["strategy", "event_id", "event_date"],
            as_index=False,
        )
        .agg(
            mean_z2=("z2", "mean"),
            max_z2=("z2", "max"),
        )
    )

    events = events.sort_values(
        ["strategy", "mean_z2"],
        ascending=[True, False],
    ).reset_index(drop=True)

    events["rank"] = (
        events.groupby("strategy").cumcount() + 1
    )

    events["year"] = events["event_date"].dt.year

    events["contribution_pct"] = (
        100
        * events["mean_z2"]
        / events.groupby("strategy")["mean_z2"].transform("sum")
    )

    return events


def calculate_concentration(df, events):
    records = []

    for strategy, name in STRATEGIES.items():
        sample = events.loc[
            events["strategy"] == strategy
        ].sort_values("rank")

        top = sample.head(TOP_N)
        remaining = sample.iloc[TOP_N:]

        total_mean = sample["mean_z2"].mean()
        remaining_mean = remaining["mean_z2"].mean()

        top_share = (
            top["mean_z2"].sum()
            / sample["mean_z2"].sum()
        ) * 100

        top_ids = top["event_id"].tolist()

        top_daily = df.loc[
            (df["strategy"] == strategy)
            & df["event_id"].isin(top_ids)
        ]

        unique_future_days = top_daily["future_date"].nunique()

        records.append({
            "strategy": strategy,
            "strategy_name": name,
            "n_events": len(sample),
            "overall_mean_z2": total_mean,
            "overall_median_z2": sample["mean_z2"].median(),
            "top5_mean_z2": top["mean_z2"].mean(),
            "mean_excluding_top5": remaining_mean,
            "top5_contribution_pct": top_share,
            "top5_unique_future_days": unique_future_days,
            "top5_repeated_future_days": len(top_daily) - unique_future_days,
            "top5_events_in_2020": int((top["year"] == 2020).sum()),
        })

    return pd.DataFrame(records)


def print_results(events, summary):
    print("\nGARCH-FHS EVENT CONCENTRATION DIAGNOSTICS")
    print("=" * 70)

    for strategy, name in STRATEGIES.items():
        sample = events.loc[
            events["strategy"] == strategy
        ].head(TOP_N)

        stats = summary.loc[
            summary["strategy"] == strategy
        ].iloc[0]

        print(f"\n{name}")
        print("-" * 70)

        display = sample[
            [
                "rank",
                "event_date",
                "mean_z2",
                "max_z2",
                "contribution_pct",
            ]
        ].copy()

        display["event_date"] = (
            display["event_date"].dt.strftime("%Y-%m-%d")
        )

        print(
            display.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

        print(
            f"\nAll-event mean M: "
            f"{stats['overall_mean_z2']:.3f}"
        )

        print(
            f"Mean excluding top 5: "
            f"{stats['mean_excluding_top5']:.3f}"
        )

        print(
            f"Top 5 share of total M: "
            f"{stats['top5_contribution_pct']:.2f}%"
        )

        print(
            f"Unique future trading days in top 5 windows: "
            f"{int(stats['top5_unique_future_days'])}/25"
        )

        print(
            f"Top 5 events occurring in 2020: "
            f"{int(stats['top5_events_in_2020'])}"
        )

    print("\n" + "=" * 70)
    print("CROSS-STRATEGY SUMMARY")

    columns = [
        "strategy",
        "n_events",
        "overall_mean_z2",
        "mean_excluding_top5",
        "top5_contribution_pct",
        "top5_repeated_future_days",
    ]

    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def main():
    df = load_data()

    events = calculate_event_statistics(df)
    summary = calculate_concentration(df, events)

    events.to_csv(
        RANKING_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print_results(events, summary)

    print("\nSAVED OUTPUTS")
    print(RANKING_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
