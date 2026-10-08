
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
RETURNS_PATH = ROOT / "data/interim/qpm/portfolio_returns_main.csv"

DAILY_OUTPUT = DATA_DIR / "var_backtest_daily.csv"
SUMMARY_OUTPUT = DATA_DIR / "var_backtest_summary.csv"

MODELS = {
    "rolling_normal": "rolling_normal_var_es_forecasts.csv",
    "historical_simulation": "historical_simulation_var_es_forecasts.csv",
    "garch_normal": "garch_normal_var_es_forecasts.csv",
    "garch_student_t": "garch_student_t_var_es_forecasts.csv",
    "garch_fhs": "garch_fhs_var_es_forecasts.csv",
}

STRATEGIES = {
    "ew_static",
    "minvar_static",
    "minvar_regime",
    "meanvar_static",
    "meanvar_regime",
}

LEVELS = {0.95, 0.99}
KEYS = ["Date", "strategy", "confidence_level"]

REQUIRED = KEYS + [
    "strategy_name",
    "window_days",
    "estimation_start_date",
    "estimation_end_date",
    "var_forecast",
    "es_forecast",
]


def load_forecasts(model, filename):

    path = DATA_DIR / filename

    df = pd.read_csv(
        path,
        parse_dates=[
            "Date",
            "estimation_start_date",
            "estimation_end_date",
        ],
    )

    missing = sorted(set(REQUIRED) - set(df.columns))

    if missing:
        raise ValueError(
            f"{filename}: missing columns {missing}"
        )

    if (
        df[REQUIRED].isna().any().any()
        or df.duplicated(KEYS).any()
    ):
        raise ValueError(
            f"{filename}: missing values or duplicate keys"
        )

    if not (
        df["estimation_end_date"] < df["Date"]
    ).all():
        raise ValueError(
            f"{filename}: look-ahead detected"
        )

    if set(df["confidence_level"].unique()) != LEVELS:
        raise ValueError(
            f"{filename}: unexpected confidence levels"
        )

    if set(df["strategy"].unique()) != STRATEGIES:
        raise ValueError(
            f"{filename}: unexpected strategies"
        )

    values = df[
        ["window_days", "var_forecast", "es_forecast"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{filename}: non-finite values"
        )

    if (
        df["es_forecast"]
        < df["var_forecast"] - 1e-12
    ).any():
        raise ValueError(
            f"{filename}: ES below VaR"
        )

    if "information_end_date" in df.columns:

        cutoff = pd.to_datetime(
            df["information_end_date"]
        )

        if (
            cutoff.isna().any()
            or (cutoff >= df["Date"]).any()
        ):
            raise ValueError(
                f"{filename}: invalid information cutoff"
            )

    if (
        "model" in df.columns
        and not df["model"].eq(model).all()
    ):
        raise ValueError(
            f"{filename}: unexpected model labels"
        )

    return df[REQUIRED].assign(model=model)


def check_previous_results(summary):

    if not SUMMARY_OUTPUT.exists():
        return

    previous = pd.read_csv(SUMMARY_OUTPUT)

    keys = [
        "model",
        "strategy",
        "confidence_level",
    ]

    needed = keys + [
        "n_days",
        "n_exceedances",
    ]

    if not set(needed).issubset(previous.columns):
        raise ValueError(
            "Previous summary has missing comparison columns"
        )

    if previous.duplicated(keys).any():
        raise ValueError(
            "Previous summary contains duplicate keys"
        )

    matched = previous[needed].merge(
        summary[needed],
        on=keys,
        how="left",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )

    if matched[
        ["n_days_new", "n_exceedances_new"]
    ].isna().any().any():
        raise ValueError(
            "An existing model is missing from the new summary"
        )

    unchanged = (
        matched["n_days_old"].eq(
            matched["n_days_new"]
        )
        & matched["n_exceedances_old"].eq(
            matched["n_exceedances_new"]
        )
    )

    if not unchanged.all():

        changed = matched.loc[
            ~unchanged,
            [
                "model",
                "strategy",
                "confidence_level",
            ],
        ]

        raise ValueError(
            "Previously calculated backtest results changed:\n"
            + changed.to_string(index=False)
        )

    print(
        f"Previous results unchanged: {len(matched)} groups"
    )


def main():

    sources = {
        model: load_forecasts(model, filename)
        for model, filename in MODELS.items()
    }

    common_keys = sources[
        "historical_simulation"
    ][KEYS].copy()

    for model in MODELS:

        common_keys = common_keys.merge(
            sources[model][KEYS],
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

    n_days = common_keys["Date"].nunique()

    expected_keys = (
        n_days * len(STRATEGIES) * len(LEVELS)
    )

    if (
        n_days != 3839
        or len(common_keys) != expected_keys
    ):
        raise ValueError(
            "Expected 3,839 complete common forecast dates"
        )

    for model in MODELS:

        if (
            model != "rolling_normal"
            and len(sources[model]) != len(common_keys)
        ):
            raise ValueError(
                f"{model}: forecast coverage differs "
                "from common sample"
            )

    forecasts = pd.concat(
        [
            source.merge(
                common_keys,
                on=KEYS,
                how="inner",
                validate="one_to_one",
            )
            for source in sources.values()
        ],
        ignore_index=True,
    )

    if len(forecasts) != (
        len(MODELS) * len(common_keys)
    ):
        raise ValueError(
            "Incomplete forecasts on common sample"
        )

    returns = pd.read_csv(
        RETURNS_PATH,
        parse_dates=["Date"],
    )

    if (
        returns["Date"].isna().any()
        or returns["Date"].duplicated().any()
    ):
        raise ValueError(
            "Invalid realized-return dates"
        )

    if not STRATEGIES.issubset(returns.columns):
        raise ValueError(
            "Missing realized-return strategy columns"
        )

    actual = returns[
        ["Date", *sorted(STRATEGIES)]
    ].melt(
        id_vars="Date",
        var_name="strategy",
        value_name="realized_return",
    )

    daily = forecasts.merge(
        actual,
        on=["Date", "strategy"],
        how="left",
        validate="many_to_one",
    )

    actual_values = daily[
        "realized_return"
    ].to_numpy(dtype=float)

    if not np.isfinite(actual_values).all():
        raise ValueError(
            "Missing or non-finite realized returns"
        )

    daily["realized_loss"] = (
        -daily["realized_return"]
    )

    daily["exceedance"] = (
        daily["realized_loss"]
        > daily["var_forecast"]
    ).astype(int)

    daily = daily.sort_values(
        [
            "Date",
            "strategy",
            "confidence_level",
            "model",
        ]
    ).reset_index(drop=True)

    summary = daily.groupby(
        [
            "model",
            "strategy",
            "confidence_level",
            "window_days",
        ],
        as_index=False,
    ).agg(
        n_days=("exceedance", "size"),
        n_exceedances=("exceedance", "sum"),
    )

    summary["expected_rate"] = (
        1 - summary["confidence_level"]
    )

    summary["observed_rate"] = (
        summary["n_exceedances"]
        / summary["n_days"]
    )

    summary["expected_count"] = (
        summary["n_days"]
        * summary["expected_rate"]
    )

    summary["coverage_ratio"] = (
        summary["observed_rate"]
        / summary["expected_rate"]
    )

    summary["rate_gap_pp"] = (
        100
        * (
            summary["observed_rate"]
            - summary["expected_rate"]
        )
    )

    summary = summary.sort_values(
        [
            "confidence_level",
            "strategy",
            "model",
        ]
    ).reset_index(drop=True)

    expected_groups = (
        len(MODELS)
        * len(STRATEGIES)
        * len(LEVELS)
    )

    if len(summary) != expected_groups:
        raise ValueError(
            "Incomplete model/strategy/level groups"
        )

    if not summary["n_days"].eq(n_days).all():
        raise ValueError(
            "Inconsistent group sample lengths"
        )

    check_previous_results(summary)

    DAILY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        DAILY_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        f"Common sample: {n_days:,} dates | "
        f"{common_keys['Date'].min().date()} to "
        f"{common_keys['Date'].max().date()}"
    )

    print(
        f"Daily rows: {len(daily):,} | "
        f"Summary rows: {len(summary)}"
    )

    for confidence in sorted(LEVELS):

        section = summary[
            summary["confidence_level"] == confidence
        ]

        print(
            f"\n=== {confidence:.0%} VaR coverage ==="
        )

        display_columns = [
            "model",
            "strategy",
            "n_exceedances",
            "observed_rate",
            "coverage_ratio",
        ]

        print(
            section[display_columns].to_string(
                index=False,
                formatters={
                    "observed_rate": lambda x: f"{x:.2%}",
                    "coverage_ratio": lambda x: f"{x:.2f}",
                },
            )
        )

    print(f"\nSaved: {DAILY_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
