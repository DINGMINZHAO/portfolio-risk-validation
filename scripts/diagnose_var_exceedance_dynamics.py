
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/exceedance_dynamics"

INPUT_PATH = DATA_DIR / "var_backtest_daily.csv"
BACKTEST_SUMMARY_PATH = DATA_DIR / "var_backtest_summary.csv"

MODELS = {
    "rolling_normal": "Rolling Normal",
    "historical_simulation": "Historical Simulation",
    "garch_normal": "GARCH-Normal",
    "garch_student_t": "GARCH-Student-t",
    "garch_fhs": "GARCH-FHS",
}

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime-aware",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime-aware",
}

GARCH_FILES = {
    "garch_normal": "garch_normal_var_es_forecasts.csv",
    "garch_student_t": "garch_student_t_var_es_forecasts.csv",
    "garch_fhs": "garch_fhs_var_es_forecasts.csv",
}

LEVELS = (0.95, 0.99)
ROLLING_DAYS = 63
LAGS = (1, 2, 5, 10)
EXPECTED_DAYS = 3839

FHS_PURE_OOS_START = pd.Timestamp("2014-09-08")

KEYS = [
    "Date",
    "model",
    "strategy",
    "confidence_level",
]


def read_daily():

    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"],
    )

    required = set(KEYS) | {
        "realized_return",
        "realized_loss",
        "var_forecast",
        "exceedance",
    }

    missing = sorted(required - set(data.columns))

    if missing:
        raise ValueError(
            f"Missing daily backtest columns: {missing}"
        )

    if (
        data[list(required)].isna().any().any()
        or data.duplicated(KEYS).any()
    ):
        raise ValueError(
            "Missing values or duplicate backtest keys"
        )

    if set(data["model"]) != set(MODELS):
        raise ValueError("Expected all five models")

    if set(data["strategy"]) != set(STRATEGIES):
        raise ValueError("Expected all five strategies")

    if set(data["confidence_level"]) != set(LEVELS):
        raise ValueError("Expected 95% and 99% levels")

    numeric = data[
        [
            "realized_return",
            "realized_loss",
            "var_forecast",
            "exceedance",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite daily values")

    hits = data["exceedance"].to_numpy(dtype=int)

    if not np.isin(hits, [0, 1]).all():
        raise ValueError("Invalid exceedance indicators")

    if not np.allclose(
        data["realized_loss"],
        -data["realized_return"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Loss and return mismatch")

    expected = (
        data["realized_loss"] > data["var_forecast"]
    ).astype(int)

    if not np.array_equal(
        expected.to_numpy(),
        hits,
    ):
        raise ValueError("Exceedance mismatch")

    dates = np.sort(data["Date"].unique())

    if (
        len(dates) != EXPECTED_DAYS
        or len(data) != EXPECTED_DAYS * 5 * 5 * 2
    ):
        raise ValueError("Unexpected backtest coverage")

    data = data.sort_values(
        KEYS
    ).reset_index(drop=True)

    for key, group in data.groupby(
        ["model", "strategy", "confidence_level"]
    ):

        if not np.array_equal(
            group["Date"].to_numpy(),
            dates,
        ):
            raise ValueError(
                f"Date mismatch: {key}"
            )

    loss_range = data.groupby(
        ["Date", "strategy"]
    )["realized_loss"].agg(["min", "max"])

    if (
        (loss_range["max"] - loss_range["min"]).abs()
        > 1e-12
    ).any():
        raise ValueError(
            "Realized losses differ across models"
        )

    return data


def read_sigma():

    pieces = []

    for model, filename in GARCH_FILES.items():

        df = pd.read_csv(
            DATA_DIR / filename,
            parse_dates=[
                "Date",
                "information_end_date",
            ],
        )

        required = [
            "Date",
            "strategy",
            "confidence_level",
            "sigma_hat",
            "information_end_date",
        ]

        if (
            df[required].isna().any().any()
            or df.duplicated(
                ["Date", "strategy", "confidence_level"]
            ).any()
        ):
            raise ValueError(
                f"Invalid GARCH forecasts: {filename}"
            )

        if (
            df["information_end_date"] >= df["Date"]
        ).any():
            raise ValueError(
                f"Look-ahead detected: {filename}"
            )

        if (
            not np.isfinite(df["sigma_hat"]).all()
            or (df["sigma_hat"] <= 0).any()
        ):
            raise ValueError(
                f"Invalid sigma: {filename}"
            )

        wide = df.pivot(
            index=["Date", "strategy"],
            columns="confidence_level",
            values="sigma_hat",
        )

        if not np.allclose(
            wide[0.95],
            wide[0.99],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Sigma differs across levels: {model}"
            )

        part = df.loc[
            df["confidence_level"] == 0.95,
            ["Date", "strategy", "sigma_hat"],
        ].copy()

        part["model"] = model
        pieces.append(part)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def max_streak(hits):

    best = 0
    current = 0

    for hit in hits:

        if hit:
            current += 1
        else:
            current = 0

        best = max(best, current)

    return best


def any_next_hits(hits, horizon):

    return np.array(
        [
            np.any(hits[i + 1:i + horizon + 1])
            for i in range(len(hits) - horizon)
        ],
        dtype=bool,
    )


def build_diagnostics(data):

    summary_rows = []
    lag_rows = []
    annual_rows = []
    post_rows = []
    enriched = []

    groups = data.groupby(
        ["model", "strategy", "confidence_level"],
        sort=True,
    )

    for (model, strategy, confidence), group in groups:

        group = group.sort_values(
            "Date"
        ).copy().reset_index(drop=True)

        hits = group["exceedance"].to_numpy(dtype=int)

        rolling = (
            pd.Series(hits)
            .rolling(
                ROLLING_DAYS,
                min_periods=ROLLING_DAYS,
            )
            .sum()
        )

        peak_end = int(rolling.idxmax())
        peak_start = peak_end - ROLLING_DAYS + 1

        after_hit = hits[1:][hits[:-1] == 1]
        after_no_hit = hits[1:][hits[:-1] == 0]

        summary_rows.append({
            "model": model,
            "strategy": strategy,
            "confidence_level": confidence,
            "n_days": len(hits),
            "n_exceedances": int(hits.sum()),
            "observed_rate": float(hits.mean()),
            "nominal_rate": 1 - confidence,
            "adjacent_pairs": int(
                np.sum(hits[1:] * hits[:-1])
            ),
            "max_consecutive_exceedances": max_streak(hits),
            "p_next_hit_given_hit": (
                float(after_hit.mean())
                if len(after_hit) else np.nan
            ),
            "p_next_hit_given_no_hit": (
                float(after_no_hit.mean())
                if len(after_no_hit) else np.nan
            ),
            "max_hits_in_63_days": int(
                rolling.iloc[peak_end]
            ),
            "peak_window_start": group["Date"].iloc[
                peak_start
            ],
            "peak_window_end": group["Date"].iloc[
                peak_end
            ],
        })

        group["hits_63"] = rolling
        enriched.append(group)

        for lag in LAGS:

            previous = hits[:-lag]
            current = hits[lag:]

            affected = previous == 1
            unaffected = previous == 0

            future_any = any_next_hits(
                hits,
                lag,
            )

            lag_rows.append({
                "model": model,
                "strategy": strategy,
                "confidence_level": confidence,
                "horizon_days": lag,
                "n_initial_hits": int(affected.sum()),
                "p_hit_exact_day_given_hit": (
                    float(current[affected].mean())
                    if affected.any() else np.nan
                ),
                "p_hit_exact_day_given_no_hit": (
                    float(current[unaffected].mean())
                    if unaffected.any() else np.nan
                ),
                "p_any_hit_next_h_days_given_hit": (
                    float(future_any[affected].mean())
                    if affected.any() else np.nan
                ),
                "p_any_hit_next_h_days_given_no_hit": (
                    float(future_any[unaffected].mean())
                    if unaffected.any() else np.nan
                ),
                "independent_expected_any_next_h": (
                    1 - confidence ** lag
                ),
            })

        annual = (
            group.assign(year=group["Date"].dt.year)
            .groupby("year", as_index=False)
            .agg(
                n_days=("exceedance", "size"),
                n_exceedances=("exceedance", "sum"),
            )
        )

        annual["model"] = model
        annual["strategy"] = strategy
        annual["confidence_level"] = confidence

        annual["observed_rate"] = (
            annual["n_exceedances"] / annual["n_days"]
        )

        annual_rows.append(annual)

        if model == "garch_fhs":

            post = group.loc[
                group["Date"] >= FHS_PURE_OOS_START
            ]

            h = post["exceedance"].to_numpy(dtype=int)

            post_rows.append({
                "strategy": strategy,
                "confidence_level": confidence,
                "n_days": len(h),
                "n_exceedances": int(h.sum()),
                "observed_rate": float(h.mean()),
                "adjacent_pairs": int(
                    np.sum(h[1:] * h[:-1])
                ),
                "start_date": post["Date"].min(),
                "end_date": post["Date"].max(),
            })

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(lag_rows),
        pd.concat(annual_rows, ignore_index=True),
        pd.DataFrame(post_rows),
        pd.concat(enriched, ignore_index=True),
    )


def summarize_sigma_response(data, sigma):

    subset = data.loc[
        data["model"].isin(GARCH_FILES)
    ].copy()

    subset = subset.merge(
        sigma,
        on=["Date", "model", "strategy"],
        how="left",
        validate="many_to_one",
    )

    if subset["sigma_hat"].isna().any():
        raise ValueError("Missing GARCH sigma values")

    rows = []

    groups = subset.groupby(
        ["model", "strategy", "confidence_level"],
        sort=True,
    )

    for (model, strategy, level), group in groups:

        group = group.sort_values(
            "Date"
        ).reset_index(drop=True)

        hits = group["exceedance"].to_numpy(dtype=int)
        vol = group["sigma_hat"].to_numpy(dtype=float)
        var = group["var_forecast"].to_numpy(dtype=float)

        before_change = 100 * (
            vol[1:-1] / vol[:-2] - 1
        )

        after_change = 100 * (
            vol[2:] / vol[1:-1] - 1
        )

        after_var_change_pp = 100 * (
            var[2:] - var[1:-1]
        )

        hit_mask = hits[1:-1] == 1
        nonhit_mask = hits[1:-1] == 0

        for label, mask in [
            ("hit_day", hit_mask),
            ("nonhit_day", nonhit_mask),
        ]:

            rows.append({
                "model": model,
                "strategy": strategy,
                "confidence_level": level,
                "event_type": label,
                "n_days": int(mask.sum()),
                "median_sigma_change_before_pct": float(
                    np.median(before_change[mask])
                ),
                "median_sigma_change_after_pct": float(
                    np.median(after_change[mask])
                ),
                "mean_sigma_change_before_pct": float(
                    np.mean(before_change[mask])
                ),
                "mean_sigma_change_after_pct": float(
                    np.mean(after_change[mask])
                ),
                "fraction_sigma_rose_after": float(
                    np.mean(after_change[mask] > 0)
                ),
                "median_next_day_var_change_pp": float(
                    np.median(after_var_change_pp[mask])
                ),
            })

    return pd.DataFrame(rows)


def verify_existing_summary(summary):

    if not BACKTEST_SUMMARY_PATH.exists():
        print("Previous summary not found; skipped reconciliation")
        return

    old = pd.read_csv(
        BACKTEST_SUMMARY_PATH
    )

    columns = [
        "model",
        "strategy",
        "confidence_level",
        "n_days",
        "n_exceedances",
    ]

    old = old[columns]

    comparison = old.merge(
        summary[columns],
        on=["model", "strategy", "confidence_level"],
        how="outer",
        validate="one_to_one",
        suffixes=("_old", "_new"),
        indicator=True,
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError("Backtest group mismatch")

    for column in ["n_days", "n_exceedances"]:

        if not comparison[
            f"{column}_old"
        ].eq(
            comparison[f"{column}_new"]
        ).all():
            raise ValueError(
                f"Backtest count mismatch: {column}"
            )

    print(
        f"Independent count reconciliation: "
        f"{len(comparison)}/50 groups match"
    )


def plot_rolling(enriched, confidence):

    fig, axes = plt.subplots(
        5, 1,
        figsize=(15, 13),
        sharex=True,
    )

    for ax, (strategy, name) in zip(
        axes,
        STRATEGIES.items(),
    ):

        sample = enriched.loc[
            (enriched["strategy"] == strategy)
            & (enriched["confidence_level"] == confidence)
        ]

        for model, label in MODELS.items():

            one = sample.loc[
                sample["model"] == model
            ]

            ax.plot(
                one["Date"],
                one["hits_63"],
                label=label,
                linewidth=0.95,
                alpha=0.88,
            )

        ax.axhline(
            ROLLING_DAYS * (1 - confidence),
            color="gray",
            linestyle="--",
            linewidth=1,
        )

        ax.set_ylabel("Hits / 63 days")
        ax.set_title(
            name,
            loc="left",
            fontsize=10,
        )
        ax.grid(alpha=0.2)

    axes[0].legend(
        ncol=3,
        fontsize=8,
        loc="upper right",
    )

    axes[-1].xaxis.set_major_locator(
        mdates.YearLocator(2)
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    fig.suptitle(
        f"{confidence:.0%} VaR: Rolling 63-day exceedance counts",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    path = FIGURE_DIR / (
        f"var{int(confidence * 100)}_rolling_63day_counts.png"
    )

    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_peak_windows(enriched, summary):

    paths = []

    for strategy, name in STRATEGIES.items():

        item = summary.loc[
            (summary["model"] == "garch_fhs")
            & (summary["strategy"] == strategy)
            & (summary["confidence_level"] == 0.99)
        ].iloc[0]

        start = item["peak_window_start"]
        end = item["peak_window_end"]

        selection = enriched.loc[
            (enriched["strategy"] == strategy)
            & (enriched["confidence_level"] == 0.99)
            & enriched["Date"].between(start, end)
        ]

        fig, ax = plt.subplots(
            figsize=(13, 5.5)
        )

        reference = selection.loc[
            selection["model"] == "garch_fhs"
        ]

        ax.plot(
            reference["Date"],
            100 * reference["realized_loss"],
            color="black",
            alpha=0.68,
            linewidth=1.2,
            label="Actual loss",
        )

        for model, label in MODELS.items():

            one = selection.loc[
                selection["model"] == model
            ]

            ax.plot(
                one["Date"],
                100 * one["var_forecast"],
                linewidth=1.3,
                label=f"{label} VaR",
            )

        breaches = reference.loc[
            reference["exceedance"] == 1
        ]

        ax.scatter(
            breaches["Date"],
            100 * breaches["realized_loss"],
            color="crimson",
            marker="x",
            s=55,
            linewidth=1.6,
            label=f"FHS exceedances ({len(breaches)})",
            zorder=5,
        )

        ax.set_title(
            f"{name}: Peak FHS window "
            f"({start.date()} to {end.date()})",
            fontsize=11,
        )

        ax.set_ylabel("Daily loss and 99% VaR (%)")
        ax.grid(alpha=0.2)

        ax.legend(
            ncol=3,
            fontsize=8,
        )

        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(interval=2)
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%Y-%m-%d")
        )

        fig.autofmt_xdate()
        fig.tight_layout()

        path = FIGURE_DIR / (
            f"var99_peak_window_{strategy}.png"
        )

        fig.savefig(
            path,
            dpi=160,
            bbox_inches="tight",
        )

        plt.close(fig)

        paths.append(path)

    return paths


def main():

    data = read_daily()
    sigma = read_sigma()

    (
        summary,
        lags,
        years,
        post,
        enriched,
    ) = build_diagnostics(data)

    response = summarize_sigma_response(
        data,
        sigma,
    )

    verify_existing_summary(summary)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "var_exceedance_timing_summary.csv": summary,
        "var_exceedance_lag_dependence.csv": lags,
        "var_exceedance_by_year.csv": years,
        "var_exceedance_volatility_response.csv": response,
        "var_fhs_post_initial_pool_summary.csv": post,
    }

    for filename, table in outputs.items():

        table.to_csv(
            DATA_DIR / filename,
            index=False,
        )

    plots = [
        plot_rolling(enriched, level)
        for level in LEVELS
    ]

    plots.extend(
        plot_peak_windows(enriched, summary)
    )

    print("\nFIVE-MODEL EXCEEDANCE TIMING DIAGNOSTICS")

    print(
        f"Input rows: {len(data):,} | "
        f"Days: {data['Date'].nunique():,}"
    )

    print(
        f"Timing groups: {len(summary)} | "
        f"Lag groups: {len(lags)}"
    )

    print(
        f"Volatility-response groups: {len(response)}"
    )

    print("\n99% VAR: EXCEEDANCE CLUSTERING")

    display = summary.loc[
        summary["confidence_level"] == 0.99,
        [
            "strategy",
            "model",
            "n_exceedances",
            "adjacent_pairs",
            "max_consecutive_exceedances",
            "max_hits_in_63_days",
            "peak_window_start",
            "peak_window_end",
        ],
    ]

    print(
        display.to_string(index=False)
    )

    print("\n99% VAR: FHS VOLATILITY RESPONSE (%)")

    fhs = response.loc[
        (response["model"] == "garch_fhs")
        & (response["confidence_level"] == 0.99),
        [
            "strategy",
            "event_type",
            "n_days",
            "median_sigma_change_before_pct",
            "median_sigma_change_after_pct",
            "fraction_sigma_rose_after",
        ],
    ]

    print(
        fhs.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSAVED OUTPUTS")

    for filename in outputs:
        print(DATA_DIR / filename)

    print(
        f"Saved {len(plots)} figures in {FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
