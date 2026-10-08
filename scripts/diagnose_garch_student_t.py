
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
RETURNS_PATH = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
FORECAST_PATH = DATA_DIR / "garch_student_t_var_es_forecasts.csv"
BACKTEST_PATH = DATA_DIR / "var_backtest_summary.csv"
FIGURE_DIR = ROOT / "reports/figures/qpm/garch_student_t_diagnostics"

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime-aware",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime-aware",
}


def load_data():
    pred = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["Date", "information_end_date"],
    )

    required = {
        "Date", "strategy", "model", "confidence_level",
        "information_end_date", "mu_hat", "sigma_hat",
        "nu", "var_forecast",
    }

    if not required.issubset(pred.columns):
        raise ValueError(
            f"Missing forecast fields: {sorted(required - set(pred.columns))}"
        )

    keys = ["Date", "strategy", "confidence_level"]

    if pred[keys].isna().any().any() or pred.duplicated(keys).any():
        raise ValueError("Missing or duplicate forecast keys")

    if set(pred["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected strategies")

    if set(pred["confidence_level"]) != {0.95, 0.99}:
        raise ValueError("Expected only 95% and 99% forecasts")

    if not pred["model"].eq("garch_student_t").all():
        raise ValueError("Unexpected forecast model")

    if (
        pred["information_end_date"].isna().any()
        or (pred["information_end_date"] >= pred["Date"]).any()
    ):
        raise ValueError("Invalid information cutoff: possible look-ahead")

    p95 = pred.loc[pred["confidence_level"] == 0.95].copy()
    p99 = pred.loc[pred["confidence_level"] == 0.99].copy()

    matched = p95.merge(
        p99[
            ["Date", "strategy", "mu_hat", "sigma_hat", "nu", "var_forecast"]
        ],
        on=["Date", "strategy"],
        how="outer",
        validate="one_to_one",
        suffixes=("_95", "_99"),
        indicator=True,
    )

    if not matched["_merge"].eq("both").all():
        raise ValueError("95% and 99% forecast coverage differs")

    for column in ["mu_hat", "sigma_hat", "nu"]:
        a = matched[f"{column}_95"].to_numpy(dtype=float)
        b = matched[f"{column}_99"].to_numpy(dtype=float)

        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Non-finite values in {column}")

        if not np.allclose(a, b, rtol=0, atol=1e-12):
            raise ValueError(f"{column} differs across confidence levels")

    n_days = matched["Date"].nunique()

    if n_days != 3839 or len(matched) != n_days * len(STRATEGIES):
        raise ValueError(
            "Expected 3,839 trading days for each of five strategies"
        )

    if not matched.groupby("strategy")["Date"].nunique().eq(n_days).all():
        raise ValueError("A strategy has missing forecast dates")

    returns = pd.read_csv(
        RETURNS_PATH,
        parse_dates=["Date"],
    )

    if returns["Date"].isna().any() or returns["Date"].duplicated().any():
        raise ValueError("Invalid portfolio return dates")

    if not set(STRATEGIES).issubset(returns.columns):
        raise ValueError("Missing portfolio return columns")

    actual = returns[
        ["Date", *STRATEGIES]
    ].melt(
        id_vars="Date",
        var_name="strategy",
        value_name="realized_return",
    )

    df = matched.merge(
        actual,
        on=["Date", "strategy"],
        how="left",
        validate="one_to_one",
    )

    values = df[
        [
            "realized_return",
            "mu_hat_95",
            "sigma_hat_95",
            "nu_95",
            "var_forecast_95",
            "var_forecast_99",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite values")

    if (
        (df["sigma_hat_95"] <= 0).any()
        or (df["nu_95"] <= 2).any()
    ):
        raise ValueError("Invalid GARCH volatility or Student-t nu")

    df = df.rename(
        columns={
            "mu_hat_95": "mu_hat",
            "sigma_hat_95": "sigma_hat",
            "nu_95": "nu",
        }
    )

    df["z"] = (
        (df["realized_return"] - df["mu_hat"])
        / df["sigma_hat"]
    )

    t_scale = np.sqrt(
        (df["nu"] - 2.0) / df["nu"]
    )

    df["pit"] = stats.t.cdf(
        df["z"] / t_scale,
        df=df["nu"],
    )

    if not np.isfinite(df["pit"]).all():
        raise ValueError("Non-finite PIT values")

    df["normal_score"] = stats.norm.ppf(
        np.clip(
            df["pit"],
            np.finfo(float).eps,
            1 - np.finfo(float).eps,
        )
    )

    for level, tail in [(95, 0.05), (99, 0.01)]:
        pit_exceed = df["pit"] < tail

        var_exceed = (
            -df["realized_return"]
            > df[f"var_forecast_{level}"]
        )

        if not np.array_equal(pit_exceed, var_exceed):
            raise ValueError(
                f"PIT/VaR exceedance mismatch at {level}%"
            )

    columns = [
        "Date",
        "strategy",
        "realized_return",
        "mu_hat",
        "sigma_hat",
        "nu",
        "z",
        "pit",
        "normal_score",
    ]

    return df[columns].sort_values(
        ["strategy", "Date"]
    ).reset_index(drop=True)


def diagnose(group):
    z = group["z"].to_numpy(dtype=float)
    u = group["pit"].to_numpy(dtype=float)
    g = group["normal_score"].to_numpy(dtype=float)

    ks = stats.kstest(u, "uniform")

    lb_z = acorr_ljungbox(
        z, lags=[10], return_df=True
    )

    lb_z2 = acorr_ljungbox(
        z ** 2, lags=[10], return_df=True
    )

    lb_g = acorr_ljungbox(
        g, lags=[10], return_df=True
    )

    lb_g2 = acorr_ljungbox(
        g ** 2, lags=[10], return_df=True
    )

    return {
        "n_days": len(group),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=1)),
        "z_skew": float(stats.skew(z, bias=False)),
        "z_excess_kurtosis": float(
            stats.kurtosis(z, fisher=True, bias=False)
        ),
        "pit_ks_statistic": float(ks.statistic),
        "pit_ks_pvalue_naive": float(ks.pvalue),
        "pit_under_01": float(np.mean(u < 0.01)),
        "pit_under_05": float(np.mean(u < 0.05)),
        "pit_over_95": float(np.mean(u > 0.95)),
        "pit_over_99": float(np.mean(u > 0.99)),
        "lb_z_p10": float(
            lb_z["lb_pvalue"].iloc[0]
        ),
        "lb_z2_p10": float(
            lb_z2["lb_pvalue"].iloc[0]
        ),
        "lb_normal_score_p10": float(
            lb_g["lb_pvalue"].iloc[0]
        ),
        "lb_normal_score2_p10": float(
            lb_g2["lb_pvalue"].iloc[0]
        ),
    }


def plot_one(group, name, path):
    u = group["pit"].to_numpy(dtype=float)
    g = group["normal_score"].to_numpy(dtype=float)
    dates = group["Date"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].hist(
        u,
        bins=np.linspace(0, 1, 21),
        density=True,
        alpha=0.75,
        edgecolor="white",
    )

    axes[0, 0].axhline(
        1, linestyle="--", color="gray"
    )
    axes[0, 0].set_title(
        "PIT histogram (ideal: uniform)"
    )
    axes[0, 0].set_xlabel("PIT")

    expected = (
        np.arange(1, len(u) + 1) - 0.5
    ) / len(u)

    axes[0, 1].plot(
        expected, np.sort(u), linewidth=1
    )
    axes[0, 1].plot(
        [0, 1], [0, 1], "--", color="gray"
    )
    axes[0, 1].set_title(
        "PIT Q-Q plot vs Uniform(0,1)"
    )
    axes[0, 1].set_xlabel(
        "Theoretical uniform quantile"
    )
    axes[0, 1].set_ylabel(
        "Observed PIT quantile"
    )

    thresholds = np.array(
        [0.005, 0.01, 0.015, 0.025, 0.05, 0.075, 0.10]
    )

    frequencies = np.array(
        [np.mean(u < p) for p in thresholds]
    )

    axes[0, 2].plot(
        thresholds * 100,
        frequencies * 100,
        "o-",
        label="Observed",
    )
    axes[0, 2].plot(
        [0, 10],
        [0, 10],
        "--",
        color="gray",
        label="Ideal",
    )
    axes[0, 2].set_title(
        "Left-tail probability calibration"
    )
    axes[0, 2].set_xlabel(
        "Nominal lower-tail probability (%)"
    )
    axes[0, 2].set_ylabel(
        "Observed frequency (%)"
    )
    axes[0, 2].legend()

    theoretical, observed = stats.probplot(
        g,
        dist="norm",
        fit=False,
    )

    axes[1, 0].scatter(
        theoretical,
        observed,
        s=6,
        alpha=0.4,
    )

    bounds = [
        min(theoretical[0], observed[0]),
        max(theoretical[-1], observed[-1]),
    ]

    axes[1, 0].plot(
        bounds, bounds, "--", color="gray"
    )

    axes[1, 0].set_title(
        "PIT normal-score Q-Q plot"
    )
    axes[1, 0].set_xlabel(
        "Standard normal quantiles"
    )
    axes[1, 0].set_ylabel(
        "Transformed PIT quantiles"
    )

    lags = np.arange(1, 21)
    g_series = pd.Series(g)

    acf_g = [
        g_series.autocorr(lag=int(k))
        for k in lags
    ]

    acf_g2 = [
        (g_series ** 2).autocorr(lag=int(k))
        for k in lags
    ]

    axes[1, 1].plot(
        lags,
        acf_g,
        "o-",
        markersize=3,
        label="Normal score",
    )

    axes[1, 1].plot(
        lags,
        acf_g2,
        "o-",
        markersize=3,
        label="Score squared",
    )

    bound = 1.96 / np.sqrt(len(g))

    axes[1, 1].axhline(
        bound, color="gray", linestyle="--"
    )
    axes[1, 1].axhline(
        -bound, color="gray", linestyle="--"
    )
    axes[1, 1].axhline(
        0, color="gray", linewidth=0.7
    )
    axes[1, 1].set_title(
        "ACF after probability transform"
    )
    axes[1, 1].set_xlabel(
        "Trading-day lag"
    )
    axes[1, 1].legend()

    rolling = (
        pd.Series(g ** 2)
        .rolling(63, min_periods=63)
        .mean()
    )

    axes[1, 2].plot(
        dates,
        rolling,
        linewidth=0.9,
    )
    axes[1, 2].axhline(
        1, linestyle="--", color="gray"
    )

    axes[1, 2].set_title(
        "63-day mean squared normal score"
    )
    axes[1, 2].set_ylabel(
        "Mean squared normal score"
    )

    for ax in axes.ravel():
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{name} | GARCH-Student-t OOS diagnostics",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def verify_previous_backtest(summary):
    if not BACKTEST_PATH.exists():
        print(
            "Previous VaR summary not found; "
            "skipped count reconciliation"
        )
        return

    previous = pd.read_csv(BACKTEST_PATH)

    needed = {
        "model", "strategy", "confidence_level",
        "n_days", "n_exceedances",
    }

    if not needed.issubset(previous.columns):
        raise ValueError(
            "Backtest summary is missing required columns"
        )

    previous = previous.loc[
        previous["model"] == "garch_student_t"
    ]

    if len(previous) != 10:
        raise ValueError(
            "Expected ten Student-t backtest groups"
        )

    for _, row in previous.iterrows():
        match = summary.loc[
            summary["strategy"] == row["strategy"]
        ]

        if (
            len(match) != 1
            or row["n_days"] != int(
                match["n_days"].iloc[0]
            )
        ):
            raise ValueError(
                "Mismatch in backtest sample sizes"
            )

        if np.isclose(row["confidence_level"], 0.95):
            tail_col = "pit_under_05"
        elif np.isclose(row["confidence_level"], 0.99):
            tail_col = "pit_under_01"
        else:
            raise ValueError(
                "Unexpected backtest confidence level"
            )

        implied_count = int(
            round(
                match[tail_col].iloc[0]
                * row["n_days"]
            )
        )

        if implied_count != row["n_exceedances"]:
            raise ValueError(
                "PIT exceedances do not match "
                "previous VaR backtest"
            )

    print(
        "PIT and existing VaR backtest: "
        "10/10 groups matched"
    )


def main():
    df = load_data()

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for strategy, name in STRATEGIES.items():
        group = df.loc[
            df["strategy"] == strategy
        ].sort_values("Date")

        records.append({
            "strategy": strategy,
            **diagnose(group),
        })

        plot_one(
            group,
            name,
            FIGURE_DIR / f"{strategy}.png",
        )

    summary = pd.DataFrame(records)

    verify_previous_backtest(summary)

    daily_output = (
        DATA_DIR / "garch_student_t_pit_daily.csv"
    )

    summary_output = (
        DATA_DIR / "garch_student_t_pit_diagnostics_summary.csv"
    )

    df.to_csv(
        daily_output,
        index=False,
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print(
        "\nGARCH-STUDENT-T "
        "OUT-OF-SAMPLE PIT DIAGNOSTICS"
    )

    print(
        f"Dates: {df.Date.min().date()} "
        f"to {df.Date.max().date()}"
    )

    print(
        f"Rows: {len(df):,} | "
        f"Days per strategy: {df.Date.nunique():,}"
    )

    print("\nDIAGNOSTIC SUMMARY")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )

    print(f"\nSaved: {daily_output}")
    print(f"Saved: {summary_output}")
    print(f"Saved 5 figures in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
