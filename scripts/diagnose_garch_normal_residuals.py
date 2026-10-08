
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


ROOT = Path(__file__).resolve().parents[1]

RETURNS_PATH = (
    ROOT / "data/interim/qpm/portfolio_returns_main.csv"
)

FORECAST_PATH = (
    ROOT / "data/processed/qpm/garch_normal_var_es_forecasts.csv"
)

OUTPUT_DIR = ROOT / "data/processed/qpm"

FIGURE_DIR = (
    ROOT / "reports/figures/qpm/garch_normal_diagnostics"
)

STRATEGIES = {
    "ew_static": "Equal Weight",
    "minvar_static": "MinVar Static",
    "minvar_regime": "MinVar Regime-aware",
    "meanvar_static": "MeanVar Static",
    "meanvar_regime": "MeanVar Regime-aware",
}


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        parse_dates=["Date"]
    )

    predictions = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["Date", "information_end_date"]
    )

    keys = ["Date", "strategy", "confidence_level"]

    if (
        returns["Date"].isna().any()
        or returns["Date"].duplicated().any()
    ):
        raise ValueError("Invalid return dates")

    if (
        predictions[keys].isna().any().any()
        or predictions.duplicated(keys).any()
    ):
        raise ValueError("Invalid forecast keys")

    if set(predictions["strategy"]) != set(STRATEGIES):
        raise ValueError("Unexpected strategies")

    if set(predictions["confidence_level"]) != {0.95, 0.99}:
        raise ValueError("Unexpected confidence levels")

    if not predictions["model"].eq("garch_normal").all():
        raise ValueError("Unexpected forecast model")

    if not (
        predictions["information_end_date"] < predictions["Date"]
    ).all():
        raise ValueError("Look-ahead detected")

    p95 = predictions[
        predictions["confidence_level"] == 0.95
    ].copy()

    p99 = predictions[
        predictions["confidence_level"] == 0.99
    ].copy()

    matching = p95.merge(
        p99[["Date", "strategy", "mu_hat", "sigma_hat"]],
        on=["Date", "strategy"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_99"),
        indicator=True,
    )

    if not matching["_merge"].eq("both").all():
        raise ValueError("95% and 99% forecasts differ in coverage")

    if not np.allclose(
        matching[["mu_hat", "sigma_hat"]].to_numpy(dtype=float),
        matching[["mu_hat_99", "sigma_hat_99"]].to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("Different mu or sigma across levels")

    dates = sorted(p95["Date"].unique())

    if (
        len(dates) != 3839
        or len(p95) != 3839 * len(STRATEGIES)
    ):
        raise ValueError("Unexpected GARCH forecast coverage")

    if not (
        p95.groupby("strategy")["Date"].nunique().eq(3839)
    ).all():
        raise ValueError("Incomplete strategy coverage")

    actual = returns[
        ["Date", *STRATEGIES]
    ].melt(
        id_vars="Date",
        var_name="strategy",
        value_name="realized_return",
    )

    df = p95.merge(
        actual,
        on=["Date", "strategy"],
        how="left",
        validate="one_to_one",
    )

    values = df[
        ["realized_return", "mu_hat", "sigma_hat"]
    ].to_numpy(dtype=float)

    if (
        not np.isfinite(values).all()
        or (df["sigma_hat"] <= 0).any()
    ):
        raise ValueError("Invalid returns or GARCH forecasts")

    df["z"] = (
        (df["realized_return"] - df["mu_hat"])
        / df["sigma_hat"]
    )

    return df[
        [
            "Date",
            "strategy",
            "realized_return",
            "mu_hat",
            "sigma_hat",
            "z",
        ]
    ].sort_values(
        ["strategy", "Date"]
    ).reset_index(drop=True)


def describe_one(group):

    z = group["z"].to_numpy()

    q01, q05 = stats.norm.ppf([0.01, 0.05])

    lb_z = acorr_ljungbox(
        z,
        lags=[10],
        return_df=True,
    )

    lb_z2 = acorr_ljungbox(
        z ** 2,
        lags=[10],
        return_df=True,
    )

    return {
        "n_days": len(z),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z, ddof=1)),
        "z_skew": float(stats.skew(z, bias=False)),
        "z_excess_kurtosis": float(
            stats.kurtosis(z, fisher=True, bias=False)
        ),
        "z_quantile_01": float(np.quantile(z, 0.01)),
        "z_quantile_05": float(np.quantile(z, 0.05)),
        "left_tail_rate_01": float(np.mean(z < q01)),
        "left_tail_rate_05": float(np.mean(z < q05)),
        "acf_z2_lag1": float(
            pd.Series(z ** 2).autocorr(lag=1)
        ),
        "lb_z_p10": float(
            lb_z["lb_pvalue"].iloc[0]
        ),
        "lb_z2_p10": float(
            lb_z2["lb_pvalue"].iloc[0]
        ),
    }


def plot_one(group, name, path):

    z = group["z"].to_numpy()
    dates = group["Date"].to_numpy()

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9)
    )

    # Distribution versus normal

    x = np.linspace(-5, 5, 500)

    axes[0, 0].hist(
        z,
        bins=np.linspace(-5, 5, 61),
        density=True,
        alpha=0.65,
    )

    axes[0, 0].plot(
        x,
        stats.norm.pdf(x),
        linewidth=2,
        label="N(0, 1)",
    )

    axes[0, 0].set_xlim(-5, 5)
    axes[0, 0].set_title(
        "Standardized innovations vs normal"
    )
    axes[0, 0].set_xlabel("z")
    axes[0, 0].legend()

    # Normal Q-Q plot

    theoretical, observed = stats.probplot(
        z,
        dist="norm",
        fit=False,
    )

    axes[0, 1].scatter(
        theoretical,
        observed,
        s=8,
        alpha=0.45,
    )

    ends = [
        min(theoretical[0], observed[0]),
        max(theoretical[-1], observed[-1]),
    ]

    axes[0, 1].plot(
        ends,
        ends,
        "--",
        color="gray",
        linewidth=1,
    )

    axes[0, 1].set_title(
        "Normal Q-Q plot against N(0,1)"
    )
    axes[0, 1].set_xlabel(
        "Theoretical normal quantile"
    )
    axes[0, 1].set_ylabel(
        "Observed standardized quantile"
    )

    # ACF of standardized innovations

    lags = np.arange(1, 21)

    acf_z = [
        pd.Series(z).autocorr(lag=int(k))
        for k in lags
    ]

    acf_z2 = [
        pd.Series(z ** 2).autocorr(lag=int(k))
        for k in lags
    ]

    axes[1, 0].plot(
        lags,
        acf_z,
        marker="o",
        markersize=3,
        label="z",
    )

    axes[1, 0].plot(
        lags,
        acf_z2,
        marker="o",
        markersize=3,
        label="z squared",
    )

    bound = 1.96 / np.sqrt(len(z))

    axes[1, 0].axhline(
        bound,
        linestyle="--",
        linewidth=1,
        color="gray",
    )

    axes[1, 0].axhline(
        -bound,
        linestyle="--",
        linewidth=1,
        color="gray",
    )

    axes[1, 0].axhline(
        0,
        color="gray",
        linewidth=0.7,
    )

    axes[1, 0].set_title(
        "ACF of standardized innovations"
    )
    axes[1, 0].set_xlabel(
        "Lag in trading days"
    )
    axes[1, 0].legend()

    # Rolling standardized variance

    rolling = (
        pd.Series(z ** 2)
        .rolling(63, min_periods=63)
        .mean()
    )

    axes[1, 1].plot(
        dates,
        rolling,
        linewidth=0.9,
    )

    axes[1, 1].axhline(
        1,
        linestyle="--",
        linewidth=1,
        color="gray",
        label="Reference: 1",
    )

    axes[1, 1].set_title(
        "63-day rolling mean of squared z"
    )

    axes[1, 1].set_ylabel(
        "Mean squared standardized innovation"
    )

    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"{name}: GARCH-Normal OOS Diagnostics",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    df = load_data()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for strategy, name in STRATEGIES.items():

        group = df[
            df["strategy"] == strategy
        ].sort_values("Date")

        result = describe_one(group)

        rows.append({
            "strategy": strategy,
            **result,
        })

        plot_one(
            group,
            name,
            FIGURE_DIR / f"{strategy}.png",
        )

    summary = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_DIR / "garch_normal_standardized_innovations.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "garch_normal_residual_diagnostics_summary.csv",
        index=False,
    )

    pd.set_option("display.width", 200)

    print(
        "\nGARCH-NORMAL OUT-OF-SAMPLE "
        "STANDARDIZED INNOVATIONS"
    )

    print(
        f"Date range: {df.Date.min().date()} "
        f"to {df.Date.max().date()}"
    )

    print(
        f"Observations: {len(df):,}"
    )

    print("\nRESIDUAL DIAGNOSTICS SUMMARY")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nSaved CSVs to: {OUTPUT_DIR}")
    print(f"Saved 5 figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
