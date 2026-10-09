
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/qpm"
FIGURE_DIR = ROOT / "reports/figures/qpm/density_estimation"

GMM_PATH = DATA_DIR / "garch_gmm_minvar_static_daily.csv"
FHS_PATH = DATA_DIR / "garch_fhs_var_es_forecasts.csv"

DAILY_OUTPUT = DATA_DIR / "gmm_fhs_tail_daily_comparison.csv"
EVENT_OUTPUT = DATA_DIR / "gmm_fhs_99_tail_events.csv"
FIGURE_OUTPUT = FIGURE_DIR / "gmm_fhs_tail_event_comparison.png"

STRATEGY = "minvar_static"
EXPECTED_DAYS = 2831


def read_data():
    data = pd.read_csv(
        GMM_PATH,
        parse_dates=["Date"],
    )

    source = pd.read_csv(
        FHS_PATH,
        parse_dates=["Date", "information_end_date"],
    )

    data = data.loc[
        data["strategy"] == STRATEGY
    ].copy()

    source = source.loc[
        (source["strategy"] == STRATEGY)
        & (source["model"] == "garch_fhs")
        & np.isclose(source["confidence_level"], 0.99)
    ].copy()

    required = [
        "Date",
        "realized_return",
        "realized_loss",
        "mu_hat",
        "sigma_hat",
        "fhs_var_99",
        "gmm_var_99",
        "fhs_es_99",
        "gmm_es_99",
        "fhs_hit_99",
        "gmm_hit_99",
    ]

    missing = sorted(set(required) - set(data.columns))

    if missing:
        raise ValueError(f"Missing GMM columns: {missing}")

    if len(data) != EXPECTED_DAYS:
        raise ValueError("Unexpected GMM sample length.")

    if data["Date"].duplicated().any():
        raise ValueError("Duplicate GMM dates.")

    if source["Date"].duplicated().any():
        raise ValueError("Duplicate FHS dates.")

    source = source[
        [
            "Date",
            "information_end_date",
            "mu_hat",
            "sigma_hat",
            "var_forecast",
            "es_forecast",
            "initial_in_sample_residuals",
        ]
    ].rename(
        columns={
            "mu_hat": "source_mu",
            "sigma_hat": "source_sigma",
            "var_forecast": "source_var",
            "es_forecast": "source_es",
        }
    )

    data = data.merge(
        source,
        on="Date",
        how="left",
        validate="one_to_one",
    )

    data = data.sort_values(
        "Date"
    ).reset_index(drop=True)

    if data.isna().any().any():
        raise ValueError("Missing values after joining.")

    if not data["initial_in_sample_residuals"].eq(0).all():
        raise ValueError("Non-OOS residual pools detected.")

    if (
        data["information_end_date"] >= data["Date"]
    ).any():
        raise ValueError("Look-ahead detected.")

    checks = [
        ("mu_hat", "source_mu"),
        ("sigma_hat", "source_sigma"),
        ("fhs_var_99", "source_var"),
        ("fhs_es_99", "source_es"),
    ]

    for left, right in checks:
        if not np.allclose(
            data[left],
            data[right],
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(
                f"Independent FHS reconciliation failed: {left}"
            )

    if not np.allclose(
        data["realized_loss"],
        -data["realized_return"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Realized loss mismatch.")

    if (data["sigma_hat"] <= 0).any():
        raise ValueError("Invalid conditional volatility.")

    for model in ["fhs", "gmm"]:
        var = data[f"{model}_var_99"]
        es = data[f"{model}_es_99"]

        if (es < var - 1e-10).any():
            raise ValueError(f"ES below VaR: {model}")

        hits = (
            data["realized_loss"] > var
        ).astype(int)

        if not np.array_equal(
            hits.to_numpy(),
            data[f"{model}_hit_99"].to_numpy(dtype=int),
        ):
            raise ValueError(
                f"Incorrect exceedance flags: {model}"
            )

    return data


def build_comparison(data):
    result = data.copy()

    fhs_hit = result["fhs_hit_99"].astype(bool)
    gmm_hit = result["gmm_hit_99"].astype(bool)

    result["event_type"] = np.select(
        [
            fhs_hit & gmm_hit,
            fhs_hit & ~gmm_hit,
            ~fhs_hit & gmm_hit,
        ],
        [
            "both",
            "fhs_only",
            "gmm_only",
        ],
        default="neither",
    )

    result["loss_pct"] = (
        100 * result["realized_loss"]
    )

    for model in ["fhs", "gmm"]:
        result[f"{model}_var_pct"] = (
            100 * result[f"{model}_var_99"]
        )

        result[f"{model}_es_pct"] = (
            100 * result[f"{model}_es_99"]
        )

        result[f"{model}_excess_pct"] = (
            100 * (
                result["realized_loss"]
                - result[f"{model}_var_99"]
            )
        )

    result["delta_var_bp"] = (
        10000
        * (
            result["gmm_var_99"]
            - result["fhs_var_99"]
        )
    )

    result["delta_es_bp"] = (
        10000
        * (
            result["gmm_es_99"]
            - result["fhs_es_99"]
        )
    )

    events = result.loc[
        result["event_type"] != "neither"
    ].copy()

    return result, events


def print_summary(result, events):
    both = int(
        (result["event_type"] == "both").sum()
    )

    fhs_only = int(
        (result["event_type"] == "fhs_only").sum()
    )

    gmm_only = int(
        (result["event_type"] == "gmm_only").sum()
    )

    union = len(events)

    print("\nGMM vs. FHS: 99% TAIL EVENT STUDY")
    print("=" * 85)

    print(f"Strategy: {STRATEGY}")
    print(f"Trading days: {len(result)}")

    print(
        f"Period: {result['Date'].min().date()} "
        f"to {result['Date'].max().date()}"
    )

    print("\nEVENT OVERLAP")

    print(f"FHS hits:       {both + fhs_only}")
    print(f"GMM hits:       {both + gmm_only}")
    print(f"Both models:    {both}")
    print(f"FHS only:       {fhs_only}")
    print(f"GMM only:       {gmm_only}")
    print(f"Union:          {union}")

    if union:
        print(
            f"Jaccard overlap: {both / union:.4f}"
        )

    print("\nDAILY FORECAST DIFFERENCES")

    print(
        "Positive differences mean GMM "
        "predicts higher tail risk."
    )

    print(
        "Mean GMM - FHS VaR: "
        f"{result['delta_var_bp'].mean():+.4f} bps"
    )

    print(
        "Mean absolute VaR difference: "
        f"{result['delta_var_bp'].abs().mean():.4f} bps"
    )

    print(
        "Mean GMM - FHS ES: "
        f"{result['delta_es_bp'].mean():+.4f} bps"
    )

    print(
        "Mean absolute ES difference: "
        f"{result['delta_es_bp'].abs().mean():.4f} bps"
    )

    if union:
        print("\nON UNION EXCEEDANCE DAYS")

        print(
            "Mean GMM - FHS VaR: "
            f"{events['delta_var_bp'].mean():+.4f} bps"
        )

        print(
            "Mean GMM - FHS ES: "
            f"{events['delta_es_bp'].mean():+.4f} bps"
        )

        print("\nALL UNION EVENTS")

        columns = [
            "Date",
            "event_type",
            "loss_pct",
            "fhs_var_pct",
            "gmm_var_pct",
            "delta_var_bp",
            "fhs_es_pct",
            "gmm_es_pct",
            "delta_es_bp",
        ]

        display = events[columns].copy()

        display["Date"] = (
            display["Date"].dt.strftime("%Y-%m-%d")
        )

        print(
            display.to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}",
            )
        )

    print("\nAPRIL 2025: DAILY FORECAST COMPARISON")

    april = result.loc[
        result["Date"].between(
            "2025-04-01",
            "2025-04-17",
        )
    ].copy()

    april_columns = [
        "Date",
        "event_type",
        "loss_pct",
        "fhs_var_pct",
        "gmm_var_pct",
        "delta_var_bp",
        "fhs_es_pct",
        "gmm_es_pct",
        "delta_es_bp",
    ]

    april = april[april_columns]

    april["Date"] = (
        april["Date"].dt.strftime("%Y-%m-%d")
    )

    print(
        april.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def plot_comparison(result, events):
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True,
    )

    dates = result["Date"]

    axes[0].plot(
        dates,
        result["delta_var_bp"],
        linewidth=1,
        color="#2563a6",
    )

    axes[0].axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_title(
        "99% VaR Forecast Difference: GMM - FHS",
        loc="left",
    )

    axes[0].set_ylabel("Basis points")

    axes[1].plot(
        dates,
        result["delta_es_bp"],
        linewidth=1,
        color="#7c3aed",
    )

    axes[1].axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
    )

    axes[1].set_title(
        "99% ES Forecast Difference: GMM - FHS",
        loc="left",
    )

    axes[1].set_ylabel("Basis points")

    colors = {
        "both": "#c0392b",
        "fhs_only": "#d97706",
        "gmm_only": "#15803d",
    }

    names = {
        "both": "Both exceed",
        "fhs_only": "FHS only",
        "gmm_only": "GMM only",
    }

    for event_type in colors:
        sample = events.loc[
            events["event_type"] == event_type
        ]

        if len(sample):
            axes[0].scatter(
                sample["Date"],
                sample["delta_var_bp"],
                s=28,
                color=colors[event_type],
                zorder=5,
                label=names[event_type],
            )

    axes[0].legend(
        frameon=False,
        ncol=3,
    )

    axes[2].scatter(
        events["Date"],
        events["loss_pct"],
        marker="x",
        s=45,
        color="#34495e",
        label="Realized loss",
    )

    axes[2].scatter(
        events["Date"],
        events["fhs_var_pct"],
        marker="o",
        s=23,
        color="#2563a6",
        label="FHS 99% VaR",
    )

    axes[2].scatter(
        events["Date"],
        events["gmm_var_pct"],
        marker="^",
        s=28,
        color="#d97706",
        label="GMM 99% VaR",
    )

    axes[2].set_title(
        "Realized Loss and VaR on Union Exceedance Days",
        loc="left",
    )

    axes[2].set_ylabel("Portfolio loss (%)")
    axes[2].set_xlabel("Trading date")

    axes[2].legend(
        frameon=False,
        ncol=3,
    )

    for ax in axes:
        ax.grid(alpha=0.2)

    axes[-1].xaxis.set_major_locator(
        mdates.YearLocator(2)
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    fig.suptitle(
        "GARCH-GMM vs. GARCH-FHS: Tail Event Comparison\n"
        f"{STRATEGY} | 2014-2025",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        FIGURE_OUTPUT,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    data = read_data()

    print("Independent FHS reconciliation passed.")

    result, events = build_comparison(data)

    DAILY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        DAILY_OUTPUT,
        index=False,
    )

    events.to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    plot_comparison(result, events)

    print_summary(result, events)

    print("\nSAVED OUTPUTS")
    print(DAILY_OUTPUT)
    print(EVENT_OUTPUT)
    print(FIGURE_OUTPUT)

    print("\nTail event study completed.")


if __name__ == "__main__":
    main()
