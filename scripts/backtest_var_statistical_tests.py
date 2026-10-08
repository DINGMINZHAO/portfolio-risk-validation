from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data/processed/qpm/var_backtest_daily.csv"
)

OUTPUT_PATH = (
    ROOT / "data/processed/qpm/var_backtest_statistical_tests.csv"
)

SIGNIFICANCE = 0.05


def bernoulli_log_likelihood(successes, trials, probability):
    if trials == 0:
        return 0.0

    total = 0.0

    if successes > 0:
        if probability <= 0:
            return -np.inf

        total += successes * np.log(probability)

    failures = trials - successes

    if failures > 0:
        if probability >= 1:
            return -np.inf

        total += failures * np.log1p(-probability)

    return float(total)


def maximum_log_likelihood(successes, trials):
    if trials == 0:
        return 0.0

    probability = successes / trials

    return bernoulli_log_likelihood(
        successes, trials, probability
    )


def test_one_series(hits, confidence_level):
    hits = np.asarray(hits, dtype=int)

    n = len(hits)

    if n < 2 or not np.isin(hits, [0, 1]).all():
        raise ValueError("Invalid exceedance sequence.")

    n_exceedances = int(hits.sum())

    expected_probability = 1 - confidence_level

    # Kupiec unconditional coverage test

    ll_null = bernoulli_log_likelihood(
        n_exceedances,
        n,
        expected_probability
    )

    ll_fitted = maximum_log_likelihood(
        n_exceedances,
        n
    )

    lr_uc = max(
        0.0,
        2 * (ll_fitted - ll_null)
    )

    p_uc = float(chi2.sf(lr_uc, df=1))

    # Christoffersen independence test

    previous = hits[:-1]
    current = hits[1:]

    states = previous * 2 + current

    n00, n01, n10, n11 = map(
        int,
        np.bincount(states, minlength=4)
    )

    from_zero = n00 + n01
    from_one = n10 + n11

    p01 = (
        n01 / from_zero
        if from_zero > 0
        else np.nan
    )

    p11 = (
        n11 / from_one
        if from_one > 0
        else np.nan
    )

    if from_zero == 0 or from_one == 0:
        lr_ind = np.nan
        p_ind = np.nan

        lr_cc = np.nan
        p_cc = np.nan

    else:
        ll_independent = maximum_log_likelihood(
            n01 + n11,
            n - 1
        )

        ll_markov = (
            maximum_log_likelihood(n01, from_zero)
            + maximum_log_likelihood(n11, from_one)
        )

        lr_ind = max(
            0.0,
            2 * (ll_markov - ll_independent)
        )

        p_ind = float(
            chi2.sf(lr_ind, df=1)
        )

        # Christoffersen conditional coverage test

        lr_cc = lr_uc + lr_ind

        p_cc = float(
            chi2.sf(lr_cc, df=2)
        )

    return {
        "n_days": n,
        "n_exceedances": n_exceedances,
        "expected_rate": expected_probability,
        "observed_rate": n_exceedances / n,

        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,

        "p01": p01,
        "p11": p11,

        "lr_uc": lr_uc,
        "kupiec_p_value": p_uc,

        "lr_ind": lr_ind,
        "independence_p_value": p_ind,

        "lr_cc": lr_cc,
        "conditional_coverage_p_value": p_cc,
    }


def main():
    data = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"]
    )

    required = [
        "Date",
        "model",
        "strategy",
        "confidence_level",
        "var_forecast",
        "realized_return",
        "exceedance"
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    keys = [
        "model",
        "strategy",
        "confidence_level",
        "Date"
    ]

    if (
        data[keys].isna().any().any()
        or data.duplicated(keys).any()
    ):
        raise ValueError(
            "Missing or duplicate backtest keys."
        )

    numeric = data[
        [
            "confidence_level",
            "var_forecast",
            "realized_return",
            "exceedance"
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite numeric values.")

    if not data["confidence_level"].between(
        0, 1, inclusive="neither"
    ).all():
        raise ValueError("Invalid confidence levels.")

    if not data["exceedance"].isin([0, 1]).all():
        raise ValueError(
            "Exceedance values must be 0 or 1."
        )

    calculated_hits = (
        -data["realized_return"]
        > data["var_forecast"]
    ).astype(int)

    if not calculated_hits.eq(
        data["exceedance"]
    ).all():
        raise ValueError(
            "Exceedance flags do not match actual losses."
        )

    data = data.sort_values(keys).reset_index(drop=True)

    records = []
    reference_dates = None

    groups = data.groupby(
        ["model", "strategy", "confidence_level"],
        sort=True
    )

    for (model, strategy, confidence), group in groups:
        dates = group["Date"].to_numpy()

        if reference_dates is None:
            reference_dates = dates

        elif not np.array_equal(dates, reference_dates):
            raise ValueError(
                "Model groups do not share the same dates."
            )

        hits = group["exceedance"].to_numpy()

        result = test_one_series(
            hits,
            confidence
        )

        result["model"] = model
        result["strategy"] = strategy
        result["confidence_level"] = confidence

        result["start_date"] = (
            group["Date"].iloc[0].date()
        )

        result["end_date"] = (
            group["Date"].iloc[-1].date()
        )

        records.append(result)

    summary = pd.DataFrame(records)

    columns = [
        "model",
        "strategy",
        "confidence_level",
        "start_date",
        "end_date",
        "n_days",
        "n_exceedances",
        "expected_rate",
        "observed_rate",
        "n00",
        "n01",
        "n10",
        "n11",
        "p01",
        "p11",
        "lr_uc",
        "kupiec_p_value",
        "lr_ind",
        "independence_p_value",
        "lr_cc",
        "conditional_coverage_p_value"
    ]

    summary = summary[columns]

    summary = summary.sort_values(
        ["confidence_level", "strategy", "model"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nVAR STATISTICAL BACKTEST RESULTS")
    print(f"Significance level: {SIGNIFICANCE:.0%}")

    for confidence, group in summary.groupby(
        "confidence_level"
    ):
        print(
            f"\nConfidence level: {confidence:.0%}"
        )

        display_columns = [
            "model",
            "strategy",
            "n_exceedances",
            "observed_rate",
            "p01",
            "p11",
            "kupiec_p_value",
            "independence_p_value",
            "conditional_coverage_p_value"
        ]

        print(
            group[display_columns].to_string(
                index=False,
                formatters={
                    "observed_rate": lambda x: f"{x:.2%}",
                    "p01": lambda x: f"{x:.3%}",
                    "p11": lambda x: f"{x:.3%}",
                    "kupiec_p_value": lambda x: f"{x:.4g}",
                    "independence_p_value": lambda x: f"{x:.4g}",
                    "conditional_coverage_p_value": lambda x: f"{x:.4g}"
                }
            )
        )

    print(f"\nSaved results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()