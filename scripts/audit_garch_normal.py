from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]

RETURNS = ROOT / "data/interim/qpm/portfolio_returns_main.csv"
FORECAST = ROOT / "data/processed/qpm/garch_normal_var_es_forecasts.csv"
PARAMS = ROOT / "data/processed/qpm/garch_normal_fit_parameters.csv"

r = pd.read_csv(RETURNS)

p = pd.read_csv(
    FORECAST,
    parse_dates=[
        "Date",
        "estimation_start_date",
        "estimation_end_date",
        "information_end_date"
    ]
)

f = pd.read_csv(
    PARAMS,
    parse_dates=[
        "refit_date",
        "estimation_start_date",
        "estimation_end_date"
    ]
)

if "Date" not in r.columns:
    r = r.rename(columns={r.columns[0]: "Date"})

r["Date"] = pd.to_datetime(r["Date"])
r = r.sort_values("Date").reset_index(drop=True)

print("\n=== 1. DATA INTEGRITY ===")

assert not r["Date"].duplicated().any()
assert not p.duplicated(
    ["Date", "strategy", "confidence_level"]
).any()
assert not f.duplicated(
    ["strategy", "refit_date"]
).any()

assert not p.isna().any().any()
assert not f.isna().any().any()

assert np.isfinite(
    p.select_dtypes("number").to_numpy()
).all()

assert np.isfinite(
    f.select_dtypes("number").to_numpy()
).all()

assert p["window_days"].eq(1008).all()
assert p["refit_every_days"].eq(21).all()

strategies = sorted(f["strategy"].unique())
dates = pd.DatetimeIndex(sorted(p["Date"].unique()))
refits = pd.DatetimeIndex(dates[::21])

assert len(strategies) == 5
assert set(strategies) == set(p["strategy"])
assert set(strategies) <= set(r.columns)

assert len(dates) == 3839
assert len(refits) == 183
assert len(p) == 38390
assert len(f) == 915

assert set(f["refit_date"]) == set(refits)
assert p.groupby(
    ["Date", "strategy"]
).size().eq(2).all()

assert set(p["confidence_level"].unique()) == {
    0.95, 0.99
}

assert (p["Date"] > p["information_end_date"]).all()
assert (
    p["information_end_date"] >= p["estimation_end_date"]
).all()

print("Data integrity: PASS")


print("\n=== 2. VAR / ES FORMULA ===")

z = norm.ppf(p["confidence_level"].to_numpy())

var = (
    z * p["sigma_hat"].to_numpy()
    - p["mu_hat"].to_numpy()
)

es = (
    norm.pdf(z)
    / (1 - p["confidence_level"].to_numpy())
    * p["sigma_hat"].to_numpy()
    - p["mu_hat"].to_numpy()
)

var_error = np.max(
    np.abs(var - p["var_forecast"].to_numpy())
)

es_error = np.max(
    np.abs(es - p["es_forecast"].to_numpy())
)

print("Maximum VaR error:", var_error)
print("Maximum ES error:", es_error)

assert var_error < 1e-12
assert es_error < 1e-12
assert (p["sigma_hat"] > 0).all()
assert (
    p["es_forecast"] >= p["var_forecast"]
).all()

print("VaR / ES formulas: PASS")


print("\n=== 3. ROLLING WINDOW AND LEAKAGE ===")

rv = r.set_index("Date")

date_index = pd.Series(
    np.arange(len(r)),
    index=r["Date"]
)

window_results = []

for row in f.itertuples(index=False):

    j = date_index.loc[row.refit_date]

    correct = (
        j >= 1008
        and r.loc[
            j - 1008, "Date"
        ] == row.estimation_start_date
        and r.loc[
            j - 1, "Date"
        ] == row.estimation_end_date
    )

    window_results.append(correct)

print(
    "Correct rolling windows:",
    sum(window_results),
    "/",
    len(window_results)
)

assert all(window_results)

information_results = []

for date, info_date in p[
    ["Date", "information_end_date"]
].drop_duplicates().itertuples(index=False, name=None):

    j = date_index.loc[date]

    correct = (
        r.loc[j - 1, "Date"] == info_date
    )

    information_results.append(correct)

print(
    "Correct information cutoff:",
    sum(information_results),
    "/",
    len(information_results)
)

assert all(information_results)

print("Rolling windows and date alignment: PASS")


print("\n=== 4. GARCH VARIANCE RECURSION ===")

for strategy in strategies:

    d = p[
        (p["strategy"] == strategy)
        & (p["confidence_level"] == 0.95)
    ].set_index("Date").reindex(dates)

    fitted = f[
        f["strategy"] == strategy
    ].set_index("refit_date").reindex(
        dates,
        method="ffill"
    )

    actual_pct = (
        rv[strategy].reindex(dates).to_numpy() * 100
    )

    sigma_pct = (
        d["sigma_hat"].to_numpy() * 100
    )

    mu_pct = fitted["mu_pct"].to_numpy()
    omega = fitted["omega_pct2"].to_numpy()
    alpha = fitted["alpha"].to_numpy()
    beta = fitted["beta"].to_numpy()

    expected_next = (
        omega[:-1]
        + alpha[:-1] * (
            actual_pct[:-1] - mu_pct[:-1]
        ) ** 2
        + beta[:-1] * sigma_pct[:-1] ** 2
    )

    observed_next = sigma_pct[1:] ** 2

    inside_block = (
        np.arange(1, len(dates)) % 21 != 0
    )

    error = np.abs(
        expected_next[inside_block]
        - observed_next[inside_block]
    )

    print(
        strategy,
        "| max error:",
        float(error.max()),
        "| errors > 1e-8:",
        int((error > 1e-8).sum())
    )


print("\n=== 5. GARCH REFIT CONVERGENCE ===")

results = []

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    for row in f.itertuples(index=False):

        j = date_index.loc[row.refit_date]

        y = (
            r.iloc[j - 1008:j][row.strategy]
            .to_numpy(dtype=float) * 100
        )

        try:

            model = arch_model(
                y,
                mean="Constant",
                vol="GARCH",
                p=1,
                o=0,
                q=1,
                dist="normal",
                rescale=False
            )

            fit = model.fit(
                disp="off",
                show_warning=False
            )

            theta = fit.params

            archived = np.array([
                row.mu_pct,
                row.omega_pct2,
                row.alpha,
                row.beta
            ])

            reproduced = np.array([
                theta["mu"],
                theta["omega"],
                theta["alpha[1]"],
                theta["beta[1]"]
            ])

            difference = float(
                np.max(np.abs(archived - reproduced))
            )

            results.append([
                row.strategy,
                row.refit_date,
                int(fit.convergence_flag),
                difference,
                str(fit.optimization_result.message)
            ])

        except Exception as exc:

            results.append([
                row.strategy,
                row.refit_date,
                -999,
                np.nan,
                str(exc)
            ])

out = pd.DataFrame(
    results,
    columns=[
        "strategy",
        "refit_date",
        "convergence_flag",
        "max_parameter_difference",
        "message"
    ]
)

print("\nConvergence status counts:")

print(
    out["convergence_flag"]
    .value_counts()
    .sort_index()
)

print("\nParameter reproduction:")

print(
    out["max_parameter_difference"]
    .describe()
)

print("\nLargest parameter discrepancies:")

print(
    out.sort_values(
        "max_parameter_difference",
        ascending=False
    ).head(10).to_string(index=False)
)


print("\n=== 6. PERSISTENCE BOUNDARY CHECK ===")

edge = f.loc[
    np.isclose(
        f["alpha_plus_beta"],
        1.0,
        atol=1e-10,
        rtol=0
    ),
    ["strategy", "refit_date"]
]

print("Near-unit persistence fits:", len(edge))

boundary_results = out.merge(
    edge,
    on=["strategy", "refit_date"],
    how="inner"
)

print(
    boundary_results.to_string(index=False)
)

print("\n=== AUDIT FINISHED ===")