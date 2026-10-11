# Portfolio Risk Assessment and Model Validation

*An Independent Evaluation of Five Portfolio Strategies*

Hi, welcome! 👋

I use portfolio weights from the open-source [qpm-project](https://github.com/AhmadMujtaba200210/qpm-project) to reconstruct daily returns for five portfolio strategies. I then measure their risk, test Value at Risk (VaR) and Expected Shortfall (ES) forecasts, and see how they perform under market stress.

The main question is simple: **How do these portfolios hold up when markets get rough?**

**[Read the full report](reports/Beyond_Optimization_Research_Report.md)**

## At a glance

| | Scope |
|---|---|
| **Portfolios** | Equal Weight, MinVar Static, MinVar Regime, MeanVar Static, MeanVar Regime |
| **Assets** | SPY, EFA, EEM, TLT, IEF, GLD, VNQ, DBC |
| **Risk models** | Rolling Normal, Historical Simulation, GARCH-Normal, GARCH-Student-t, GARCH-FHS |
| **Validation** | One-day VaR/ES forecasts at 95% and 99%; Kupiec and Christoffersen tests; ES residuals and FZ0 scoring |
| **Stress tests** | COVID crash (2020), inflation and rate hikes (2022), April trade shock (2025), and three hypothetical 20-day scenarios |
| **Sample** | 2006–2025 portfolio analysis; 3,839 VaR/ES backtesting days (2010–2025) |

## Key findings

- **Portfolio risk:** All four optimized portfolios had lower volatility and smaller maximum drawdowns than Equal Weight over the full sample. MinVar Static had the lowest annualized volatility (**6.94%**), while MeanVar Regime had the highest annualized return (**9.16%**).
- **VaR backtesting:** GARCH-FHS had 99% VaR exceedance rates of **0.99%–1.20%**, close to the expected 1%. But all five portfolios failed the exceedance independence test at both 95% and 99%.
- **VaR and ES together:** GARCH-FHS had the lowest average FZ0 score among the five models for every portfolio at both confidence levels.
- **Stress testing:** The optimized portfolios generally lost less during the three selected historical shocks. They also lost less than Equal Weight in all three hypothetical scenarios using the May 2025 weights. Regime-based constraints did not always help.

## 1. Portfolio risk and return

I compare the five portfolios using annualized return, annualized volatility, and maximum drawdown. Equal Weight is the benchmark.

![Portfolio risk and return comparison](reports/figures/qpm/portfolio_risk_return.png)

| Portfolio | Annualized return | Annualized volatility | Maximum drawdown |
|---|---:|---:|---:|
| Equal Weight | 6.83% | 12.62% | −37.01% |
| MinVar Static | 5.70% | **6.94%** | −18.95% |
| MinVar Regime | 5.73% | 7.05% | −19.49% |
| MeanVar Static | 9.09% | 9.28% | −19.34% |
| MeanVar Regime | **9.16%** | 8.88% | **−18.94%** |

## 2. VaR and ES model validation

I compare five methods for forecasting one-day VaR and ES. All five models use the same **3,839 backtesting days**, from September 3, 2010, to December 8, 2025.

I check **VaR coverage, exceedance independence, conditional coverage, exceedance timing, and joint VaR–ES scores**.

![VaR backtesting results](reports/figures/qpm/var_backtesting_summary.png)

GARCH-FHS had exceedance rates close to the target levels, but VaR breaches still tended to happen in clusters. So I look at both *how often* losses exceed VaR and *when* they happen.

You can also see the [VaR–ES FZ0 score comparison](reports/figures/qpm/joint_var_es_fz0_comparison.png).

## 3. Stress testing

For **historical stress tests**, I compare portfolio returns during the 2020 COVID crash, the 2022 inflation and rate hikes, and the April 2025 trade policy shock.

![Historical stress event return paths](reports/figures/qpm/historical_stress_paths.png)

For **hypothetical stress tests**, I apply three 20-day asset return shocks—Equity Crash, Inflation and Rate Shock, and Joint Historical Worst—to portfolio weights from April 1 (Calm) and May 1 (Stressed), 2025.

[See the hypothetical stress test results](reports/figures/qpm/hypothetical_stress_snapshots.png).

## Workflow and tools

| Stage | What I did |
|---|---|
| **Data checks** | Checked prices and weights, rebuilt daily returns, and compared them with the upstream backtest |
| **Risk measurement** | Calculated annualized returns, volatility, maximum drawdown, and losses during market stress |
| **Risk forecasting** | Built Rolling Normal, Historical Simulation, GARCH-Normal, GARCH-Student-t, and GARCH-FHS forecasts |
| **Model validation** | Tested VaR coverage and independence, checked PIT and residuals, and compared ES forecasts using FZ0 scores |
| **Stress testing** | Compared historical market shocks, hypothetical scenarios, and Static versus Regime allocations |
| **Results** | Saved scripts, summary CSVs, figures, and a full report |

**Tools:** Python, pandas, NumPy, SciPy, ARCH/GARCH modeling, Matplotlib.

## Where to find things

- [`scripts/`](scripts/) — Code for data checks, risk models, backtesting, and stress tests.
- [`data/interim/qpm/`](data/interim/qpm/) — Daily asset and portfolio returns, plus portfolio weights.
- [`data/processed/qpm/`](data/processed/qpm/) — Risk forecasts, backtesting results, and summary statistics.
- [`reports/figures/qpm/`](reports/figures/qpm/) — Figures and diagnostics.
- [`reports/Beyond_Optimization_Research_Report.md`](reports/Beyond_Optimization_Research_Report.md) — Full report with methods, results, tables, and figures.

## Data source and reproducibility

The portfolio weights and optimization rules come from the open-source [qpm-project](https://github.com/AhmadMujtaba200210/qpm-project). My work starts with those portfolios and focuses on measuring and testing their risk.

I use a fixed version of the upstream data from commit [`96ab45f`](https://github.com/AhmadMujtaba200210/qpm-project/commit/96ab45ff7bdc12ee9e220d1ade5138e571f520e4). The [source manifest](data/external/qpm/upstream/source_manifest.json) lists the source files and hashes. The [`download_qpm_data.py`](scripts/download_qpm_data.py) script downloads the original inputs. Third-party raw data and large daily forecast files are not included in this repository.

---

Thanks for stopping by! If you'd like to see the methods, numbers, and tests in more detail, take a look at the [full report](reports/Beyond_Optimization_Research_Report.md).
