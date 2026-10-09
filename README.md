# Beyond Optimization
## Independent Portfolio Risk Assessment

**Research Question:** Does portfolio optimization deliver robust risk reduction across different risk measures and market conditions?

## Overview

I reconstruct daily portfolio returns using externally developed portfolio weights and build an independent risk modeling and validation framework.

I implement five risk models to estimate Value at Risk (VaR) and Expected Shortfall (ES), evaluate forecast reliability, and analyze tail-risk dynamics.

I investigate how portfolio optimization affects downside risk across different risk measures and market conditions, identifying the extent and limitations of risk reduction.

## Key Findings

GARCH-FHS produced the most accurate 99% VaR coverage among the five risk models. The exceedance rates ranged from 0.99% to 1.20% across the five portfolios.

However, the exceedances were not independent. All five portfolios failed the Christoffersen independence test at both the 95% and 99% VaR levels.

GARCH-Normal underestimated tail risk at the 99% VaR level. Exceedance rates ranged from 1.82% to 2.19%.

GARCH-Student-t reduced these rates to 1.41%–1.64%, but all five portfolios still failed the 99% Kupiec test.

Rolling Normal also underestimated tail risk at the 99% VaR level. Exceedance rates ranged from 1.93% to 2.14%.

Historical Simulation achieved rates of 0.89%–1.09%. However, its exceedances still showed significant temporal clustering.

## Data and Portfolio Strategies

I evaluate five portfolio strategies: Equal Weight, Static Minimum Variance, Regime-Based Minimum Variance, Static Mean-Variance, and Regime-Based Mean-Variance.

The portfolios consist of eight ETFs: SPY, EFA, EEM, TLT, IEF, GLD, VNQ, and DBC.

The common risk backtesting period runs from September 3, 2010, to December 8, 2025, with 3,839 trading days.

## Risk Models

I estimate one-day-ahead VaR and ES using five methods: Rolling Normal, Historical Simulation, GARCH-Normal, GARCH-Student-t, and GARCH-FHS.

Rolling Normal uses a 252-day window, while the other models use 1,008 days of historical data.

I re-estimate the GARCH parameters every 21 trading days and update the conditional volatility forecasts daily.

## Model Validation

VaR forecasts are evaluated against realized portfolio losses at the 95% and 99% confidence levels.

The Kupiec test assesses unconditional coverage, while the Christoffersen test examines exceedance independence.

Additional diagnostics include standardized residuals, probability integral transforms (PIT), and VaR exceedance timing analysis.

## Data Provenance

Portfolio weights and reference backtest data come from [AhmadMujtaba200210/qpm-project](https://github.com/AhmadMujtaba200210/qpm-project).

The upstream data are linked to commit `96ab45ff7bdc12ee9e220d1ade5138e571f520e4`. The source manifest records the original file URLs and SHA-256 hashes.

Third-party raw datasets are excluded from this repository.

## Repository Structure

- `scripts/` — Risk models, backtesting, and diagnostic scripts.
- `data/interim/qpm/` — Reconstructed portfolio returns, asset returns, and portfolio weights.
- `data/processed/qpm/` — Risk model outputs and statistical test results.
- `outputs/figures/` — Portfolio return diagnostics.
- `reports/figures/` — Risk model diagnostics and backtesting figures.
- `requirements.txt` — Python dependencies.


## Research Status

Five baseline VaR and ES forecasting models have been implemented and evaluated across five portfolio strategies.

VaR validation includes unconditional coverage, exceedance independence, PIT diagnostics, and tail-event analysis. ES validation includes exceedance residuals and joint VaR–ES scoring.

GARCH-FHS achieved the lowest mean joint VaR–ES score among the five baseline models across all portfolios at both the 95% and 99% confidence levels. Exceedance clustering remains an important limitation.

Exploratory distribution modeling includes KDE, Gaussian mixtures, and Student-t mixtures. These methods did not demonstrate a material predictive improvement over GARCH-FHS in the evaluated minimum-variance portfolio.

The next stages are historical and hypothetical stress testing, followed by cross-strategy portfolio risk assessment.

Additional model experiments on previously examined data are treated as exploratory.


## Reproducibility

Python dependencies are listed in `requirements.txt`.

The original data sources and file hashes are documented in `data/external/qpm/upstream/source_manifest.json`.

Risk modeling, backtesting, and diagnostic scripts are available in `scripts/`.
