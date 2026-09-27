# TreasuryShield-MRM v1.2 Methodology Sources

Verified 2026-09-27. This review governs the v1.2 diagnostic hardening pass; it does not alter frozen project facts, the v1 champion, or locked predictions.

## Model risk management

The Federal Reserve's [SR 26-2 Revised Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) superseded SR 11-7 in April 2026. It emphasizes risk-based model governance, effective challenge, validation, ongoing monitoring, outcome analysis, model limitations, and controls proportionate to model exposure and intended use. v1.2 therefore treats clustered inference, reproduction limits, monitoring escalation, and corrected semantic lineage as validation evidence—not as model retuning.

## Backtesting and panel-aware inference

The Federal Reserve's [Review of Backtesting and Backtesting Procedures](https://www.federalreserve.gov/econres/feds/a-review-of-backtesting-and-backtesting-procedures.htm) distinguishes unconditional coverage, exception independence, multi-quantile tests, and loss-function-based validation, and notes the importance of test power. A cross-bank row ordering is not a temporal Bernoulli sequence, so v1.2 limits transition analysis to within-bank adjacent-quarter observations and uses quarter-block, bank-cluster, and two-way clustered inference.

The current statsmodels documentation for [`cov_cluster_2groups`](https://www.statsmodels.org/dev/generated/statsmodels.stats.sandwich_covariance.cov_cluster_2groups.html) describes the additive two-way CRV1 estimator: bank-cluster covariance plus quarter-cluster covariance minus intersection covariance. v1.2 uses the algebraically equivalent intercept-only estimator for mean paired loss and weighted-influence statistics, with finite-cluster corrections and explicit low-resolution warnings where time clusters are few.

## Liquidity risk and stress testing

Federal Reserve [SR 10-6](https://www.federalreserve.gov/boarddocs/srletters/2010/sr1006.htm), revised in 2023, emphasizes cash-flow projections, diversified funding, liquidity stress tests, liquid-asset cushions, and actionable contingency funding plans. The Federal Reserve's [liquidity-risk-management topic page](https://www.federalreserve.gov/supervisionreg/topics/liquidity_risk.htm) also points to current stress-testing guidance. v1.2 consequently reuses the actual v1 public-data proxy engine, including recovery and reverse-stress routines, while preserving the project's express limitation that these are not regulatory liquidity metrics.

## LightGBM and SHAP

The current LightGBM [`Booster` documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html) documents native model save/load, JSON dumps, predictions, and split/gain feature importance. The current SHAP [`TreeExplainer` documentation](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html) identifies LightGBM as a supported tree model. v1.2 labels TreeSHAP as exact only if a strict frozen-prediction reproduction gate passes; otherwise it leaves v1.1 surrogate evidence clearly labeled as historical diagnostic evidence.

## Application boundary

These sources support the hardening methods and governance framing. They do not imply regulatory approval, production fitness, or equivalence between TreasuryShield's public-data liquidity proxies and internal bank liquidity measures.
