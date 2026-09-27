# TreasuryShield-MRM v1.2.0: Validation-Hardened Research Release

This release publishes the completed TreasuryShield-MRM research code, public manifests, reports, and validation-hardened evidence. It preserves the v1 and v1.1 frozen scientific record and does not retune or replace the selected model.

## Headline evidence

- **Frozen champion:** B8 LightGBM quantile.
- **Q05 calibration:** 3.975% exception rate versus 5% nominal, with a whole-quarter-block 95% CI of 3.501%–4.441%. The locked-period result is conservatively miscalibrated. The original frozen v1 metric remains 3.995%; v1.2's aligned hardening analysis reports 3.975%.
- **Selective review:** a 10% review budget captured 28.61% of severe tail misses, a 2.86x lift over random 10% review. This is miss capture, not an accuracy increase.
- **Robust slicing:** 19 material slices retain robust Q05 coverage deviation after exposure, sample-size, time-span, persistence, clustered-inference, and multiplicity gates.
- **Challenger inference:** the earlier nominal GRU reversal in one asset slice does not survive robust quarter/two-way inference and Holm adjustment; it is not a stable win.
- **Stress attribution:** Peak Funding Need proxy variation is 83.22% scenario assumptions, 11.27% model choice, and 5.51% model-by-scenario interaction.
- **Monitoring:** the historical state-machine backtest has 100% recall and 45.8% precision, exposing a material false-alert tradeoff.
- **Exact reproduction:** the frozen B8 recipe reproduces all 30,087 locked rows across seven quantiles with maximum absolute prediction difference 0. Exact TreeSHAP analysis is therefore permitted by the reproduction gate.
- **Local validation:** 39 tests passed. Designated v1 and v1.1 files passed byte-level integrity checks.

## Validation boundary

The final holdout covers eight 2024+ forecast-origin quarters, so time-cluster inference is necessarily low-resolution. v1.2 is post-hoc validation hardening under frozen prelock policies, not a second model-selection exercise. The independent project recommendation remains **ACCEPT WITH LIMITATIONS**.

The public repository excludes raw regulatory filings, processed bank-level data, prediction-level outputs, fitted model artifacts, machine state, and the local DuckDB warehouse. GitHub Actions therefore runs a nine-test portable code-health subset on Windows and Ubuntu; it does not claim to rerun the complete empirical study. The complete local environment passed the 39-test validation suite.

## Public-data and proxy caveats

The study uses public Call Reports, public institution/event data, H.8 aggregates, public macro series/vintages, and a separate FR Y-9C extension. It has no access to private bank data. Peak Funding Need, liquidity runway, reverse-stress, and recovery outputs are research proxies—not regulatory liquidity metrics or production decisions. Public sources cannot recover contractual cash-flow ladders, depositor concentration, collateral encumbrance, committed facilities, institution-specific behavioral assumptions, or every historical filing amendment vintage.

## Included release asset

`TreasuryShield-MRM-v1.2.0-results.zip` is a curated public-safe bundle containing these release notes, key reports, selected figures, and public manifests. `TreasuryShield-MRM-v1.2.0-results.zip.sha256` contains its SHA-256 digest.

No license has been selected for this repository; licensing remains unchanged.
