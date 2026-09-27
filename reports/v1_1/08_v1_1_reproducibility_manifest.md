# TreasuryShield v1.1 reproducibility manifest

Generated: 2026-09-27T20:09:47.094147+00:00

## Scientific boundary

The v1 selection, prediction files, locked result, preregistration, reports, and results database were hashed before this extension. New evidence is stored under the v1.1 namespaces. Sequential recalibration is an exploratory pseudo-live replay with strict target maturity checks.

## Execution

`python -m treasuryshield.v11.pipeline`

## Evidence

- Freeze: `artifacts/v1_1_analysis/v1_freeze_manifest.json`
- Diagnostic database: `artifacts/v1_1_analysis/results_v1_1.duckdb`
- Audit: `artifacts/v1_1_analysis/v1_1_audit.json`
- Cutoffs: `artifacts/v1_1_analysis/slice_cutoffs_prelock.json`
- State: `.treasuryshield/v1_1_run_state.json`

## Assumptions

PIT uses a monotone piecewise-linear approximation from seven stored quantiles. Stress attribution is a common-random-number diagnostic on identical sampled profiles. Explanation reason codes use a prelock-fitted surrogate where persisted fitted LightGBM objects are unavailable; stored v1 gain importance is retained separately. No causal interpretation is warranted.
