# Changelog

All notable public releases of TreasuryShield-MRM are documented here.

## [1.2.0] - 2026-09-27

### Added

- Cluster-aware panel calibration with whole-quarter, bank-cluster, and two-way uncertainty views.
- Corrected slice definitions, materiality gates, persistence checks, and Holm-adjusted inference.
- Re-evaluation of the earlier nominal GRU slice reversal.
- Frozen-threshold selective-review confidence analysis and random-review benchmark.
- Actual v1 stress-engine variance attribution using common profiles and random numbers.
- Five-state monitoring policy with reachable `REDEVELOP` state and de-escalation hysteresis.
- Exact seven-quantile B8 LightGBM reconstruction, parity gate, native importance, and TreeSHAP evidence.
- Public README, release notes, curated-release workflow, and portable Windows/Linux CI.

### Preserved

- B8 LightGBM remains the frozen champion.
- v1 and v1.1 designated frozen artifacts remain byte-identical.
- No model retuning, holdout reuse, or favorable historical-results rewriting was performed.

### Clarified

- The locked Q05 result is conservative miscalibration, not exact calibration.
- Selective-review capture is not an accuracy improvement.
- The eight-quarter final-period sample limits temporal inference.
- Public CI is a portable code-health subset, distinct from the 39-test local research validation.

## [1.1] - 2026-09-27

- Added post-model calibration, error/slice analysis, drift and OOD diagnostics, selective review, stress attribution, and monitoring replay while preserving v1 evidence.

## [1.0] - 2026-09-27

- Completed the frozen public-data research lifecycle: acquisition, point-in-time controls, modeling, locked validation, independent challenge, stress/recovery analysis, monitoring, warehouse, figures, and reports.
