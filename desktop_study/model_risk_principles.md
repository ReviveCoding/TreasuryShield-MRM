# Model-risk principles for this study

Reviewed 2026-09-27. [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) supersedes SR 11-7. The [revised interagency guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm) describes effective challenge, conceptual soundness, outcome analysis, ongoing monitoring, governance, documentation, and inventory in a risk-based framework. Its applicability language and the model's intended use matter; this project does not issue a supervisory opinion.

Study controls:

1. State the research-only purpose and proxy boundaries before fitting models.
2. Preserve source files with hashes and distinguish restated from pseudo-point-in-time panels.
3. Fit preprocessing, peers, regimes, calibration, and tuning strictly within historical training windows.
4. Register the primary metrics and final locked holdout before inspecting that holdout.
5. Benchmark against simple and methodologically distinct challengers. Complexity has a governance cost.
6. Have independent validation code recompute metrics from stored predictions and record weaknesses, not just wins.
7. Monitor coverage, tail misses, input drift, and challenger disagreement; use escalation only when backtests support it.
8. Record limitations and unresolved findings with affected use, severity, owner, and residual risk.

The [revised guidance](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm) permits validation rigor to vary with materiality and use. TreasuryShield will still apply a demanding research standard because the hypothetical decisions concern liquidity stress. Its project acceptance labels are internal to this study.
