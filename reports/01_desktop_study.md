# TreasuryShield-MRM desktop study

Reviewed 2026-09-27. This is public-data research for bank funding and
liquidity model risk. It is not an internal liquidity measure or a regulatory
compliance opinion. Detailed matrices and URLs are under `desktop_study/`.

## Current supervisory context

The [Federal Reserve's SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)
supersedes SR 11-7 and places model risk management in a risk-based framework.
Its [interagency guidance text](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm)
covers conceptual soundness, effective challenge, outcomes, monitoring,
inventory, governance, and documentation. This project translates those ideas
into a separate validator implementation, baseline and challenger suite,
pre-registered locked holdout, model inventory, findings register, and
monitoring simulation. The guidance's applicability depends on institution,
materiality, and use; this study makes no claim that one model is approved.

[SR 10-6](https://www.federalreserve.gov/boarddocs/srletters/2010/sr1006.htm)
and its [interagency liquidity policy statement](https://www.federalreserve.gov/frrs/guidance/interagency-policy-statement-on-funding-and-liquidity-risk-management.htm)
motivate cash-flow projections, funding diversification, liquidity cushions,
stress assumptions, and contingency funding actions. [SR 12-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1207.htm)
motivates multiple stress methods, reverse stress, and capital-liquidity
interactions. Public filings do not expose the institution's contractual cash
flows, collateral schedules, intraday liquidity, depositor identities, or
committed funding limits; TreasuryShield's stress and recovery calculations
are therefore transparent proxies with sensitivity ranges.

## Treasury and liquidity questions

1. What is the one-, two-, and four-quarter distribution of a bank's deposit
   change, especially the severe runoff tail?
2. Does deposit contraction coincide with borrowing substitution or loss of
   reported liquid-asset buffers?
3. Do funding composition, deposit cost, balance-sheet dynamics, peer
   position, macro conditions, and filtered regimes improve forecasts?
4. How do institution and system-wide shocks combine under alternative
   dependence assumptions, and which practical funding or asset actions can
   close a modeled gap at reasonable cost?
5. Which model remains reliable across GFC, pandemic, rate tightening, and
   recent periods, and where does it fail?

These are predictive and scenario questions, not causal claims. A published
Call Report quarter is usable only after a defensible release lag. The
[FFIEC filing instructions](https://www.ffiec.gov/sites/default/files/data/reporting-forms/FFIEC031_FFIEC041_202503_i.pdf)
set filing deadlines after quarter-end; this study uses a uniform 45-day
research proxy and makes amendment-vintage limitations explicit.

## Official public sources and observed acquisition

| Source | Role | Observed result and boundary |
| --- | --- | --- |
| [New York Fed harmonized Call Reports](https://www.newyorkfed.org/research/banking_research/balance-sheets-income-statements) | Long Core bank history | Downloaded balance-sheet and income ZIPs, dictionary, and construction README. The ZIP contents extend through 2026Q1; the webpage and README lag that observed file coverage. Source amounts are in USD thousands. |
| [FFIEC CDR quarterly bulk](https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx) | Modern Enhanced features | Acquired all 38 quarterly public files 2017Q1–2026Q2. Parsed 031, 041, and 051 filer types separately with field-presence tracking. Direct uninsured estimates are structurally absent for many filers. |
| [FDIC BankFind](https://api.fdic.gov/banks/docs) | IDs, events, failures | Acquired institution metadata, FI-level history, and failure listings. A missing Call Report row is never assigned a failure label. Event-to-RSSD matches carry confidence. |
| [Federal Reserve H.8](https://www.federalreserve.gov/releases/h8/about.htm) | System context and scenario shape | Weekly published commercial-bank aggregates can shape market scenarios; they are not bank-specific weekly observations. |
| [FRED](https://fred.stlouisfed.org/) and [ALFRED real-time data](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) | Rates, curve, volatility, labor, activity, inflation, and vintages | Current series and 262 quarter-origin historical-vintage snapshots were acquired without credentials. CFNAI vintages in this route begin in 2012. Revised and vintage-aware modes remain separate. |
| [Chicago Fed FR Y-9C history](https://www.chicagofed.org/banking/financial-institution-reports/bhc-data) and [NIC download page](https://www.ffiec.gov/npw/FinancialReport/FinancialDataDownload) | Holding-company external validation | Chicago Fed historical files were acquired for 2001Q1–2021Q1. NIC's current endpoint returned a CAPTCHA in this environment. FR Y-9C observations remain a separate consolidated holding-company unit. |

All actual counts, hashes, vintage dates, and quality results are in
`artifacts/results.duckdb`, `data/manifests/raw_manifest.csv`, and the linked
data-quality report. The source and literature bibliographies are in
`desktop_study/source_manifest.csv` and
`desktop_study/literature_matrix.csv`.

## Forecast and model-risk design

The target is bank-level log deposit growth. Future borrowing and liquid-asset
changes provide related funding and buffer outcomes. Entity events identify
M&A contamination, with zero, one, two, and no-quarter exclusion sensitivities.
The Core Panel carries stable long-history variables; the Enhanced Panel adds
form-specific deposit and funding composition when reported. The historical
insured-deposit proxy and direct FFIEC uninsured estimate are distinct concepts.

Train and evaluation windows are temporal. The GFC, pandemic, 2022–23, and
2024+ periods have separate folds, with the final period locked before model
selection. Random splitting, full-sample preprocessing, future-smoothed HMM
states, revised macro vintages, and uncontrolled M&A are measured as negative
controls only. The [pre-registration](preregistration.md) fixes the primary
Q05 pinball endpoint and model-choice gates. An HMM contributes **filtered**
probabilities at each forecast origin; future-smoothed states cannot enter a
predictive feature.

Historical-peer and linear models remain in the suite beside quantile forests,
CUDA XGBoost, LightGBM, hierarchical Student-t, and CUDA neural models.
RAHQ-C is a testable candidate, not a presumed winner. Calibration,
quantile-crossing correction, block-bootstrap uncertainty, subgroup checks,
perturbations, stress arithmetic, and independent implementation review are
required before recommending a model. The independent validation report must
give affected use, materiality, and remediation for observed failures.

## Literature and traceability

The source-specific review and the requirement-to-artifact matrix are in
`desktop_study/data_source_review.md` and
`desktop_study/traceability_matrix.csv`. The literature matrix covers
[Koenker and Bassett's quantile regression](https://www.jstor.org/stable/1913643),
[Meinshausen's quantile forests](https://www.jmlr.org/papers/v7/meinshausen06a.html),
[Hamilton's regime models](https://www.jstor.org/stable/1912559),
[Gibbs and Candès' adaptive conformal inference](https://arxiv.org/abs/2106.00170),
and bank-failure evidence. The study uses these methods as candidate tools;
their assumptions, especially exchangeability under regime shift and
comparability across legal bank entities, are tested or explicitly limited.
