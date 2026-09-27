# TreasuryShield-MRM preregistration

Frozen on 2026-09-27 UTC before inspecting any 2024+ deposit-growth outcome,
model error, stress result, or holdout leaderboard. Changes after the hash is
recorded require a dated append-only amendment with a reason and a second hash.

## Data and unit of analysis

- Raw snapshot: `fbc0b7b5a2c46fbcc0b5` (425 validated official files).
- Primary unit: one U.S. commercial bank RSSD ID and Call Report quarter, using
  New York Fed harmonized balance sheet and income statement data. FFIEC 031,
  041, and 051 schedules form the Enhanced Panel from 2017Q1. FR Y-9C holding
  companies are a separate external-validation extension and will never be
  pooled with bank observations.
- Core features: `data/manifests/feature_quality.json`, SHA256
  `B48F3EDBFE6DA4ECB53644A7D2A608553F48D3834F520F2F123A053826BD14F3`.
  Official-field registry: `data/manifests/feature_registry.csv`, SHA256
  `49D784A1130BE7020D8C25B2FB03B8EDDE2464CCC0F34F35BAC2FC5F80FC81E3`.
  Target registry: `data/manifests/target_registry.csv`, SHA256
  `773A7077EAF8470CBAE5BB7F545BDD0AD46C613AD1C7D1B3AB0FA4CBD0B55834`.
- Pre-lock EDA: `data/manifests/eda_prelock.json`, SHA256
  `1CF8BFE9604C67F25470D1F400759C3306BB44616239C61EEBCC06D11E71B2A3`.
  This EDA includes origins through 2023Q4 only.

## Information clock and outcomes

The forecast origin for a bank-quarter is report quarter end plus 45 calendar
days. This is a conservative public availability proxy, not the exact historical
filing or amendment timestamp. All model features must have an information date
on or before origin. Macro real-time experiments use quarter-origin ALFRED
vintages for unemployment, CPI, and CFNAI where snapshots exist. Market and H.8
series use explicit conservative lags; their historical source revisions are not
fully reconstructed. H.8 is system-level context, not bank-level weekly data.

Primary outcome is `log(deposits[t+h]/deposits[t])` at 1, 2, and 4 quarters;
one quarter is primary. Deposits must be positive at both dates, and the exact
future quarter must exist for the same RSSD ID. Other outcomes are borrowing
change scaled by origin assets and changes in narrow and broad liquid-asset
ratios. FDIC failure is secondary external validity. Severe liquidity-event
thresholds are fitted only within each training partition and frozen before
evaluating its test partition.

The primary supervised rule excludes M&A and branch-transfer events during
the forecast path plus one quarter on either side. Sensitivities use zero, two,
and no exclusion quarters. Disappearance is never labeled failure. Raw stress
observations are never deleted solely because their changes are large.

## Splits and final holdout

All temporal training selections require the **outcome availability date** to
be no later than the training cutoff. Historical origin dates alone are
insufficient for a multi-quarter label. Model fitting, imputation, scaling,
winsorization, learned peers, regimes, calibration, and hyperparameters are
recomputed inside the relevant training partition.

| Test | Training information cutoff | Test origins |
| --- | --- | --- |
| GFC regime | 2006Q4 outcome reporting cutoff | 2007Q3–2009Q4 |
| Pandemic regime | 2019Q4 outcome reporting cutoff | 2020Q1–2021Q4 |
| Tightening/regional stress | 2021Q4 outcome reporting cutoff | 2022Q1–2023Q4 |
| Final locked holdout | 2023Q4 outcome reporting cutoff | 2024Q1 onward through latest observable target in the frozen New York Fed data |

Additional expanding/rolling temporal folds are wholly pre-2024. A random
split is a negative control only. The 2022–23 regime can inform the final
pre-2024 model choice and is therefore not presented as independent of that
choice. The final 2024+ holdout remains locked until candidate configurations,
selection rules, preprocessing, and acceptance gates are fixed. No tuning or
ablation choices may be made from that holdout.

## Model and preprocessing protocol

Compare all prescribed simple and complex candidates: unconditional and peer
historical quantiles, bank persistence, OLS, panel fixed effects, linear
quantile, quantile random forest, XGBoost quantile on CUDA, LightGBM quantile
on CPU, Bayesian hierarchical Student-t, and PyTorch quantile MLP/GRU on CUDA.
RAHQ-C(Tree) and RAHQ-C(NN) add rule or learned peers, lagged dynamics, macro
context, filtered regime probabilities, and rolling calibration to their
respective backbones. Keep their uncalibrated versions. HMM inputs use filtered
`P(state_t | information_through_t)` only. The HMM is fitted inside training
folds. Compare rule, 2-, 3-, and 4-state alternatives. Neural sequence models
may use up to eight prior observed quarters, with missing-history masks.

Numeric missing values receive train-fitted median imputation plus missingness
indicators. Linear models additionally use train-fitted 0.5th/99.5th percentile
clipping and scaling. Trees retain raw extremes unless a particular sensitivity
changes that choice. No method sees an evaluation label during fitting or
calibration. Peer P1 uses predeclared asset size bands (<$100m, $100m–$1bn,
$1bn–$10bn, >=$10bn), residential-loan share, and brokered-deposit share;
learned P2 clusters are fitted only on training data. Source amounts are USD
thousands. `1 - insured/domestic deposits` is a historical proxy, distinct from
the direct FFIEC estimate of uninsured domestic deposits.

Hyperparameter search uses bounded Optuna trials and inner temporal folds;
GPU trials are sequential on the single RTX GPU. Expensive CPU challengers may
use a deterministic training subsample, declared in each configuration, but all
eligible test rows are scored. Seeds and trial results are stored. After model
choice, retraining on all permissible pre-holdout outcomes is allowed with
frozen settings. Historical calibration uses only past out-of-sample residuals.

## Hypotheses, comparisons, and metrics

Primary question: do RAHQ-C candidates materially improve severe deposit
runoff forecasts over the historical peer, linear quantile, and uncalibrated
XGBoost challengers while remaining calibrated and governable? A null or a
simple-challenger win is a valid outcome.

The primary endpoint is equal-bank Q05 pinball loss for one-quarter deposit
growth, assessed in pre-2024 stress folds and the final locked period. The
primary tail checks are Q01 pinball, Q05 coverage error, frequency and severity
of outcomes below predicted Q05, and weighted interval score. Secondary
metrics include median absolute error, RMSE, quantile crossing, stress-event
PR-AUC/Brier/Recall@K, deposit- and asset-weighted versions, horizon 2/4,
and subgroup results. Statistical comparisons use paired resampling of whole
time blocks (all banks within a selected block), 95% intervals and fixed seeds.
Bank-cluster and two-way resampling are sensitivity checks. Report absolute and
relative differences; secondary multiple comparisons are exploratory.

## Model choice gates

RAHQ-C may be preferred only if, relative to the best credible simple
challenger, it achieves at least a 3% lower primary Q05 pinball loss on the
pre-2024 stress-fold average **and** the locked holdout, without a paired 95%
block interval indicating material harm; reduces severe tail underestimation;
keeps absolute Q05 empirical coverage error within 2 percentage points of the
nominal 5% and no worse than the challenger by more than 1 point; keeps median
absolute error within 5%; and shows no >10% primary-tail deterioration in a
predeclared material size/liquidity/business-model subgroup. Ablations,
perturbations, runtime, memory, calibration, and operational complexity must
support the increment. Any failed gate is reported. If performance is
statistically and economically indistinguishable, select the simpler credible
model. A distinct challenger may be selected outright.

Primary stress test uses the frozen predictive distribution to generate
institution, market, and combined deposit outflow scenarios with disclosed
dependence assumptions. Coverage-gap and action-cost calculations are public
data proxies. Run narrow/broad liquidity, haircuts, funding capacity, action
timing, capital-liquidity interaction, reverse-stress, and recovery-ordering
sensitivities. The independent validator evaluates all material weaknesses,
including where the selected model fails.
