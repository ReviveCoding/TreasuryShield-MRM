# Frozen Model Evaluation Metrics

Frozen model: `core_B_1q_final_locked_B8_lightgbm_quantile`; selected before scoring 2024+ forecast origins. Snapshot `fbc0b7b5a2c46fbcc0b5`; independent reconciliation `artifacts/independent_final_review.json`.

Final equal-bank Q05 pinball: **0.004323**; empirical Q05 coverage: **0.0400** (nominal 0.0500). These values are rows in `artifacts/results.duckdb.metrics`.

Paired whole-time-block comparisons (selected minus reference; negative favors selected):

| Reference model       |   Selected minus reference Q05 |   Relative difference (%) |   95% low |   95% high |   Bootstrap draws |
|:----------------------|-------------------------------:|--------------------------:|----------:|-----------:|------------------:|
| B0_unconditional      |                      -0.001216 |                -21.946113 | -0.001301 |  -0.001133 |              1000 |
| B1_peer               |                      -0.001092 |                -20.162644 | -0.001203 |  -0.000985 |              1000 |
| B2_persistence        |                      -0.000925 |                -17.627527 | -0.001035 |  -0.000819 |              1000 |
| B3_ols                |                      -0.001874 |                -30.241463 | -0.002098 |  -0.001653 |              1000 |
| B4_panel_fe           |                      -0.004012 |                -48.135964 | -0.004290 |  -0.003732 |              1000 |
| B5_linear_quantile    |                      -0.000206 |                 -4.539941 | -0.000225 |  -0.000185 |              1000 |
| B6_quantile_forest    |                      -0.000484 |                -10.060598 | -0.000529 |  -0.000439 |              1000 |
| B7_xgb_quantile       |                      -0.000372 |                 -7.926824 | -0.000405 |  -0.000340 |              1000 |
| B9_bayesian_student_t |                      -0.002464 |                -36.302370 | -0.002725 |  -0.002207 |              1000 |
| B10_mlp               |                      -0.000554 |                -11.360117 | -0.000624 |  -0.000481 |              1000 |
| B10_gru               |                      -0.000126 |                 -2.842435 | -0.000270 |   0.000016 |              1000 |
| RAHQ_tree             |                      -0.000266 |                 -5.802445 | -0.000292 |  -0.000241 |              1000 |
| RAHQ_nn               |                      -0.000443 |                 -9.293199 | -0.000584 |  -0.000303 |              1000 |
| RAHQ_C_tree           |                      -0.000457 |                 -9.560917 | -0.000473 |  -0.000441 |              1000 |
| RAHQ_C_nn             |                      -0.000576 |                -11.762768 | -0.000715 |  -0.000439 |              1000 |

Independent project recommendation: **ACCEPT WITH LIMITATIONS**. The holdout has few time blocks; source filings are restated, and public liquidity proxies lack contractual cash flows. No single crisis, seed, or weighted aggregate is offered as a standalone production claim.
