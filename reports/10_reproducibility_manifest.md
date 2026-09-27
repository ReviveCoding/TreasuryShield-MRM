# Reproducibility manifest

Generated 2026-09-27T12:36:45.250092+00:00. Snapshot `fbc0b7b5a2c46fbcc0b5`. Frozen selection SHA256 `d01a8e6345dcfe06b56fc0785654fb171e55dc06dfb8791ce5b17bd1f5092baf`.

## Execution

Run stages are checkpointed in `.treasuryshield/run_state.json`. Raw official files are read-only after validation and indexed by `data/manifests/raw_manifest.csv`. The final comparison roster and all code hashes at freeze are in `artifacts/model_selection.json`. Numerical claims are in `artifacts/results.duckdb`; predictions, configs, stress draws, monitoring paths, and validation checks retain artifact hashes.

Experiment registry status counts: `{"completed": 40}`. Stored model runs: 197. Figures: 19.

## Key hashes

| Artifact                                     | SHA256                                                           |
|:---------------------------------------------|:-----------------------------------------------------------------|
| reports/preregistration.md                   | 1a500e432944e0058cceb18b987405a3fcb54a8076955bba1aac825a382c4d12 |
| artifacts/model_selection.json               | d01a8e6345dcfe06b56fc0785654fb171e55dc06dfb8791ce5b17bd1f5092baf |
| data/manifests/raw_manifest.csv              | 4dde5c9553c379330d7c306a65ae6a416836bcf7645a3520a8775ed3bb5b8cd1 |
| data/manifests/feature_registry_complete.csv | a43494a9d5facb701328f871921327c36e081d0bd4aeefb7c24270e09052f8ba |
| data/manifests/target_registry.csv           | 773a7077eaf8470cbae5bb7f545bdd0ad46c613ad1c7d1b3ab0fa4cbd0b55834 |
| artifacts/independent_final_review.json      | 958e741ef9776da71c9ad610720e3239627f2acfe9658c38f72fee5dedd8d17a |
| artifacts/validation_opinion.json            | 0100b94e8e4544f9c06c05e809137a52faa156297a4475056b3f2d0259d65eb5 |
| figures/figure_manifest.json                 | feac9b4d6d80749cac5d26c316cd1bfe480cece7607a5cb518f0a5d1aef3e6cb |

## Software

| Package         | Version      |
|:----------------|:-------------|
| python          | 3.12.10      |
| numpy           | 2.5.3        |
| pandas          | 3.0.6        |
| tabulate        | 0.9.0        |
| duckdb          | 1.5.5        |
| pyarrow         | 25.0.1       |
| xgboost         | 3.4.1        |
| lightgbm        | 4.7.0        |
| torch           | 2.14.0+cu132 |
| scikit-learn    | 1.9.1        |
| hmmlearn        | 0.3.3        |
| optuna          | 5.0.0        |
| matplotlib      | 3.11.2       |
| quantile-forest | 1.4.2        |

The paired CPU/GPU timings, actual CUDA tensor/booster checks, and resource peaks are stored in `artifacts/hardware_benchmarks.json`, `artifacts/gpu_smoke.json`, and model configs. The local RTX GPU was used for compatible neural and XGBoost work.

## Reproduction path

1. Validate the frozen raw manifest and preregistration hashes.
2. Rebuild processed panels, features, and targets using `src/treasuryshield/`.
3. Execute the frozen model roster with the recorded seeds/configs; never use final labels for selection.
4. Reconcile stored predictions and metrics with the independent validator.
5. Rebuild stress, monitoring, figures, and reports from the warehouse and artifact manifests.

Public source restatements can change if a new raw snapshot is acquired; reproduction of this run uses the frozen snapshot and recorded package versions.
