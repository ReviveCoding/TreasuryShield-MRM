# TreasuryShield-MRM v1.2 Reproducibility Manifest

The v1 and v1.1 byte freeze is `artifacts/v1_2_hardening/freeze_manifest_v1_v11.json`. The namespaced warehouse is `artifacts/v1_2_hardening/results_v1_2.duckdb`. All v1.2 analyses use frozen predictions and prelock thresholds; only exact B8 reproduction refits the frozen recipe. Run `.\.venv\Scripts\python.exe -m treasuryshield.v12.pipeline`, the complete pytest suite, then `.\.venv\Scripts\python.exe -m treasuryshield.v12.finalize`.

Frozen model: B8_lightgbm_quantile. Data snapshot: fbc0b7b5a2c46fbcc0b5.
