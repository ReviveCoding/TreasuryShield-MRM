from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.v11.calibration import calibration_rows, exception_tests, pit_values

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "v1_1_analysis"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sequential_recalibration_uses_only_strictly_matured_outcomes() -> None:
    replay = pd.read_parquet(ART / "sequential_recalibration.parquet")
    assert replay.strictly_prior.all()
    assert (pd.to_datetime(replay.max_used_target_availability) < pd.to_datetime(replay.forecast_origin)).all()


def test_slice_cutoffs_are_frozen_from_prelock_data() -> None:
    registry = pd.read_csv(ART / "slice_registry.csv")
    assert registry.cutoff_fixed_prelock.all()
    assert registry.cutoff_source.str.contains("pre-2024|source metadata", regex=True).all()


def test_intersection_sample_size_gate() -> None:
    frame = pd.read_parquet(ART / "intersection_metrics.parquet")
    assert not frame.empty
    assert (frame.n >= 1000).all()


def test_quantile_calibration_calculation() -> None:
    frame = pd.DataFrame({q: [-1.0, 1.0] for q in ["q01","q05","q10","q25","q50","q75","q90"]})
    frame["actual"] = [-2.0, 2.0]
    rows = calibration_rows(frame, "test")
    q05 = next(row for row in rows if row["quantile"] == "q05")
    assert q05["empirical_coverage"] == 0.5
    assert np.isclose(q05["pinball"], 0.5)


def test_exception_tests_are_finite_and_report_runs() -> None:
    result = exception_tests(np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0] * 10), .05)
    assert result["applicable"]
    assert result["unconditional_coverage"]["lr"] >= 0
    assert result["runs"]["maximum"] == 2


def test_pit_monotone_interpolation() -> None:
    quantiles = [0, 1, 2, 3, 4, 5, 6]
    frame = pd.DataFrame([{**dict(zip(["q01","q05","q10","q25","q50","q75","q90"], quantiles)), "actual": y} for y in (-1, 1, 3, 5, 7)])
    pit = pit_values(frame)
    assert np.all(np.diff(pit) >= 0)
    assert np.all((pit >= 0) & (pit <= 1))


def test_ood_isolation_forest_fitted_prelock_only() -> None:
    metadata = json.loads((ART / "ood_fit_metadata.json").read_text())
    assert metadata["fit_population"] == "pre-2024 only"
    assert metadata["prelock_rows"] > 0


def test_selective_review_thresholds_frozen_prelock() -> None:
    policy = json.loads((ART / "selective_policy_freeze.json").read_text())
    assert policy["fit_population"] == "pre-2024 only"
    assert set(policy["thresholds"]) == {"0.01", "0.05", "0.1", "0.2"}


def test_challenger_prediction_rows_are_identically_aligned() -> None:
    base = pd.read_parquet(ROOT / "artifacts/predictions/core_B_1q_final_locked_B8_lightgbm_quantile.parquet", columns=["bank_id","report_period","actual"])
    for model in ("B1_peer", "B5_linear_quantile", "B7_xgb_quantile", "B10_gru", "RAHQ_tree", "RAHQ_C_tree", "RAHQ_C_nn"):
        other = pd.read_parquet(ROOT / f"artifacts/predictions/core_B_1q_final_locked_{model}.parquet", columns=["bank_id","report_period","actual"])
        assert base[["bank_id","report_period"]].equals(other[["bank_id","report_period"]])
        assert np.allclose(base.actual, other.actual, equal_nan=True)


def test_stress_attribution_uses_common_random_numbers() -> None:
    frame = pd.read_parquet(ART / "stress_attribution.parquet")
    metadata = json.loads((ART / "stress_common_random_numbers.json").read_text())
    assert frame.common_random_numbers.all()
    assert metadata["draw_matrix_shape"][1] == 128
    assert frame.profiles.nunique() == 1


def test_v1_artifacts_are_byte_identical_to_freeze() -> None:
    manifest = json.loads((ART / "v1_freeze_manifest.json").read_text())
    for relative, metadata in manifest["frozen_files"].items():
        path = ROOT / relative
        assert path.exists(), relative
        assert digest(path) == metadata["sha256"], relative
