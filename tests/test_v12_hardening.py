from __future__ import annotations

import inspect
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from treasuryshield.v12 import analytics, stress_attribution
from treasuryshield.v12.common import ROOT, sha256
from treasuryshield.v12.inference import (
    cluster_bootstrap_mean, clustered_summary, holm_adjust, two_way_cluster_mean,
)
from treasuryshield.v12.monitoring import next_state


def test_frozen_v1_and_v11_files_unchanged() -> None:
    manifest = json.loads((ROOT / "artifacts/v1_2_hardening/freeze_manifest_v1_v11.json").read_text())
    for group in ("frozen_v1_files", "frozen_v1_1_files"):
        for relative, metadata in manifest[group].items():
            assert sha256(ROOT / relative) == metadata["sha256"]


def test_no_cross_bank_temporal_sequence() -> None:
    source = inspect.getsource(analytics.run_calibration)
    assert '.groupby("bank_id")' in source
    assert 'sort_values(["bank_id", "forecast_origin"])' in source
    assert 'sort_values(["forecast_origin", "bank_id"])' not in source


def test_within_bank_transition_requires_adjacent_quarter() -> None:
    source = inspect.getsource(analytics.run_calibration)
    assert "np.diff(periods) == 1" in source


def test_quarter_block_bootstrap_resamples_whole_clusters() -> None:
    values = np.array([0, 0, 10, 10], float)
    groups = np.array(["q1", "q1", "q2", "q2"])
    lo, hi = cluster_bootstrap_mean(values, groups, reps=999, seed=1)
    assert lo == 0 and hi == 10


def test_bank_cluster_bootstrap_is_deterministic() -> None:
    values = np.arange(12, dtype=float)
    groups = np.repeat(np.arange(4), 3)
    assert cluster_bootstrap_mean(values, groups, reps=99, seed=4) == cluster_bootstrap_mean(values, groups, reps=99, seed=4)


def test_two_way_cluster_alignment_and_center() -> None:
    values = np.array([-1., 0., 1., 2.])
    result = two_way_cluster_mean(values, np.array([1,1,2,2]), np.array([1,2,1,2]))
    assert result["mean"] == .5
    assert result["banks"] == 2 and result["quarters"] == 2


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust(np.array([.03, .001, .02]))
    order = np.argsort([.03, .001, .02])
    assert np.all(np.diff(adjusted[order]) >= 0)


def test_slice_registry_names_match_formulas() -> None:
    registry = pd.read_csv(ROOT / "artifacts/v1_2_hardening/slice_registry_corrected.csv")
    borrow = registry.loc[registry.display_name.eq("borrowings_assets_decile")].iloc[0]
    report = registry.loc[registry.display_name.eq("report_form")].iloc[0]
    assert "borrowings_asset_ratio" in borrow.exact_formula
    assert "ffiec_form" in report.exact_formula


def test_slice_cutoffs_are_prelock() -> None:
    registry = pd.read_csv(ROOT / "artifacts/v1_2_hardening/slice_registry_corrected.csv")
    assert registry.cutoff_source.str.contains("pre-2024|FFIEC|feature registry|entity-event|source metadata|frozen prediction|documented", regex=True).all()


def test_material_slice_gates() -> None:
    policy = json.loads((ROOT / "artifacts/v1_2_hardening/material_slice_policy.json").read_text())
    assert policy["minimum_rows"] == 1000 and policy["minimum_quarters"] >= 6


def test_challenger_rows_are_identically_aligned() -> None:
    source = inspect.getsource(analytics.run_slice_inference)
    assert "base.index.intersection(other.index)" in source
    assert "np.allclose(selected.actual, opponent.actual" in source


def test_selective_thresholds_remain_v11_frozen() -> None:
    frozen = json.loads((ROOT / "artifacts/v1_1_analysis/selective_policy_freeze.json").read_text())["thresholds"]
    result = pd.read_parquet(ROOT / "artifacts/v1_2_hardening/selective_review_clustered.parquet")
    combined = result.query("record_type=='policy' and policy=='combined'")
    for row in combined.itertuples():
        assert row.threshold == frozen[str(row.review_budget)]


def test_random_review_benchmark_expectation() -> None:
    result = pd.read_parquet(ROOT / "artifacts/v1_2_hardening/selective_review_clustered.parquet")
    random = result.query("policy=='random'")
    assert np.allclose(random.random_expected_capture, random.review_budget)
    assert np.allclose(random.lift_over_random, 1)


def test_actual_v1_stress_engine_is_called() -> None:
    source = inspect.getsource(stress_attribution.run_stress_attribution)
    assert "_simulate(" in source and "calculate(" in source and "optimize_recovery(" in source


def test_common_random_numbers_and_profiles() -> None:
    frame = pd.read_parquet(ROOT / "artifacts/v1_2_hardening/stress_engine_attribution.parquet")
    assert frame.profile_hash.nunique() == 1 and frame.draw_hash.nunique() == 1


def test_monitoring_redevelop_is_reachable_and_hysteretic() -> None:
    assert next_state("RECALIBRATE", 3, 2, 0, 0, 0) == "REDEVELOP"
    assert next_state("REDEVELOP", 0, 0, 0, 0, 3) == "REDEVELOP"
    assert next_state("REDEVELOP", 0, 0, 0, 0, 4) == "REVIEW"


def test_monitoring_thresholds_are_frozen_prelock() -> None:
    policy = json.loads((ROOT / "artifacts/v1_2_hardening/monitoring_state_machine.json").read_text())
    assert policy["fit_population"] == "pre-2024 only"
    assert all(value["source"].startswith("pre-2024") for value in policy["thresholds"].values())


def test_lightgbm_exact_label_obeys_reproduction_gate() -> None:
    reproduction = json.loads((ROOT / "artifacts/v1_2_hardening/lightgbm_reproduction.json").read_text())
    shap = pd.read_parquet(ROOT / "artifacts/v1_2_hardening/lightgbm_shap.parquet")
    if not reproduction["prediction_parity_passed"]:
        assert shap.empty or not shap.exact_label.fillna(False).any()
    else:
        assert not shap.empty and shap.exact_label.all()


def test_v12_database_required_tables_populated() -> None:
    required = {"v12_calibration_clustered", "v12_exception_persistence", "v12_slice_definitions",
                "v12_slice_inference", "v12_challenger_inference", "v12_selective_review",
                "v12_stress_engine_attribution", "v12_stress_variance", "v12_monitoring",
                "v12_lightgbm_reproduction", "v12_explanations", "v12_findings"}
    con = duckdb.connect(str(ROOT / "artifacts/v1_2_hardening/results_v1_2.duckdb"), read_only=True)
    try:
        present = {row[0] for row in con.execute("show tables").fetchall()}
        assert required <= present
        assert all(con.execute(f"select count(*) from {name}").fetchone()[0] > 0 for name in required)
    finally:
        con.close()

