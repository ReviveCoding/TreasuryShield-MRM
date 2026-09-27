"""Evidence warehouse. Reports query this database rather than loose claims."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from treasuryshield.orchestration.state import sha256_file

GROUPS = [
    "random vs temporal split", "temporal vs regime holdout", "availability-aware point-in-time data",
    "revised FRED vs ALFRED macro", "M&A handling", "mean vs quantile prediction",
    "historical peer vs OLS vs linear quantile", "linear vs nonlinear quantile",
    "Core vs Enhanced panel", "no-peer vs rule-peer vs learned-peer",
    "no-macro vs macro", "no-regime vs rule vs HMM",
    "hard regime vs filtered probabilities", "no calibration vs global vs peer vs regime weighted",
    "quantile crossing methods", "separate vs shared multi-horizon",
    "deposit composition ablation", "rate sensitivity and beta ablation",
    "dynamics ablation", "GFC holdout", "pandemic holdout", "2022-23 holdout",
    "final locked 2024+ holdout", "bank size robustness", "business model robustness",
    "liquidity subgroup robustness", "uninsured deposit subgroup robustness",
    "missingness perturbation", "stale-data perturbation", "measurement-noise perturbation",
    "scenario dependence", "capital-liquidity interaction sensitivity",
    "reverse stress", "recovery action comparison", "recovery optimization",
    "static vs scheduled vs trigger retraining", "model disagreement failure signal",
    "FDIC distress and failure external validity", "Y-9C BHC external validity",
    "CPU/GPU benchmark",
]


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    path = root / "artifacts" / "results.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE IF NOT EXISTS experiment_registry (experiment_id VARCHAR PRIMARY KEY, description VARCHAR, status VARCHAR, evidence_paths VARCHAR, updated_at TIMESTAMP)")
    con.execute("CREATE TABLE IF NOT EXISTS experiments (run_id VARCHAR PRIMARY KEY, experiment_id VARCHAR, panel VARCHAR, mode VARCHAR, target VARCHAR, horizon INTEGER, split VARCHAR, model_id VARCHAR, seed INTEGER, config_json VARCHAR, config_hash VARCHAR, snapshot_id VARCHAR, status VARCHAR, runtime_seconds DOUBLE, peak_ram_bytes BIGINT, peak_vram_bytes BIGINT, artifact_path VARCHAR, artifact_sha256 VARCHAR, error VARCHAR, created_at TIMESTAMP)")
    con.execute("CREATE TABLE IF NOT EXISTS predictions (run_id VARCHAR, bank_id BIGINT, report_period VARCHAR, forecast_origin DATE, horizon INTEGER, panel VARCHAR, model_id VARCHAR, quantile DOUBLE, prediction DOUBLE, actual DOUBLE, assets DOUBLE, deposits DOUBLE, peer_group INTEGER, regime VARCHAR, snapshot_id VARCHAR, config_hash VARCHAR, seed INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS metrics (run_id VARCHAR, experiment_id VARCHAR, panel VARCHAR, mode VARCHAR, split VARCHAR, horizon INTEGER, model_id VARCHAR, metric VARCHAR, quantile VARCHAR, weighting VARCHAR, value DOUBLE, n BIGINT, snapshot_id VARCHAR, config_hash VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS subgroup_metrics (run_id VARCHAR, subgroup_family VARCHAR, subgroup VARCHAR, metric VARCHAR, quantile VARCHAR, weighting VARCHAR, value DOUBLE, n BIGINT)")
    con.execute("CREATE TABLE IF NOT EXISTS bootstrap_results (comparison_id VARCHAR, split VARCHAR, panel VARCHAR, horizon INTEGER, challenger VARCHAR, reference VARCHAR, metric VARCHAR, quantile VARCHAR, estimate DOUBLE, ci_low DOUBLE, ci_high DOUBLE, replicates INTEGER, seed INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS data_quality (scope VARCHAR, metric VARCHAR, value DOUBLE, detail VARCHAR, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS data_coverage (report_period VARCHAR, panel VARCHAR, institutions BIGINT, assets_thousands DOUBLE, deposits_thousands DOUBLE, source_artifact VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS ablations (experiment_id VARCHAR, variant VARCHAR, split VARCHAR, panel VARCHAR, horizon INTEGER, metric VARCHAR, value DOUBLE, reference_value DOUBLE, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS robustness (experiment_id VARCHAR, perturbation VARCHAR, level VARCHAR, split VARCHAR, panel VARCHAR, horizon INTEGER, metric VARCHAR, value DOUBLE, reference_value DOUBLE, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS stress_runs (run_id VARCHAR, bank_id BIGINT, scenario VARCHAR, horizon INTEGER, statistic VARCHAR, value DOUBLE, config_hash VARCHAR, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS reverse_stress (run_id VARCHAR, bank_id BIGINT, scenario VARCHAR, driver VARCHAR, break_even_value DOUBLE, config_hash VARCHAR, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS recovery_runs (run_id VARCHAR, bank_id BIGINT, strategy VARCHAR, scenario VARCHAR, funding_gap DOUBLE, action_cost DOUBLE, residual_shortfall DOUBLE, config_hash VARCHAR, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS monitoring (run_id VARCHAR, report_period VARCHAR, policy VARCHAR, state VARCHAR, metric VARCHAR, value DOUBLE, alert BOOLEAN, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS validation_findings (finding_id VARCHAR PRIMARY KEY, pillar VARCHAR, evidence VARCHAR, affected_use VARCHAR, severity VARCHAR, materiality VARCHAR, recommendation VARCHAR, owner VARCHAR, status VARCHAR, residual_risk VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS model_inventory (model_id VARCHAR PRIMARY KEY, family VARCHAR, panel VARCHAR, status VARCHAR, selected BOOLEAN, config_hash VARCHAR, artifact_path VARCHAR, training_snapshot VARCHAR, limitations VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS hardware_benchmarks (workload VARCHAR, device VARCHAR, runtime_seconds DOUBLE, input_rows BIGINT, metadata_json VARCHAR, artifact_path VARCHAR)")
    con.execute("CREATE TABLE IF NOT EXISTS trials (run_id VARCHAR, trial_number INTEGER, config_json VARCHAR, validation_metric DOUBLE, status VARCHAR, runtime_seconds DOUBLE, seed INTEGER)")
    return con


def initialize(root: Path) -> None:
    con = connect(root)
    for number, title in enumerate(GROUPS, 1):
        con.execute("INSERT INTO experiment_registry VALUES (?, ?, 'pending', '', current_timestamp) ON CONFLICT (experiment_id) DO NOTHING",
                    [f"E{number:02d}", title])
    coverage = pd.read_parquet(root / "artifacts" / "eda_prelock_quarterly.parquet")
    coverage["panel"] = "core"
    coverage["source_artifact"] = "artifacts/eda_prelock_quarterly.parquet"
    con.execute("DELETE FROM data_coverage WHERE panel = 'core'")
    con.register("incoming_coverage", coverage[["report_period", "panel", "institutions", "assets_thousands", "deposits_thousands", "source_artifact"]])
    con.execute("INSERT INTO data_coverage SELECT * FROM incoming_coverage")
    con.unregister("incoming_coverage")
    quality_paths = list((root / "data" / "manifests").glob("*_quality.json")) + [root / "data" / "manifests" / "eda_prelock.json"]
    con.execute("DELETE FROM data_quality")
    rows = []
    for path in quality_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rows.append((path.stem, key, float(value), "", str(path.relative_to(root)).replace("\\", "/")))
    con.executemany("INSERT INTO data_quality VALUES (?, ?, ?, ?, ?)", rows)
    for artifact in ["gpu_smoke.json", "xgb_gpu_smoke.json"]:
        path = root / "artifacts" / artifact
        if path.exists():
            info = json.loads(path.read_text(encoding="utf-8"))
            con.execute("DELETE FROM hardware_benchmarks WHERE artifact_path = ?", [str(path.relative_to(root)).replace("\\", "/")])
            # Exact timed rows are inserted by the benchmark study; this row
            # merely preserves the verified hardware-operation evidence.
            con.execute("INSERT INTO hardware_benchmarks VALUES (?, ?, NULL, NULL, ?, ?)",
                        [artifact.removesuffix(".json"), "verified", json.dumps(info), str(path.relative_to(root)).replace("\\", "/")])
    con.close()
    print(f"Initialized {len(GROUPS)} experiment groups in {root / 'artifacts' / 'results.duckdb'}")


if __name__ == "__main__":
    initialize(Path(__file__).resolve().parents[3])
