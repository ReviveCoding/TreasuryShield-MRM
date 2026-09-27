"""Fail-closed completion audit against the repository's execution contract."""

from __future__ import annotations

import csv
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash


REQUIRED_REPORTS = [
    "01_desktop_study.md", "02_data_lineage_and_quality.html", "03_eda_report.html",
    "04_model_development_report.html", "05_stress_testing_report.html",
    "06_independent_validation_report.html", "07_monitoring_and_governance_report.html",
    "08_executive_summary.html", "09_technical_appendix.html",
    "10_reproducibility_manifest.md", "resume_safe_metrics.md",
]
REQUIRED_TABLES = [
    "data_quality", "data_coverage", "experiments", "predictions", "metrics",
    "subgroup_metrics", "bootstrap_results", "ablations", "robustness",
    "stress_runs", "reverse_stress", "recovery_runs", "monitoring",
    "validation_findings", "model_inventory", "hardware_benchmarks",
    "budgeting_results", "feature_importance", "classification_results",
    "classification_calibration", "experiment_registry",
]


def run(root: Path) -> None:
    output = root / "artifacts" / "completion_audit.json"
    selection_path = root / "artifacts" / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = {"selection_sha256": sha256_file(selection_path),
              "results_sha256": sha256_file(root / "artifacts" / "results.duckdb"),
              "figure_manifest_sha256": sha256_file(root / "figures" / "figure_manifest.json"),
              "source_sha256": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        failures = []
        evidence = {}
        state = RunState(root).data
        if state.get("data_snapshot_id") != selection["data_snapshot_id"]:
            failures.append("Run-state snapshot differs from frozen selection")
        incomplete_stages = [name for name, item in state["stages"].items()
                             if name != "completion_audit" and item["status"] != "completed"]
        if incomplete_stages:
            failures.append(f"Incomplete/failed stages: {incomplete_stages}")
        missing_reports = [name for name in REQUIRED_REPORTS if not (root / "reports" / name).is_file()]
        if missing_reports:
            failures.append(f"Missing reports: {missing_reports}")
        evidence["required_reports"] = len(REQUIRED_REPORTS) - len(missing_reports)
        if sha256_file(root / "reports" / "preregistration.md") != (
                root / "reports" / "preregistration.sha256").read_text(encoding="ascii").split()[0]:
            failures.append("Preregistration hash changed")
        if sha256_file(selection_path) != (
                root / "artifacts" / "model_selection.sha256").read_text(encoding="ascii").split()[0]:
            failures.append("Model-selection freeze hash changed")

        with (root / "data" / "manifests" / "raw_manifest.csv").open(
                encoding="utf-8-sig", newline="") as handle:
            raw = list(csv.DictReader(handle))
        raw_bad = []
        raw_mutable = []
        for item in raw:
            path = root / item["local_filename"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raw_bad.append(item["local_filename"])
            elif os.name == "nt" and not (path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY):
                raw_mutable.append(item["local_filename"])
        if raw_bad:
            failures.append(f"Raw files missing or hash-mismatched: {raw_bad[:10]} (n={len(raw_bad)})")
        if raw_mutable:
            failures.append(f"Validated raw files not read-only: {raw_mutable[:10]} (n={len(raw_mutable)})")
        evidence["validated_raw_files"] = len(raw) - len(raw_bad)
        evidence["read_only_raw_files"] = len(raw) - len(raw_mutable)

        figures_path = root / "figures" / "figure_manifest.json"
        figures = json.loads(figures_path.read_text(encoding="utf-8"))["figures"]
        if [item["number"] for item in figures] != list(range(1, 20)):
            failures.append("Required figure numbering/count incomplete")
        bad_figures = []
        verified_source_hashes: dict[str, str] = {}
        for item in figures:
            path = root / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                bad_figures.append(item["path"])
            for source in item["sources"]:
                source_path = root / source["path"]
                if source_path.is_file() and source["path"] not in verified_source_hashes:
                    verified_source_hashes[source["path"]] = sha256_file(source_path)
                if not source_path.is_file() or verified_source_hashes.get(source["path"]) != source["sha256"]:
                    bad_figures.append(source["path"])
        if bad_figures:
            failures.append(f"Figure or figure-source hash mismatch: {bad_figures[:10]}")
        evidence["figures_verified"] = 19 - len(set(x for x in bad_figures if x.startswith("figures/")))

        con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        missing_tables = sorted(set(REQUIRED_TABLES) - tables)
        if missing_tables:
            failures.append(f"Missing result tables: {missing_tables}")
        inventory_fields = {row[1] for row in con.execute("PRAGMA table_info('model_inventory')").fetchall()}
        required_inventory_fields = {"model_id", "purpose", "owner_role", "inputs", "outputs",
                                     "methodology", "version", "limitations",
                                     "validation_status", "monitoring_status"}
        if required_inventory_fields - inventory_fields:
            failures.append(f"Model inventory fields missing: {sorted(required_inventory_fields - inventory_fields)}")
        registry = con.execute("SELECT experiment_id,status FROM experiment_registry ORDER BY experiment_id").fetchall()
        incomplete_experiments = [(eid, status) for eid, status in registry if status != "completed"]
        if len(registry) != 40 or incomplete_experiments:
            failures.append(f"Experiment registry incomplete: {len(registry)} groups, {incomplete_experiments}")
        evidence["completed_experiment_groups"] = len(registry) - len(incomplete_experiments)
        nonfinite_metrics = con.execute("SELECT COUNT(*) FROM metrics WHERE value IS NULL OR isnan(value) OR isinf(value)").fetchone()[0]
        if nonfinite_metrics:
            failures.append(f"Nonfinite/placeholder metric values: {nonfinite_metrics}")
        final = con.execute("SELECT run_id,created_at FROM experiments WHERE split='final_locked'").fetchall()
        roster = {item["run_id"] for item in selection["final_comparison_roster"]}
        if {item[0] for item in final} != roster:
            failures.append("Final holdout runs differ from frozen comparison roster")
        for run_id, _ in final:
            info = json.loads((root / "artifacts" / "predictions" / f"{run_id}.json").read_text(
                encoding="utf-8"))
            if info["created_at"] <= selection["frozen_at"]:
                failures.append(f"Final run predates selection freeze: {run_id}")
            if info["snapshot_id"] != selection["data_snapshot_id"]:
                failures.append(f"Final run snapshot mismatch: {run_id}")
        evidence["frozen_final_roster_runs"] = len(final)
        stored = con.execute("""SELECT value FROM metrics WHERE run_id=? AND metric='pinball'
            AND quantile='Q05' AND weighting='equal_bank'""", [selection["final_run_id"]]).fetchone()
        coverage = con.execute("""SELECT value FROM metrics WHERE run_id=? AND metric='coverage'
            AND quantile='Q05' AND weighting='equal_bank'""", [selection["final_run_id"]]).fetchone()
        opinion = json.loads((root / "artifacts" / "validation_opinion.json").read_text(encoding="utf-8"))
        if stored is None or coverage is None or not np.isclose(stored[0], opinion["final_q05_pinball"]) or not np.isclose(
                coverage[0], opinion["final_q05_coverage"]):
            failures.append("Headline final metrics do not reconcile to DuckDB")
        for table in ("data_quality", "data_coverage", "predictions", "metrics", "subgroup_metrics",
                      "bootstrap_results", "ablations", "robustness", "stress_runs",
                      "reverse_stress", "recovery_runs", "monitoring", "validation_findings",
                      "model_inventory", "hardware_benchmarks", "budgeting_results", "feature_importance",
                      "classification_results", "classification_calibration"):
            if table in tables and con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
                failures.append(f"Required evidence table empty: {table}")
        con.close()

        torch = json.loads((root / "artifacts" / "gpu_smoke.json").read_text(encoding="utf-8"))
        xgb = json.loads((root / "artifacts" / "xgb_gpu_smoke.json").read_text(encoding="utf-8"))
        benchmark = json.loads((root / "artifacts" / "hardware_benchmarks.json").read_text(encoding="utf-8"))
        if not torch.get("cuda_available") or torch.get("peak_vram_bytes", 0) <= 0 or not xgb.get(
                "cuda_recorded_device", "").startswith("cuda"):
            failures.append("Actual RTX/PyTorch/XGBoost CUDA execution not verified")
        pairs = {(item["workload"], item["device"]) for item in benchmark["matched_workloads"]}
        if len(pairs) != 4:
            failures.append("Matched CPU/GPU benchmark pairs incomplete")
        evidence["verified_gpu_name"] = torch.get("device_name")

        required_sources = list((root / "src" / "treasuryshield").rglob("*.py"))
        todos = []
        for path in required_sources:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("#") and ("TO" + "DO" in line or "FIX" + "ME" in line):
                    todos.append(f"{path.relative_to(root)}:{line_number}")
        if todos:
            failures.append(f"Unresolved source TODO/FIXME: {todos[:20]}")
        command = [sys.executable, "-m", "pytest", "-q"]
        tested = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=300)
        (root / "artifacts" / "pytest_final.txt").write_text(
            tested.stdout + tested.stderr, encoding="utf-8")
        if tested.returncode:
            failures.append(f"pytest failed: {tested.returncode}")
        evidence["pytest_output"] = tested.stdout.strip()
        audit = {"audited_at": datetime.now(timezone.utc).isoformat(),
                 "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                 "checks": evidence, "failures": failures,
                 "verified_complete": not failures,
                 "limits": ["A passing audit verifies stored execution artifacts, not fitness for production use.",
                            "Independent project recommendation remains in validation_opinion.json."]}
        output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        if failures:
            raise AssertionError("Completion audit failed: " + "; ".join(failures))
        print("Completion contract verified", flush=True)

    RunState(root).run("completion_audit", "VERIFYING", config,
                       ["artifacts/completion_audit.json", "artifacts/pytest_final.txt"],
                       execute, resume=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    if result["verified_complete"]:
        state = RunState(root)
        state.data["completion_verified"] = True
        state.set_status("COMPLETE")


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
