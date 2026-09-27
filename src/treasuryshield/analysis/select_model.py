"""Freeze a prelock selection and fixed final comparison roster before scoring."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash


SPLITS = ("gfc", "pandemic", "tightening")
CANDIDATES = (
    ("B7_xgb_quantile", {}),
    ("B8_lightgbm_quantile", {}),
    ("B8_lightgbm_quantile", {"calibration_kind": "global"}),
    ("B8_lightgbm_quantile", {"calibration_kind": "peer"}),
    ("B8_lightgbm_quantile", {"calibration_kind": "adaptive_global"}),
    ("RAHQ_C_tree", {}),
    ("RAHQ_C_tree", {"calibration_kind": "adaptive_global"}),
    ("RAHQ_C_nn", {}),
    ("RAHQ_C_nn", {"calibration_kind": "adaptive_global"}),
)


def _run_id(model: str, split: str, config: dict) -> str:
    base = f"core_B_1q_{split}_{model}"
    return base + ("_" + stable_hash(config)[:8] if config else "")


def run(root: Path) -> None:
    output = root / "artifacts" / "model_selection.json"
    frozen_hash_path = root / "artifacts" / "model_selection.sha256"
    if output.exists():
        if not frozen_hash_path.is_file() or sha256_file(output) != frozen_hash_path.read_text(
                encoding="ascii").split()[0]:
            raise RuntimeError("Frozen model selection hash mismatch")
        print("Verified existing frozen model selection", flush=True)
        return
    prereg = root / "reports" / "preregistration.md"
    if sha256_file(prereg) != (root / "reports" / "preregistration.sha256").read_text(
            encoding="ascii").split()[0]:
        raise RuntimeError("Preregistration hash changed")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
    if con.execute("SELECT COUNT(*) FROM experiments WHERE split='final_locked'").fetchone()[0]:
        raise RuntimeError("Final locked holdout has already been scored")
    metric_table = con.execute("""SELECT run_id,metric,quantile,weighting,value,n
        FROM metrics WHERE split IN ('gfc','pandemic','tightening') AND panel='core'
        AND mode='B' AND horizon=1""").df()
    experiment_table = con.execute("""SELECT run_id, config_hash, artifact_path, runtime_seconds,
        peak_vram_bytes FROM experiments WHERE split IN ('gfc','pandemic','tightening')
        AND panel='core' AND mode='B' AND horizon=1 AND status='completed'""").df()
    con.close()
    candidates = []
    for model, model_config in CANDIDATES:
        runs = [_run_id(model, split, model_config) for split in SPLITS]
        if any(run_id not in experiment_table.run_id.values for run_id in runs):
            raise ValueError(f"Selection prerequisite missing: {runs}")
        scores = []
        for split, run_id in zip(SPLITS, runs):
            sub = metric_table.loc[metric_table.run_id.eq(run_id) &
                                   metric_table.weighting.eq("equal_bank")]
            def metric(name: str, quantile: str) -> float:
                row = sub.loc[sub.metric.eq(name) & sub["quantile"].eq(quantile), "value"]
                if len(row) != 1:
                    raise ValueError(f"Missing metric {name} {quantile} for {run_id}")
                return float(row.iloc[0])
            scores.append({"split": split, "run_id": run_id,
                           "q05_pinball": metric("pinball", "Q05"),
                           "q01_pinball": metric("pinball", "Q01"),
                           "q05_coverage": metric("coverage", "Q05"),
                           "median_mae": metric("mae", "Q50"),
                           "tail_underestimate_severity": metric("tail_underestimate_severity", "Q05")})
        info = experiment_table.loc[experiment_table.run_id.isin(runs)]
        candidates.append({"model_id": model, "model_config": model_config,
                           "class": "proposed" if model.startswith("RAHQ") else "challenger",
                           "episodes": scores,
                           "prelock_equal_episode_mean_q05_pinball": float(np.mean(
                               [item["q05_pinball"] for item in scores])),
                           "maximum_absolute_q05_coverage_error": float(max(
                               abs(item["q05_coverage"] - .05) for item in scores)),
                           "mean_runtime_seconds": float(info.runtime_seconds.mean()),
                           "maximum_peak_vram_bytes": int(info.peak_vram_bytes.max())})
    simple = min((candidate for candidate in candidates if candidate["class"] == "challenger"),
                 key=lambda item: item["prelock_equal_episode_mean_q05_pinball"])
    eligible_proposed = []
    for candidate in candidates:
        if candidate["class"] != "proposed":
            continue
        improvements = []
        for episode, reference in zip(candidate["episodes"], simple["episodes"]):
            improvements.append((reference["q05_pinball"] - episode["q05_pinball"]) /
                                reference["q05_pinball"])
        mean_improvement = (
            (simple["prelock_equal_episode_mean_q05_pinball"] -
             candidate["prelock_equal_episode_mean_q05_pinball"]) /
            simple["prelock_equal_episode_mean_q05_pinball"])
        candidate["improvement_vs_best_simple"] = mean_improvement
        candidate["worst_episode_improvement_vs_best_simple"] = min(improvements)
        candidate["passes_prelock_primary_gate"] = bool(
            mean_improvement >= .03 and min(improvements) >= -.10 and
            candidate["maximum_absolute_q05_coverage_error"] <= .02)
        if candidate["passes_prelock_primary_gate"]:
            eligible_proposed.append(candidate)
    selected = (min(eligible_proposed,
                    key=lambda item: item["prelock_equal_episode_mean_q05_pinball"])
                if eligible_proposed else simple)
    selected_model = selected["model_id"]
    selected_config = selected["model_config"]
    fixed_comparators = [
        ("B0_unconditional", {}), ("B1_peer", {}), ("B2_persistence", {}),
        ("B3_ols", {}), ("B4_panel_fe", {}), ("B5_linear_quantile", {}),
        ("B6_quantile_forest", {}), ("B7_xgb_quantile", {}),
        ("B8_lightgbm_quantile", {}), ("B9_bayesian_student_t", {}),
        ("B10_mlp", {}), ("B10_gru", {}),
        ("RAHQ_tree", {}), ("RAHQ_nn", {}),
        ("RAHQ_C_tree", {}), ("RAHQ_C_nn", {}),
    ]
    final_roster = list(dict.fromkeys((selected_model, json.dumps(selected_config, sort_keys=True))
                                    for selected_model, selected_config in [
                                        (selected_model, selected_config), *fixed_comparators]))
    final_roster = [{"model_id": model, "model_config": json.loads(cfg),
                     "run_id": _run_id(model, "final_locked", json.loads(cfg))}
                    for model, cfg in final_roster]
    final_run = _run_id(selected_model, "final_locked", selected_config)
    source_hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
                     for path in sorted((root / "src" / "treasuryshield").rglob("*.py"))}
    payload = {"frozen_at": datetime.now(timezone.utc).isoformat(),
               "data_snapshot_id": RunState(root).data["data_snapshot_id"],
               "preregistration_sha256": sha256_file(prereg),
               "feature_registry_sha256": sha256_file(root / "data" / "manifests" / "feature_registry.csv"),
               "target_registry_sha256": sha256_file(root / "data" / "manifests" / "target_registry.csv"),
               "prelock_candidate_scores": candidates,
               "selection_rule": "If a proposed model passes the preregistered 3% mean Q05, worst-episode and coverage prelock gates, choose the best proposed; otherwise choose the lowest mean Q05 credible challenger. Full validation gates remain for use recommendation.",
               "selected_model_id": selected_model, "selected_model_config": selected_config,
               "selected_prelock_run_ids": [_run_id(selected_model, split, selected_config)
                                            for split in SPLITS],
               "final_run_id": final_run, "stress_run_id": final_run,
               "final_comparison_roster": final_roster,
               "source_hashes_at_freeze": source_hashes,
               "final_holdout_scored_at_freeze": False}
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    frozen_hash_path.write_text(sha256_file(output) + "  model_selection.json\n", encoding="ascii")
    RunState(root).run("model_selection", "SELECTION",
                       {"selection_sha256": sha256_file(output),
                        "preregistration_sha256": payload["preregistration_sha256"]},
                       ["artifacts/model_selection.json", "artifacts/model_selection.sha256"],
                       lambda: None, resume=True)
    print(f"Frozen selection: {selected_model} {selected_config}; final roster {len(final_roster)}", flush=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
