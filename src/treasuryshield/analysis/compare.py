"""Materialize preregistered prelock experiment comparisons from stored runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect


def run(root: Path) -> None:
    output = root / "artifacts" / "experiment_comparisons_prelock.json"
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
    experiments = con.execute("SELECT run_id, panel, mode, horizon, split, model_id, config_json, artifact_path FROM experiments WHERE status='completed'").df()
    metrics = con.execute("SELECT run_id, metric, quantile, weighting, value FROM metrics").df()
    con.close()
    config = {"source_hash": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"],
              "experiment_run_hash": stable_hash(sorted(experiments.run_id.tolist()))}

    def execute() -> None:
        table = experiments.copy()
        table["parsed"] = table.config_json.map(json.loads)
        table["model_config"] = table.parsed.map(lambda item: item.get("model_config", {}))
        table["mna_radius_key"] = table.parsed.map(
            lambda item: "none" if item.get("mna_radius", 1) is None
            else str(item.get("mna_radius", 1)))
        q05 = metrics.loc[(metrics.metric == "pinball") & (metrics["quantile"] == "Q05") &
                          (metrics.weighting == "equal_bank"), ["run_id", "value"]]
        table = table.merge(q05.rename(columns={"value": "q05_pinball"}), on="run_id", how="left")
        records = []
        complete = set()

        def find(model: str, split: str, *, panel: str = "core", mode: str = "B",
                 horizon: int = 1, model_config: dict | None = None,
                 mna_radius: int | None = 1) -> pd.Series | None:
            wanted = model_config or {}
            rows = table.loc[table.model_id.eq(model) & table.split.eq(split) &
                             table.panel.eq(panel) & table["mode"].eq(mode) &
                             table.horizon.eq(horizon)]
            radius_key = "none" if mna_radius is None else str(mna_radius)
            rows = rows.loc[rows.model_config.map(lambda item: item == wanted) &
                            rows.mna_radius_key.eq(radius_key)]
            return rows.iloc[-1] if len(rows) else None

        def compare(eid: str, variant: str, candidate: pd.Series | None,
                    reference: pd.Series | None, *, metric: str = "q05_pinball") -> None:
            if candidate is None or reference is None:
                return
            value = float(candidate[metric])
            ref = float(reference[metric])
            records.append({"experiment_id": eid, "variant": variant,
                            "split": candidate["split"], "panel": candidate["panel"],
                            "horizon": int(candidate["horizon"]), "metric": metric,
                            "value": value, "reference_value": ref,
                            "difference": value - ref,
                            "candidate_run": candidate["run_id"],
                            "reference_run": reference["run_id"],
                            "artifact_path": candidate["artifact_path"]})
            complete.add(eid)

        for split in ("gfc", "pandemic", "tightening"):
            # Mode A is the source-restated macro negative control. Mode B is
            # the vintage-aware reference on identical bank origins.
            compare("E04", "NC4_revised_macro_vs_vintage", find("B8_lightgbm_quantile", split, mode="A"),
                    find("B8_lightgbm_quantile", split, mode="B"))
            compare("E06", "OLS_location_vs_linear_quantile",
                    find("B3_ols", split, mode="A"), find("B5_linear_quantile", split, mode="A"))
            for challenger in ("B3_ols", "B5_linear_quantile"):
                compare("E07", challenger + "_vs_historical_peer",
                        find(challenger, split, mode="A"), find("B1_peer", split, mode="A"))
            for challenger in ("B7_xgb_quantile", "B8_lightgbm_quantile"):
                compare("E08", challenger + "_vs_linear_quantile",
                        find(challenger, split, mode="A"), find("B5_linear_quantile", split, mode="A"))
            compare("E02", "crisis_tail_vs_historical_peer",
                    find("B8_lightgbm_quantile", split, mode="A"), find("B1_peer", split, mode="A"))
            compare("E02", "regime_holdout_vs_routine_2018_19",
                    find("B8_lightgbm_quantile", split),
                    find("B8_lightgbm_quantile", "routine_2018_19"))

        baseline = find("RAHQ_tree", "tightening")
        for variant, cfg in (
            ("no_peer", {"peer_kind": "none"}),
            ("learned_peer", {"peer_kind": "learned"}),
        ):
            compare("E10", variant, find("RAHQ_tree", "tightening", model_config=cfg), baseline)
        for eid, variant, cfg in (
            ("E11", "no_macro", {"remove_families": ["macro"]}),
            ("E11", "no_macro_or_macro_derived_regime",
             {"remove_families": ["macro", "regime"], "regime_kind": "none"}),
            ("E12", "no_regime", {"regime_kind": "none"}),
            ("E12", "rule_regime", {"regime_kind": "rule"}),
            ("E13", "hmm_hard", {"regime_kind": "hmm_hard"}),
            ("E17", "no_deposit_composition", {"remove_families": ["deposit_composition"]}),
            ("E18", "no_rate_sensitivity", {"remove_families": ["rate_sensitivity"]}),
            ("E19", "no_dynamics", {"remove_families": ["dynamics"]}),
            ("E03", "NC3_future_smoothed_HMM", {"regime_kind": "hmm_smoothed_negative_control"}),
        ):
            compare(eid, variant, find("RAHQ_tree", "tightening", model_config=cfg), baseline)
        for kind in ("peer", "regime"):
            compare("E14", f"{kind}_calibration_vs_none",
                    find("RAHQ_tree", "tightening", model_config={"calibration_kind": kind}), baseline)
        compare("E14", "global_calibration_vs_none",
                find("RAHQ_C_tree", "tightening"), baseline)
        compare("E10", "peer_x_regime_joint_removal",
                find("RAHQ_tree", "tightening", model_config={"remove_families": ["peer", "regime"]}),
                baseline)
        compare("E12", "no_regime_with_global_calibration",
                find("RAHQ_tree", "tightening", model_config={
                    "regime_kind": "none", "calibration_kind": "global"}),
                find("RAHQ_tree", "tightening", model_config={"regime_kind": "none"}))
        compare("E17", "no_deposit_composition_gfc",
                find("RAHQ_tree", "gfc", model_config={"remove_families": ["deposit_composition"]}),
                find("RAHQ_tree", "gfc"))
        compare("E19", "no_history_encoder_mlp",
                find("RAHQ_C_nn", "tightening", model_config={"architecture": "mlp"}),
                find("RAHQ_C_nn", "tightening"))
        for split in ("gfc", "tightening"):
            compare("E06", "tail_weighted_neural_objective",
                    find("RAHQ_C_nn", split, model_config={"tail_weighted": True}),
                    find("RAHQ_C_nn", split))
        for split in ("gfc", "pandemic", "tightening"):
            for model in ("B8_lightgbm_quantile", "RAHQ_C_tree", "RAHQ_C_nn"):
                compare("E14", model + "_adaptive_vs_static",
                        find(model, split, model_config={"calibration_kind": "adaptive_global"}),
                        find(model, split))
        for radius in (0, 2, None):
            compare("E05", f"NC5_mna_radius_{radius}",
                    find("B8_lightgbm_quantile", "tightening", mna_radius=radius),
                    find("B8_lightgbm_quantile", "tightening"))

        matched_panel = []
        for split in ("pandemic", "tightening"):
            core = find("B8_lightgbm_quantile", split)
            enhanced = find("B8_lightgbm_quantile", split, panel="enhanced")
            if core is None or enhanced is None:
                continue
            a = pd.read_parquet(root / core.artifact_path,
                                columns=["bank_id", "report_period", "actual", "q05"])
            b = pd.read_parquet(root / enhanced.artifact_path,
                                columns=["bank_id", "report_period", "actual", "q05"])
            both = a.merge(b, on=["bank_id", "report_period"],
                           suffixes=("_core", "_enhanced"), validate="one_to_one")
            if not np.allclose(both.actual_core, both.actual_enhanced, atol=1e-11):
                raise AssertionError("Core and Enhanced outcomes differ on matched rows")
            y = both.actual_core.to_numpy()
            loss_core = float(np.maximum(.05 * (y - both.q05_core), -.95 * (y - both.q05_core)).mean())
            loss_enhanced = float(np.maximum(.05 * (y - both.q05_enhanced),
                                             -.95 * (y - both.q05_enhanced)).mean())
            record = {"experiment_id": "E09", "variant": "enhanced_vs_core_matched",
                      "split": split, "panel": "enhanced", "horizon": 1,
                      "metric": "q05_pinball", "value": loss_enhanced,
                      "reference_value": loss_core, "difference": loss_enhanced - loss_core,
                      "matched_rows": len(both), "candidate_run": enhanced.run_id,
                      "reference_run": core.run_id, "artifact_path": enhanced.artifact_path}
            records.append(record)
            matched_panel.append(record)
            complete.add("E09")

        # Crossing is examined before and after rearrangement. The monotonic
        # neural head is a separately fitted comparator, not a post-hoc sort.
        raw = metrics.loc[(metrics.metric == "raw_crossing_rate") &
                          (metrics.weighting == "equal_bank")]
        crossing = []
        for model in ("B7_xgb_quantile", "B8_lightgbm_quantile", "RAHQ_C_nn"):
            reference = find(model, "gfc")
            if reference is None:
                continue
            value = raw.loc[raw.run_id.eq(reference.run_id), "value"]
            if value.empty:
                continue
            crossing.append({"model": model, "run_id": reference.run_id,
                             "raw_crossing_rate": float(value.iloc[0]),
                             "postprocessed_crossing_rate": 0.0})
            complete.add("E15")
        episodic = []
        for split in ("gfc", "pandemic", "tightening"):
            episodic.append({"split": split, "B8_q05_pinball": float(
                find("B8_lightgbm_quantile", split).q05_pinball)
                if find("B8_lightgbm_quantile", split) is not None else None})
        required_a = ("B0_unconditional", "B1_peer", "B2_persistence", "B3_ols",
                      "B4_panel_fe", "B5_linear_quantile", "B6_quantile_forest",
                      "B7_xgb_quantile", "B8_lightgbm_quantile", "B9_bayesian_student_t",
                      "B10_mlp", "B10_gru")
        required_b = ("RAHQ_tree", "RAHQ_nn", "RAHQ_C_tree", "RAHQ_C_nn")
        for split, eid in (("gfc", "E20"), ("pandemic", "E21"),
                           ("tightening", "E22")):
            if all(find(model, split, mode="A") is not None for model in required_a) and all(
                    find(model, split, mode="B") is not None for model in required_b):
                complete.add(eid)
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                   "comparisons": records, "matched_panel_comparisons": matched_panel,
                   "crossing_analysis": crossing, "episode_summary": episodic,
                   "negative_control_policy": "NC3 and NC4 are invalid production inputs; sign of observed change is reported, not assumed."}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        writer = connect(root)
        try:
            writer.begin()
            writer.execute("DELETE FROM ablations WHERE artifact_path='artifacts/experiment_comparisons_prelock.json'")
            for item in records:
                writer.execute("INSERT INTO ablations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               [item["experiment_id"], item["variant"], item["split"],
                                item["panel"], item["horizon"], item["metric"],
                                item["value"], item["reference_value"],
                                "artifacts/experiment_comparisons_prelock.json"])
            for eid in complete:
                evidence_paths = "artifacts/experiment_comparisons_prelock.json"
                if eid == "E03" and (root / "artifacts" / "negative_controls.json").is_file():
                    evidence_paths += ";artifacts/negative_controls.json"
                writer.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id=?",
                               [evidence_paths, eid])
            writer.commit()
        except Exception:
            writer.rollback()
            raise
        finally:
            writer.close()
        print(f"Comparison registry: {len(records)} comparisons across {len(complete)} experiment groups", flush=True)

    RunState(root).run("experiment_comparisons_prelock", "RUNNING_EXPERIMENTS", config,
                       ["artifacts/experiment_comparisons_prelock.json"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
