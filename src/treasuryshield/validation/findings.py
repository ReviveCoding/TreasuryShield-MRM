"""Independent findings, inventory, and project-use recommendation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect


def run(root: Path) -> None:
    selection = json.loads((root / "artifacts" / "model_selection.json").read_text(encoding="utf-8"))
    final = json.loads((root / "artifacts" / "independent_final_review.json").read_text(encoding="utf-8"))
    stress = json.loads((root / "artifacts" / "stress" /
                         f"{selection['stress_run_id']}_manifest.json").read_text(encoding="utf-8"))
    external_fdic = json.loads((root / "artifacts" / "fdic_external_validity.json").read_text(encoding="utf-8"))
    external_y9c = json.loads((root / "artifacts" / "y9c_external_validity.json").read_text(encoding="utf-8"))
    output = root / "artifacts" / "validation_opinion.json"
    limitations_path = root / "artifacts" / "limitations_register.json"
    config = {"selection_sha256": sha256_file(root / "artifacts" / "model_selection.json"),
              "final_review_sha256": sha256_file(root / "artifacts" / "independent_final_review.json"),
              "stress_sha256": sha256_file(root / "artifacts" / "stress" /
                                            f"{selection['stress_run_id']}_manifest.json"),
              "source_hash": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        diagnostic_con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
        hmm_row = diagnostic_con.execute("""SELECT value FROM monitoring
            WHERE run_id='regime_trace_prelock'
            AND metric='last_training_log_likelihood_delta'""").fetchone()
        diagnostic_con.close()
        hmm_note = (f" The 3-state HMM's last tightening training likelihood step was "
                    f"{hmm_row[0]:.6f}, a small negative step despite the library convergence flag."
                    if hmm_row is not None and hmm_row[0] < 0 else "")
        metrics = final["selected_final_metrics"]
        def value(metric: str, quantile: str, weighting: str = "equal_bank") -> float:
            matches = [item["value"] for item in metrics if item["metric"] == metric and
                       item["quantile"] == quantile and item["weighting"] == weighting]
            if len(matches) != 1:
                raise ValueError(f"Missing final {metric} {quantile} {weighting}")
            return float(matches[0])
        coverage = value("coverage", "Q05")
        q05 = value("pinball", "Q05")
        largest_quarter = max(final["selected_quarterly_failures"],
                              key=lambda item: abs(item["q05_breach_frequency"] - .05))
        material_subgroups = [item for item in final["selected_final_subgroups"]
                              if item["weighting"] == "equal_bank" and item["n"] >= 1000]
        worst_subgroup = max(material_subgroups,
                             key=lambda item: abs(item["q05_coverage"] - .05))
        severe_shift = abs(coverage - .05) > .02
        quarterly_shift = abs(largest_quarter["q05_breach_frequency"] - .05) > .05
        opinion = ("REMEDIATE BEFORE USE" if severe_shift or quarterly_shift
                   else "ACCEPT WITH LIMITATIONS")
        findings = [
            {"finding_id": "TS-F01", "pillar": "Data quality / point in time",
             "evidence": "data/manifests/core_quality.json; data/manifests/feature_quality.json",
             "affected_use": "Historical and prospective forecasts", "severity": "HIGH",
             "materiality": "Exact original Call Report filing and amendment vintages are unavailable; a 45-day public lag and restated bank values are used.",
             "recommendation": "Acquire true filing vintages before production point-in-time use and rerun locked tests.",
             "owner": "Data governance", "status": "OPEN", "residual_risk": "Potential revision optimism"},
            {"finding_id": "TS-F02", "pillar": "Data coverage",
             "evidence": "data/manifests/ffiec_quality.json; data/manifests/feature_registry_complete.csv",
             "affected_use": "Uninsured deposit segmentation", "severity": "HIGH",
             "materiality": "Direct FFIEC uninsured values have structural form-dependent gaps; Core history uses an insured-deposit proxy.",
             "recommendation": "Restrict interpretation to reported fields and show proxy/direct results separately.",
             "owner": "Data governance", "status": "OPEN", "residual_risk": "Subgroup misclassification"},
            {"finding_id": "TS-F03", "pillar": "Outcome / calibration",
             "evidence": f"artifacts/predictions/{selection['final_run_id']}_metrics.json; artifacts/independent_final_review.json",
             "affected_use": "Q05 deposit-runoff alert thresholds", "severity": "HIGH" if severe_shift else "MEDIUM",
             "materiality": f"Final Q05 coverage {coverage:.4f} versus nominal 0.0500; worst quarter {largest_quarter['report_period']} breach frequency {largest_quarter['q05_breach_frequency']:.4f}.",
             "recommendation": "Review rolling recalibration, set conservative escalation limits, and validate on later untouched periods.",
             "owner": "Model development", "status": "OPEN", "residual_risk": "Regime-dependent tail miscalibration"},
            {"finding_id": "TS-F04", "pillar": "Model stability",
             "evidence": "artifacts/model_selection.json; artifacts/experiment_comparisons_prelock.json; artifacts/results.duckdb.monitoring",
             "affected_use": "Proposed RAHQ-C selection", "severity": "MEDIUM",
             "materiality": "Proposed variants differ materially by crisis and horizon; calibration can worsen the tightening fold." + hmm_note,
             "recommendation": "Retain simple challengers, require all preregistered gates before promoting RAHQ-C.",
             "owner": "Model development", "status": "OPEN", "residual_risk": "Model-selection instability"},
            {"finding_id": "TS-F05", "pillar": "Stress testing",
             "evidence": f"artifacts/stress/{selection['stress_run_id']}_manifest.json",
             "affected_use": "Liquidity and recovery decisions", "severity": "HIGH",
             "materiality": "Public Call Reports omit contractual cash flows, depositor concentration, collateral eligibility, intraday liquidity, and committed funding rights.",
             "recommendation": "Treat PFN/reverse stress/recovery outputs as research proxies; obtain internal cash-flow and collateral data before action use.",
             "owner": "Treasury risk", "status": "OPEN", "residual_risk": "Scenario and action-capacity model risk"},
            {"finding_id": "TS-F06", "pillar": "External validity",
             "evidence": "artifacts/fdic_external_validity.json; artifacts/y9c_external_validity.json",
             "affected_use": "Failure inference and holding-company transfer", "severity": "MEDIUM",
             "materiality": "FDIC events are secondary labels; Chicago Fed BHC test ends in 2020 and is a distinct consolidated unit.",
             "recommendation": "Do not pool BHC and bank rows; repeat external validation with current consolidated vintages if available.",
             "owner": "Independent validation", "status": "OPEN", "residual_risk": "Scope transfer uncertainty"},
            {"finding_id": "TS-F07", "pillar": "Operational monitoring",
             "evidence": "artifacts/monitoring_prelock.json; artifacts/disagreement_prelock.json",
             "affected_use": "Alerting and retraining", "severity": "MEDIUM",
             "materiality": "Historical alerts and retraining policies are simulated using public data and fixed project thresholds; live lead time is unproven.",
             "recommendation": "Shadow-run with outcome maturity checks and maintain escalation ownership before live use.",
             "owner": "Model operations", "status": "OPEN", "residual_risk": "False alerts or missed deterioration"},
            {"finding_id": "TS-F08", "pillar": "Subgroup outcome",
             "evidence": "artifacts/independent_final_review.json; artifacts/results.duckdb.subgroup_metrics",
             "affected_use": "Bank-level segmentation", "severity": "HIGH" if abs(worst_subgroup["q05_coverage"] - .05) > .05 else "MEDIUM",
             "materiality": f"Worst material final group {worst_subgroup['family']}={worst_subgroup['group']} has Q05 coverage {worst_subgroup['q05_coverage']:.4f} (n={worst_subgroup['n']}).",
             "recommendation": "Review source coverage and group-specific tail errors before segment-level thresholds; retain equal-bank results alongside weights.",
             "owner": "Independent validation", "status": "OPEN", "residual_risk": "Uneven tail error across bank segments"},
        ]
        limitations = [
            {"id": "L01", "topic": "Historical amendments", "source": "NY Fed harmonized restated Call Reports", "effect": "Pseudo point-in-time bank inputs may differ from original filings."},
            {"id": "L02", "topic": "Availability", "source": "FFIEC filing deadlines", "effect": "Uniform 45-day lag does not capture individual filing delays."},
            {"id": "L03", "topic": "ALFRED coverage", "source": "ALFRED snapshots", "effect": "Selected macro series have real-time vintages, not every market/H.8 series."},
            {"id": "L04", "topic": "Uninsured deposits", "source": "FFIEC forms and Core proxy", "effect": "Structural direct-field missingness and proxy error limit subgroup claims."},
            {"id": "L05", "topic": "Stress resources", "source": "Public Call Reports/H.8", "effect": "No internal contractual runoff, collateral, or facility rights."},
            {"id": "L06", "topic": "Dependence tails", "source": "Quarterly residuals and assumed copulas", "effect": "Daily co-stress and tail dependence are not directly observed."},
            {"id": "L07", "topic": "Final OOT length", "source": "Frozen data snapshot", "effect": "Few independent time blocks; confidence intervals are coarse."},
            {"id": "L08", "topic": "External labels", "source": "FDIC and Chicago Fed", "effect": "Failure is sparse; BHC records are historical and a different unit."},
            {"id": "L09", "topic": "Neural stability", "source": "Prelock RAHQ-C runs", "effect": "Monotonic NN required train-only initialization and remained regime-sensitive."},
            {"id": "L10", "topic": "Recovery assumptions", "source": "Stress engine configuration", "effect": "Activation lag, haircut, capacity, and cost are sensitivity inputs, not contracts."},
            {"id": "L11", "topic": "HMM optimization", "source": "Forward-filtered regime trace and DuckDB monitoring diagnostic", "effect": "The final training likelihood step can be slightly negative even when the library reports convergence; regime-state stability remains a sensitivity."},
        ]
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                   "selected_run_id": selection["final_run_id"],
                   "project_recommendation": opinion,
                   "label_policy": "TreasuryShield project label, not JPMorgan terminology or a supervisory approval.",
                   "final_q05_pinball": q05, "final_q05_coverage": coverage,
                   "worst_quarter": largest_quarter,
                   "worst_material_subgroup": worst_subgroup,
                   "findings": findings,
                   "external_validity_artifacts_checked": bool(external_fdic and external_y9c),
                   "stress_banks": stress["banks"]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        limitations_path.write_text(json.dumps({"created_at": payload["created_at"],
                                                "limitations": limitations}, indent=2), encoding="utf-8")
        reader = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
        inventory_rows = reader.execute("""SELECT model_id, any_value(panel) AS panel,
            count(*) AS runs, arg_max(config_hash, created_at),
            arg_max(artifact_path, created_at) FROM experiments
            WHERE status='completed' GROUP BY model_id""").fetchall()
        reader.close()
        writer = connect(root)
        try:
            inventory_columns = {row[1] for row in writer.execute("PRAGMA table_info('model_inventory')").fetchall()}
            for name in ("purpose", "owner_role", "inputs", "outputs", "methodology",
                         "version", "validation_status", "monitoring_status"):
                if name not in inventory_columns:
                    writer.execute(f"ALTER TABLE model_inventory ADD COLUMN {name} VARCHAR")
            writer.begin()
            writer.execute("DELETE FROM validation_findings")
            for item in findings:
                writer.execute("INSERT INTO validation_findings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               [item[key] for key in ("finding_id", "pillar", "evidence", "affected_use",
                                                      "severity", "materiality", "recommendation", "owner",
                                                      "status", "residual_risk")])
            writer.execute("DELETE FROM model_inventory")
            for model_id, panel, runs, model_hash, artifact_path in inventory_rows:
                selected_flag = model_id == selection["selected_model_id"]
                family = ("RAHQ-C proposed" if model_id.startswith("RAHQ_C") else
                          "RAHQ proposed uncalibrated" if model_id.startswith("RAHQ") else
                          "baseline/challenger")
                writer.execute("""INSERT INTO model_inventory (
                    model_id,family,panel,status,selected,config_hash,artifact_path,
                    training_snapshot,limitations,purpose,owner_role,inputs,outputs,
                    methodology,version,validation_status,monitoring_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                               [model_id, family, panel, "validated" if selected_flag else "challenger",
                                selected_flag, model_hash, artifact_path, config["snapshot"],
                                "See validation findings and model development report",
                                "Forecast severe and central quarterly deposit change",
                                "Model development; independent validation challenge",
                                f"{panel} bank, macro, and contextual features as configured",
                                "Q01/Q05/Q10/Q25/Q50/Q75/Q90 log-deposit-growth",
                                family, "0.1.0",
                                "Final locked comparison and independent review",
                                "Historical shadow-policy simulation; no live approval"])
            writer.commit()
        except Exception:
            writer.rollback()
            raise
        finally:
            writer.close()
        print(f"Validation opinion: {opinion}; {len(findings)} open findings", flush=True)

    RunState(root).run("validation_findings", "VALIDATION", config,
                       ["artifacts/validation_opinion.json", "artifacts/limitations_register.json"],
                       execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
