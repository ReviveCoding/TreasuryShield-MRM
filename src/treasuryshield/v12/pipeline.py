from __future__ import annotations

import html
import json
from datetime import datetime, timezone

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytics import (
    build_corrected_slices, load_evidence, run_calibration, run_selective_review,
    run_slice_inference,
)
from .common import ART, DB, FIG, QCOLS, QLEVELS, REP, ROOT, ensure_dirs, sha256, write_json
from .explain import reproduce_lightgbm
from .monitoring import run_monitoring
from .stress_attribution import run_stress_attribution


def checkpoint(stage: str, status: str = "completed", details: dict | None = None) -> None:
    path = ROOT / ".treasuryshield" / "v1_2_run_state.json"
    path.parent.mkdir(exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else {"schema_version": "1.2", "stages": {}}
    state["status"] = "RUNNING"
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["stages"][stage] = {"status": status, "at": state["updated_at_utc"], **(details or {})}
    write_json(path, state)


def completed(stage: str) -> bool:
    path = ROOT / ".treasuryshield" / "v1_2_run_state.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("stages", {}).get(stage, {}).get("status") == "completed"


def approximate_pit(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in final.itertuples(index=False):
        values = np.asarray([getattr(row, q) for q in QCOLS], float)
        pit = float(np.interp(row.actual, values, QLEVELS, left=0, right=1))
        rows.append({"bank_id": row.bank_id, "report_period": row.report_period, "pit": pit,
                     "lower_tail": pit < .10, "upper_tail": pit > .90,
                     "ood_score": row.ood_score,
                     "pit_method": "approximate piecewise interpolation across seven stored quantiles"})
    out = pd.DataFrame(rows)
    out["ood_decile"] = pd.qcut(out.ood_score.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    out["pit_bin"] = pd.cut(out.pit, np.linspace(0, 1, 11), include_lowest=True).astype(str)
    out.to_parquet(ART / "pit_diagnostics_approximate.parquet", index=False)
    return out


def reconciliation(calibration: pd.DataFrame, selective: pd.DataFrame, slices: pd.DataFrame) -> dict:
    frozen_metrics = json.loads((ROOT / "artifacts" / "predictions" / "core_B_1q_final_locked_B8_lightgbm_quantile_metrics.json").read_text())
    q05_pinball = next(x["value"] for x in frozen_metrics if x["metric"] == "pinball" and x["quantile"] == "Q05" and x["weighting"] == "equal_bank")
    q05_coverage = next(x["value"] for x in frozen_metrics if x["metric"] == "coverage" and x["quantile"] == "Q05" and x["weighting"] == "equal_bank")
    v11_selective = pd.read_parquet(ROOT / "artifacts" / "v1_1_analysis" / "selective_review.parquet")
    v11_challenger = pd.read_parquet(ROOT / "artifacts" / "v1_1_analysis" / "challenger_slice_deltas.parquet")
    old_reversals = v11_challenger.loc[v11_challenger.classification.eq("challenger_wins")]
    matches = []
    for old in old_reversals.itertuples(index=False):
        corrected_name = "prelock_asset_decile" if old.slice_name == "asset_decile" else old.slice_name
        corrected_value = (str(old.slice_value).replace("Q", "D", 1)
                           if old.slice_name == "asset_decile" else str(old.slice_value))
        matched = slices.loc[(slices.challenger == old.challenger) & (slices.slice_name == corrected_name)
                             & (slices.slice_value.astype(str) == corrected_value)]
        matches.append({"challenger": old.challenger, "slice_name": old.slice_name,
                        "slice_value": str(old.slice_value), "v1_1_classification": old.classification,
                        "corrected_v1_2_slice_name": corrected_name, "corrected_v1_2_slice_value": corrected_value,
                        "v1_2_classification": matched.classification.iloc[0] if len(matched) else "corrected_equivalent_unavailable",
                        "delta_q05_pinball": float(matched.delta_q05_pinball_selected_minus_challenger.iloc[0]) if len(matched) else None,
                        "quarter_ci": [float(matched.quarter_ci_low.iloc[0]), float(matched.quarter_ci_high.iloc[0])] if len(matched) else None,
                        "bank_ci": [float(matched.bank_ci_low.iloc[0]), float(matched.bank_ci_high.iloc[0])] if len(matched) else None,
                        "two_way_ci": [float(matched.two_way_ci_low.iloc[0]), float(matched.two_way_ci_high.iloc[0])] if len(matched) else None,
                        "quarter_direction_persistence": float(matched.quarter_direction_persistence.iloc[0]) if len(matched) else None,
                        "holm_adjusted_p": float(matched.holm_adjusted_p.iloc[0]) if len(matched) else None})
    result = {
        "frozen_selected_model": json.loads((ROOT / "artifacts" / "model_selection.json").read_text())["selected_model_id"],
        "frozen_q05_pinball": q05_pinball, "frozen_q05_coverage": q05_coverage,
        "v1_2_recomputed_q05_pinball": float(calibration.query("quantile=='Q05' and weighting=='equal_bank' and metric=='pinball'").estimate.iloc[0]),
        "v1_2_recomputed_q05_coverage": float(calibration.query("quantile=='Q05' and weighting=='equal_bank' and metric=='exception_rate'").estimate.iloc[0]),
        "v1_1_10pct_capture": float(v11_selective.loc[v11_selective.review_budget.eq(.1), "tail_miss_capture"].iloc[0]),
        "v1_2_10pct_capture": float(selective.query("policy=='combined' and review_budget==0.1").severe_tail_miss_capture.iloc[0]),
        "v1_1_nominal_challenger_reversals": matches,
    }
    write_json(ART / "evidence_reconciliation.json", result)
    write_json(ART / "challenger_reversal_reevaluation.json", {"prior_reversals": matches})
    return result


def build_findings(calibration: pd.DataFrame, persistence: pd.DataFrame, slices: pd.DataFrame,
                   selective: pd.DataFrame, decomposition: dict, monitoring: dict,
                   reproduction: dict) -> pd.DataFrame:
    q05 = calibration.query("quantile=='Q05' and weighting=='equal_bank' and metric=='coverage_error'").iloc[0]
    repeat = persistence.query("quantile=='Q05' and scope=='aggregate_within_bank'").iloc[0]
    robust_weak = slices.loc[slices.classification.eq("challenger_wins_robustly")]
    calibration_weak = slices.loc[slices.calibration_classification.ne("calibration_indistinguishable_or_unstable")].drop_duplicates(["slice_name", "slice_value"])
    review = selective.query("policy=='combined' and review_budget==0.1").iloc[0]
    topics = [
        ("V12-01", "Panel-aware global calibration", f"Q05 coverage error {q05.estimate:.6f}; quarter CI [{q05.quarter_ci_low:.6f}, {q05.quarter_ci_high:.6f}] and bank CI [{q05.bank_ci_low:.6f}, {q05.bank_ci_high:.6f}].", "Tail calibration and decision overlays", "High", "Retain monitoring and calibration limitation", "Validation", "Open", "Clustered inference remains low-resolution across final quarters"),
        ("V12-02", "Within-bank/quarter exception clustering", f"Within-bank adjacent-quarter Q05 repeat probability {repeat.p_exception_after_exception:.6f} versus post-nonexception {repeat.p_exception_after_nonexception:.6f}.", "Exception backtesting", "Moderate", "Use within-bank and quarter-block diagnostics", "Validation", "Closed-method", "Individual-bank tests remain underpowered"),
        ("V12-03", "Robust material slice calibration", f"{len(calibration_weak)} unique material slices retain robust Q05 calibration deviations; {len(robust_weak)} challenger-winning comparisons survive clustered inference and Holm adjustment.", "Subgroup use", "Moderate", "Route persistent weak slices to review", "Validation", "Open", "Multiple-testing and sparse-source limits remain"),
        ("V12-04", "Re-evaluated challenger reversal", "The v1.1 nominal reversal is reclassified using three clustered methods and multiplicity control in challenger_reversal_reevaluation.json.", "Champion/challenger governance", "Moderate", "Use v1.2 robust wording", "Validation", "Closed-method", "Post-hoc slice evidence does not alter champion selection"),
        ("V12-05", "Selective-review lift and uncertainty", f"Frozen combined 10% review captures {review.severe_tail_miss_capture:.3%}; lift over random is {review.lift_over_random:.2f}x.", "Human review routing", "Moderate", "Use as residual-risk routing, not accuracy improvement", "Operations", "Open", "Quarter stability varies"),
        ("V12-06", "Slice definition and provenance correction", "Peer deterioration, borrowings/assets, size, M&A distance, uninsured provenance, and report-form semantics are corrected in the v1.2 registry.", "Reporting and interpretation", "Moderate", "Use corrected v1.2 labels", "Data governance", "Closed-method", "Some direct measures remain unavailable"),
        ("V12-07", "Actual-engine model/scenario attribution", f"Scenario share {decomposition['scenario_assumption_variation_share']:.3%}; model share {decomposition['model_variation_share']:.3%}.", "Stress proxy use", "High", "Govern dominant assumptions and interaction", "Treasury validation", "Open", "Public-data proxy constraints remain"),
        ("V12-08", "Monitoring state machine effectiveness", f"Backtest precision {monitoring['backtest']['precision']:.3f}, recall {monitoring['backtest']['recall']:.3f}; all five states and hysteresis are implemented.", "Ongoing monitoring", "High", "Apply fixed rules prospectively", "Model monitoring", "Open", "No genuinely future post-2026Q1 outcomes"),
        ("V12-09", "Exact LightGBM explainability status", f"Reproduction parity passed={reproduction['prediction_parity_passed']}; max absolute error {reproduction['maximum_absolute_difference']:.3g}.", "Model explanation", "Moderate", "Use exact label only under passed gate", "Development", "Closed-method", "TreeSHAP is predictive, not causal"),
        ("V12-10", "Residual public-data limitations", "Bank-report amendment vintages, internal cash flows, collateral, concentration, committed funding, and behavioral assumptions remain unavailable.", "All uses", "High", "Maintain public-data-proxy boundary", "Model owner", "Open", "Irreducible without internal or new future data"),
    ]
    frame = pd.DataFrame(topics, columns=["finding_id", "topic", "evidence", "affected_use", "materiality", "remediation", "owner_role", "status", "residual_risk"])
    frame.to_parquet(ART / "findings_register.parquet", index=False)
    return frame


def persist_database(tables: dict[str, pd.DataFrame]) -> None:
    if DB.exists():
        DB.unlink()
    con = duckdb.connect(str(DB))
    try:
        for name, frame in tables.items():
            con.register("incoming", frame)
            con.execute(f"create table {name} as select * from incoming")
            con.unregister("incoming")
    finally:
        con.close()


def _savefig(name: str) -> None:
    plt.tight_layout(); plt.savefig(FIG / name, dpi=170, bbox_inches="tight"); plt.close()


def make_figures(calibration: pd.DataFrame, quarterly: pd.DataFrame, persistence: pd.DataFrame,
                 slices: pd.DataFrame, selective: pd.DataFrame, stress: pd.DataFrame,
                 scenarios: pd.DataFrame, decomposition: dict, monitoring: pd.DataFrame,
                 reproduction: dict, importance: pd.DataFrame, shap_frame: pd.DataFrame) -> list[str]:
    plt.style.use("seaborn-v0_8-whitegrid")
    names = []
    def done(filename: str): names.append(filename); _savefig(filename)
    plt.errorbar(range(len(quarterly)), quarterly.exception_rate, yerr=np.sqrt(.05*.95/quarterly.n_banks), fmt="o-"); plt.axhline(.05, color="black", ls="--"); plt.xticks(range(len(quarterly)), quarterly.quarter, rotation=45); plt.ylabel("Q05 exception rate"); done("01_q05_coverage_by_quarter.png")
    curve = calibration.query("weighting=='equal_bank' and metric=='exception_rate'").sort_values("alpha"); plt.plot(curve.alpha, curve.estimate, "o-", label="observed"); plt.fill_between(curve.alpha, curve.quarter_ci_low, curve.quarter_ci_high, alpha=.25, label="quarter-block 95% CI"); plt.plot([0,1],[0,1],"k--"); plt.legend(); plt.xlabel("Nominal quantile"); plt.ylabel("Exception rate"); done("02_seven_quantile_calibration.png")
    p = persistence.query("scope=='aggregate_within_bank'"); x=np.arange(len(p)); plt.bar(x-.18,p.p_exception_after_nonexception,.36,label="after nonexception"); plt.bar(x+.18,p.p_exception_after_exception,.36,label="after exception"); plt.xticks(x,p["quantile"]); plt.legend(); plt.ylabel("Transition probability"); done("03_within_bank_persistence.png")
    heat = slices.pivot_table(index="slice_name", columns="slice_value", values="selected_q05_coverage_error", aggfunc="mean"); plt.imshow(heat.fillna(0),aspect="auto",cmap="coolwarm"); plt.colorbar(label="Q05 coverage error"); plt.yticks(range(len(heat)),heat.index,fontsize=6); plt.xticks(range(len(heat.columns)),[str(x).replace("$", "\\$") for x in heat.columns],rotation=90,fontsize=6); done("04_material_slice_coverage_heatmap.png")
    forest=slices.loc[slices.material_gates_pass].sort_values("delta_q05_pinball_selected_minus_challenger").head(30); y=np.arange(len(forest)); plt.errorbar(forest.delta_q05_pinball_selected_minus_challenger,y,xerr=[forest.delta_q05_pinball_selected_minus_challenger-forest.quarter_ci_low,forest.quarter_ci_high-forest.delta_q05_pinball_selected_minus_challenger],fmt="o"); plt.axvline(0,color="black"); forest_labels=(forest.challenger+" | "+forest.slice_name+"="+forest.slice_value).str.slice(0,65).str.replace("$", "\\$", regex=False); plt.yticks(y,forest_labels,fontsize=6); done("05_clustered_delta_forest.png")
    old=json.loads((ART/"challenger_reversal_reevaluation.json").read_text())["prior_reversals"]; labels=[f"{x['challenger']}:{x['slice_name']}" for x in old] or ["none located"]; values=[0 if x.get("v1_2_classification")!="challenger_wins_robustly" else 1 for x in old] or [0]; plt.bar(labels,values); plt.ylabel("Robust reversal survives (1=yes)"); plt.xticks(rotation=30,ha="right"); done("06_prior_reversal_reevaluation.png")
    combined=selective.query("record_type=='policy' and policy=='combined'"); plt.errorbar(combined.review_budget,combined.severe_tail_miss_capture,yerr=[combined.severe_tail_miss_capture-combined.capture_quarter_ci_low,combined.capture_quarter_ci_high-combined.severe_tail_miss_capture],fmt="o-"); plt.xlabel("Review budget"); plt.ylabel("Severe-miss capture"); done("07_selective_review_capture_ci.png")
    plt.plot(combined.review_budget,combined.severe_tail_miss_capture,"o-",label="frozen combined"); plt.plot(combined.review_budget,combined.random_expected_capture,"k--",label="random expectation"); plt.legend(); plt.xlabel("Review budget"); plt.ylabel("Capture"); done("08_review_lift_random.png")
    ten=selective.query("record_type=='policy' and review_budget==0.1"); plt.bar(ten.policy,ten.severe_tail_miss_capture); plt.xticks(rotation=30,ha="right"); plt.ylabel("10% severe-miss capture"); done("09_signal_policy_comparison.png")
    reg=slices.groupby("slice_name").selected_q05_coverage_error.apply(lambda x: x.abs().max()).sort_values(); plt.barh(reg.index,reg.values); plt.xlabel("Maximum absolute Q05 coverage error"); done("10_corrected_slice_coverage.png")
    model=stress.groupby("model").peak_funding_need_proxy.mean().sort_values(); plt.barh(model.index,model.values); plt.xlabel("Mean p95 peak funding need proxy"); done("11_actual_engine_model_attribution.png")
    lhs=scenarios.query("analysis=='model_x_scenario' and model=='B8_lightgbm_quantile'"); drivers=["deposit_runoff_multiplier","funding_access_constraint","securities_haircut","commitment_draw_proxy","capital_liquidity_interaction","recovery_capacity","recovery_cost_rate"]; corr=[abs(lhs[d].corr(lhs.peak_funding_need_proxy)) for d in drivers]; plt.barh(drivers,corr); plt.xlabel("|correlation| with PFN across LHS"); done("12_scenario_assumption_tornado.png")
    parts=[decomposition["model_variation_share"],decomposition["scenario_assumption_variation_share"],decomposition["model_scenario_interaction_share"]]; plt.bar(["model","scenario","interaction"],parts); plt.ylabel("Share of balanced cell variation"); done("13_variance_decomposition.png")
    state_map={s:i for i,s in enumerate(["GREEN","WATCH","REVIEW","RECALIBRATE","REDEVELOP"])}; plt.step(range(len(monitoring)),monitoring.state.map(state_map),where="mid"); plt.yticks(list(state_map.values()),list(state_map)); plt.xticks(range(len(monitoring)),monitoring.quarter,rotation=90,fontsize=6); done("14_monitoring_state_timeline.png")
    bt=json.loads((ART/"monitoring_state_machine.json").read_text())["backtest"]; plt.bar(["true alerts","false alerts","misses"],[bt["true_alerts"],bt["false_alerts"],bt["missed_deterioration"]]); done("15_monitoring_alert_performance.png")
    errors=pd.read_parquet(ART/"lightgbm_reproduction_errors.parquet"); plt.boxplot([errors[f"error_{q}"].to_numpy() for q in QCOLS],tick_labels=[q.upper() for q in QCOLS]); plt.ylabel("Reproduced minus frozen prediction"); done("16_lightgbm_reproduction_error.png")
    if reproduction["prediction_parity_passed"] and not shap_frame.empty:
        global_shap=shap_frame.query("scope=='global'").nlargest(20,"mean_abs_shap").sort_values("mean_abs_shap"); plt.barh(global_shap.feature,global_shap.mean_abs_shap); plt.xlabel("Mean |TreeSHAP|, Q05"); done("17_exact_q05_shap.png")
        stability=shap_frame.query("scope=='quarter'").pivot(index="group",columns="feature",values="mean_abs_shap"); top=stability.mean().nlargest(10).index; plt.imshow(stability[top],aspect="auto"); plt.colorbar(label="Mean |SHAP|"); plt.xticks(range(len(top)),top,rotation=90,fontsize=6); plt.yticks(range(len(stability)),stability.index,fontsize=6); done("18_shap_stability.png")
    else:
        plt.text(.5,.5,"Exact reproduction gate failed\nNo exact SHAP claim",ha="center",va="center"); plt.axis("off"); done("17_reproduction_failure_diagnostic.png")
        gain=importance.query("quantile=='Q05'").nlargest(20,"gain") if not importance.empty else pd.DataFrame();
        if not gain.empty: plt.barh(gain.feature,gain.gain)
        else: plt.text(.5,.5,"Surrogate remains historical v1.1 evidence",ha="center")
        done("18_explanation_status.png")
    write_json(FIG / "figure_manifest.json", {"figures": names, "source_database": DB.relative_to(ROOT).as_posix()})
    return names


def _table(frame: pd.DataFrame, n: int = 30) -> str:
    return frame.head(n).to_html(index=False, border=0, float_format=lambda x: f"{x:.6g}")


def write_reports(calibration, quarterly, persistence, slices, registry, selective, stress, scenarios,
                  decomposition, monitoring_policy, monitoring, reproduction, importance, shap_frame,
                  findings, reconciliation) -> list[str]:
    style = "<style>body{font-family:Segoe UI,Arial;max-width:1200px;margin:35px auto;color:#17202a}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px;border-bottom:1px solid #ddd;text-align:left}h1,h2{color:#123b5d}.tag{background:#e8f1f8;padding:3px 7px;border-radius:4px}</style>"
    boundary = "<p><span class='tag'>Frozen v1 confirmatory evidence</span> remains unchanged. <span class='tag'>v1.1 diagnostic evidence</span> is historical. All new results below are <span class='tag'>v1.2 post-hoc hardening evidence</span> under fixed prelock policies; none is a new model-selection holdout.</p>"
    q05 = calibration.query("quantile=='Q05' and weighting=='equal_bank'")
    robust = slices.classification.value_counts().rename_axis("classification").reset_index(name="count")
    calibration_weak = slices.loc[slices.calibration_classification.ne("calibration_indistinguishable_or_unstable")].drop_duplicates(["slice_name", "slice_value"])
    review10 = selective.query("policy=='combined' and review_budget==0.1")
    conclusions = f"<ul><li>B8 remains the frozen champion; v1.2 did not retune or replace it.</li><li>Q05 clustered calibration evidence is shown below across only {quarterly.quarter.nunique()} final time clusters, so time inference is low-resolution.</li><li>{len(calibration_weak)} unique material slices retain robust Q05 coverage deviations; challenger comparisons are {html.escape(robust.to_dict('records').__str__())}.</li><li>Frozen combined 10% review capture is {review10.severe_tail_miss_capture.iloc[0]:.2%}, {review10.lift_over_random.iloc[0]:.2f}x random expectation.</li><li>Actual-engine scenario variation share is {decomposition['scenario_assumption_variation_share']:.2%}, versus {decomposition['model_variation_share']:.2%} for model choice.</li><li>Exact LightGBM prediction parity passed: {reproduction['prediction_parity_passed']}.</li></ul>"
    reports = {
        "01_validation_hardening_report.html": ("Validation Hardening Report", conclusions + "<h2>Findings</h2>" + _table(findings)),
        "02_panel_calibration_report.html": ("Panel Calibration and Backtesting", "<p>Cross-bank rows are never treated as temporal transitions. Approximate PIT is based on seven stored quantiles and pooled IID goodness-of-fit p-values are not headlined.</p><h2>Clustered calibration</h2>" + _table(q05, 50) + "<h2>Quarter series</h2>" + _table(quarterly) + "<h2>Within-bank adjacent-quarter persistence</h2>" + _table(persistence.query("scope=='aggregate_within_bank'"))),
        "03_robust_slicing_report.html": ("Robust Slicing and Selective Review", "<p>Definitions use corrected semantics, prelock cutoffs, minimum materiality gates, three clustered inference approaches, and Holm adjustment.</p><h2>Registry</h2>" + _table(registry, 50) + "<h2>Challenger inference summary</h2>" + _table(robust) + "<h2>Material calibration weaknesses</h2>" + _table(calibration_weak[["slice_name","slice_value","selected_q05_coverage_error","coverage_quarter_ci_low","coverage_quarter_ci_high","coverage_bank_ci_low","coverage_bank_ci_high","coverage_two_way_ci_low","coverage_two_way_ci_high","coverage_holm_adjusted_p","calibration_classification"]], 50) + "<h2>Prior reversal</h2><pre>" + html.escape(json.dumps(reconciliation["v1_1_nominal_challenger_reversals"], indent=2)) + "</pre><h2>Selective review</h2>" + _table(selective.query("review_budget==0.1"), 20)),
        "04_stress_engine_attribution_report.html": ("Actual v1 Stress-Engine Attribution", f"<p>The adapter calls the actual frozen v1 engine functions with common profiles and random draws. Scenario dominance is {'confirmed' if decomposition['scenario_dominates_model'] else 'not confirmed'} for mean PFN in the balanced design.</p><pre>{html.escape(json.dumps(decomposition, indent=2))}</pre><h2>Model results</h2>" + _table(stress.groupby("model").agg(pfn=("peak_funding_need_proxy","mean"),shortfall=("shortfall_probability","mean"),cost=("recovery_cost","mean")).reset_index())),
        "05_monitoring_state_machine_report.html": ("Monitoring State Machine", f"<p>Thresholds are frozen from pre-2024 quarter evidence. REDEVELOP is reachable after repeated structural OOD, repeated robust reversal, or persistence after RECALIBRATE. Hysteresis requires two clean quarters for ordinary de-escalation and four from REDEVELOP.</p><pre>{html.escape(json.dumps(monitoring_policy['backtest'], indent=2))}</pre>" + _table(monitoring, 50)),
        "06_lightgbm_explainability_report.html": ("LightGBM Reproduction and Explainability", f"<pre>{html.escape(json.dumps(reproduction, indent=2))}</pre><p>TreeSHAP and native importance are called exact only when strict prediction parity passes. Attributions are predictive and non-causal.</p><h2>Importance</h2>" + _table(importance.query("quantile=='Q05'").sort_values("gain",ascending=False), 30)),
        "07_v1_2_executive_addendum.html": ("TreasuryShield-MRM v1.2 Executive Addendum", conclusions + "<h2>Residual risks</h2><p>Exact historical amendment vintages, internal contractual cash flows, depositor concentration, collateral encumbrance, committed facilities, and institution-specific behavioral assumptions remain unavailable. No further locked model-selection analysis is justified before genuinely new future outcomes; prospective monitoring is justified.</p>"),
    }
    paths = []
    for filename, (title, body) in reports.items():
        path = REP / filename
        path.write_text(f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>{style}<h1>{html.escape(title)}</h1>{boundary}{body}<p><small>Generated {datetime.now(timezone.utc).isoformat()} from {DB.relative_to(ROOT).as_posix()} and frozen source artifacts.</small></p>", encoding="utf-8")
        paths.append(filename)
    manifest = REP / "08_v1_2_reproducibility_manifest.md"
    manifest.write_text("# TreasuryShield-MRM v1.2 Reproducibility Manifest\n\n"
                        "The v1 and v1.1 byte freeze is `artifacts/v1_2_hardening/freeze_manifest_v1_v11.json`. "
                        "The namespaced warehouse is `artifacts/v1_2_hardening/results_v1_2.duckdb`. "
                        "All v1.2 analyses use frozen predictions and prelock thresholds; only exact B8 reproduction refits the frozen recipe. "
                        "Run `.\\.venv\\Scripts\\python.exe -m treasuryshield.v12.pipeline`, the complete pytest suite, then `.\\.venv\\Scripts\\python.exe -m treasuryshield.v12.finalize`.\n\n"
                        f"Frozen model: {reconciliation['frozen_selected_model']}. Data snapshot: fbc0b7b5a2c46fbcc0b5.\n",
                        encoding="utf-8")
    paths.append(manifest.name)
    return paths


def main() -> None:
    ensure_dirs(); checkpoint("freeze", details={"manifest": "artifacts/v1_2_hardening/freeze_manifest_v1_v11.json"})
    pre, final, models = load_evidence(); checkpoint("evidence_loading", details={"prelock_rows": len(pre), "locked_rows": len(final)})
    if completed("panel_calibration") and all((ART / name).exists() for name in ["calibration_clustered.parquet", "calibration_by_quarter.parquet", "exception_persistence.parquet"]):
        calibration = pd.read_parquet(ART / "calibration_clustered.parquet")
        quarterly = pd.read_parquet(ART / "calibration_by_quarter.parquet")
        persistence = pd.read_parquet(ART / "exception_persistence.parquet")
    else:
        calibration, quarterly, persistence = run_calibration(pre, final); checkpoint("panel_calibration")
    final, registry = build_corrected_slices(pre, final)
    if completed("robust_slicing") and (ART / "slice_inference.parquet").exists():
        slices = pd.read_parquet(ART / "slice_inference.parquet")
    else:
        slices = run_slice_inference(final, models, registry); checkpoint("robust_slicing", details={"comparisons": len(slices)})
    if completed("selective_review") and (ART / "selective_review_clustered.parquet").exists():
        selective = pd.read_parquet(ART / "selective_review_clustered.parquet")
    else:
        selective = run_selective_review(pre, final); checkpoint("selective_review")
    pit = approximate_pit(final)
    if completed("stress_engine_attribution") and (ART / "stress_engine_attribution.parquet").exists():
        stress = pd.read_parquet(ART / "stress_engine_attribution.parquet")
        scenarios = pd.read_parquet(ART / "stress_scenario_design.parquet")
        decomposition = json.loads((ART / "stress_variance_decomposition.json").read_text())
    else:
        stress, scenarios, decomposition = run_stress_attribution(); checkpoint("stress_engine_attribution")
    if completed("monitoring_state_machine") and (ART / "monitoring_backtest.parquet").exists():
        monitoring_policy = json.loads((ART / "monitoring_state_machine.json").read_text())
        monitoring = pd.read_parquet(ART / "monitoring_backtest.parquet")
    else:
        monitoring_policy, monitoring = run_monitoring(pre, final, slices); checkpoint("monitoring_state_machine")
    if completed("lightgbm_reproduction") and (ART / "lightgbm_reproduction.json").exists():
        reproduction = json.loads((ART / "lightgbm_reproduction.json").read_text())
        importance = pd.read_parquet(ART / "lightgbm_exact_importance.parquet")
        shap_frame = pd.read_parquet(ART / "lightgbm_shap.parquet")
    else:
        reproduction, importance, shap_frame = reproduce_lightgbm(final); checkpoint("lightgbm_reproduction", details={"parity": reproduction["prediction_parity_passed"]})
    recon = reconciliation(calibration, selective, slices)
    findings = build_findings(calibration, persistence, slices, selective, decomposition, monitoring_policy, reproduction)
    explanation_table = pd.concat([
        shap_frame.assign(explanation_type="shap"),
        importance.rename(columns={"quantile":"group", "gain":"mean_abs_shap"}).assign(scope="importance", explanation_type="native_importance")
    ], ignore_index=True, sort=False)
    reproduction_table = pd.DataFrame(reproduction["per_quantile"])
    reproduction_table["prediction_parity_passed"] = reproduction["prediction_parity_passed"]
    variance_table = pd.DataFrame([{k:v for k,v in decomposition.items() if not isinstance(v,(dict,list))}])
    persist_database({
        "v12_calibration_clustered": calibration, "v12_exception_persistence": persistence,
        "v12_slice_definitions": registry, "v12_slice_inference": slices,
        "v12_challenger_inference": slices.copy(), "v12_selective_review": selective,
        "v12_stress_engine_attribution": pd.concat([stress, scenarios], ignore_index=True, sort=False),
        "v12_stress_variance": variance_table, "v12_monitoring": monitoring,
        "v12_lightgbm_reproduction": reproduction_table, "v12_explanations": explanation_table,
        "v12_findings": findings, "v12_pit_approximate": pit,
    }); checkpoint("database")
    figures = make_figures(calibration, quarterly, persistence, slices, selective, stress, scenarios,
                           decomposition, monitoring, reproduction, importance, shap_frame)
    reports = write_reports(calibration, quarterly, persistence, slices, registry, selective, stress,
                            scenarios, decomposition, monitoring_policy, monitoring, reproduction,
                            importance, shap_frame, findings, recon)
    checkpoint("reporting", details={"figures": len(figures), "reports": len(reports)})
    print(json.dumps({"locked_rows":len(final),"slice_comparisons":len(slices),"figures":len(figures),
                      "reports":len(reports),"lightgbm_parity":reproduction["prediction_parity_passed"],
                      "database":str(DB.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
