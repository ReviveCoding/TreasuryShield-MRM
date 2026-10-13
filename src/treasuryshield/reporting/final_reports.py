"""Evidence-first professional reports, generated only after final validation."""

from __future__ import annotations

import html
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import duckdb
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file


CSS = """<style>
body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;margin:40px auto;padding:0 25px;color:#203047;line-height:1.52}
h1,h2,h3{color:#123a60}h1{border-bottom:3px solid #176c91;padding-bottom:9px}h2{margin-top:34px;border-bottom:1px solid #bccad2}
table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0 20px}th,td{border:1px solid #ccd5dc;padding:7px 9px;text-align:left}th{background:#e7f0f5}
tr:nth-child(even){background:#f8fafc}.callout{background:#eaf3f7;border-left:5px solid #176c91;padding:13px 18px;margin:18px 0}
.risk{background:#fff3ee;border-left:5px solid #b45a42;padding:13px 18px;margin:18px 0}figure{margin:22px 0}img{max-width:100%;border:1px solid #d5dee4}
figcaption{font-size:12px;color:#56677a}small{color:#56677a}a{color:#12628c}
</style>"""


def _page(title: str, body: str, *, subtitle: str = "") -> str:
    sources = ("<h2>Official source references</h2><p>"
               "<a href='https://www.newyorkfed.org/research/banking_research/balance-sheets-income-statements'>New York Fed harmonized Call Reports</a>; "
               "<a href='https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx'>FFIEC CDR</a>; "
               "<a href='https://api.fdic.gov/banks/docs'>FDIC BankFind</a>; "
               "<a href='https://www.federalreserve.gov/releases/h8/about.htm'>Federal Reserve H.8</a>; "
               "<a href='https://fred.stlouisfed.org/docs/api/fred/realtime_period.html'>ALFRED vintages</a>; "
               "<a href='https://www.chicagofed.org/banking/financial-institution-reports/bhc-data'>Chicago Fed FR Y-9C</a>. "
               "Exact downloaded URLs, retrieval times, and checksums appear in the raw manifest and "
               "<a href='01_desktop_study.md'>desktop study</a>.</p>")
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>" + CSS +
            f"<title>{html.escape(title)}</title><body><h1>{html.escape(title)}</h1>" +
            f"<p><small>{html.escape(subtitle)}</small></p>" + body + sources + "</body></html>")


def _table(frame: pd.DataFrame, *, digits: int = 6) -> str:
    if frame.empty:
        raise ValueError("Required report table is empty")
    view = frame.copy()
    for name in view.select_dtypes(include="number").columns:
        view[name] = view[name].round(digits)
    return view.to_html(index=False, escape=True, border=0, classes="evidence")


def _fig(number: int, caption: str) -> str:
    return (f"<figure><img src='../figures/figure_{number:02d}.png' alt='{html.escape(caption)}'>"
            f"<figcaption>Figure {number}. {html.escape(caption)}. Source paths and checksums: "
            "<a href='../figures/figure_manifest.json'>figure manifest</a>.</figcaption></figure>")


def _metric(metrics: pd.DataFrame, run_id: str, metric: str,
            quantile: str = "Q05", weighting: str = "equal_bank") -> float:
    rows = metrics.loc[metrics.run_id.eq(run_id) & metrics.metric.eq(metric) &
                       metrics["quantile"].eq(quantile) & metrics.weighting.eq(weighting), "value"]
    if len(rows) != 1:
        raise ValueError(f"Expected one stored metric {run_id} {metric} {quantile} {weighting}, got {len(rows)}")
    return float(rows.iloc[0])


def run(root: Path) -> None:
    selection_path = root / "artifacts" / "model_selection.json"
    if sha256_file(selection_path) != (root / "artifacts" / "model_selection.sha256").read_text(
            encoding="ascii").split()[0]:
        raise RuntimeError("Frozen selection has changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["final_run_id"]
    review = json.loads((root / "artifacts" / "independent_final_review.json").read_text(encoding="utf-8"))
    opinion = json.loads((root / "artifacts" / "validation_opinion.json").read_text(encoding="utf-8"))
    stress_manifest = json.loads((root / "artifacts" / "stress" /
                                  f"{selected}_manifest.json").read_text(encoding="utf-8"))
    limitations = json.loads((root / "artifacts" / "limitations_register.json").read_text(encoding="utf-8"))
    monitoring = json.loads((root / "artifacts" / "monitoring_prelock.json").read_text(encoding="utf-8"))
    figure_manifest = json.loads((root / "figures" / "figure_manifest.json").read_text(encoding="utf-8"))
    if len(figure_manifest["figures"]) != 19:
        raise ValueError("Nineteen required figures must be rendered before reporting")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
    metrics = con.execute("SELECT * FROM metrics").df()
    experiments = con.execute("SELECT * FROM experiments").df()
    registry = con.execute("SELECT * FROM experiment_registry ORDER BY experiment_id").df()
    ablations = con.execute("SELECT * FROM ablations").df()
    robustness = con.execute("SELECT * FROM robustness").df()
    subgroup = con.execute("SELECT * FROM subgroup_metrics").df()
    bootstrap = con.execute("SELECT * FROM bootstrap_results").df()
    stress = con.execute("SELECT * FROM stress_runs WHERE run_id=?", [selected]).df()
    reverse = con.execute("SELECT * FROM reverse_stress WHERE run_id=?", [selected]).df()
    recovery = con.execute("SELECT * FROM recovery_runs WHERE run_id=?", [selected]).df()
    monitor_table = con.execute("SELECT * FROM monitoring").df()
    findings = con.execute("SELECT * FROM validation_findings ORDER BY finding_id").df()
    inventory = con.execute("SELECT * FROM model_inventory ORDER BY model_id").df()
    hardware = con.execute("SELECT * FROM hardware_benchmarks").df()
    quality = con.execute("SELECT * FROM data_quality").df()
    trials = con.execute("SELECT * FROM trials").df()
    budget = con.execute("SELECT * FROM budgeting_results").df()
    classification = con.execute("SELECT * FROM classification_results").df()
    con.close()
    if selected not in experiments.run_id.values:
        raise ValueError("Selected final run absent from evidence warehouse")
    if not all(item["run_id"] in experiments.run_id.values
               for item in selection["final_comparison_roster"]):
        raise ValueError("Frozen comparison roster incomplete")
    if len(stress.loc[stress.bank_id.ge(0)]) == 0 or findings.empty:
        raise ValueError("Stress or independent findings missing")
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    snapshot = selection["data_snapshot_id"]
    subtitle = f"Frozen public-data snapshot {snapshot} | Generated {datetime.now(timezone.utc).date()} UTC"

    def persist(filename: str, title: str, body: str) -> None:
        (reports / filename).write_text(_page(title, body, subtitle=subtitle), encoding="utf-8")

    primary = pd.DataFrame([{
        "Model": item["model_id"] + (" adaptive" if item["model_config"] else ""),
        "GFC Q05": item["episodes"][0]["q05_pinball"],
        "Pandemic Q05": item["episodes"][1]["q05_pinball"],
        "Tightening Q05": item["episodes"][2]["q05_pinball"],
        "Equal episode mean": item["prelock_equal_episode_mean_q05_pinball"],
        "Max coverage error": item["maximum_absolute_q05_coverage_error"],
        "Improvement vs best simple (%)": (100 * item["improvement_vs_best_simple"]
                                           if "improvement_vs_best_simple" in item else None),
        "Proposed prelock gate": item.get("passes_prelock_primary_gate"),
    } for item in selection["prelock_candidate_scores"]])
    final_table = pd.DataFrame([{
        "Model": item["model_id"] + (" adaptive" if item["model_config"] else ""),
        "Run ID": item["run_id"], "Q05 pinball": _metric(metrics, item["run_id"], "pinball"),
        "Q05 coverage": _metric(metrics, item["run_id"], "coverage"),
        "Q01 pinball": _metric(metrics, item["run_id"], "pinball", "Q01"),
        "Median MAE": _metric(metrics, item["run_id"], "mae", "Q50"),
    } for item in selection["final_comparison_roster"]])
    final_selected = final_table.loc[final_table["Run ID"].eq(selected)].iloc[0]
    weighted_view = metrics.loc[metrics.run_id.eq(selected) &
                                metrics.metric.isin(("pinball", "coverage", "mae", "weighted_interval_score")) &
                                metrics["quantile"].isin(("Q05", "Q50", "all")),
                                ["metric", "quantile", "weighting", "value", "n"]]
    comparison = bootstrap.loc[bootstrap.comparison_id.str.startswith(selected + "__vs__") &
                               bootstrap.comparison_id.str.endswith("__time_block")].copy()
    comparison["reference_run"] = comparison.comparison_id.map(
        lambda item: item.split("__vs__", 1)[1].rsplit("__", 1)[0])
    comparison["relative_difference_pct"] = comparison.apply(
        lambda item: 100 * item.estimate /
        _metric(metrics, item.reference_run, "pinball"), axis=1)
    comparison_view = comparison[["reference", "estimate", "relative_difference_pct",
                                  "ci_low", "ci_high", "replicates"]].rename(
        columns={"reference": "Reference model", "estimate": "Selected minus reference Q05",
                 "relative_difference_pct": "Relative difference (%)",
                 "ci_low": "95% low", "ci_high": "95% high", "replicates": "Bootstrap draws"})
    sensitivity_view = bootstrap.loc[bootstrap.comparison_id.str.startswith(selected + "__vs__") &
                                     ~bootstrap.comparison_id.str.endswith("__time_block"),
                                     ["comparison_id", "estimate", "ci_low", "ci_high"]]
    q05_final = float(final_selected["Q05 pinball"])
    coverage_final = float(final_selected["Q05 coverage"])
    prelock_2q4q = metrics.loc[metrics.run_id.isin(
        [f"core_B_{h}q_{split}_{selection['selected_model_id']}" for h in (2, 4)
         for split in ("gfc", "pandemic", "tightening")]) &
        metrics.metric.eq("pinball") & metrics["quantile"].eq("Q05") &
        metrics.weighting.eq("equal_bank"), ["split", "horizon", "value", "n"]].rename(
            columns={"split": "Episode", "horizon": "Quarters", "value": "Q05 pinball", "n": "Rows"})
    hpo_view = trials.groupby("run_id").agg(
        trials=("trial_number", "size"),
        completed=("status", lambda values: int((values == "COMPLETE").sum())),
        pruned=("status", lambda values: int((values == "PRUNED").sum())),
        best_inner_q05=("validation_metric", "min")).reset_index()

    report4 = (
        "<div class='callout'><b>Prelock decision.</b> " + html.escape(selection["selected_model_id"]) +
        " was frozen before any 2024+ forecast-origin holdout scoring. " +
        html.escape(selection["selection_rule"]) + "</div>"
        "<h2>Scope and temporal design</h2><p>One commercial bank RSSD ID and reporting quarter is the unit. "
        "The primary target is 1Q log deposit growth, with 2Q and 4Q secondary targets. Bank reports are "
        "assumed public after 45 days; model, peer, regime, preprocessing, and calibration inputs are fitted "
        "inside historical training partitions. Mode B uses available ALFRED vintages for selected macro series. "
        "The GFC, pandemic, and tightening test periods are prelock episodes; only the 2024+ forecast-origin "
        "window is independent of model choice.</p>"
        "<h2>Candidate development</h2>" + _table(primary) + _fig(8, "Q05 loss across historical episodes") +
        "<h3>Routine temporal versus crisis holdouts</h3>" +
        _table(ablations.loc[ablations.experiment_id.eq("E02") &
                         ablations.variant.eq("regime_holdout_vs_routine_2018_19"),
                         ["split", "value", "reference_value"]]) +
        "<h3>Bounded inner temporal tuning</h3>" + _table(hpo_view) +
        "<p>Optuna configurations and every completed/pruned trial are stored in the warehouse. Inner "
        "2016–19 folds apply to pandemic and later design work, not to the earlier GFC fit.</p>" +
        "<p>The 2022–23 tightening episode exposed substantial calibration and neural-model instability. "
        "No model is presumed to win because it is more complex. The full simple and strong challenger suite remains "
        "in the prediction warehouse.</p>"
        "<h2>Longer horizons</h2>" + _table(prelock_2q4q) +
        "<p>Horizon-specific estimates and a separate shared-horizon CUDA backbone are recorded in "
        "<a href='../artifacts/shared_horizon_tightening.json'>the shared-horizon artifact</a>.</p>"
        "<h2>Locked holdout comparison</h2>" + _table(final_table) + _fig(6, "Final holdout quantile coverage") +
        "<h2>Calibration, negative controls, and ablations</h2>" +
        _table(ablations.loc[ablations.experiment_id.isin(["E01", "E03", "E04", "E05", "E10", "E11", "E12", "E13", "E14", "E17", "E18", "E19"]),
                         ["experiment_id", "variant", "split", "value", "reference_value"]].tail(35)) +
        _fig(10, "Change in Q05 loss under recorded ablations") +
        "<p>Random future-label training, full-sample preprocessing, smoothed HMM states, revised macro data, "
        "and uncontrolled M&A rows are labeled invalid negative controls. The sign of their measured effect is "
        "retained; an invalid method is never a production candidate.</p>"
        "<h2>Robustness and compute</h2>" +
        _table(robustness[["perturbation", "level", "value", "reference_value"]].head(20)) +
        _fig(11, "Locked subgroup Q05 loss") + _fig(18, "Feature gain across fold cutoffs") +
        _fig(19, "Measured CPU and CUDA training time") +
        "<p>Raw crossing rates, post-hoc rearrangement, and the monotonic neural head were compared. "
        "Model-specific configs, training rows, resource peaks, and quantile metrics are stored under "
        "<code>artifacts/predictions/</code> and in <code>artifacts/results.duckdb</code>.</p>"
    )
    persist("04_model_development_report.html", "Model development report", report4)

    stress_table = stress.loc[stress.bank_id.ge(0) & stress.statistic.isin(
        ["shortfall_probability", "p95_peak_funding_need", "median_runway_proxy_days"])].groupby(
            ["scenario", "statistic"]).value.agg(["mean", "count"]).reset_index().rename(
                columns={"scenario": "Scenario", "statistic": "Metric", "mean": "Mean across banks", "count": "Banks"})
    dependence = stress.loc[stress.bank_id.eq(-1) & stress.scenario.str.startswith("J"),
                            ["scenario", "statistic", "value"]].rename(
                                columns={"scenario": "Dependence", "statistic": "Metric", "value": "Value"})
    definitions = stress.loc[stress.bank_id.eq(-1) & stress.scenario.str.contains("cash"),
                             ["scenario", "statistic", "value"]]
    capital = stress.loc[stress.scenario.str.startswith("capital_loss_") &
                         stress.statistic.eq("peak_funding_need_proxy")].groupby("scenario").value.agg(
                             ["mean", "count"]).reset_index()
    recovery_table = recovery.groupby("strategy")[["funding_gap", "action_cost", "residual_shortfall"]].mean().reset_index()
    report5 = (
        "<div class='risk'><b>Scope.</b> Every output in this report is a public-data research proxy. "
        "It is not LCR, NSFR, internal liquidity stress, or a contractual recovery capacity.</div>"
        f"<p>Stress distribution: <code>{html.escape(selected)}</code>; latest forecast origin "
        f"{html.escape(stress_manifest['latest_forecast_origin_period'])}. The engine evaluated "
        f"{int(stress_manifest['banks'])} bank profiles and a stratified {int(stress_manifest['profile_banks'])}" 
        "-bank set for dependence, reverse stress, and recovery sensitivity.</p>"
        "<h2>Scenario construction and outcomes</h2><p>Institution-specific runoff, an aggregate H.8 market "
        "shock/path, and their combination produce 7/30/90-day net outflow proxies. Cash and haircut-adjusted "
        "securities provide a limited resource proxy. Forecast quantiles are interpolated; unobserved contractual "
        "cash flows are never inferred.</p>" + _table(stress_table) + _fig(13, "Scenario gap path and draw uncertainty") +
        _fig(14, "Peak funding-need draw distributions") +
        "<h2>Dependence and liquidity definition</h2><p>J0 independent, J1 Gaussian, J2 Student-t, and J3 "
        "empirical quarter-block residual draws are compared on identical sampled bank profiles. Differences are "
        "scenario-model sensitivity, not estimated causal effects.</p>" + _table(dependence) + _table(definitions) +
        "<h2>Capital interaction, reverse stress, and recovery</h2>" + _table(capital) +
        f"<p>Reverse-stress records: {len(reverse)} bank/assumption combinations. "
        "A breaking runoff above 100% is marked unreachable rather than presented as feasible.</p>" +
        _fig(15, "Reachable reverse-stress frontier") + _table(recovery_table) +
        _fig(16, "Heuristic and optimized action costs") +
        "<p>The recovery optimizer minimizes assumed action cost subject to gap, capacity, haircut, and "
        "timing constraints. Facility and collateral inputs are assumptions, not rights. "
        "Material data limits and action sensitivities are in the validation findings and "
        "<a href='../artifacts/limitations_register.json'>limitations register</a>.</p>"
    )
    persist("05_stress_testing_report.html", "Stress testing and recovery report", report5)

    worst = opinion["worst_quarter"]
    worst_group = opinion["worst_material_subgroup"]
    finding_view = findings[["finding_id", "pillar", "severity", "affected_use", "materiality",
                             "recommendation", "status"]]
    subgroup_view = subgroup.loc[subgroup.run_id.eq(selected) &
                                 subgroup.metric.eq("pinball") & subgroup.weighting.eq("equal_bank"),
                                 ["subgroup_family", "subgroup", "value", "n"]]
    report6 = (
        f"<div class='risk'><b>Executive Validation Opinion: {html.escape(opinion['project_recommendation'])}.</b> "
        "This is a TreasuryShield project label, not a supervisory or JPMorgan designation. The selected model "
        "is suitable for controlled research and shadow monitoring only under the stated limitations.</div>"
        "<h2>Scope</h2><p>The validator independently rejoined final predictions to source-derived "
        "future targets, recalculated quantile metrics, checked source and target clocks, and compared fixed "
        "challengers on identical bank-origin rows.</p><h2>Intended Use</h2><p>Public-data research on deposit-runoff tail "
        "forecasting and scenario sensitivity.</p>"
        "<h2>Data Review</h2><p>Core and Enhanced Call Reports, ALFRED/FRED, H.8, FDIC events, and separate "
        "FR Y-9C observations are documented in the data report. Harmonized bank histories are restated and "
        "lack exact amendment vintages; the 45-day public lag remains a material approximation.</p>"
        "<h2>Conceptual Soundness</h2><p>Quantiles target conditional deposit "
        "growth rather than an internal cash-flow measure. All final prediction rows reconciled to the target "
        f"table ({review['feature_clock']['rows']} feature-clock joins; zero future-dated bank features and "
        "zero future ALFRED vintages). Predictions had no postprocessed quantile crossing.</p>"
        "<h2>Implementation Verification</h2><p>Independent equations, source-target joins, frozen hashes, "
        "and public-availability clocks reconciled to the development warehouse.</p>"
        "<h2>Benchmarking</h2>" + _table(final_table) + _table(comparison_view) +
        "<h3>Selected model under equal-bank and balance-sheet weights</h3>" + _table(weighted_view) +
        _fig(9, "Paired block-bootstrap intervals") +
        f"<p>The selected final Q05 pinball loss is {q05_final:.6f}; empirical Q05 coverage is "
        f"{coverage_final:.4f} against nominal 0.0500. The largest quarterly coverage departure is "
        f"{html.escape(worst['report_period'])} ({worst['q05_breach_frequency']:.4f}, "
        f"n={worst['rows']}). These failures limit threshold interpretation even if aggregate pinball improves.</p>"
        "<h2>OOT Analysis</h2><p>The final fixed roster was evaluated on 2024+ forecast origins after selection.</p>"
        "<h2>Regime Robustness</h2>" + _table(primary) +
        "<h2>Outcome Analysis</h2>" +
        _fig(7, "Quarterly Q05 tail exceedance") +
        "<p>The crisis folds are deliberately heterogeneous; 2022–23 influenced prelock model selection. "
        "The final window is the sole untouched forecast-origin holdout. FDIC distress/failure and historical "
        "Y-9C results are secondary external-validity checks and do not establish causal failure prediction "
        "or direct holding-company portability.</p>"
        "<h2>Sensitivity</h2>" + _table(subgroup_view.head(30)) +
        "<h3>Bank-cluster and crossed bank/time-block uncertainty</h3>" + _table(sensitivity_view) +
        f"<p>The largest material final subgroup coverage departure is {html.escape(worst_group['family'])}="
        f"{html.escape(worst_group['group'])}: {worst_group['q05_coverage']:.4f} (n={worst_group['n']}).</p>" +
        "<p>Prelock missingness, stale inputs, measurement noise, Enhanced-panel, M&A, calibration, and "
        "bank-cluster/two-way bootstrap sensitivity are stored in the warehouse. Subgroup conclusions are "
        "qualified where direct uninsured fields are structurally absent.</p>"
        "<h2>Stress Testing</h2><p>Dependence, capital interaction, cash-only versus "
        "cash-plus-securities, reverse runoff, and recovery optimization were executed on the selected distribution. "
        "Public sources omit contractual capacity and depositor concentration, so action recommendations require "
        "internal data.</p><h2>Reverse Stress Testing</h2><p>Breaking runoff frontiers and unreachable "
        "combinations are recorded separately by haircut and access assumption.</p>"
        "<h2>Operational Performance</h2><p>The historical policy study compares static, scheduled, "
        "quarterly recalibration, event-triggered, and performance-triggered procedures. It records alert "
        "duration, missed deterioration, false alerts, compute burden, input drift, and challenger disagreement. "
        "Live response times remain untested.</p><h2>Monitoring</h2><p>Escalation states require mature "
        "outcomes and explicit challenger review.</p>"
        "<h2>Limitations</h2><p>Source restatements, public-data coverage, time-block length, and scenario "
        "assumptions constrain interpretation.</p><h2>Findings</h2>" + _table(finding_view) +
        "<h2>Remediation</h2><p>Each open finding names an owner and concrete evidence needed before use expands.</p>"
        "<h2>Residual Risk</h2>" +
        f"<p>{len(limitations['limitations'])} limitations are catalogued with affected interpretation in "
        "<a href='../artifacts/limitations_register.json'>the limitations register</a>. The project-specific "
        f"use recommendation is <b>{html.escape(opinion['project_recommendation'])}</b>; findings remain open "
        "until independent remediation evidence is reviewed.</p>"
    )
    persist("06_independent_validation_report.html", "Independent validation report", report6)

    policy = monitor_table.loc[monitor_table.report_period.eq("2017Q1-2023Q4")].pivot_table(
        index="policy", columns="metric", values="value", aggfunc="first").reset_index()
    disagreement = monitor_table.loc[monitor_table.run_id.eq("disagreement_prelock")].pivot_table(
        index="report_period", columns="metric", values="value", aggfunc="first").reset_index()
    report7 = (
        "<h2>Historical deployment simulation</h2><p>For 2017Q1–2023Q4, each forecast uses only labels whose "
        "public-availability proxy precedes the current origin. The Q05 LightGBM policy study holds input "
        "definition and training sample cap fixed while comparing static, annual, quarterly calibration, "
        "event, and performance triggers.</p>" + _table(policy) + _fig(17, "Monitoring states through time") +
        "<h2>Signals and thresholds</h2><p>The project-internal states GREEN, WATCH, REVIEW, RECALIBRATE, "
        "and REDEVELOP respond to matured rolling coverage, rate/volatility events, and population stability "
        "index. False-alert and missed-quarter counts use a disclosed retrospective deterioration label; "
        "they are not live performance guarantees.</p>"
        "<h2>Challenger disagreement</h2>" + _table(disagreement) + _fig(12, "High-disagreement tail-miss lift") +
        "<p>Disagreement uses a fixed B1/B5/B7/B8 Q05 spread on identical prelock bank origins. It is an "
        "escalation signal only if its lift persists across episodes. Input and residual drift, regime change, "
        "and challenger gaps must be reviewed with mature outcomes.</p>"
        "<h2>Governance</h2><p>The model inventory identifies owners by role and retains all simple challengers. "
        "A material calibration or data-quality finding invokes human review, documented remediation, and a "
        "new independent validation before any expanded use. Source restatements, missingness, and H.8 market "
        "proxies remain standing monitoring limits.</p>" +
        _table(inventory[["model_id", "family", "panel", "owner_role", "version",
                          "validation_status", "monitoring_status", "selected"]])
    )
    persist("07_monitoring_and_governance_report.html", "Monitoring and governance report", report7)

    report8 = (
        f"<div class='callout'><b>Selected candidate:</b> {html.escape(selection['selected_model_id'])}. "
        f"<b>Independent project recommendation:</b> {html.escape(opinion['project_recommendation'])}. "
        "The 2024+ forecast-origin holdout was scored only after selection was frozen.</div>"
        "<h2>What was built and evaluated</h2><p>Validated official bank, macro, event, and historical "
        "holding-company sources feed a point-in-time controlled Core and Enhanced panel. Eleven baseline and "
        "challenger families, tree and neural proposed models, prelock ablations/robustness, a locked holdout, "
        "independent checks, stress/recovery proxies, and monitoring policies are stored with configs and hashes.</p>"
        f"<h2>Locked result</h2><p>Selected Q05 pinball {q05_final:.6f}; empirical Q05 coverage "
        f"{coverage_final:.4f} versus 0.0500 nominal. See the full fixed-roster results and uncertainty below.</p>" +
        _table(final_table[["Model", "Q05 pinball", "Q05 coverage", "Median MAE"]]) +
        _fig(8, "Historical crisis comparison") + _fig(7, "Final tail exceedance") +
        "<h2>Decision limits</h2><p>Coverage varies by period, and public-source liquidity stress lacks "
        "contractual flows and collateral rights. The independent findings identify required remediation before "
        "operational decisions. This work does not certify a regulatory liquidity metric or production model.</p>"
    )
    persist("08_executive_summary.html", "TreasuryShield-MRM executive summary", report8)

    budget_view = budget.loc[budget.metric.eq("mae_fraction_current_assets"),
                             ["target", "model", "value", "n"]]
    event_view = classification.loc[classification.weighting.eq("equal_bank") &
                                    classification.metric.isin(("prevalence", "pr_auc", "brier",
                                                                  "recall_at_top5pct", "precision_at_top5pct"))]
    event_view = event_view.pivot_table(index=["split", "definition", "model"], columns="metric",
                                        values="value", aggfunc="first").reset_index()
    hardware_view = hardware.loc[hardware.runtime_seconds.notna(),
                                  ["workload", "device", "runtime_seconds", "input_rows"]]
    report9 = (
        "<h2>Information clock and universe</h2><p>Commercial bank RSSD ID and quarter are the main unit. "
        "Assets, deposits, call-report lags, source vintage, report form, and M&A flags are carried explicitly. "
        "Bank-origin features must be available by the forecast origin; labels require their future quarter and "
        "availability date. The Core historical uninsured quantity is a proxy, distinct from FFIEC direct values.</p>"
        "<h2>Forecast target and score</h2><p>For horizon h, y(i,t,h)=log(Deposits(i,t+h)/Deposits(i,t)). "
        "Q05 pinball is max(0.05(y−q),−0.95(y−q)); lower-tail coverage is the proportion y≤q. "
        "Results include equal-bank, asset-weighted, and deposit-weighted metrics, MAE/RMSE, tail severity, "
        "weighted interval score, and quantile-crossing checks. M&A events are excluded on the primary path.</p>"
        "<h2>Model fitting and inference</h2><p>Numerical medians, clipping/scaling, peers, HMM parameters, "
        "and model weights are fit within training folds. Forward HMM probabilities depend only on current "
        "and earlier macro information. Adaptive calibration, where used, updates only from targets published "
        "strictly before the current origin; its algorithm was fixed before final scoring. The invalid negative "
        "controls intentionally violate these rules and are isolated.</p>"
        "<h2>Bootstrap and uncertainty</h2><p>Primary paired inference resamples circular four-quarter time "
        "blocks with all banks in each selected block. Bank-cluster and crossed bank/time-block resampling are "
        "sensitivity checks. Few holdout time blocks limit precision; a confidence interval is not evidence "
        "of stationarity.</p>"
        "<h2>Stress and optimization equations</h2><p>Peak funding need is max over 7/30/90-day "
        "(net-outflow proxy minus liquid-resource proxy), floored at zero for shortfall. Four dependence methods "
        "vary the joint deposit, borrowing, and liquid-buffer draws. Reverse stress solves the minimum "
        "deposit runoff for a 30-day shortfall under a haircut/access grid. Recovery minimizes assumed cost "
        "under liquidity gap, capacity, timing, and haircut constraints.</p>"
        "<h2>Secondary four-quarter budgeting extension</h2><p>Persistence, four-quarter trend, ridge, "
        "bank-only boosted, and regime-aware boosted estimates are tested on 2022–23 origins. The NII field "
        "is explicitly partial: loan/security interest income less selected funding interest expense. "
        "Other-borrowing interest expense is unavailable in the modern harmonized source and is omitted, "
        "so this proxy may overstate complete NII. "
        "An equal-dollar accounting projection enforces assets = liabilities + equity and reports the "
        "accuracy change from reconciliation.</p>" + _table(budget_view.head(35)) +
        "<h2>Training-frozen funding-stress event</h2><p>A peer-relative co-stress label combines severe "
        "deposit contraction with borrowing substitution or liquid-buffer deterioration. An alternative "
        "union definition tests label sensitivity. Thresholds are estimated from each training fold only. "
        "PR-AUC, Brier score, calibration bins, Recall@top 5%, and Precision@top 5% are secondary "
        "classification diagnostics, not observed liquidity failures.</p>" + _table(event_view) +
        "<h2>Hardware and implementation</h2><p>XGBoost quantile and PyTorch neural models executed on the "
        "detected RTX CUDA device; the paired CPU/GPU workloads are recorded below. A small PyTorch workload "
        "can be slower on GPU because transfer/launch overhead matters. Windows LightGBM and empirical-Bayes "
        "Student-t challengers use CPU.</p>" + _table(hardware_view) +
        "<h2>Source and code trace</h2><p>Exact official URLs, acquisition dates, and field mappings are in "
        "<a href='01_desktop_study.md'>the desktop study</a>, "
        "<a href='02_data_lineage_and_quality.html'>data lineage</a>, raw manifest, registries, and "
        "reproducibility manifest. The figure manifest records source hashes for every plot.</p>"
    )
    persist("09_technical_appendix.html", "Technical appendix", report9)

    package_names = ["python", "numpy", "pandas", "tabulate", "duckdb", "pyarrow", "xgboost", "lightgbm",
                     "torch", "scikit-learn", "hmmlearn", "optuna", "matplotlib", "quantile-forest"]
    versions = {"python": platform.python_version()}
    for name in package_names:
        if name == "python":
            continue
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    package_path = root / "artifacts" / "environment_packages.json"
    package_path.write_text(json.dumps(versions, indent=2), encoding="utf-8")
    key_paths = [root / "reports" / "preregistration.md", selection_path,
                 root / "data" / "manifests" / "raw_manifest.csv",
                 root / "data" / "manifests" / "feature_registry_complete.csv",
                 root / "data" / "manifests" / "target_registry.csv",
                 root / "artifacts" / "independent_final_review.json",
                 root / "artifacts" / "validation_opinion.json",
                 root / "figures" / "figure_manifest.json"]
    hashes = pd.DataFrame([{"Artifact": str(path.relative_to(root)).replace("\\", "/"),
                            "SHA256": sha256_file(path)} for path in key_paths])
    status = registry.status.value_counts().to_dict()
    manifest_md = (
        "# Reproducibility manifest\n\n"
        f"Generated {datetime.now(timezone.utc).isoformat()}. Snapshot `{snapshot}`. "
        f"Frozen selection SHA256 `{sha256_file(selection_path)}`.\n\n"
        "## Execution\n\n"
        "Run stages are checkpointed in `.treasuryshield/run_state.json`. Raw official files are "
        "read-only after validation and indexed by `data/manifests/raw_manifest.csv`. "
        "The final comparison roster and all code hashes at freeze are in `artifacts/model_selection.json`. "
        "Numerical claims are in `artifacts/results.duckdb`; predictions, configs, stress draws, "
        "monitoring paths, and validation checks retain artifact hashes.\n\n"
        f"Experiment registry status counts: `{json.dumps(status, sort_keys=True)}`. "
        f"Stored model runs: {len(experiments)}. Figures: {len(figure_manifest['figures'])}.\n\n"
        "## Key hashes\n\n" + hashes.to_markdown(index=False) + "\n\n"
        "## Software\n\n" + pd.DataFrame(list(versions.items()), columns=["Package", "Version"]).to_markdown(index=False) +
        "\n\nThe paired CPU/GPU timings, actual CUDA tensor/booster checks, and resource peaks are "
        "stored in `artifacts/hardware_benchmarks.json`, `artifacts/gpu_smoke.json`, and model configs. "
        "The local RTX GPU was used for compatible neural and XGBoost work.\n\n"
        "## Reproduction path\n\n"
        "1. Validate the frozen raw manifest and preregistration hashes.\n"
        "2. Rebuild processed panels, features, and targets using `src/treasuryshield/`.\n"
        "3. Execute the frozen model roster with the recorded seeds/configs; never use final labels for selection.\n"
        "4. Reconcile stored predictions and metrics with the independent validator.\n"
        "5. Rebuild stress, monitoring, figures, and reports from the warehouse and artifact manifests.\n\n"
        "Public source restatements can change if a new raw snapshot is acquired; reproduction of this run "
        "uses the frozen snapshot and recorded package versions.\n"
    )
    (reports / "10_reproducibility_manifest.md").write_text(manifest_md, encoding="utf-8")

    safe_pairs = comparison_view.copy()
    safe_md = (
        "# Frozen Model Evaluation Metrics\n\n"
        f"Frozen model: `{selected}`; selected before scoring 2024+ forecast origins. "
        f"Snapshot `{snapshot}`; independent reconciliation `artifacts/independent_final_review.json`.\n\n"
        f"Final equal-bank Q05 pinball: **{q05_final:.6f}**; empirical Q05 coverage: "
        f"**{coverage_final:.4f}** (nominal 0.0500). These values are rows in "
        "`artifacts/results.duckdb.metrics`.\n\n"
        "Paired whole-time-block comparisons (selected minus reference; negative favors selected):\n\n" +
        safe_pairs.to_markdown(index=False, floatfmt=".6f") + "\n\n"
        f"Independent project recommendation: **{opinion['project_recommendation']}**. "
        "The holdout has few time blocks; source filings are restated, and public liquidity proxies lack "
        "contractual cash flows. No single crisis, seed, or weighted aggregate is offered as a standalone "
        "production claim.\n"
    )
    (reports / "frozen_model_metrics.md").write_text(safe_md, encoding="utf-8")
    print("Generated six final HTML reports plus reproducibility and model-evaluation reports", flush=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    outputs = [f"reports/{name}" for name in (
        "04_model_development_report.html", "05_stress_testing_report.html",
        "06_independent_validation_report.html", "07_monitoring_and_governance_report.html",
        "08_executive_summary.html", "09_technical_appendix.html",
        "10_reproducibility_manifest.md", "frozen_model_metrics.md")]
    RunState(project_root).run(
        "final_reports", "REPORTING",
        {"results_sha256": sha256_file(project_root / "artifacts" / "results.duckdb"),
         "figures_sha256": sha256_file(project_root / "figures" / "figure_manifest.json"),
         "selection_sha256": sha256_file(project_root / "artifacts" / "model_selection.json"),
         "source_sha256": sha256_file(Path(__file__))},
        outputs + ["artifacts/environment_packages.json"],
        lambda: run(project_root), resume=True)
