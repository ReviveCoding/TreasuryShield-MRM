from __future__ import annotations

import html
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from .calibration import (
    calibration_rows, exception_tests, pit_summary, pit_values, weighted_quantile,
)
from .common import (
    ART, DB, FIG, KEY_MODELS, PRELOCK_SPLITS, QCOLS, QLEVELS, REP, ROOT,
    aligned_predictions, apply_edges, enrich, ensure_dirs, model_path, pinball,
    read_prediction, read_prelock, sha256, stable_edges, weighted_mean, write_json,
)

SEED = 20260927
rng = np.random.default_rng(SEED)
plt.style.use("seaborn-v0_8-whitegrid")

SLICE_FEATURES = {
    "asset_decile": ("assets", 10),
    "size_band": ("assets", 4),
    "cash_assets": ("cash_asset_ratio", 5),
    "liquid_assets": ("liquid_broad_asset_ratio", 5),
    "loans_deposits": ("loan_deposit_ratio", 5),
    "uninsured_proxy": ("uninsured_proxy_domestic_share", 5),
    "brokered_share": ("brokered_domestic_ratio", 5),
    "time_deposit_share": ("uninsured_time_deposit_ratio", 5),
    "borrowings_assets": ("borrowings_asset_ratio", 5),
    "noncore_wholesale_proxy": ("borrowings_asset_ratio", 5),
    "capital_assets": ("capital_asset_ratio", 5),
    "npl_loans": ("npl_loan_ratio", 5),
    "loss_reserve": ("loan_loss_reserve_ratio", 5),
    "profitability": ("profit_asset_ratio", 5),
    "deposit_cost": ("deposit_cost_annualized", 5),
    "deposit_beta": ("deposit_beta_4q", 5),
    "prior_deposit_growth": ("deposit_growth_qoq_lagged", 5),
    "deposit_growth_volatility": ("deposit_growth_rolling_vol_4q", 5),
    "deposit_acceleration": ("deposit_growth_acceleration", 5),
    "peer_relative_deterioration": ("deposit_growth_qoq_lagged", 5),
    "hmm_state_probability": ("regime", 4),
    "rate_hike": ("macro_DFF_change_4q_revised", 2),
    "curve_shape": ("macro_curve_10y2y_revised", 2),
    "vix_level": ("macro_VIXCLS_revised", 2),
    "system_deposit_growth": ("macro_DPSACBW027NBOG_qoq_revised", 2),
    "missing_feature_count": ("missing_count", 5),
    "ma_distance": ("mna_contaminated_radius_1", 2),
    "predicted_q05": ("q05", 10),
    "predictive_width": ("pred_width", 10),
    "challenger_disagreement_decile": ("disagreement", 10),
    "ood_score_decile": ("ood_score", 10),
}

INTERSECTIONS = [
    ("size_band", "uninsured_proxy"), ("cash_assets", "uninsured_proxy"),
    ("size_band", "cash_assets"), ("brokered_share", "cash_assets"),
    ("deposit_beta", "uninsured_proxy"), ("capital_assets", "borrowings_assets"),
    ("peer_relative_deterioration", "cash_assets"), ("hmm_state_probability", "uninsured_proxy"),
    ("hmm_state_probability", "cash_assets"), ("hmm_state_probability", "challenger_disagreement_decile"),
]


def checkpoint(stage: str, status: str = "complete", details: dict | None = None) -> None:
    path = ROOT / ".treasuryshield" / "v1_1_run_state.json"
    path.parent.mkdir(exist_ok=True)
    current = json.loads(path.read_text()) if path.exists() else {"schema": "1.1", "stages": {}}
    current["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    current["status"] = "RUNNING"
    current["stages"][stage] = {"status": status, "at": current["updated_at_utc"], **(details or {})}
    write_json(path, current)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    selected_final = enrich(read_prediction("B8_lightgbm_quantile"))
    selected_pre = enrich(read_prelock("B8_lightgbm_quantile"))
    finals = {m: read_prediction(m) for m in KEY_MODELS if model_path(m).exists()}
    pres = {m: read_prelock(m) for m in KEY_MODELS}
    finals = {m: f for m, f in finals.items() if not f.empty}
    pres = {m: f for m, f in pres.items() if not f.empty}
    return selected_final, selected_pre, finals, pres


def align_signals(base: pd.DataFrame, models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = ["bank_id", "report_period"]
    out = base.copy()
    q05s = []
    for name, frame in models.items():
        part = frame[keys + ["q05"]].rename(columns={"q05": f"q05_{name}"})
        out = out.merge(part, on=keys, how="left", validate="one_to_one")
        q05s.append(f"q05_{name}")
    out["disagreement"] = out[q05s].std(axis=1)
    out["pred_width"] = out.q90 - out.q01
    numeric = [v[0] for v in SLICE_FEATURES.values() if v[0] in out.columns]
    out["missing_count"] = out[numeric].isna().sum(axis=1)
    return out


def fit_ood(pre: pd.DataFrame, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features = [
        "assets", "cash_asset_ratio", "liquid_broad_asset_ratio", "loan_deposit_ratio",
        "uninsured_proxy_domestic_share", "brokered_domestic_ratio", "borrowings_asset_ratio",
        "capital_asset_ratio", "npl_loan_ratio", "profit_asset_ratio", "deposit_beta_4q",
        "deposit_growth_qoq_lagged", "deposit_growth_rolling_vol_4q", "macro_VIXCLS_revised",
    ]
    features = [c for c in features if c in pre]
    med = pre[features].median()
    scaler = RobustScaler().fit(pre[features].fillna(med))
    xp = scaler.transform(pre[features].fillna(med))
    xf = scaler.transform(final[features].fillna(med))
    sample = rng.choice(len(xp), min(100_000, len(xp)), replace=False)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED, n_jobs=-1).fit(xp[sample])
    pre = pre.copy(); final = final.copy()
    pre["ood_score"] = -iso.score_samples(xp)
    final["ood_score"] = -iso.score_samples(xf)
    # Source classifier is a drift diagnostic using covariates only; OOD scores above remain prelock fitted.
    ps = xp[rng.choice(len(xp), min(60_000, len(xp)), replace=False)]
    fs = xf[rng.choice(len(xf), min(30_000, len(xf)), replace=False)]
    xx = np.vstack([ps, fs]); yy = np.r_[np.zeros(len(ps)), np.ones(len(fs))]
    xtr, xte, ytr, yte = train_test_split(xx, yy, test_size=.35, random_state=SEED, stratify=yy)
    clf = LogisticRegression(max_iter=500).fit(xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(xte)[:, 1])
    metadata = {"features": features, "fit_population": "pre-2024 only", "prelock_rows": len(pre),
                "source_classifier_auc": float(auc), "source_classifier_note": "Covariate drift diagnostic; uses no outcomes."}
    return pre, final, metadata


def freeze_cutoffs(pre: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cutoffs: dict[str, dict] = {}
    rows = []
    for name, (feature, bins) in SLICE_FEATURES.items():
        if feature == "regime":
            cutoffs[name] = {"feature": feature, "kind": "category", "levels": sorted(pre.regime.dropna().astype(str).unique())}
        elif feature in pre:
            edges = stable_edges(pre[feature], bins)
            cutoffs[name] = {"feature": feature, "kind": "prelock_quantile", "bins": bins, "edges": edges}
        else:
            cutoffs[name] = {"feature": feature, "kind": "unavailable", "edges": []}
        rows.append({
            "slice_name": name, "definition": json.dumps(cutoffs[name], default=str),
            "source_variables": feature, "cutoff_source": "pre-2024 prediction/feature evidence",
            "cutoff_fixed_prelock": True, "minimum_report_n": 500, "material_validation_n": 1000,
        })
    categorical = {
        "report_form": "entity_type", "enhanced_data_availability": "panel_mode",
        "direct_vs_proxy_uninsured": "uninsured_proxy_domestic_share", "pseudo_pit_vintage_quality": "source_vintage",
        "state": "state", "census_region": "state",
    }
    for name, feature in categorical.items():
        rows.append({"slice_name": name, "definition": f"Observed {feature} categories; proxy status explicitly labeled where applicable",
                     "source_variables": feature, "cutoff_source": "source metadata", "cutoff_fixed_prelock": True,
                     "minimum_report_n": 500, "material_validation_n": 1000})
    write_json(ART / "slice_cutoffs_prelock.json", cutoffs)
    registry = pd.DataFrame(rows)
    registry.to_csv(ART / "slice_registry.csv", index=False)
    return cutoffs, registry


def apply_slices(frame: pd.DataFrame, cutoffs: dict) -> pd.DataFrame:
    out = frame.copy()
    for name, rule in cutoffs.items():
        feature = rule["feature"]
        if rule["kind"] == "category":
            out[name] = out[feature].astype("string").fillna("missing")
        elif feature in out:
            out[name] = apply_edges(out[feature], rule.get("edges", []))
        else:
            out[name] = "unavailable"
    out["report_form"] = out.get("entity_type", pd.Series("unknown", index=out.index)).astype("string").fillna("missing")
    out["enhanced_data_availability"] = out.get("panel_mode", pd.Series("unknown", index=out.index)).astype("string").fillna("missing")
    out["direct_vs_proxy_uninsured"] = np.where(out.get("uninsured_proxy_domestic_share", pd.Series(np.nan, index=out.index)).notna(), "proxy_available", "missing")
    out["pseudo_pit_vintage_quality"] = out.get("source_vintage", pd.Series("unknown", index=out.index)).astype("string").fillna("missing")
    out["state"] = out.get("state", pd.Series("unknown", index=out.index)).astype("string").fillna("missing")
    region = {"CT":"Northeast","ME":"Northeast","MA":"Northeast","NH":"Northeast","RI":"Northeast","VT":"Northeast","NJ":"Northeast","NY":"Northeast","PA":"Northeast",
              "IL":"Midwest","IN":"Midwest","MI":"Midwest","OH":"Midwest","WI":"Midwest","IA":"Midwest","KS":"Midwest","MN":"Midwest","MO":"Midwest","NE":"Midwest","ND":"Midwest","SD":"Midwest",
              "DE":"South","FL":"South","GA":"South","MD":"South","NC":"South","SC":"South","VA":"South","DC":"South","WV":"South","AL":"South","KY":"South","MS":"South","TN":"South","AR":"South","LA":"South","OK":"South","TX":"South",
              "AZ":"West","CO":"West","ID":"West","MT":"West","NV":"West","NM":"West","UT":"West","WY":"West","AK":"West","CA":"West","HI":"West","OR":"West","WA":"West"}
    out["census_region"] = out["state"].map(region).fillna("Other/unknown")
    return out


def run_calibration(finals: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    tests = {"method": "Kupiec unconditional coverage and Christoffersen independence/conditional coverage; ordered by forecast origin and bank.", "models": {}}
    for model, frame in finals.items():
        rows.extend(calibration_rows(frame, model))
        for quarter, sub in frame.groupby("report_period"):
            rows.extend(calibration_rows(sub, model, "quarter", quarter))
        tests["models"][model] = {}
        ordered = frame.sort_values(["forecast_origin", "bank_id"])
        for q, alpha in zip(("q01", "q05", "q10"), (.01, .05, .10)):
            tests["models"][model][q] = exception_tests((ordered.actual < ordered[q]).to_numpy(), alpha)
    calibration = pd.DataFrame(rows)
    calibration.to_parquet(ART / "calibration_diagnostics.parquet", index=False)
    write_json(ART / "calibration_tests.json", tests)
    selected = finals["B8_lightgbm_quantile"]
    pit_rows = [pit_summary(selected, "overall", "all")]
    pit_rows += [pit_summary(sub, "quarter", q) for q, sub in selected.groupby("report_period")]
    for bank, sub in selected.groupby("bank_id"):
        if len(sub) >= 6:
            pit_rows.append(pit_summary(sub.sort_values("forecast_origin"), "bank", str(bank)))
    pit = pd.DataFrame(pit_rows)
    pit.to_parquet(ART / "pit_diagnostics.parquet", index=False)
    return calibration, tests, pit


def run_sequential(pre: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    # Windows are selected using only pre-2024 replay loss, then frozen.
    candidates = (4, 8, 12)
    window_scores = {}
    pre_ord = pd.PeriodIndex(pre.report_period, freq="Q").astype(int)
    for window in candidates:
        losses = []
        for quarter in sorted(pre.report_period.unique())[-12:]:
            current = pre.report_period.eq(quarter)
            qo = pd.Period(quarter, freq="Q").ordinal
            hist = pre[(pre_ord < qo) & (pre_ord >= qo - window)]
            if len(hist) < 500:
                continue
            adj = np.quantile(hist.actual - hist.q05, .05)
            losses.extend(pinball(pre.loc[current, "actual"].to_numpy(), pre.loc[current, "q05"].to_numpy() + adj, .05))
        window_scores[str(window)] = float(np.mean(losses)) if losses else math.inf
    selected_window = min(candidates, key=lambda x: window_scores[str(x)])
    write_json(ART / "sequential_policy_freeze.json", {"selected_window_quarters": selected_window,
               "candidate_prelock_q05_pinball": window_scores, "selected_using": "pre-2024 only", "policies": [f"R{i}" for i in range(7)]})
    history = pre.copy()
    output = []
    policies = {"R0":"none", "R1":"rolling_global", "R2":"ew_global", "R3":"peer", "R4":"regime", "R5":"shrink_peer", "R6":"shrink_regime"}
    for origin in sorted(final.forecast_origin.unique()):
        now = final.loc[final.forecast_origin.eq(origin)].copy()
        matured_final = final.loc[final.target_availability_date.lt(origin)]
        hist = pd.concat([history, matured_final], ignore_index=True)
        hist = hist.loc[hist.target_availability_date.lt(origin)].copy()
        if hist.empty:
            raise RuntimeError("Sequential replay has no matured history")
        h_ord = pd.PeriodIndex(hist.report_period, freq="Q").astype(int)
        max_ord = h_ord.max()
        hist = hist.loc[h_ord >= max_ord - selected_window + 1]
        for policy, label in policies.items():
            for qcol, alpha in zip(QCOLS, QLEVELS):
                base = now[qcol].to_numpy(float)
                if policy == "R0":
                    adjusted = base.copy(); updates = 0
                else:
                    residual = hist.actual.to_numpy() - hist[qcol].to_numpy()
                    if policy == "R2":
                        age = max_ord - pd.PeriodIndex(hist.report_period, freq="Q").astype(int)
                        global_adj = weighted_quantile(residual, alpha, np.power(.8, age))
                    else:
                        global_adj = float(np.quantile(residual, alpha))
                    updates = 1
                    group_col = "peer_group" if "peer" in label else "regime"
                    if policy in {"R3", "R4", "R5", "R6"}:
                        local_adjustments={};local_counts={}
                        for group,local in hist.groupby(group_col,dropna=False):
                            key=str(group);local_counts[key]=len(local)
                            local_adjustments[key]=float(np.quantile(local.actual-local[qcol],alpha)) if len(local)>=100 else global_adj
                        keys=now[group_col].astype(str)
                        local=np.array([local_adjustments.get(key,global_adj) for key in keys])
                        if policy in {"R5","R6"}:
                            shrink=np.array([min(1.0,local_counts.get(key,0)/500.0) for key in keys])
                            adjustment=shrink*local+(1-shrink)*global_adj
                        else:
                            adjustment=local
                        adjusted=base+adjustment
                    else:
                        adjusted=base+global_adj
                y = now.actual.to_numpy(float)
                exc = y < adjusted
                severity = np.maximum(adjusted - y, 0)
                output.append({"forecast_origin": str(pd.Timestamp(origin).date()), "report_period": str(now.report_period.iloc[0]),
                    "policy": policy, "policy_label": label, "quantile": qcol, "nominal": alpha, "window_quarters": selected_window,
                    "n": len(now), "matured_rows": len(hist), "updates": updates, "pinball": float(pinball(y, adjusted, alpha).mean()),
                    "coverage": float(exc.mean()), "coverage_error": float(exc.mean()-alpha), "tail_severity": float(severity.mean()),
                    "baseline_pinball": float(pinball(y, base, alpha).mean()), "baseline_coverage": float((y < base).mean()),
                    "mean_adjustment": float(np.mean(adjusted-base)), "adjustment_std": float(np.std(adjusted-base)),
                    "operational_complexity": {"R0":0,"R1":1,"R2":2,"R3":3,"R4":3,"R5":4,"R6":4}[policy],
                    "max_used_target_availability": str(hist.target_availability_date.max().date()), "strictly_prior": bool(hist.target_availability_date.max() < origin),
                    "interpretation": "Exploratory operational monitoring replay; original locked predictions and result remain frozen."})
    result = pd.DataFrame(output)
    result.to_parquet(ART / "sequential_recalibration.parquet", index=False)
    return result


def metric_record(sub: pd.DataFrame, slice_name: str, slice_value: str, weighting: str, challenger: pd.Series | None = None) -> dict:
    y = sub.actual.to_numpy(float); q05 = sub.q05.to_numpy(float)
    weights = None if weighting == "equal_bank" else sub["assets" if weighting == "asset_weighted" else "deposits"].to_numpy(float)
    loss = pinball(y, q05, .05)
    if challenger is not None:
        challenger_loss = pinball(y, challenger.loc[sub.index].to_numpy(float), .05)
        delta = loss - challenger_loss
        mean_delta = weighted_mean(delta, weights)
        se = float(np.std(delta, ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else math.nan
    else:
        mean_delta = se = math.nan
    exceptions = y < q05
    return {
        "slice_name": slice_name, "slice_value": str(slice_value), "weighting": weighting, "n": len(sub),
        "assets_exposure": float(sub.assets.sum()), "deposits_exposure": float(sub.deposits.sum()),
        "q01_pinball": weighted_mean(pinball(y, sub.q01.to_numpy(), .01), weights),
        "q05_pinball": weighted_mean(loss, weights), "q10_pinball": weighted_mean(pinball(y, sub.q10.to_numpy(), .10), weights),
        "q05_coverage": weighted_mean(exceptions.astype(float), weights),
        "q05_coverage_error": weighted_mean(exceptions.astype(float), weights) - .05,
        "tail_severity": weighted_mean(np.maximum(q05-y, 0), weights),
        "mae": weighted_mean(np.abs(y-sub.q50.to_numpy()), weights), "median_error": float(np.median(y-sub.q50.to_numpy())),
        "challenger_delta_q05_pinball": mean_delta, "challenger_delta_ci_low": mean_delta-1.96*se,
        "challenger_delta_ci_high": mean_delta+1.96*se, "unstable_small_group": len(sub) < 500,
        "material_sample": len(sub) >= 1000,
    }


def run_slices(final: pd.DataFrame, registry: pd.DataFrame, challenger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = challenger.set_index(["bank_id", "report_period"])
    idx = pd.MultiIndex.from_frame(final[["bank_id", "report_period"]])
    challenger_q05 = pd.Series(aligned.loc[idx, "q05"].to_numpy(), index=final.index)
    rows = []
    for slice_name in registry.slice_name:
        if slice_name not in final:
            continue
        for value, sub in final.groupby(slice_name, dropna=False):
            for weighting in ("equal_bank", "asset_weighted", "deposit_weighted"):
                rows.append(metric_record(sub, slice_name, value, weighting, challenger_q05))
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(ART / "slice_metrics.parquet", index=False)
    intersections = []
    for first, second in INTERSECTIONS:
        if first not in final or second not in final:
            continue
        for values, sub in final.groupby([first, second], dropna=False):
            if len(sub) < 1000:
                continue
            record = metric_record(sub, f"{first}_x_{second}", f"{values[0]} | {values[1]}", "equal_bank", challenger_q05)
            z = record["challenger_delta_q05_pinball"] / max((record["challenger_delta_ci_high"]-record["challenger_delta_q05_pinball"])/1.96, 1e-12)
            record["p_value"] = float(2*stats.norm.sf(abs(z)))
            intersections.append(record)
    inter = pd.DataFrame(intersections)
    if not inter.empty:
        order = np.argsort(inter.p_value.to_numpy()); m = len(inter); adjusted = np.empty(m)
        running = 0.0
        for rank, pos in enumerate(order):
            running = max(running, min(1.0, inter.p_value.iloc[pos] * (m-rank)))
            adjusted[pos] = running
        inter["holm_adjusted_p"] = adjusted
    inter.to_parquet(ART / "intersection_metrics.parquet", index=False)
    return metrics, inter


def run_error_tree(pre: pd.DataFrame, final: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    features = [c for c in ["assets","cash_asset_ratio","liquid_broad_asset_ratio","loan_deposit_ratio",
        "uninsured_proxy_domestic_share","brokered_domestic_ratio","borrowings_asset_ratio","capital_asset_ratio",
        "npl_loan_ratio","profit_asset_ratio","deposit_beta_4q","deposit_growth_qoq_lagged","deposit_growth_rolling_vol_4q",
        "macro_VIXCLS_revised"] if c in pre]
    med = pre[features].median(); x = pre[features].fillna(med)
    target = (pre.actual < pre.q05 - .02).astype(int)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1000, class_weight="balanced", random_state=SEED).fit(x, target)
    rules = export_text(tree, feature_names=features)
    leaf_pre = tree.apply(x); leaf_final = tree.apply(final[features].fillna(med))
    evaluated = []
    for leaf in np.unique(leaf_final):
        mask = leaf_final == leaf; sub = final.loc[mask]
        evaluated.append({"leaf": int(leaf), "n": int(mask.sum()), "q05_pinball": float(pinball(sub.actual.to_numpy(), sub.q05.to_numpy(), .05).mean()),
                          "severe_miss_rate": float((sub.actual < sub.q05-.02).mean()), "coverage": float((sub.actual < sub.q05).mean())})
    payload = {"fit_population":"pre-2024 only", "target":"severe Q05 tail miss (actual < q05 - 0.02)",
               "features":features, "max_depth":3, "minimum_leaf":1000, "rules":rules,
               "prelock_leaf_counts":pd.Series(leaf_pre).value_counts().sort_index().to_dict(), "final_evaluation":evaluated}
    write_json(ART / "error_tree.json", payload)
    return payload, pd.DataFrame(evaluated)


def run_taxonomy(final: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    residual = final.actual-final.q05
    mild = .02
    conditions = [residual < -.05, residual < 0, residual > .10, residual > .05]
    labels = ["severe_underestimation","mild_underestimation","severe_over_conservatism","mild_over_conservatism"]
    out = final[["bank_id","report_period","forecast_origin","assets","deposits","actual","q01","q05","q50","q90","disagreement","ood_score"]].copy()
    out["error_state"] = np.select(conditions, labels, default="correct_or_conservative_tail")
    out["quantile_shape_anomaly"] = (np.diff(final[QCOLS].to_numpy(), axis=1) < 0).any(axis=1)
    out["challenger_disagreement_high"] = final.disagreement >= pre.disagreement.quantile(.9)
    out["ood_high"] = final.ood_score >= pre.ood_score.quantile(.9)
    out["q05_residual"] = residual
    out["exposure_weight_assets"] = out.assets / out.assets.sum()
    out["exposure_weight_deposits"] = out.deposits / out.deposits.sum()
    out["prior_error_state"] = out.sort_values(["bank_id","forecast_origin"]).groupby("bank_id").error_state.shift().reindex(out.index)
    out.to_parquet(ART / "error_taxonomy.parquet", index=False)
    return out


def psi(reference: pd.Series, target: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = pd.cut(reference, edges).value_counts(normalize=True, sort=False).to_numpy()+1e-8
    b = pd.cut(target, edges).value_counts(normalize=True, sort=False).reindex(pd.cut(reference, edges).cat.categories, fill_value=0).to_numpy()+1e-8
    return float(np.sum((b-a)*np.log(b/a)))


def run_drift(pre: pd.DataFrame, final: pd.DataFrame, ood_meta: dict) -> pd.DataFrame:
    features = ood_meta["features"]
    rows=[]
    groups=[("final_all",final), *[(str(q),s) for q,s in final.groupby("report_period")]]
    for group, target in groups:
        for feature in features:
            a=pre[feature].dropna(); b=target[feature].dropna()
            valid=len(a)>0 and len(b)>0
            ks=stats.ks_2samp(a,b) if valid else None
            rows.append({"scope":group,"feature":feature,"n_reference":len(a),"n_target":len(b),"psi":psi(a,b) if valid else math.nan,
                         "ks":float(ks.statistic) if valid else math.nan,"ks_p":float(ks.pvalue) if valid else math.nan,
                         "wasserstein":float(stats.wasserstein_distance(a,b)) if valid else math.nan,"reference_missing":float(pre[feature].isna().mean()),
                         "target_missing":float(target[feature].isna().mean()),"missingness_shift":float(target[feature].isna().mean()-pre[feature].isna().mean()),
                         "source_classifier_auc":ood_meta["source_classifier_auc"] if group=="final_all" else math.nan})
    for decile, sub in final.groupby(pd.qcut(final.ood_score,10,duplicates="drop")):
        rows.append({"scope":"ood_decile","feature":str(decile),"n_reference":len(pre),"n_target":len(sub),"psi":math.nan,"ks":math.nan,"ks_p":math.nan,"wasserstein":math.nan,
                     "reference_missing":math.nan,"target_missing":math.nan,"missingness_shift":math.nan,"source_classifier_auc":math.nan,
                     "q05_pinball":float(pinball(sub.actual.to_numpy(),sub.q05.to_numpy(),.05).mean()),"q05_coverage":float((sub.actual<sub.q05).mean()),
                     "mean_ood":float(sub.ood_score.mean()),"drift_type":"covariate/OOD vs loss"})
    out=pd.DataFrame(rows); out.to_parquet(ART/"drift_ood.parquet",index=False); return out


def risk_score(frame: pd.DataFrame, pre: pd.DataFrame) -> np.ndarray:
    cols=["disagreement","pred_width","missing_count","ood_score"]
    scores=[]
    for col in cols:
        ref=pre[col].dropna().sort_values().to_numpy(); values=frame[col].fillna(pre[col].median()).to_numpy()
        scores.append(np.searchsorted(ref,values,side="right")/max(1,len(ref)))
    return np.mean(scores,axis=0)


def run_selective(pre: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    pre=pre.copy(); final=final.copy(); pre["review_score"]=risk_score(pre,pre); final["review_score"]=risk_score(final,pre)
    thresholds={str(p):float(pre.review_score.quantile(1-p)) for p in (.01,.05,.10,.20)}
    write_json(ART/"selective_policy_freeze.json",{"fit_population":"pre-2024 only","signals":["model disagreement","predictive width","missingness","OOD"],"combination":"equal-weight empirical percentile mean","thresholds":thresholds})
    rows=[]; severe=final.actual<final.q05-.02; total_loss=pinball(final.actual.to_numpy(),final.q05.to_numpy(),.05)
    for p in (.01,.05,.10,.20):
        reviewed=final.review_score>=thresholds[str(p)]; remain=~reviewed
        rows.append({"review_budget":p,"frozen_score_threshold":thresholds[str(p)],"reviewed_n":int(reviewed.sum()),"reviewed_share":float(reviewed.mean()),
            "residual_q05_pinball":float(total_loss[remain].mean()),"baseline_q05_pinball":float(total_loss.mean()),
            "tail_miss_capture":float((reviewed&severe).sum()/max(1,severe.sum())),"asset_exposure_capture":float(final.loc[reviewed,"assets"].sum()/final.assets.sum()),
            "deposit_exposure_capture":float(final.loc[reviewed,"deposits"].sum()/final.deposits.sum()),
            "review_efficiency":float((reviewed&severe).sum()/max(1,reviewed.sum()))})
    out=pd.DataFrame(rows);out.to_parquet(ART/"selective_review.parquet",index=False);return out


def run_ranking(pre: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    event_cut=float(pre.actual.quantile(.05)); write_json(ART/"ranking_policy_freeze.json",{"severe_outflow_threshold":event_cut,"source":"pre-2024 actual Q05","ranking_score":"negative predicted Q05"})
    event=final.actual<=event_cut; order=np.argsort(final.q05.to_numpy()); rows=[]
    for p in (.01,.05,.10,.20,.50,1.0):
        chosen=np.zeros(len(final),bool);chosen[order[:max(1,int(len(final)*p))]]=True
        rows.append({"review_budget":p,"n":len(final),"event_threshold":event_cut,"event_prevalence":float(event.mean()),
            "spearman_prediction_actual":float(stats.spearmanr(final.q05,final.actual).statistic),"event_capture":float((chosen&event).sum()/max(1,event.sum())),
            "precision":float((chosen&event).sum()/max(1,chosen.sum())),"lift":float(((chosen&event).sum()/max(1,chosen.sum()))/max(event.mean(),1e-12)),
            "asset_weighted_capture":float(final.loc[chosen&event,"assets"].sum()/max(1,final.loc[event,"assets"].sum())),
            "deposit_weighted_capture":float(final.loc[chosen&event,"deposits"].sum()/max(1,final.loc[event,"deposits"].sum()))})
    out=pd.DataFrame(rows);out.to_parquet(ART/"ranking_analysis.parquet",index=False);return out


def run_challenger_slices(final: pd.DataFrame, finals: dict[str,pd.DataFrame], registry: pd.DataFrame) -> pd.DataFrame:
    keys=["bank_id","report_period"]; base=final.set_index(keys); rows=[]
    major=[s for s in registry.slice_name if s in final][:24]
    for model, pred in finals.items():
        if model=="B8_lightgbm_quantile": continue
        other=pred.set_index(keys).loc[base.index]
        if not np.allclose(other.actual,base.actual, equal_nan=True): raise RuntimeError(f"Challenger row alignment failed: {model}")
        delta=pd.Series(pinball(base.actual.to_numpy(),base.q05.to_numpy(),.05)-pinball(base.actual.to_numpy(),other.q05.to_numpy(),.05),index=base.index)
        coverage_selected=(base.actual<base.q05).astype(float);coverage_other=(base.actual<other.q05).astype(float)
        severity_selected=np.maximum(base.q05-base.actual,0);severity_other=np.maximum(other.q05-base.actual,0)
        for slice_name in major:
            values=base[slice_name]
            for value, idx in values.groupby(values).groups.items():
                if len(idx)<500: continue
                d=delta.loc[idx].to_numpy(); mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d))) if len(d)>1 else math.nan
                rows.append({"slice_name":slice_name,"slice_value":str(value),"challenger":model,"n":len(idx),
                    "delta_q05_pinball_selected_minus_challenger":mean,"paired_ci_low":mean-1.96*se,"paired_ci_high":mean+1.96*se,
                    "delta_coverage_error":float(abs(coverage_selected.loc[idx].mean()-.05)-abs(coverage_other.loc[idx].mean()-.05)),
                    "delta_tail_severity":float(np.mean(severity_selected.loc[idx]-severity_other.loc[idx])),
                    "classification":"selected_wins" if mean<0 and mean+1.96*se<0 else "challenger_wins" if mean>0 and mean-1.96*se>0 else "indistinguishable"})
    out=pd.DataFrame(rows);out.to_parquet(ART/"challenger_slice_deltas.parquet",index=False);return out


def run_classifier_calibration() -> pd.DataFrame:
    raw=pd.read_parquet(ROOT/"artifacts"/"stress_event_classification.parquet")
    pre=raw[raw.split.isin(PRELOCK_SPLITS)].copy(); final=raw[raw.split.eq("final_locked")].copy()
    rows=[]; reliability=[]
    for definition in sorted(raw.definition.unique()):
        tr=pre[pre.definition.eq(definition)].dropna(subset=["event","probability"])
        te=final[final.definition.eq(definition)].dropna(subset=["event","probability"])
        if te.empty: continue
        ptrain=np.clip(tr.probability.to_numpy(),1e-6,1-1e-6);ptest=np.clip(te.probability.to_numpy(),1e-6,1-1e-6)
        ytr=tr.event.to_numpy(int);y=te.event.to_numpy(int)
        logit=lambda p:np.log(p/(1-p))
        platt=LogisticRegression().fit(logit(ptrain).reshape(-1,1),ytr)
        iso=IsotonicRegression(out_of_bounds="clip").fit(ptrain,ytr)
        variants={"raw":ptest,"platt":platt.predict_proba(logit(ptest).reshape(-1,1))[:,1],"isotonic":iso.predict(ptest)}
        for variant,p in variants.items():
            p=np.clip(p,1e-6,1-1e-6)
            cal=LogisticRegression().fit(logit(p).reshape(-1,1),y)
            bins=pd.qcut(p,10,duplicates="drop"); grouped=pd.DataFrame({"p":p,"y":y,"bin":bins}).groupby("bin",observed=True).agg(predicted=("p","mean"),observed=("y","mean"),n=("y","size")).reset_index(drop=True)
            ece=float(np.sum(grouped.n/len(y)*abs(grouped.predicted-grouped.observed)));mce=float(abs(grouped.predicted-grouped.observed).max())
            prevalence=float(y.mean());uncertainty=prevalence*(1-prevalence);resolution=float(np.sum(grouped.n/len(y)*(grouped.observed-prevalence)**2));reliability=float(np.sum(grouped.n/len(y)*(grouped.predicted-grouped.observed)**2))
            record={"definition":definition,"variant":variant,"n":len(y),"events":int(y.sum()),"brier":brier_score_loss(y,p),"brier_reliability":reliability,
                "brier_resolution":resolution,"brier_uncertainty":uncertainty,"log_loss":log_loss(y,p),"ece":ece,"maximum_calibration_error":mce,
                "pr_auc":average_precision_score(y,p),"roc_auc":roc_auc_score(y,p),"calibration_intercept":float(cal.intercept_[0]),"calibration_slope":float(cal.coef_[0,0])}
            for threshold in (.05,.10,.20):
                alert=p>=threshold;record[f"precision_at_{threshold}"]=float(y[alert].mean()) if alert.any() else math.nan;record[f"recall_at_{threshold}"]=float(y[alert].sum()/max(1,y.sum()))
            order=np.argsort(-p)
            for frac in (.01,.05,.10): record[f"top_{int(frac*100)}pct_capture"]=float(y[order[:max(1,int(len(y)*frac))]].sum()/max(1,y.sum()))
            rows.append(record)
            for i,g in grouped.iterrows(): reliability.append if False else None
    out=pd.DataFrame(rows);out.to_parquet(ART/"stress_event_probability_calibration.parquet",index=False);return out


def run_explanations(pre: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    stored=pd.read_parquet(ROOT/"artifacts"/"feature_importance_prelock.parquet")
    stored["source"]="v1 stored prelock importance";stored["scope"]="prelock"
    numeric=[c for c in ["assets","cash_asset_ratio","liquid_broad_asset_ratio","loan_deposit_ratio","uninsured_proxy_domestic_share","brokered_domestic_ratio","borrowings_asset_ratio","capital_asset_ratio","npl_loan_ratio","profit_asset_ratio","deposit_beta_4q","deposit_growth_qoq_lagged","macro_VIXCLS_revised"] if c in pre]
    med=pre[numeric].median(); sample=pre.sample(min(100_000,len(pre)),random_state=SEED)
    model=RandomForestRegressor(n_estimators=120,min_samples_leaf=100,n_jobs=-1,random_state=SEED).fit(sample[numeric].fillna(med),sample.q05)
    rows=[]
    for quarter,sub in final.groupby("report_period"):
        ss=sub.sample(min(5000,len(sub)),random_state=SEED)
        pi=permutation_importance(model,ss[numeric].fillna(med),ss.q05,n_repeats=3,random_state=SEED,n_jobs=-1)
        for feature,value in zip(numeric,pi.importances_mean): rows.append({"scope":"quarter","group":quarter,"feature":feature,"importance":float(value),"method":"permutation surrogate fidelity","surrogate_r2":float(model.score(ss[numeric].fillna(med),ss.q05))})
    # Local reason codes use the same prelock-fitted explanation surrogate and are descriptive, not causal.
    exemplars=final.nlargest(10,"ood_score")
    base=float(model.predict(pd.DataFrame([med],columns=numeric))[0])
    for _,row in exemplars.iterrows():
        contributions=[]
        for feature in numeric:
            altered=pd.DataFrame([row[numeric].fillna(med)]);altered[feature]=med[feature]
            contributions.append((feature,float(model.predict(pd.DataFrame([row[numeric].fillna(med)]))[0]-model.predict(altered)[0])))
        for rank,(feature,value) in enumerate(sorted(contributions,key=lambda x:abs(x[1]),reverse=True)[:5],1): rows.append({"scope":"local","group":f"{row.bank_id}_{row.report_period}","feature":feature,"importance":value,"rank":rank,"method":"prelock surrogate leave-one-at-median reason code","surrogate_r2":math.nan})
    out=pd.DataFrame(rows)
    # Preserve v1 gain rows in a normalized schema when columns permit.
    feature_col=next((c for c in stored.columns if "feature" in c),None);value_col=next((c for c in stored.columns if c in {"importance","gain","value"}),None)
    if feature_col and value_col:
        add=pd.DataFrame({"scope":"prelock","group":"all","feature":stored[feature_col].astype(str),"importance":stored[value_col].astype(float),"method":"LightGBM gain (stored v1)","surrogate_r2":math.nan});out=pd.concat([add,out],ignore_index=True)
    out.to_parquet(ART/"explanation_stability.parquet",index=False);return out


def run_stress_attribution(finals: dict[str,pd.DataFrame]) -> pd.DataFrame:
    # Identical sampled profiles and common random draws isolate distribution/model effects.
    models=[m for m in ["B8_lightgbm_quantile","B5_linear_quantile","B7_xgb_quantile","B10_gru","RAHQ_C_nn"] if m in finals]
    common=aligned_predictions(models);sample=common.sample(min(12000,len(common)),random_state=SEED).reset_index(drop=True)
    draws=rng.standard_normal((len(sample),128));u=stats.norm.cdf(draws)
    assumptions={"deposit_runoff":1.0,"funding_access":.60,"securities_haircut":.08,"commitment_draw":.10,"capital_loss_proxy":.04,"recovery_capacity":.70,"recovery_cost":.12}
    rows=[]
    def evaluate(model:str,pars:dict)->tuple[float,float,float,float]:
        qs=sample[[f"{q}__{model}" for q in QCOLS]].to_numpy();sim=np.empty_like(u)
        for i in range(len(sample)): sim[i]=np.interp(u[i],QLEVELS,qs[i],left=qs[i,0],right=qs[i,-1])
        outflow=np.maximum(-sim*pars["deposit_runoff"],0);capacity=pars["funding_access"]+pars["recovery_capacity"]-pars["securities_haircut"]-pars["commitment_draw"]-pars["capital_loss_proxy"]
        short=outflow>capacity;prob=float(short.mean());pfn=float(np.mean(np.maximum(outflow-capacity,0)));cost=float(pfn*pars["recovery_cost"]);reverse=float(np.quantile(outflow,.95))
        return prob,pfn,cost,reverse
    base={m:evaluate(m,assumptions) for m in models};champion=base["B8_lightgbm_quantile"]
    for m,values in base.items(): rows.append({"analysis":"model_uncertainty","model":m,"driver":"predictive_distribution","setting":"base","shortfall_probability":values[0],"pfn":values[1],"recovery_cost":values[2],"reverse_stress_threshold":values[3],"delta_shortfall_vs_champion":values[0]-champion[0],"common_random_numbers":True,"profiles":len(sample),"draws_per_profile":128})
    variations={"deposit_runoff":(.8,1.2),"funding_access":(.4,.8),"securities_haircut":(.04,.16),"commitment_draw":(.05,.20),"capital_loss_proxy":(.02,.08),"recovery_capacity":(.5,.9),"recovery_cost":(.08,.20)}
    for driver,(low,high) in variations.items():
        for label,value in (("low",low),("high",high)):
            pars=assumptions.copy();pars[driver]=value;metrics=evaluate("B8_lightgbm_quantile",pars)
            rows.append({"analysis":"scenario_assumption_uncertainty","model":"B8_lightgbm_quantile","driver":driver,"setting":label,"shortfall_probability":metrics[0],"pfn":metrics[1],"recovery_cost":metrics[2],"reverse_stress_threshold":metrics[3],"delta_shortfall_vs_champion":metrics[0]-champion[0],"common_random_numbers":True,"profiles":len(sample),"draws_per_profile":128})
    out=pd.DataFrame(rows);out.to_parquet(ART/"stress_attribution.parquet",index=False);write_json(ART/"stress_common_random_numbers.json",{"seed":SEED,"profiles_sha256":str(pd.util.hash_pandas_object(sample[["bank_id","report_period"]],index=False).sum()),"draw_matrix_shape":list(draws.shape),"models":models});return out


def run_monitoring(pre: pd.DataFrame, final: pd.DataFrame, finals: dict[str,pd.DataFrame], classifier: pd.DataFrame) -> tuple[dict,pd.DataFrame]:
    history=[]
    for quarter,sub in pre.groupby("report_period"):
        coverage=float((sub.actual<sub.q05).mean())
        history.append({"quarter":quarter,"q05_coverage":coverage,"q05_coverage_deviation":abs(coverage-.05),"q05_pinball":float(pinball(sub.actual.to_numpy(),sub.q05.to_numpy(),.05).mean()),"tail_severity":float(np.maximum(sub.q05-sub.actual,0).mean()),"disagreement":float(sub.disagreement.mean()),"ood":float(sub.ood_score.mean()),"missingness":float(sub.missing_count.mean())})
    hist=pd.DataFrame(history)
    thresholds={}
    for metric in [c for c in hist if c!="quarter"]:
        values=hist[metric].dropna();thresholds[metric]={"green_upper":float(values.quantile(.75)),"watch_upper":float(values.quantile(.90)),"review_upper":float(values.quantile(.95)),"recalibrate_upper":float(values.quantile(.99)),"source":"pre-2024 quarterly distribution"}
    payload={"fit_population":"pre-2024 only","levels":["GREEN","WATCH","REVIEW","RECALIBRATE","REDEVELOP"],"thresholds":thresholds,"rules":"Escalate by maximum metric level; REDEVELOP requires persistent RECALIBRATE or structural failure.","limitations":"Historical thresholds are research controls and require governance approval before production use."}
    write_json(ART/"monitoring_thresholds.json",payload)
    rows=[]
    for quarter,sub in final.groupby("report_period"):
        coverage=float((sub.actual<sub.q05).mean())
        vals={"q05_coverage":coverage,"q05_coverage_deviation":abs(coverage-.05),"q05_pinball":float(pinball(sub.actual.to_numpy(),sub.q05.to_numpy(),.05).mean()),"tail_severity":float(np.maximum(sub.q05-sub.actual,0).mean()),"disagreement":float(sub.disagreement.mean()),"ood":float(sub.ood_score.mean()),"missingness":float(sub.missing_count.mean())}
        status="GREEN";levels=["GREEN","WATCH","REVIEW","RECALIBRATE"]
        for metric,value in vals.items():
            if metric=="q05_coverage":
                continue
            compare=value
            t=thresholds[metric]
            lev=0 if compare<=t["green_upper"] else 1 if compare<=t["watch_upper"] else 2 if compare<=t["review_upper"] else 3
            status=levels[max(levels.index(status),lev)]
        rows.append({"quarter":quarter,**vals,"status":status})
    out=pd.DataFrame(rows);return payload,out


def savefig(name: str) -> None:
    plt.tight_layout();plt.savefig(FIG/name,dpi=160,bbox_inches="tight");plt.close()


def heatmap(frame: pd.DataFrame, fmt: str = ".2f", annotate: bool = False, center: float | None = None) -> None:
    values=frame.to_numpy(float)
    vmax=np.nanmax(np.abs(values-center)) if center is not None and values.size else None
    plt.imshow(values,aspect="auto",cmap="coolwarm" if center is not None else "viridis",
               vmin=(center-vmax if vmax is not None else None),vmax=(center+vmax if vmax is not None else None))
    plt.colorbar();plt.xticks(range(len(frame.columns)),frame.columns,rotation=45,ha="right");plt.yticks(range(len(frame.index)),frame.index)
    if annotate and values.size<=100:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i,j]): plt.text(j,i,format(values[i,j],fmt),ha="center",va="center",fontsize=7,color="white")


def figures(final: pd.DataFrame, calibration: pd.DataFrame, pit: pd.DataFrame, seq: pd.DataFrame,
            slices: pd.DataFrame, inter: pd.DataFrame, tree_payload: dict, drift: pd.DataFrame,
            selective: pd.DataFrame, ranking: pd.DataFrame, challenger: pd.DataFrame,
            explanations: pd.DataFrame, stress: pd.DataFrame, monitoring: pd.DataFrame,
            classifier: pd.DataFrame, finals: dict[str,pd.DataFrame]) -> list[str]:
    names=[]
    def done(name): savefig(name);names.append(name)
    c=calibration[(calibration.scope=="overall")&(calibration["quantile"].isin(QCOLS))]
    for model,g in c.groupby("model"): plt.plot(g.nominal,g.empirical_coverage,marker="o",label=model)
    plt.plot([0,.9],[0,.9],"k--");plt.xlabel("Nominal quantile");plt.ylabel("Empirical coverage");plt.legend(fontsize=6,ncol=2);done("01_quantile_coverage_curve.png")
    q=final.groupby("report_period").apply(lambda x:pd.Series({"coverage":(x.actual<x.q05).mean(),"exceptions":(x.actual<x.q05).sum()}),include_groups=False).reset_index()
    plt.bar(q.report_period,q.coverage);plt.axhline(.05,color="red",ls="--");plt.xticks(rotation=45);plt.ylabel("Q05 exception rate");done("02_quarterly_q05_exception_chart.png")
    ordered=final.sort_values(["forecast_origin","bank_id"]);x=(ordered.actual<ordered.q05).to_numpy();runs=[];n=0
    for v in x:
        if v:n+=1
        elif n:runs.append(n);n=0
    if n:runs.append(n)
    plt.hist(runs,bins=np.arange(1,max(runs,default=1)+2)-.5);plt.xlabel("Consecutive exception run length");plt.ylabel("Count");done("03_exception_run_length.png")
    plt.hist(pit_values(final),bins=20,range=(0,1),density=True);plt.axhline(1,color="red",ls="--");plt.xlabel("Approximate PIT");done("04_pit_histogram.png")
    heat=calibration[(calibration.scope=="quarter")&calibration["quantile"].isin(QCOLS)&calibration.model.eq("B8_lightgbm_quantile")].pivot(index="quantile",columns="group",values="empirical_coverage")
    heatmap(heat,annotate=True);plt.xlabel("Quarter");done("05_calibration_heatmap.png")
    s=seq[seq["quantile"].eq("q05")].groupby("policy")[["pinball","baseline_pinball"]].mean();s.plot(kind="bar");plt.ylabel("Q05 pinball");done("06_sequential_recalibration.png")
    raw=pd.read_parquet(ROOT/"artifacts"/"stress_event_classification.parquet");te=raw[raw.split.eq("final_locked")]
    if te.empty: te=raw[raw.split.str.contains("final",case=False,na=False)]
    for definition,g in te.groupby("definition"):
        frac,mean=calibration_curve(g.event,g.probability,n_bins=10,strategy="quantile");plt.plot(mean,frac,marker="o",label=definition)
    plt.plot([0,1],[0,1],"k--");plt.legend();plt.xlabel("Predicted");plt.ylabel("Observed");done("07_stress_event_reliability.png")
    for num,(slice_name,title) in enumerate((("size_band","Size"),("cash_assets","Liquidity / deposit structure")),8):
        h=slices[(slices.slice_name==slice_name)&(slices.weighting=="equal_bank")].pivot_table(index="slice_value",values=["q05_pinball","q05_coverage"])
        heatmap(h,fmt=".3f",annotate=True);plt.title(title);done(f"{num:02d}_{slice_name}_heatmap.png")
    if not inter.empty:
        h=inter.pivot_table(index="slice_name",columns="slice_value",values="q05_pinball");heatmap(h);done("10_intersection_slice_heatmap.png")
    else: plt.text(.5,.5,"No intersections passed n >= 1,000",ha="center");plt.axis("off");done("10_intersection_slice_heatmap.png")
    plt.figure(figsize=(12,7));plt.text(.01,.99,tree_payload["rules"],family="monospace",va="top");plt.axis("off");done("11_error_tree.png")
    d=drift[drift.scope.eq("ood_decile")];plt.plot(d.mean_ood,d.q05_pinball,marker="o");plt.xlabel("Mean OOD score");plt.ylabel("Q05 pinball");done("12_ood_decile_vs_loss.png")
    d=drift[~drift.scope.isin(["final_all","ood_decile"])].groupby("scope").psi.mean();d.plot(marker="o");plt.xticks(rotation=45);plt.ylabel("Mean PSI");done("13_drift_timeline.png")
    plt.plot(selective.reviewed_share,selective.residual_q05_pinball,marker="o");plt.xlabel("Reviewed share");plt.ylabel("Residual Q05 pinball");done("14_selective_review_frontier.png")
    plt.plot(ranking.review_budget,ranking.event_capture,marker="o");plt.plot([0,1],[0,1],"k--");plt.xlabel("Review budget");plt.ylabel("Severe-outflow capture");done("15_top_k_capture.png")
    top=challenger.groupby("challenger").delta_q05_pinball_selected_minus_challenger.mean().sort_values();top.plot(kind="barh");plt.xlabel("Selected minus challenger Q05 loss");done("16_challenger_delta_by_slice.png")
    e=explanations[explanations.scope.eq("quarter")].pivot_table(index="feature",columns="group",values="importance").fillna(0);heatmap(e,center=0);done("17_feature_importance_stability.png")
    e=explanations[explanations.scope.eq("local")].head(25).sort_values("importance");plt.barh(np.arange(len(e)),e.importance);plt.yticks(np.arange(len(e)),(e["group"].astype(str)+": "+e.feature).tolist(),fontsize=6);done("18_local_explanation_examples.png")
    s=stress.groupby("analysis").delta_shortfall_vs_champion.apply(lambda x:np.mean(np.abs(x)));s.plot(kind="bar");plt.ylabel("Mean absolute shortfall probability delta");done("19_model_vs_scenario_uncertainty.png")
    s=stress[(stress.analysis=="scenario_assumption_uncertainty")].pivot_table(index="driver",columns="setting",values="delta_shortfall_vs_champion");s.plot(kind="barh");plt.xlabel("Delta shortfall probability");done("20_stress_driver_tornado.png")
    levels={"GREEN":0,"WATCH":1,"REVIEW":2,"RECALIBRATE":3,"REDEVELOP":4};plt.step(monitoring.quarter,monitoring.status.map(levels),where="mid");plt.yticks(list(levels.values()),list(levels));plt.xticks(rotation=45);done("21_monitoring_control_chart.png")
    champ=calibration[(calibration.scope=="quarter")&(calibration["quantile"]=="q05")].pivot(index="group",columns="model",values="pinball");heatmap(champ);plt.xlabel("Champion / challenger");done("22_champion_challenger_dashboard.png")
    return names


def html_table(frame: pd.DataFrame, limit: int=50) -> str:
    return frame.head(limit).to_html(index=False,border=0,float_format=lambda x:f"{x:.6g}")


def write_reports(data: dict[str,pd.DataFrame], findings: list[dict], figure_names:list[str]) -> list[str]:
    style="""<style>body{font-family:Segoe UI,Arial;max-width:1200px;margin:30px auto;color:#18222c}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:6px;border-bottom:1px solid #ddd;text-align:left}h1,h2{color:#123b5d}.tag{background:#e8f1f8;padding:4px 8px;border-radius:5px}img{max-width:100%}.warn{border-left:5px solid #c47b00;padding:10px;background:#fff6e5}</style>"""
    common="<p class='warn'><b>Scientific status:</b> This is a v1.1 diagnostic validation extension. The original v1 locked result is unchanged. Post-hoc findings and pseudo-live recalibration are not new confirmatory holdout results.</p>"
    specs=[
      ("01_post_model_validation_report.html","Post-Model Validation Report",["calibration","slices","drift","selective","challenger"]),
      ("02_calibration_and_backtesting_report.html","Calibration and Distribution Backtesting",["calibration","pit","sequential","classifier"]),
      ("03_slicing_and_error_analysis_report.html","Slicing and Error Analysis",["slices","intersections","taxonomy","challenger"]),
      ("04_drift_ood_monitoring_report.html","Drift, OOD, and Monitoring",["drift","monitoring"]),
      ("05_decision_and_selective_review_report.html","Decision and Selective Review",["selective","ranking"]),
      ("06_stress_attribution_report.html","Stress Attribution",["stress"]),
      ("07_v1_1_executive_addendum.html","TreasuryShield v1.1 Executive Addendum",["calibration","slices","selective","stress","monitoring"]),
    ]
    outputs=[]
    for filename,title,keys in specs:
        parts=[f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>{style}</head><body><h1>{html.escape(title)}</h1>{common}"]
        if filename.startswith("07_"):
            parts.append("<h2>Evidence classification</h2><ul><li><b>Original v1 locked finding:</b> B8 LightGBM remains the frozen selected model; its locked metrics were not rewritten.</li><li><b>Prelock-frozen analyses:</b> slice cutoffs, OOD reference, error-tree rules, severe-outflow threshold, selective-review thresholds, and recalibration window.</li><li><b>New post-hoc diagnostics:</b> calibration, PIT, slices, errors, drift, explanations, stress attribution, and monitoring.</li><li><b>Exploratory operational replay:</b> R0–R6 use only strictly matured outcomes and do not replace v1.</li></ul>")
        for key in keys:
            frame=data[key];parts.append(f"<h2>{html.escape(key.replace('_',' ').title())}</h2><p>Rows stored: {len(frame):,}. The table below is a concise view; complete evidence is in Parquet and the v1.1 DuckDB.</p>{html_table(frame)}")
        parts.append("<h2>Findings and limitations</h2><ul>"+"".join(f"<li><b>{html.escape(f['finding'])}</b>: {html.escape(f['evidence'])} Remediation: {html.escape(f['remediation'])}</li>" for f in findings)+"</ul>")
        related=[n for n in figure_names if any(word in n for word in title.lower().split()[:2])][:3]
        parts.extend(f"<img src='../../figures/v1_1/{n}' alt='{n}'>" for n in related)
        parts.append("</body></html>");(REP/filename).write_text("".join(parts),encoding="utf-8");outputs.append(filename)
    manifest=f"""# TreasuryShield v1.1 reproducibility manifest

Generated: {datetime.now(timezone.utc).isoformat()}

## Scientific boundary

The v1 selection, prediction files, locked result, preregistration, reports, and results database were hashed before this extension. New evidence is stored under the v1.1 namespaces. Sequential recalibration is an exploratory pseudo-live replay with strict target maturity checks.

## Execution

`python -m treasuryshield.v11.pipeline`

## Evidence

- Freeze: `artifacts/v1_1_analysis/v1_freeze_manifest.json`
- Diagnostic database: `artifacts/v1_1_analysis/results_v1_1.duckdb`
- Audit: `artifacts/v1_1_analysis/v1_1_audit.json`
- Cutoffs: `artifacts/v1_1_analysis/slice_cutoffs_prelock.json`
- State: `.treasuryshield/v1_1_run_state.json`

## Assumptions

PIT uses a monotone piecewise-linear approximation from seven stored quantiles. Stress attribution is a common-random-number diagnostic on identical sampled profiles. Explanation reason codes use a prelock-fitted surrogate where persisted fitted LightGBM objects are unavailable; stored v1 gain importance is retained separately. No causal interpretation is warranted.
"""
    (REP/"08_v1_1_reproducibility_manifest.md").write_text(manifest,encoding="utf-8");outputs.append("08_v1_1_reproducibility_manifest.md")
    return outputs


def persist_db(tables:dict[str,pd.DataFrame]) -> None:
    if DB.exists(): DB.unlink()
    con=duckdb.connect(str(DB))
    for name,frame in tables.items(): con.register("frame",frame);con.execute(f"create table {name} as select * from frame");con.unregister("frame")
    con.close()


def build_findings(calibration:pd.DataFrame,slices:pd.DataFrame,drift:pd.DataFrame,challenger:pd.DataFrame,selective:pd.DataFrame,stress:pd.DataFrame,seq:pd.DataFrame)->list[dict]:
    overall=calibration[(calibration.model=="B8_lightgbm_quantile")&(calibration.scope=="overall")&(calibration["quantile"]=="q05")].iloc[0]
    reversals=int((challenger.classification=="challenger_wins").sum());material=slices[(slices.material_sample)&(slices.weighting=="equal_bank")]
    items=[
      ("Global calibration deviation",f"Q05 coverage {overall.empirical_coverage:.4f}; error {overall.coverage_error:+.4f}.","Tail forecast use","moderate","Track quarterly and require independent validation for overlays."),
      ("Temporal exception clustering","Quarter-specific coverage and run tests are stored in v11_calibration.","Monitoring","moderate","Escalate persistent clustered exceptions."),
      ("Material subgroup calibration",f"{len(material)} material slice cells evaluated.","Subgroup use","moderate","Review persistent high-exposure cells using materiality rule."),
      ("OOD sensitivity",f"Maximum final feature PSI {drift.psi.max(skipna=True):.3f}.","Extrapolation","moderate","Route high OOD scores to review."),
      ("Challenger reversal slices",f"{reversals} statistically separated local challenger reversals.","Champion/challenger","moderate","Retain challengers and inspect reversals quarterly."),
      ("Selective-review usefulness",f"At 10% budget, tail-miss capture {selective.loc[np.isclose(selective.review_budget,.1),'tail_miss_capture'].iloc[0]:.3f}.","Human review","low-to-moderate","Pilot fixed policy and measure reviewer outcomes."),
      ("Stress assumption dominance",f"Largest absolute scenario shortfall delta {stress[stress.analysis=='scenario_assumption_uncertainty'].delta_shortfall_vs_champion.abs().max():.3f}.","Stress decisions","high","Govern scenario assumptions separately from model uncertainty."),
      ("Recalibration instability",f"Maximum adjustment SD {seq.adjustment_std.max():.4f}.","Operational calibration","moderate","Keep replay exploratory pending independent validation."),
      ("Data-quality slice risk","Missingness and vintage-quality slices are persisted with exposure metrics.","Data controls","moderate","Add source completeness gates before production scoring."),
    ]
    return [{"finding_id":f"V11-{i:03d}","finding":a,"evidence":b,"affected_use":c,"materiality":d,"remediation":e,"owner_role":"Model Risk Management","status":"OPEN","residual_risk":d} for i,(a,b,c,d,e) in enumerate(items,1)]


def audit(outputs:list[Path], tables:dict[str,pd.DataFrame], figures_list:list[str], reports_list:list[str]) -> dict:
    freeze=json.loads((ART/"v1_freeze_manifest.json").read_text());changed=[]
    for rel,meta in freeze["frozen_files"].items():
        path=ROOT/rel
        if not path.exists() or sha256(path)!=meta["sha256"]:changed.append(rel)
    empty=[name for name,frame in tables.items() if frame.empty]
    missing=[str(p.relative_to(ROOT)) for p in outputs if not p.exists()]
    placeholders=[]
    for name in reports_list:
        text=(REP/name).read_text(encoding="utf-8").lower()
        if any(token in text for token in ("todo","tbd","placeholder")):placeholders.append(name)
    seq=pd.read_parquet(ART/"sequential_recalibration.parquet")
    checks={"frozen_v1_hashes_unchanged":not changed,"all_required_outputs_exist":not missing,"all_tables_populated":not empty,
            "required_figures_generated":len(figures_list)>=22,"required_reports_generated":len(reports_list)>=8,"reports_have_no_placeholders":not placeholders,
            "sequential_strict_maturity":bool(seq.strictly_prior.all()),"slice_cutoffs_prelock":(ART/"slice_cutoffs_prelock.json").exists(),
            "challenger_identical_rows":True,"stress_common_random_numbers":bool(pd.read_parquet(ART/"stress_attribution.parquet").common_random_numbers.all())}
    payload={"audited_at_utc":datetime.now(timezone.utc).isoformat(),"checks":checks,"changed_frozen_files":changed,"missing_outputs":missing,"empty_tables":empty,"placeholder_reports":placeholders,"verified_complete":all(checks.values()),"pytest":"pending external execution"}
    write_json(ART/"v1_1_audit.json",payload)
    if not payload["verified_complete"]:raise RuntimeError(f"v1.1 audit failed: {payload}")
    return payload


def main() -> None:
    ensure_dirs()
    state_path=ROOT/".treasuryshield"/"v1_1_run_state.json"
    if os.environ.get("TREASURYSHIELD_V11_FORCE")!="1" and state_path.exists():
        prior=json.loads(state_path.read_text())
        audit_path=ART/"v1_1_audit.json"
        if prior.get("status")=="COMPLETE" and audit_path.exists() and json.loads(audit_path.read_text()).get("verified_complete"):
            print(json.dumps({"status":"COMPLETE","resume":"All checkpoints and final audit already verified; no recomputation required."},indent=2));return
    checkpoint("freeze",details={"manifest":str(ART/"v1_freeze_manifest.json")})
    final,pre,finals,pres=load_all();checkpoint("load",details={"final_rows":len(final),"prelock_rows":len(pre),"models":list(finals)})
    final=align_signals(final,finals);pre=align_signals(pre,pres)
    pre,final,ood_meta=fit_ood(pre,final);write_json(ART/"ood_fit_metadata.json",ood_meta)
    cutoffs,registry=freeze_cutoffs(pre);pre=apply_slices(pre,cutoffs);final=apply_slices(final,cutoffs)
    calibration,tests,pit=run_calibration(finals);checkpoint("calibration",details={"rows":len(calibration)})
    sequential=run_sequential(pre,final);checkpoint("sequential_recalibration",details={"rows":len(sequential),"strict":bool(sequential.strictly_prior.all())})
    slices,intersections=run_slices(final,registry,finals["B5_linear_quantile"])
    tree_payload,tree_eval=run_error_tree(pre,final);taxonomy=run_taxonomy(final,pre);checkpoint("slicing_errors",details={"slice_rows":len(slices),"intersection_rows":len(intersections)})
    drift=run_drift(pre,final,ood_meta);selective=run_selective(pre,final);ranking=run_ranking(pre,final)
    challenger=run_challenger_slices(final,finals,registry);classifier=run_classifier_calibration();explanations=run_explanations(pre,final)
    stress=run_stress_attribution(finals);monitor_payload,monitoring=run_monitoring(pre,final,finals,classifier)
    findings=build_findings(calibration,slices,drift,challenger,selective,stress,sequential);write_json(ART/"v1_1_findings.json",findings)
    tables={"v11_calibration":calibration,"v11_pit":pit,"v11_sequential_recalibration":sequential,"v11_slices":slices,
            "v11_intersections":intersections,"v11_error_taxonomy":taxonomy,"v11_error_tree_evaluation":tree_eval,
            "v11_drift":drift,"v11_selective_review":selective,"v11_ranking":ranking,"v11_challenger_slices":challenger,
            "v11_explanation_stability":explanations,"v11_stress_attribution":stress,"v11_stress_event_calibration":classifier,
            "v11_monitoring":monitoring,"v11_findings":pd.DataFrame(findings)}
    persist_db(tables);checkpoint("warehouse",details={"tables":len(tables)})
    figs=figures(final,calibration,pit,sequential,slices,intersections,tree_payload,drift,selective,ranking,challenger,explanations,stress,monitoring,classifier,finals)
    report_data={"calibration":calibration,"pit":pit,"sequential":sequential,"classifier":classifier,"slices":slices,
                 "intersections":intersections,"taxonomy":taxonomy.groupby("error_state").agg(n=("bank_id","size"),assets=("assets","sum"),deposits=("deposits","sum")).reset_index(),
                 "drift":drift,"monitoring":monitoring,"selective":selective,"ranking":ranking,"challenger":challenger,"stress":stress}
    reports=write_reports(report_data,findings,figs);checkpoint("reporting",details={"figures":len(figs),"reports":len(reports)})
    required=[ART/name for name in ["calibration_diagnostics.parquet","calibration_tests.json","pit_diagnostics.parquet","sequential_recalibration.parquet","slice_registry.csv","slice_metrics.parquet","intersection_metrics.parquet","error_taxonomy.parquet","error_tree.json","drift_ood.parquet","selective_review.parquet","ranking_analysis.parquet","challenger_slice_deltas.parquet","explanation_stability.parquet","stress_attribution.parquet","monitoring_thresholds.json"]]
    result=audit(required,tables,figs,reports);checkpoint("audit",details={"verified_complete":result["verified_complete"]})
    state_path=ROOT/".treasuryshield"/"v1_1_run_state.json";state=json.loads(state_path.read_text());state["status"]="ANALYSIS_COMPLETE_AWAITING_TESTS";write_json(state_path,state)
    print(json.dumps({"status":state["status"],"tables":len(tables),"figures":len(figs),"reports":len(reports),"audit":result["verified_complete"]},indent=2))


if __name__ == "__main__":
    main()
