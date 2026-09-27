from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from treasuryshield.v11.common import enrich, model_path, read_prediction, read_prelock
from treasuryshield.v11.pipeline import align_signals, fit_ood, risk_score

from .common import ART, QCOLS, QLEVELS, ROOT, SEED, pinball, weights, write_json
from .inference import clustered_summary, clustered_weighted_summary, holm_adjust


KEY_MODELS = [
    "B5_linear_quantile", "B7_xgb_quantile", "B10_gru",
    "RAHQ_tree", "RAHQ_C_tree", "RAHQ_C_nn",
]


def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    final = enrich(read_prediction("B8_lightgbm_quantile"))
    pre = enrich(read_prelock("B8_lightgbm_quantile"))
    final_models = {"B8_lightgbm_quantile": read_prediction("B8_lightgbm_quantile")}
    pre_models = {"B8_lightgbm_quantile": read_prelock("B8_lightgbm_quantile")}
    for model in KEY_MODELS:
        if model_path(model).exists():
            final_models[model] = read_prediction(model)
        prior = read_prelock(model)
        if not prior.empty:
            pre_models[model] = prior
    pre = align_signals(pre, pre_models)
    final = align_signals(final, final_models)
    pre, final, _ = fit_ood(pre, final)
    return pre, final, final_models


def run_calibration(pre: pd.DataFrame, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for column, alpha in zip(QCOLS, QLEVELS):
        exception = (final.actual.to_numpy(float) <= final[column].to_numpy(float)).astype(float)
        loss = pinball(final.actual.to_numpy(float), final[column].to_numpy(float), float(alpha))
        severity = np.maximum(final[column].to_numpy(float) - final.actual.to_numpy(float), 0)
        for weighting in ("equal_bank", "asset_weighted", "deposit_weighted"):
            w = weights(final, weighting)
            for metric, values, expected in (
                ("exception_rate", exception, float(alpha)),
                ("coverage_error", exception - alpha, 0.0),
                ("pinball", loss, 0.0),
                ("tail_underestimation_severity", severity, 0.0),
            ):
                summary = clustered_weighted_summary(
                    values, w, final.bank_id.to_numpy(), final.report_period.to_numpy(),
                    reps=999, seed=SEED + int(alpha * 1000),
                )
                rows.append({"quantile": column.upper(), "alpha": alpha, "weighting": weighting,
                             "metric": metric, "expected": expected,
                             "assets_exposure": float(final.assets.sum()),
                             "deposits_exposure": float(final.deposits.sum()), **summary,
                             "time_cluster_resolution": "low" if summary["n_quarters"] < 12 else "ordinary"})
    calibration = pd.DataFrame(rows)
    calibration.to_parquet(ART / "calibration_clustered.parquet", index=False)

    quarter_rows = []
    for quarter, sub in final.groupby("report_period", sort=True):
        y, q = sub.actual.to_numpy(float), sub.q05.to_numpy(float)
        exc = y <= q
        quarter_rows.append({
            "quarter": quarter, "quantile": "Q05", "exception_rate": float(exc.mean()),
            "expected_exception_rate": .05, "coverage_deviation": float(exc.mean() - .05),
            "absolute_coverage_deviation": float(abs(exc.mean() - .05)),
            "pinball": float(pinball(y, q, .05).mean()),
            "tail_severity": float(np.maximum(q - y, 0).mean()), "n_banks": len(sub),
            "assets": float(sub.assets.sum()), "deposits": float(sub.deposits.sum()),
        })
    quarterly = pd.DataFrame(quarter_rows)
    quarterly.to_parquet(ART / "calibration_by_quarter.parquet", index=False)

    combined = pd.concat([pre.assign(window="prelock"), final.assign(window="locked")], ignore_index=True)
    persistence = []
    for column, alpha in zip(QCOLS[:3], QLEVELS[:3]):
        transitions = []
        bank_summaries = []
        for bank, sub in combined.sort_values(["bank_id", "forecast_origin"]).groupby("bank_id"):
            sub = sub.drop_duplicates("report_period").sort_values("report_period")
            periods = pd.PeriodIndex(sub.report_period, freq="Q").astype(int).to_numpy()
            exc = (sub.actual.to_numpy(float) <= sub[column].to_numpy(float)).astype(int)
            adjacent = np.diff(periods) == 1
            if adjacent.sum() < 3:
                continue
            prev, nxt = exc[:-1][adjacent], exc[1:][adjacent]
            transitions.extend(zip(prev.tolist(), nxt.tolist()))
            p_repeat = float(nxt[prev == 1].mean()) if (prev == 1).any() else math.nan
            bank_summaries.append((bank, len(exc), int(adjacent.sum()), float(exc.mean()), p_repeat))
        t = np.asarray(transitions, int)
        if len(t):
            p01 = float(t[t[:, 0] == 0, 1].mean()) if (t[:, 0] == 0).any() else math.nan
            p11 = float(t[t[:, 0] == 1, 1].mean()) if (t[:, 0] == 1).any() else math.nan
            persistence.append({"quantile": column.upper(), "scope": "aggregate_within_bank",
                                "bank_id": None, "observations": len(combined), "transitions": len(t),
                                "exception_rate": float((combined.actual <= combined[column]).mean()),
                                "p_exception_after_nonexception": p01,
                                "p_exception_after_exception": p11,
                                "repeat_risk_ratio": p11 / p01 if p01 > 0 else math.nan,
                                "formal_independence_test": "not forced; panel/block inference used"})
        for bank, n, nt, rate, repeat in bank_summaries:
            persistence.append({"quantile": column.upper(), "scope": "bank", "bank_id": bank,
                                "observations": n, "transitions": nt, "exception_rate": rate,
                                "p_exception_after_nonexception": math.nan,
                                "p_exception_after_exception": repeat,
                                "repeat_risk_ratio": math.nan,
                                "formal_independence_test": "underpowered at individual-bank length"})
    persistence_frame = pd.DataFrame(persistence)
    persistence_frame.to_parquet(ART / "exception_persistence.parquet", index=False)
    return calibration, quarterly, persistence_frame


def build_corrected_slices(pre: pd.DataFrame, final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre = pre.copy(); final = final.copy()
    # Genuine peer-relative deterioration uses a peer-group reference frozen on prelock observations.
    peer_reference = pre.groupby("peer_group", dropna=False).deposit_growth_qoq_lagged.median()
    final["peer_relative_deterioration"] = final.deposit_growth_qoq_lagged - final.peer_group.map(peer_reference)
    pre["peer_relative_deterioration"] = pre.deposit_growth_qoq_lagged - pre.peer_group.map(peer_reference)
    pre_edges = {
        "peer_relative_deterioration": np.unique(np.nanquantile(pre.peer_relative_deterioration, np.linspace(0, 1, 6))).tolist(),
        "borrowings_assets_decile": np.unique(np.nanquantile(pre.borrowings_asset_ratio, np.linspace(0, 1, 11))).tolist(),
        "prelock_asset_decile": np.unique(np.nanquantile(pre.assets, np.linspace(0, 1, 11))).tolist(),
        "uninsured_proxy_decile": np.unique(np.nanquantile(pre.uninsured_proxy_domestic_share.dropna(), np.linspace(0, 1, 11))).tolist(),
        "ood_decile": np.unique(np.nanquantile(pre.ood_score, np.linspace(0, 1, 11))).tolist(),
        "disagreement_decile": np.unique(np.nanquantile(pre.disagreement, np.linspace(0, 1, 11))).tolist(),
    }
    for name, edge in pre_edges.items():
        edge[0], edge[-1] = -np.inf, np.inf
        source = {"borrowings_assets_decile": "borrowings_asset_ratio", "prelock_asset_decile": "assets",
                  "uninsured_proxy_decile": "uninsured_proxy_domestic_share", "ood_decile": "ood_score",
                  "disagreement_decile": "disagreement"}.get(name, name)
        final[name] = pd.cut(final[source], edge, labels=[f"D{i}" for i in range(1, len(edge))], include_lowest=True).astype("string").fillna("missing")
    final["fixed_economic_size_band"] = pd.cut(
        final.assets, [-np.inf, 1_000_000, 10_000_000, 100_000_000, np.inf],
        labels=["under_$1bn", "$1bn_$10bn", "$10bn_$100bn", "over_$100bn"], include_lowest=True,
    ).astype("string")
    final["uninsured_measure_provenance"] = np.where(
        final.uninsured_proxy_domestic_share.notna(), "constructed_insured_deposit_proxy", "unavailable")
    final["pseudo_pit_vintage_quality"] = final.source_vintage.astype("string").fillna("unavailable")
    final["hmm_state_probability"] = final.regime.astype("string").fillna("unassigned_hard_state_only")

    enhanced = pd.read_parquet(ROOT / "data" / "features" / "enhanced_features.parquet",
                               columns=["bank_id", "report_period", "ffiec_form"])
    enhanced["report_period"] = enhanced.report_period.astype(str)
    final = final.merge(enhanced.drop_duplicates(["bank_id", "report_period"]),
                        on=["bank_id", "report_period"], how="left", validate="one_to_one")
    final["report_form"] = final.ffiec_form.astype("string").fillna("not_available_in_enhanced_link")

    events = pd.read_parquet(ROOT / "data" / "processed" / "entity_events.parquet",
                             columns=["bank_id", "event_quarter", "structural_contamination"])
    events = events.loc[events.structural_contamination.eq(True) & events.bank_id.notna() & events.event_quarter.notna()].copy()
    def stata_quarter(value: str) -> int:
        period = pd.Period(value, freq="Q")
        return (period.year - 1960) * 4 + period.quarter - 1

    event_map = {bank: pd.to_numeric(group.event_quarter, errors="coerce").dropna().astype(int).to_numpy()
                 for bank, group in events.groupby("bank_id")}
    distances = []
    for row in final[["bank_id", "report_period"]].itertuples(index=False):
        candidates = event_map.get(row.bank_id)
        if candidates is None or len(candidates) == 0:
            distances.append(np.nan)
        else:
            distances.append(int(np.min(np.abs(candidates - stata_quarter(row.report_period)))))
    final["ma_distance_quarters"] = distances
    final["ma_distance_band"] = pd.cut(final.ma_distance_quarters, [-np.inf, 1, 2, 4, 8, np.inf],
                                        labels=["0-1q", "2q", "3-4q", "5-8q", "9q+"], include_lowest=True).astype("string").fillna("no_linked_material_event")

    registry_rows = [
        ("peer_relative_deterioration", "bank deposit_growth_qoq_lagged minus prelock pooled median for frozen peer_group", "deposit_growth_qoq_lagged; peer_group", "pre-2024", "Relative lagged deterioration, not contemporaneous outcome", "Peer reference pooled across prelock episodes"),
        ("borrowings_assets_decile", "prelock-decile of borrowings_asset_ratio", "borrowings_asset_ratio", "pre-2024", "Borrowings/assets; not a complete noncore-funding measure", "Does not measure all wholesale/noncore funding"),
        ("fixed_economic_size_band", "assets in $000: <1m, 1m-10m, 10m-100m, >=100m", "assets", "documented fixed thresholds", "Approximate <$1bn, $1-10bn, $10-100bn, >=$100bn", "Public-data accounting units are $000"),
        ("prelock_asset_decile", "prelock asset decile cutoffs", "assets", "pre-2024", "Relative size rank", "Not an economic size band"),
        ("ma_distance_band", "minimum absolute quarter distance to linked structural M&A event", "entity_events.bank_id; event_quarter; structural_contamination", "entity-event table", "Temporal proximity to material M&A", "Missing means no linked event, not infinite certainty"),
        ("uninsured_measure_provenance", "constructed proxy if 1-insured_deposits/domestic_dep exists; otherwise unavailable", "insured_deposits; domestic_dep", "feature registry", "Provenance of core uninsured proxy", "Not directly reported uninsured deposits"),
        ("report_form", "actual ffiec_form from enhanced-panel linkage", "enhanced_features.ffiec_form", "FFIEC source", "Actual linked Call Report form", "Unavailable where enhanced linkage is absent"),
        ("pseudo_pit_vintage_quality", "source_vintage category", "source_vintage", "source metadata", "Restated/pseudo-PIT vintage label", "Exact amendment vintage unavailable"),
        ("hmm_state_probability", "stored hard regime label; probabilities unavailable in frozen prediction row", "regime", "frozen prediction", "Hard-state descriptive slice", "Must not be interpreted as an HMM probability"),
        ("uninsured_proxy_decile", "prelock decile of constructed uninsured proxy", "uninsured_proxy_domestic_share", "pre-2024", "Relative constructed-proxy rank", "Proxy, not direct reporting"),
        ("ood_decile", "prelock decile of prelock-fitted IsolationForest OOD score", "ood_score", "pre-2024", "Covariate OOD rank", "Research diagnostic"),
        ("disagreement_decile", "prelock decile of cross-model Q05 standard deviation", "disagreement", "pre-2024", "Model-disagreement rank", "Depends on available frozen models"),
    ]
    registry = pd.DataFrame(registry_rows, columns=["display_name", "exact_formula", "raw_canonical_source_fields", "cutoff_source", "intended_interpretation", "limitations"])
    registry["availability_period"] = "final locked 2024Q1-2026Q1 where source fields exist"
    registry.to_csv(ART / "slice_registry_corrected.csv", index=False)
    write_json(ART / "slice_cutoffs_prelock.json", pre_edges)
    return final, registry


def run_slice_inference(final: pd.DataFrame, models: dict[str, pd.DataFrame], registry: pd.DataFrame) -> pd.DataFrame:
    policy = {"minimum_rows": 1000, "minimum_unique_banks": 50, "minimum_quarters": 6,
              "minimum_absolute_q05_pinball_effect": 1e-5, "minimum_exposure_share": .001,
              "persistence_requirement": "same effect direction in at least two-thirds of represented quarters",
              "multiplicity": "Holm adjustment within challenger and slice family", "alpha": .05}
    write_json(ART / "material_slice_policy.json", policy)
    keys = ["bank_id", "report_period"]
    base = final.set_index(keys, drop=False)
    rows = []
    slice_names = [name for name in registry.display_name if name in final]
    for challenger, frame in models.items():
        if challenger == "B8_lightgbm_quantile":
            continue
        other = frame.set_index(keys)
        common = base.index.intersection(other.index)
        selected = base.loc[common]
        opponent = other.loc[common]
        if not np.allclose(selected.actual, opponent.actual, equal_nan=True):
            raise RuntimeError(f"Challenger actual alignment failed: {challenger}")
        delta_all = pinball(selected.actual, selected.q05, .05) - pinball(selected.actual, opponent.q05, .05)
        delta = pd.Series(delta_all, index=common)
        for slice_name in slice_names:
            for value, index in selected.groupby(slice_name, dropna=False).groups.items():
                sub = selected.loc[index]
                d = delta.loc[index].to_numpy(float)
                summary = clustered_summary(d, sub.bank_id.to_numpy(), sub.report_period.to_numpy(), reps=999, seed=SEED)
                selected_coverage = float((sub.actual <= sub.q05).mean())
                challenger_coverage = float((sub.actual.to_numpy(float) <= opponent.loc[index, "q05"].to_numpy(float)).mean())
                coverage_values = (sub.actual.to_numpy(float) <= sub.q05.to_numpy(float)).astype(float) - .05
                coverage_summary = clustered_summary(coverage_values, sub.bank_id.to_numpy(), sub.report_period.to_numpy(), reps=999, seed=SEED + 7)
                coverage_quarters = pd.Series(coverage_values).groupby(sub.report_period.to_numpy()).mean()
                coverage_direction = np.sign(coverage_summary["estimate"])
                coverage_persistence = float((np.sign(coverage_quarters) == coverage_direction).mean()) if coverage_direction else 0.0
                coverage_z = coverage_summary["estimate"] / coverage_summary["two_way_se"] if coverage_summary["two_way_se"] > 0 else 0.0
                coverage_p = float(2 * stats.t.sf(abs(coverage_z), max(coverage_summary["n_quarters"] - 1, 1)))
                quarter_means = pd.Series(d).groupby(sub.report_period.to_numpy()).mean()
                direction = np.sign(summary["estimate"])
                persistence = float((np.sign(quarter_means) == direction).mean()) if direction else 0.0
                tw_z = summary["estimate"] / summary["two_way_se"] if summary["two_way_se"] > 0 else 0.0
                p_value = float(2 * stats.t.sf(abs(tw_z), max(summary["n_quarters"] - 1, 1)))
                exposure_share = float(sub.assets.sum() / final.assets.sum())
                gates = (len(sub) >= policy["minimum_rows"] and sub.bank_id.nunique() >= policy["minimum_unique_banks"]
                         and sub.report_period.nunique() >= policy["minimum_quarters"]
                         and abs(summary["estimate"]) >= policy["minimum_absolute_q05_pinball_effect"]
                         and exposure_share >= policy["minimum_exposure_share"] and persistence >= 2 / 3)
                excludes = sum([summary["quarter_ci_low"] > 0 or summary["quarter_ci_high"] < 0,
                                summary["bank_ci_low"] > 0 or summary["bank_ci_high"] < 0,
                                summary["two_way_ci_low"] > 0 or summary["two_way_ci_high"] < 0])
                rows.append({"slice_name": slice_name, "slice_value": str(value), "challenger": challenger,
                             "delta_q05_pinball_selected_minus_challenger": summary.pop("estimate"),
                             **summary, "assets_exposure": float(sub.assets.sum()),
                             "deposits_exposure": float(sub.deposits.sum()), "exposure_share": exposure_share,
                             "selected_q05_coverage": selected_coverage,
                             "selected_q05_coverage_error": selected_coverage - .05,
                             "challenger_q05_coverage": challenger_coverage,
                             "coverage_quarter_ci_low": coverage_summary["quarter_ci_low"],
                             "coverage_quarter_ci_high": coverage_summary["quarter_ci_high"],
                             "coverage_bank_ci_low": coverage_summary["bank_ci_low"],
                             "coverage_bank_ci_high": coverage_summary["bank_ci_high"],
                             "coverage_two_way_ci_low": coverage_summary["two_way_ci_low"],
                             "coverage_two_way_ci_high": coverage_summary["two_way_ci_high"],
                             "coverage_two_way_se": coverage_summary["two_way_se"],
                             "coverage_quarter_direction_persistence": coverage_persistence,
                             "coverage_raw_p_value": coverage_p,
                             "quarter_direction_persistence": persistence, "cluster_methods_excluding_zero": excludes,
                             "material_gates_pass": gates, "raw_p_value": p_value})
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = np.nan
    result["coverage_holm_adjusted_p"] = np.nan
    for (_, _), idx in result.groupby(["challenger", "slice_name"]).groups.items():
        result.loc[idx, "holm_adjusted_p"] = holm_adjust(result.loc[idx, "raw_p_value"].to_numpy())
        result.loc[idx, "coverage_holm_adjusted_p"] = holm_adjust(result.loc[idx, "coverage_raw_p_value"].to_numpy())
    def classify(row: pd.Series) -> str:
        if not row.material_gates_pass or row.cluster_methods_excluding_zero < 2 or row.holm_adjusted_p > .05:
            return "indistinguishable_or_unstable"
        return "selected_wins_robustly" if row.delta_q05_pinball_selected_minus_challenger < 0 else "challenger_wins_robustly"
    result["classification"] = result.apply(classify, axis=1)
    def coverage_classify(row: pd.Series) -> str:
        basic_gates = (row.n >= policy["minimum_rows"] and row.n_banks >= policy["minimum_unique_banks"]
                       and row.n_quarters >= policy["minimum_quarters"] and row.exposure_share >= policy["minimum_exposure_share"]
                       and abs(row.selected_q05_coverage_error) >= .005
                       and row.coverage_quarter_direction_persistence >= 2 / 3)
        excludes = sum([row.coverage_quarter_ci_low > 0 or row.coverage_quarter_ci_high < 0,
                        row.coverage_bank_ci_low > 0 or row.coverage_bank_ci_high < 0,
                        row.coverage_two_way_ci_low > 0 or row.coverage_two_way_ci_high < 0])
        if basic_gates and excludes >= 2 and row.coverage_holm_adjusted_p <= .05:
            return "over_exception_miscalibrated_robustly" if row.selected_q05_coverage_error > 0 else "under_exception_miscalibrated_robustly"
        return "calibration_indistinguishable_or_unstable"
    result["calibration_classification"] = result.apply(coverage_classify, axis=1)
    result.to_parquet(ART / "slice_inference.parquet", index=False)
    return result


def run_selective_review(pre: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    frozen = json.loads((ROOT / "artifacts" / "v1_1_analysis" / "selective_policy_freeze.json").read_text())
    pre = pre.copy(); final = final.copy()
    signal_map = {"disagreement": "disagreement", "predictive_width": "pred_width",
                  "missingness": "missing_count", "ood": "ood_score"}
    final["combined"] = risk_score(final, pre)
    pre["combined"] = risk_score(pre, pre)
    for name, column in signal_map.items():
        reference = pre[column].dropna().sort_values().to_numpy()
        final[name] = np.searchsorted(reference, final[column].fillna(pre[column].median()), side="right") / len(reference)
        pre[name] = np.searchsorted(reference, pre[column].fillna(pre[column].median()), side="right") / len(reference)
    severe = (final.actual < final.q05 - .02).to_numpy()
    loss = pinball(final.actual.to_numpy(), final.q05.to_numpy(), .05)
    rows = []
    rng = np.random.default_rng(SEED)
    for budget in (.01, .05, .10, .20):
        for policy in ["combined", *signal_map]:
            threshold = (float(frozen["thresholds"][str(budget)]) if policy == "combined"
                         else float(pre[policy].quantile(1 - budget)))
            reviewed = final[policy].to_numpy() >= threshold
            capture_values = (reviewed & severe).astype(float) / max(severe.mean(), 1e-12)
            capture_ci = clustered_summary(capture_values, final.bank_id.to_numpy(), final.report_period.to_numpy(), reps=999, seed=SEED)
            quarter_capture = []
            for _, idx in final.groupby("report_period").groups.items():
                idx = np.asarray(list(idx)); denom = severe[idx].sum()
                if denom: quarter_capture.append(float((reviewed[idx] & severe[idx]).sum() / denom))
            rows.append({"record_type": "policy", "policy": policy, "review_budget": budget,
                         "threshold": threshold, "threshold_source": "frozen v1.1" if policy == "combined" else "prelock single-signal",
                         "reviewed_share": float(reviewed.mean()), "reviewed_n": int(reviewed.sum()),
                         "severe_tail_miss_capture": float((reviewed & severe).sum() / max(severe.sum(), 1)),
                         "capture_quarter_ci_low": capture_ci["quarter_ci_low"], "capture_quarter_ci_high": capture_ci["quarter_ci_high"],
                         "capture_bank_ci_low": capture_ci["bank_ci_low"], "capture_bank_ci_high": capture_ci["bank_ci_high"],
                         "quarter_capture_min": min(quarter_capture) if quarter_capture else math.nan,
                         "quarter_capture_max": max(quarter_capture) if quarter_capture else math.nan,
                         "residual_q05_pinball": float(loss[~reviewed].mean()) if (~reviewed).any() else math.nan,
                         "asset_exposure_captured": float(final.loc[reviewed, "assets"].sum() / final.assets.sum()),
                         "deposit_exposure_captured": float(final.loc[reviewed, "deposits"].sum() / final.deposits.sum()),
                         "review_precision_efficiency": float((reviewed & severe).sum() / max(reviewed.sum(), 1)),
                         "random_expected_capture": budget,
                         "lift_over_random": float((reviewed & severe).sum() / max(severe.sum(), 1) / budget),
                         "bank_review_persistence": float(final.assign(reviewed=reviewed).groupby("bank_id").reviewed.mean().std())})
        random_captures = []
        n_review = max(1, round(len(final) * budget))
        for _ in range(1000):
            chosen = rng.choice(len(final), n_review, replace=False)
            random_captures.append(float(severe[chosen].sum() / max(severe.sum(), 1)))
        rows.append({"record_type": "random_benchmark", "policy": "random", "review_budget": budget,
                     "threshold": math.nan, "threshold_source": "1000 seeded equal-probability samples",
                     "reviewed_share": n_review / len(final), "reviewed_n": n_review,
                     "severe_tail_miss_capture": float(np.mean(random_captures)),
                     "capture_quarter_ci_low": float(np.quantile(random_captures, .025)),
                     "capture_quarter_ci_high": float(np.quantile(random_captures, .975)),
                     "capture_bank_ci_low": math.nan, "capture_bank_ci_high": math.nan,
                     "quarter_capture_min": math.nan, "quarter_capture_max": math.nan,
                     "residual_q05_pinball": math.nan, "asset_exposure_captured": budget,
                     "deposit_exposure_captured": budget, "review_precision_efficiency": float(severe.mean()),
                     "random_expected_capture": budget, "lift_over_random": 1.0, "bank_review_persistence": math.nan})
    out = pd.DataFrame(rows)
    out.to_parquet(ART / "selective_review_clustered.parquet", index=False)
    return out
