from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import qmc

from treasuryshield.stress.arithmetic import (
    BankState, Scenario, calculate, optimize_recovery, reverse_runoff_threshold,
)
from treasuryshield.stress.run import (
    QCOLS, _h8_shape, _joint_uniform, _reference_joint, _simulate,
)

from .common import ART, ROOT, SEED, write_json


MODELS = [
    "B8_lightgbm_quantile", "B5_linear_quantile", "B7_xgb_quantile",
    "B10_gru", "RAHQ_tree", "RAHQ_C_tree",
]


def _path(model: str):
    return ROOT / "artifacts" / "predictions" / f"core_B_1q_final_locked_{model}.parquet"


def _profiles() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    predictions = {model: pd.read_parquet(_path(model)) for model in MODELS if _path(model).exists()}
    if len(predictions) < 5:
        raise RuntimeError("Required stress attribution models are unavailable")
    latest = min(frame.report_period.max() for frame in predictions.values())
    common_ids = None
    for frame in predictions.values():
        ids = set(frame.loc[frame.report_period.eq(latest), "bank_id"])
        common_ids = ids if common_ids is None else common_ids & ids
    champion = predictions["B8_lightgbm_quantile"]
    champion = champion.loc[champion.report_period.eq(latest) & champion.bank_id.isin(common_ids)].copy()
    source = pd.read_parquet(ROOT / "data" / "features" / "core_features.parquet",
                             columns=["bank_id", "report_period", "cash", "securities", "borrowings", "equity", "loan_deposit_ratio"])
    states = champion.merge(source, on=["bank_id", "report_period"], validate="one_to_one")
    states["loans_proxy"] = states.loan_deposit_ratio * states.deposits
    states = states.loc[states[["deposits", "assets", "cash", "securities", "borrowings", "equity", "loans_proxy"]].notna().all(axis=1)]
    states["asset_decile"] = pd.qcut(states.assets.rank(method="first"), 10, labels=False)
    states = states.groupby("asset_decile", observed=True).head(15).sort_values("bank_id").reset_index(drop=True)
    aligned = {}
    for model, frame in predictions.items():
        part = frame.loc[frame.report_period.eq(latest)].set_index("bank_id").loc[states.bank_id].reset_index()
        aligned[model] = part
    return states, aligned


def run_stress_attribution() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    states, predictions = _profiles()
    shape, shape_meta = _h8_shape(ROOT)
    raw, corr, joint_meta = _reference_joint(ROOT)
    draws = 300
    rng = np.random.default_rng(SEED)
    uniforms = _joint_uniform("J1_gaussian", len(states) * draws, corr, raw, rng).reshape(len(states), draws, 3)
    profile_hash = hashlib.sha256(pd.util.hash_pandas_object(states[["bank_id", "report_period"]], index=False).values.tobytes()).hexdigest()
    draw_hash = hashlib.sha256(uniforms.tobytes()).hexdigest()
    macro = pd.read_parquet(ROOT / "data" / "processed" / "macro_origins.parquet",
                            columns=["report_period", "macro_DPSACBW027NBOG_qoq_revised"])
    market_shift = float(macro.loc[macro.report_period.between("2001Q1", "2023Q4"), "macro_DPSACBW027NBOG_qoq_revised"].quantile(.01))
    rows = []
    for model, pred in predictions.items():
        summary, _ = _simulate(states, pred[QCOLS].to_numpy(float), uniforms, raw, shape,
                               scenario="combined", market_shift_log=market_shift)
        for state_row, result in zip(states.itertuples(index=False), summary.itertuples(index=False)):
            state = BankState(state_row.deposits, state_row.assets, state_row.cash, state_row.securities,
                              state_row.borrowings, state_row.equity, state_row.loans_proxy)
            gap = max(float(result.p95_gap_30d), 0)
            recovery = optimize_recovery(state, gap, horizon_days=30, securities_haircut=.20, secured_access_reduction=.50)
            threshold = reverse_runoff_threshold(state, Scenario(0, securities_haircut=.20, secured_access_reduction=.50),
                                                  horizon_index=1, weekly_shape=shape)
            rows.append({"analysis": "model_choice", "model": model, "bank_id": state_row.bank_id,
                         "profile_id": f"{state_row.bank_id}_{state_row.report_period}", "scenario_id": "v1_combined_base",
                         "peak_funding_need_proxy": float(result.p95_peak_funding_need),
                         "shortfall_probability": float(result.shortfall_probability),
                         "maximum_shortfall": float(max(result.p95_gap_7d, result.p95_gap_30d, result.p95_gap_90d)),
                         "buffer_utilization": float(result.mean_max_buffer_utilization),
                         "liquidity_runway_proxy": float(result.median_runway_proxy_days),
                         "recovery_cost": float(recovery["action_cost"]),
                         "recovery_residual_shortfall": float(recovery["residual_shortfall"]),
                         "optimized_action_mix": json.dumps(recovery.get("action_amounts", [])),
                         "reverse_stress_threshold": threshold,
                         "profile_hash": profile_hash, "draw_hash": draw_hash,
                         "actual_v1_engine": True})
    model_results = pd.DataFrame(rows)

    # Actual-engine assumption design: fixed LHS points reused for every model/profile.
    sampler = qmc.LatinHypercube(d=7, seed=SEED)
    design = qmc.scale(sampler.random(64),
                       [.75, .20, .05, .01, 0.0, .50, .02],
                       [1.35, .80, .35, .08, .06, 1.00, .12])
    scenario_rows = []
    for scenario_index, values in enumerate(design):
        runoff_mult, access, haircut, commitment, capital_loss, recovery_capacity, recovery_unit_cost = values
        for model, pred in predictions.items():
            q05_by_bank = pred.set_index("bank_id").q05
            outcomes = []
            costs = []
            for state_row in states.itertuples(index=False):
                state = BankState(state_row.deposits, state_row.assets, state_row.cash, state_row.securities,
                                  state_row.borrowings, state_row.equity, state_row.loans_proxy)
                runoff = min(max(1 - np.exp(float(q05_by_bank.loc[state_row.bank_id]) + market_shift), 0) * runoff_mult, 1)
                scenario = Scenario(runoff, borrowing_substitution_fraction_assets=.01,
                                    liquidity_deterioration_fraction_assets=.01,
                                    securities_haircut=haircut, secured_access_reduction=access,
                                    commitment_draw_fraction_loans=commitment,
                                    capital_asset_loss_fraction=capital_loss)
                outcome = calculate(state, scenario, weekly_shape=shape)
                recovery = optimize_recovery(state, outcome["funding_gap"][1], horizon_days=30,
                                             securities_haircut=haircut, secured_access_reduction=access)
                # Capacity and cost overlays are applied only to the actual optimizer's returned action capacities.
                action = np.asarray(recovery.get("action_amounts", [0] * 4), float) * recovery_capacity
                residual = max(outcome["funding_gap"][1] - action.sum(), 0)
                costs.append(float(action.sum() * recovery_unit_cost + residual * recovery_unit_cost * 2))
                outcomes.append(outcome)
            scenario_rows.append({
                "analysis": "model_x_scenario", "scenario_id": f"lhs_{scenario_index:03d}", "model": model,
                "deposit_runoff_multiplier": runoff_mult, "funding_access_constraint": access,
                "securities_haircut": haircut, "commitment_draw_proxy": commitment,
                "capital_liquidity_interaction": capital_loss, "recovery_capacity": recovery_capacity,
                "recovery_cost_rate": recovery_unit_cost, "dependence_specification": "J1_gaussian common base",
                "peak_funding_need_proxy": float(np.mean([o["peak_funding_need_proxy"] for o in outcomes])),
                "shortfall_probability": float(np.mean([o["peak_funding_need_proxy"] > 0 for o in outcomes])),
                "maximum_shortfall": float(np.mean([o["maximum_liquidity_shortfall"] for o in outcomes])),
                "buffer_utilization": float(np.mean([o["buffer_utilization_max"] for o in outcomes])),
                "liquidity_runway_proxy": float(np.mean([o["liquidity_runway_proxy_days"] for o in outcomes])),
                "recovery_cost": float(np.mean(costs)), "profiles": len(states), "actual_v1_engine": True,
                "profile_hash": profile_hash, "draw_hash": draw_hash,
            })
    scenarios = pd.DataFrame(scenario_rows)

    # Dependence is varied through the v1 engine's own joint-uniform implementation.
    champion = predictions["B8_lightgbm_quantile"]
    for kind in ["J0_independent", "J1_gaussian", "J2_student_t", "J3_empirical_block"]:
        local_rng = np.random.default_rng(SEED)
        u = _joint_uniform(kind, len(states) * draws, corr, raw, local_rng).reshape(len(states), draws, 3)
        summary, _ = _simulate(states, champion[QCOLS].to_numpy(float), u, raw, shape,
                               scenario="combined", market_shift_log=market_shift)
        scenarios = pd.concat([scenarios, pd.DataFrame([{
            "analysis": "dependence_oat", "scenario_id": kind, "model": "B8_lightgbm_quantile",
            "dependence_specification": kind, "peak_funding_need_proxy": float(summary.p95_peak_funding_need.mean()),
            "shortfall_probability": float(summary.shortfall_probability.mean()),
            "maximum_shortfall": float(summary[["p95_gap_7d", "p95_gap_30d", "p95_gap_90d"]].max(axis=1).mean()),
            "buffer_utilization": float(summary.mean_max_buffer_utilization.mean()),
            "liquidity_runway_proxy": float(summary.median_runway_proxy_days.mean()),
            "recovery_cost": np.nan, "profiles": len(states), "actual_v1_engine": True,
            "profile_hash": profile_hash, "draw_hash": hashlib.sha256(u.tobytes()).hexdigest(),
        }])], ignore_index=True)

    balanced = scenarios.loc[scenarios.analysis.eq("model_x_scenario")]
    pivot = balanced.pivot(index="model", columns="scenario_id", values="peak_funding_need_proxy")
    grand = float(pivot.to_numpy().mean())
    ss_model = float(pivot.shape[1] * np.sum((pivot.mean(axis=1) - grand) ** 2))
    ss_scenario = float(pivot.shape[0] * np.sum((pivot.mean(axis=0) - grand) ** 2))
    residual = pivot.to_numpy() - pivot.mean(axis=1).to_numpy()[:, None] - pivot.mean(axis=0).to_numpy()[None, :] + grand
    ss_interaction = float(np.sum(residual ** 2))
    total = ss_model + ss_scenario + ss_interaction
    decomposition = {
        "outcome": "mean peak funding need proxy across common profiles",
        "model_variation_share": ss_model / total if total else 0,
        "scenario_assumption_variation_share": ss_scenario / total if total else 0,
        "model_scenario_interaction_share": ss_interaction / total if total else 0,
        "sum_squares": {"model": ss_model, "scenario": ss_scenario, "interaction": ss_interaction},
        "scenario_dominates_model": ss_scenario > ss_model,
        "profiles": len(states), "models": list(pivot.index), "scenarios": pivot.shape[1],
        "profile_hash": profile_hash, "common_draw_hash": draw_hash,
        "v1_engine_functions": ["treasuryshield.stress.run._simulate", "stress.arithmetic.calculate",
                                "stress.arithmetic.optimize_recovery", "stress.arithmetic.reverse_runoff_threshold"],
        "h8_shape": shape_meta, "joint_reference": joint_meta,
    }
    model_results.to_parquet(ART / "stress_engine_attribution.parquet", index=False)
    scenarios.to_parquet(ART / "stress_scenario_design.parquet", index=False)
    write_json(ART / "stress_variance_decomposition.json", decomposition)
    return model_results, scenarios, decomposition

