"""Public-data scenario, dependence, reverse-stress, and recovery studies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.stress.arithmetic import (
    BankState, Scenario, calculate, heuristic_recovery,
    optimize_recovery, reverse_runoff_threshold,
)

LEVELS = np.array([.01, .05, .10, .25, .50, .75, .90])
QCOLS = ["q01", "q05", "q10", "q25", "q50", "q75", "q90"]


def _h8_shape(root: Path) -> tuple[np.ndarray, dict]:
    weekly = pd.read_csv(root / "data" / "raw" / "fred" / "DPSACBW027NBOG.csv")
    weekly["observation_date"] = pd.to_datetime(weekly["observation_date"])
    weekly["DPSACBW027NBOG"] = pd.to_numeric(weekly["DPSACBW027NBOG"], errors="coerce")
    weekly = weekly.loc[weekly["observation_date"].between("2001-01-01", "2023-12-31")].dropna().reset_index(drop=True)
    values = weekly["DPSACBW027NBOG"].to_numpy()
    rolling = (values[13:] - values[:-13]) / values[:-13]
    worst = int(np.argmin(rolling))
    start = values[worst]
    cumulative = np.maximum(start - values[worst + 1:worst + 14], 0)
    cumulative = np.maximum.accumulate(cumulative)
    if cumulative[-1] <= 0:
        shape = np.arange(1, 14, dtype=float) / 13
        fallback = True
    else:
        shape = cumulative / cumulative[-1]
        fallback = False
    return shape, {"source": "FRED H.8 aggregate DPSACBW027NBOG",
                   "historical_window": "2001-01-01 to 2023-12-31",
                   "worst_13_week_start": str(weekly.loc[worst, "observation_date"].date()),
                   "worst_13_week_change": float(rolling[worst]),
                   "linear_fallback": fallback}


def _reference_joint(root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    targets = pd.read_parquet(root / "data" / "features" / "targets.parquet",
                              columns=["report_period", "horizon_quarters", "deposit_log_growth",
                                       "borrowings_change_asset_scaled", "liquid_broad_change",
                                       "mna_contaminated_radius_1"])
    x = targets.loc[targets["report_period"].between("2001Q1", "2023Q4")
                    & targets["horizon_quarters"].eq(1)
                    & targets["mna_contaminated_radius_1"].eq(False),
                    ["report_period", "deposit_log_growth", "borrowings_change_asset_scaled", "liquid_broad_change"]].dropna()
    x = x.sample(n=min(len(x), 200_000), random_state=17)
    raw = np.column_stack([-x["deposit_log_growth"].to_numpy(),
                            x["borrowings_change_asset_scaled"].to_numpy(),
                            -x["liquid_broad_change"].to_numpy(),
                            x["report_period"].str[:4].astype(int).to_numpy() * 4 +
                            x["report_period"].str[-1].astype(int).to_numpy() - 1])
    transformed = np.column_stack([stats.norm.ppf((stats.rankdata(raw[:, j]) - .5) / len(raw))
                                   for j in range(3)])
    corr = np.corrcoef(transformed, rowvar=False)
    values, vectors = np.linalg.eigh(corr)
    corr = vectors @ np.diag(np.maximum(values, .05)) @ vectors.T
    scale = np.sqrt(np.diag(corr))
    corr = corr / scale[:, None] / scale[None, :]
    return raw, corr, {"reference_rows": len(raw), "window": "2001Q1-2023Q4, M&A clean",
                       "correlation": corr.tolist(), "variables": ["deposit_outflow_log", "borrowing_change/assets", "liquid_ratio_decline"]}


def _joint_uniform(kind: str, count: int, corr: np.ndarray, raw: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    if kind == "J0_independent":
        return rng.random((count, 3))
    if kind in {"J1_gaussian", "J2_student_t"}:
        z = rng.multivariate_normal(np.zeros(3), corr, size=count)
        if kind == "J2_student_t":
            df = 4
            z = z / np.sqrt(rng.chisquare(df, size=count)[:, None] / df)
            return stats.t.cdf(z, df)
        return stats.norm.cdf(z)
    if kind == "J3_empirical_block":
        periods = raw[:, 3].astype(int)
        starts = np.unique(periods)
        drawn_starts = rng.choice(starts, size=count)
        chosen = np.empty(count, dtype=int)
        for start in np.unique(drawn_starts):
            where = np.flatnonzero(drawn_starts == start)
            pool = np.flatnonzero((periods >= start) & (periods < start + 4))
            chosen[where] = rng.choice(pool, size=len(where))
        sampled = raw[chosen]
        return np.column_stack([np.searchsorted(np.sort(raw[:, j]), sampled[:, j], side="right") / len(raw)
                                for j in range(3)])
    raise ValueError(kind)


def _predictive_log_growth(quantiles: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    out = np.empty_like(uniforms)
    for index in range(len(quantiles)):
        out[index] = np.interp(uniforms[index], LEVELS, quantiles[index],
                               left=quantiles[index, 0], right=quantiles[index, -1])
    return out


def _simulate(states: pd.DataFrame, quantiles: np.ndarray, uniforms: np.ndarray,
              ref: np.ndarray, shape: np.ndarray, *, scenario: str,
              market_shift_log: float, securities_available: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    n, draws = uniforms.shape[:2]
    log_growth = _predictive_log_growth(quantiles, uniforms[:, :, 0])
    funding = np.interp(uniforms[:, :, 1], np.linspace(0, 1, len(ref)), np.sort(ref[:, 1]))
    buffer_decline = np.interp(uniforms[:, :, 2], np.linspace(0, 1, len(ref)), np.sort(ref[:, 2]))
    macro_shift = market_shift_log if scenario in {"market", "combined"} else 0.0
    if scenario == "institution":
        # Conditioning on severe forecast-tail draws is a scenario assumption.
        log_growth = np.minimum(log_growth, quantiles[:, 2:3])
    elif scenario == "combined":
        log_growth = np.minimum(log_growth, quantiles[:, 2:3])
    haircut = {"institution": .10, "market": .15, "combined": .20}[scenario]
    funding_reduction = {"institution": 0.0, "market": .25, "combined": .50}[scenario]
    deposits = states["deposits"].to_numpy()[:, None]
    assets = states["assets"].to_numpy()[:, None]
    loans = states["loans_proxy"].to_numpy()[:, None]
    securities = states["securities"].to_numpy()[:, None]
    cash = states["cash"].to_numpy()[:, None]
    runoff90 = deposits * np.maximum(1 - np.exp(log_growth + macro_shift), 0)
    borrowing90 = assets * np.maximum(funding, 0)
    liquid90 = assets * np.maximum(buffer_decline, 0)
    commitment90 = loans * .02
    collateral = securities * (1 - haircut)
    monetizable = collateral * (.5 + .5 * .7 * (1 - funding_reduction))
    if not securities_available:
        monetizable = np.zeros_like(monetizable)
    fractions = shape[[0, 3, 12]] / shape[-1]
    activation = np.array([.25, .70, 1.0])
    gap = np.empty((n, draws, 3), dtype=float)
    utilization = np.empty((n, draws, 3), dtype=float)
    for j in range(3):
        outflows = (runoff90 + borrowing90 + liquid90 + commitment90) * fractions[j]
        resources = cash + monetizable * activation[j]
        gap[:, :, j] = outflows - resources
        utilization[:, :, j] = outflows / np.maximum(resources, 1e-9)
    peak = np.maximum(gap.max(axis=2), 0)
    runway = np.where(gap[:, :, 0] > 0, 7,
                      np.where(gap[:, :, 1] > 0, 30,
                               np.where(gap[:, :, 2] > 0, 90, 91)))
    detail = pd.DataFrame({
        "bank_id": np.repeat(states["bank_id"].to_numpy(), draws),
        "scenario": scenario,
        "peak_funding_need_proxy": peak.ravel(),
        "gap_7d": gap[:, :, 0].ravel(),
        "gap_30d": gap[:, :, 1].ravel(),
        "gap_90d": gap[:, :, 2].ravel(),
    })
    summary = pd.DataFrame({
        "bank_id": states["bank_id"].to_numpy(), "scenario": scenario,
        "shortfall_probability": (peak > 0).mean(axis=1),
        "p50_peak_funding_need": np.quantile(peak, .5, axis=1),
        "p95_peak_funding_need": np.quantile(peak, .95, axis=1),
        "mean_peak_funding_need": peak.mean(axis=1),
        "mean_max_buffer_utilization": utilization.max(axis=2).mean(axis=1),
        "median_runway_proxy_days": np.median(runway, axis=1),
        "p95_gap_7d": np.quantile(np.maximum(gap[:, :, 0], 0), .95, axis=1),
        "p95_gap_30d": np.quantile(np.maximum(gap[:, :, 1], 0), .95, axis=1),
        "p95_gap_90d": np.quantile(np.maximum(gap[:, :, 2], 0), .95, axis=1),
    })
    return summary, detail


def build(root: Path, prediction_run_id: str, simulations: int = 500) -> None:
    path = root / "artifacts" / "predictions" / f"{prediction_run_id}.parquet"
    prediction = pd.read_parquet(path)
    latest = prediction["report_period"].max()
    prediction = prediction.loc[prediction["report_period"].eq(latest)].copy()
    source = pd.read_parquet(root / "data" / "features" / "core_features.parquet",
                             columns=["bank_id", "report_period", "cash", "securities", "borrowings",
                                      "equity", "loan_deposit_ratio"])
    states = prediction.merge(source, on=["bank_id", "report_period"], validate="one_to_one")
    states["loans_proxy"] = states["loan_deposit_ratio"] * states["deposits"]
    states = states.loc[states[["deposits", "assets", "cash", "securities", "equity", "loans_proxy"]].notna().all(axis=1)
                        & states["assets"].gt(0) & states["deposits"].gt(0)].copy()
    states = states.sort_values("bank_id").reset_index(drop=True)
    shape, shape_meta = _h8_shape(root)
    joint_raw, corr, joint_meta = _reference_joint(root)
    macro = pd.read_parquet(root / "data" / "processed" / "macro_origins.parquet",
                            columns=["report_period", "macro_DPSACBW027NBOG_qoq_revised"])
    market_shift = float(macro.loc[macro["report_period"].between("2001Q1", "2023Q4"),
                                   "macro_DPSACBW027NBOG_qoq_revised"].quantile(.01))
    config = {"prediction_run_id": prediction_run_id, "prediction_sha256": sha256_file(path),
              "draws_per_bank": simulations, "seed": 1701,
              "scenario_draw_coupling": "Common J1 Gaussian uniforms across institution, market, and combined scenarios",
              "h8_weekly_shape": shape_meta, "joint_reference": joint_meta,
              "market_shift_log": market_shift,
              "scenario_haircuts": {"institution": .10, "market": .15, "combined": .20},
              "secured_access_reduction": {"institution": 0, "market": .25, "combined": .50},
              "commitment_draw_loans": .02,
              "collateral_allocation": "50% sold, 50% pledged at 70% advance; activation 25/70/100% by 7/30/90 days",
              "scope": "research proxies, not regulatory liquidity measures"}
    config_hash = stable_hash(config)
    quantiles = states[QCOLS].to_numpy(dtype=float)
    rng = np.random.default_rng(1701)
    all_summaries = []
    samples = []
    scenario_uniforms = _joint_uniform(
        "J1_gaussian", len(states) * simulations, corr, joint_raw, rng
    ).reshape(len(states), simulations, 3)
    for scenario in ["institution", "market", "combined"]:
        summary, detail = _simulate(states, quantiles, scenario_uniforms, joint_raw, shape, scenario=scenario,
                                    market_shift_log=market_shift)
        all_summaries.append(summary)
        # Retain distribution draws for a reproducible stratified profile sample.
        sample_ids = set(states.groupby(pd.qcut(states["assets"].rank(method="first"), 10), observed=True).head(10)["bank_id"])
        samples.append(detail.loc[detail["bank_id"].isin(sample_ids)].copy())
    base_summary = pd.concat(all_summaries, ignore_index=True)
    ordered = base_summary.pivot(index="bank_id", columns="scenario",
                                 values="p95_peak_funding_need")
    if (ordered["combined"] + 1e-8 < ordered[["institution", "market"]].max(axis=1)).any():
        raise AssertionError("Combined stress is less severe on shared draws")
    sampled_states = states.groupby(pd.qcut(states["assets"].rank(method="first"), 10), observed=True).head(25)
    dependency = []
    for kind in ["J0_independent", "J1_gaussian", "J2_student_t", "J3_empirical_block"]:
        u = _joint_uniform(kind, len(sampled_states) * simulations, corr, joint_raw, rng).reshape(len(sampled_states), simulations, 3)
        summary, _ = _simulate(sampled_states, sampled_states[QCOLS].to_numpy(), u, joint_raw, shape,
                               scenario="combined", market_shift_log=market_shift)
        dependency.append({"method": kind, "banks": len(sampled_states),
                           "mean_shortfall_probability": float(summary["shortfall_probability"].mean()),
                           "mean_p95_peak_funding_need": float(summary["p95_peak_funding_need"].mean())})
    u_liquidity = _joint_uniform("J1_gaussian", len(sampled_states) * simulations,
                                 corr, joint_raw, rng).reshape(len(sampled_states), simulations, 3)
    liquidity_definition_results = []
    for definition, securities_available in (("broad_cash_plus_securities", True),
                                             ("narrow_cash_only", False)):
        defined, _ = _simulate(sampled_states, sampled_states[QCOLS].to_numpy(),
                               u_liquidity, joint_raw, shape, scenario="combined",
                               market_shift_log=market_shift,
                               securities_available=securities_available)
        liquidity_definition_results.append({"definition": definition,
                                             "banks": len(sampled_states),
                                             "mean_p95_peak_funding_need": float(
                                                 defined["p95_peak_funding_need"].mean()),
                                             "mean_shortfall_probability": float(
                                                 defined["shortfall_probability"].mean())})
    reverse_rows = []
    recovery_rows = []
    capital_rows = []
    stress_borrowing = float(np.quantile(np.maximum(joint_raw[:, 1], 0), .95))
    stress_liquid_decline = float(np.quantile(np.maximum(joint_raw[:, 2], 0), .95))
    for row in sampled_states.itertuples(index=False):
        state = BankState(deposits=row.deposits, assets=row.assets, cash=row.cash,
                          securities=row.securities, borrowings=row.borrowings,
                          equity=row.equity, loans=row.loans_proxy)
        for haircut in [0.0, .10, .20, .30]:
            for access in [0.0, .25, .50]:
                base = Scenario(deposit_runoff_fraction_90d=0,
                                securities_haircut=haircut,
                                secured_access_reduction=access)
                threshold = reverse_runoff_threshold(state, base, horizon_index=1,
                                                     weekly_shape=shape)
                reverse_rows.append({"bank_id": row.bank_id, "haircut": haircut,
                                     "access_reduction": access,
                                     "break_even_runoff_fraction": threshold if np.isfinite(threshold) else np.nan,
                                     "reachable_within_100pct_runoff": bool(np.isfinite(threshold))})
        for loss in [0.0, .02, .05]:
            interaction = Scenario(
                deposit_runoff_fraction_90d=max(1 - np.exp(row.q05 + market_shift), 0),
                borrowing_substitution_fraction_assets=stress_borrowing,
                liquidity_deterioration_fraction_assets=stress_liquid_decline,
                securities_haircut=.20, secured_access_reduction=.25,
                capital_asset_loss_fraction=loss)
            outcome = calculate(state, interaction, weekly_shape=shape)
            capital_rows.append({"bank_id": row.bank_id, "asset_loss_fraction": loss,
                                 "peak_funding_need_proxy": outcome["peak_funding_need_proxy"],
                                 "stressed_capital_asset_ratio": outcome["stressed_capital_asset_ratio"],
                                 "effective_secured_access_reduction": outcome["effective_secured_access_reduction"]})
        for scenario in ["institution", "market", "combined"]:
            gap = float(base_summary.loc[base_summary["bank_id"].eq(row.bank_id)
                                         & base_summary["scenario"].eq(scenario), "p95_gap_30d"].iloc[0])
            haircut = config["scenario_haircuts"][scenario]
            access = config["secured_access_reduction"][scenario]
            optimized = optimize_recovery(state, gap, horizon_days=30,
                                          securities_haircut=haircut, secured_access_reduction=access)
            heuristic = heuristic_recovery(state, gap, horizon_days=30,
                                           securities_haircut=haircut, secured_access_reduction=access)
            for method, outcome in [("optimized", optimized), ("heuristic", heuristic)]:
                recovery_rows.append({"bank_id": row.bank_id, "scenario": scenario,
                                      "strategy": method, "funding_gap": gap,
                                      "action_cost": outcome["action_cost"],
                                      "residual_shortfall": outcome["residual_shortfall"],
                                      "status": outcome["status"]})
    output_dir = root / "artifacts" / "stress"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{prediction_run_id}_summary.parquet"
    draws_path = output_dir / f"{prediction_run_id}_sample_draws.parquet"
    reverse_path = output_dir / f"{prediction_run_id}_reverse.parquet"
    recovery_path = output_dir / f"{prediction_run_id}_recovery.parquet"
    capital_path = output_dir / f"{prediction_run_id}_capital.parquet"
    manifest_path = output_dir / f"{prediction_run_id}_manifest.json"
    base_summary.to_parquet(summary_path, index=False, compression="zstd")
    pd.concat(samples, ignore_index=True).to_parquet(draws_path, index=False, compression="zstd")
    pd.DataFrame(reverse_rows).to_parquet(reverse_path, index=False, compression="zstd")
    pd.DataFrame(recovery_rows).to_parquet(recovery_path, index=False, compression="zstd")
    pd.DataFrame(capital_rows).to_parquet(capital_path, index=False, compression="zstd")
    manifest_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_run_id": prediction_run_id, "latest_forecast_origin_period": latest,
        "banks": len(states), "profile_banks": len(sampled_states),
        "config": config, "config_hash": config_hash,
        "dependence_results": dependency,
        "liquidity_definition_results": liquidity_definition_results,
        "reverse_unreachable_fraction": float(pd.DataFrame(reverse_rows)["reachable_within_100pct_runoff"].eq(False).mean()),
        "limitations": ["No internal cash-flow, collateral, deposit concentration, or committed funding data",
                        "Quantile interpolation flat beyond Q01/Q90; scenario shifts represent unsupported extremes",
                        "Market shock and weekly shape from aggregate H.8, not bank-specific observations",
                        "Recovery capacities and costs are assumptions, not contractual rights"],
        "paths": [str(p.relative_to(root)).replace("\\", "/") for p in [summary_path, draws_path, reverse_path, recovery_path, capital_path]],
    }, indent=2), encoding="utf-8")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    con.execute("DELETE FROM stress_runs WHERE run_id=?", [prediction_run_id])
    con.execute("DELETE FROM reverse_stress WHERE run_id=?", [prediction_run_id])
    con.execute("DELETE FROM recovery_runs WHERE run_id=?", [prediction_run_id])
    for statistic in ["shortfall_probability", "p50_peak_funding_need", "p95_peak_funding_need",
                      "mean_peak_funding_need", "mean_max_buffer_utilization",
                      "median_runway_proxy_days", "p95_gap_7d", "p95_gap_30d", "p95_gap_90d"]:
        records = base_summary[["bank_id", "scenario", statistic]].copy()
        records["run_id"] = prediction_run_id
        records["horizon"] = (int(statistic.split("_")[-1].removesuffix("d"))
                              if statistic.startswith("p95_gap_") else 90)
        records["statistic"] = statistic
        records["config_hash"] = config_hash
        records["artifact_path"] = str(summary_path.relative_to(root)).replace("\\", "/")
        con.register("incoming_stress", records)
        con.execute("INSERT INTO stress_runs SELECT run_id, bank_id, scenario, horizon, statistic, " +
                    f'"{statistic}", config_hash, artifact_path FROM incoming_stress')
        con.unregister("incoming_stress")
    reverse_frame = pd.DataFrame(reverse_rows)
    reverse_frame["run_id"] = prediction_run_id
    reverse_frame["scenario"] = "30d_liquidity_break"
    reverse_frame["driver"] = reverse_frame.apply(lambda x: f"runoff|haircut={x.haircut:.2f}|access={x.access_reduction:.2f}", axis=1)
    reverse_frame["config_hash"] = config_hash
    reverse_frame["artifact_path"] = str(reverse_path.relative_to(root)).replace("\\", "/")
    con.register("incoming_reverse", reverse_frame)
    con.execute("INSERT INTO reverse_stress SELECT run_id, bank_id, scenario, driver, break_even_runoff_fraction, config_hash, artifact_path FROM incoming_reverse")
    con.unregister("incoming_reverse")
    recovery_frame = pd.DataFrame(recovery_rows)
    recovery_frame["run_id"] = prediction_run_id
    recovery_frame["config_hash"] = config_hash
    recovery_frame["artifact_path"] = str(recovery_path.relative_to(root)).replace("\\", "/")
    con.register("incoming_recovery", recovery_frame)
    con.execute("INSERT INTO recovery_runs SELECT run_id, bank_id, strategy, scenario, funding_gap, action_cost, residual_shortfall, config_hash, artifact_path FROM incoming_recovery")
    con.unregister("incoming_recovery")
    capital_frame = pd.DataFrame(capital_rows)
    capital_frame["run_id"] = prediction_run_id
    capital_frame["scenario"] = capital_frame["asset_loss_fraction"].map(lambda x: f"capital_loss_{x:.2f}")
    capital_frame["horizon"] = 90
    capital_frame["statistic"] = "peak_funding_need_proxy"
    capital_frame["config_hash"] = config_hash
    capital_frame["artifact_path"] = str(capital_path.relative_to(root)).replace("\\", "/")
    con.register("incoming_capital", capital_frame)
    con.execute("INSERT INTO stress_runs SELECT run_id, bank_id, scenario, horizon, statistic, peak_funding_need_proxy, config_hash, artifact_path FROM incoming_capital")
    con.unregister("incoming_capital")
    for item in dependency:
        for statistic in ("mean_shortfall_probability", "mean_p95_peak_funding_need"):
            con.execute("INSERT INTO stress_runs VALUES (?, -1, ?, 90, ?, ?, ?, ?)",
                        [prediction_run_id, item["method"], statistic, item[statistic],
                         config_hash, str(manifest_path.relative_to(root)).replace("\\", "/")])
    for item in liquidity_definition_results:
        for statistic in ("mean_shortfall_probability", "mean_p95_peak_funding_need"):
            con.execute("INSERT INTO stress_runs VALUES (?, -1, ?, 90, ?, ?, ?, ?)",
                        [prediction_run_id, item["definition"], statistic, item[statistic],
                         config_hash, str(manifest_path.relative_to(root)).replace("\\", "/")])
    unreachable = float(reverse_frame["break_even_runoff_fraction"].isna().mean())
    con.execute("INSERT INTO stress_runs VALUES (?, -1, 'reverse', 30, 'unreachable_fraction', ?, ?, ?)",
                [prediction_run_id, unreachable, config_hash,
                 str(manifest_path.relative_to(root)).replace("\\", "/")])
    for experiment in ["E31", "E32", "E33", "E34", "E35"]:
        con.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id=?",
                    [str(manifest_path.relative_to(root)).replace("\\", "/"), experiment])
    con.close()
    print(f"Stress study {prediction_run_id}: {len(states)} banks, {len(sampled_states)} dependence/recovery profiles", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-run-id", required=True)
    parser.add_argument("--simulations", type=int, default=500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    prefix = f"artifacts/stress/{args.prediction_run_id}"
    RunState(root).run(
        f"stress_{args.prediction_run_id}", "STRESS_TESTING",
        {"prediction_sha256": sha256_file(root / "artifacts" / "predictions" / f"{args.prediction_run_id}.parquet"),
         "simulations": args.simulations, "seed": 1701,
         "source_sha256": {name: sha256_file(root / "src" / "treasuryshield" / name)
                           for name in ["stress/run.py", "stress/arithmetic.py"]}},
        [f"{prefix}_summary.parquet", f"{prefix}_sample_draws.parquet",
         f"{prefix}_reverse.parquet", f"{prefix}_recovery.parquet",
         f"{prefix}_capital.parquet",
         f"{prefix}_manifest.json"],
        lambda: build(root, args.prediction_run_id, args.simulations), resume=True)


if __name__ == "__main__":
    main()
