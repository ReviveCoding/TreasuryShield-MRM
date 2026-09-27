"""Historical, availability-aware tail-monitoring and retraining policy study."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.splits.temporal import deterministic_sample, load_joined


POLICIES = ("static", "annual_retrain", "quarterly_recalibration",
            "event_triggered", "performance_triggered")


def _train(history: pd.DataFrame) -> tuple[object, NumericPreprocessor, int]:
    sample = deterministic_sample(history, 120_000, 1701)
    pre = NumericPreprocessor(list(CORE_NUMERIC)).fit(sample)
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=.05, device_type="cpu", n_estimators=160,
        num_leaves=31, max_depth=6, min_child_samples=100,
        learning_rate=.06, subsample=.85, colsample_bytree=.85,
        reg_lambda=10, random_state=1701, n_jobs=8, verbosity=-1)
    model.fit(pre.transform(sample), sample.deposit_log_growth.to_numpy(dtype=float))
    return model, pre, len(sample)


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < 100 or len(current) < 100:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.maximum(np.histogram(reference, bins=edges)[0] / len(reference), 1e-6)
    b = np.maximum(np.histogram(current, bins=edges)[0] / len(current), 1e-6)
    return float(np.sum((b - a) * np.log(b / a)))


def run(root: Path) -> None:
    output = root / "artifacts" / "monitoring_prelock.json"
    predictions_path = root / "artifacts" / "monitoring_predictions_prelock.parquet"
    config = {"mode": "B", "horizon": 1, "period_start": "2017Q1",
              "period_end": "2023Q4", "model": "LightGBM Q05",
              "sample_max": 120_000, "seed": 1701,
              "snapshot": RunState(root).data["data_snapshot_id"],
              "source_hash": sha256_file(Path(__file__))}

    def execute() -> None:
        frame = load_joined(root, "core", 1, mode="B")
        frame = frame.loc[frame.report_period.le(config["period_end"])].copy()
        periods = sorted(frame.loc[frame.report_period.between(
            config["period_start"], config["period_end"]), "report_period"].unique())
        if len(periods) != 28:
            raise ValueError(f"Expected 28 monitoring quarters, found {len(periods)}")
        profiles: dict[str, tuple[object, NumericPreprocessor, int] | None] = {
            policy: None for policy in POLICIES}
        last_fit: dict[str, str] = {}
        forecasts: list[pd.DataFrame] = []
        timeline: list[dict] = []
        counts = {policy: 0 for policy in POLICIES}
        base = frame.loc[frame.report_period.lt("2017Q1")]
        reference_cash = base.cash_asset_ratio.to_numpy(dtype=float)
        reference_size = base.log_assets.to_numpy(dtype=float)
        for period in periods:
            current = frame.loc[frame.report_period.eq(period)].copy()
            origin = current.forecast_origin.min()
            if current.forecast_origin.nunique() != 1:
                raise ValueError(f"Mixed forecast origins in {period}")
            available = frame.loc[frame.target_availability_date.le(origin)
                                  & frame.report_period.lt(period)].copy()
            if available.empty:
                raise ValueError(f"No mature labels before {period}")
            matured = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
            if not matured.empty:
                matured = matured.loc[matured.target_availability_date.le(origin)]
            event = bool(current.macro_VIXCLS_revised.median() > 30 or
                         current.macro_DFF_change_4q_revised.median() > 2)
            for policy in POLICIES:
                historical_policy = (matured.loc[matured.policy.eq(policy)]
                                     if not matured.empty else pd.DataFrame())
                recent_periods = (sorted(historical_policy.report_period.unique())[-4:]
                                  if not historical_policy.empty else [])
                recent = (historical_policy.loc[historical_policy.report_period.isin(recent_periods)]
                          if recent_periods else pd.DataFrame())
                recent_coverage = (float(np.mean(recent.actual <= recent.q05))
                                   if not recent.empty else float("nan"))
                recent_pinball = (float(np.maximum(.05 * (recent.actual - recent.q05),
                                                   -.95 * (recent.actual - recent.q05)).mean())
                                  if not recent.empty else float("nan"))
                recent_residual_mean = (float(np.mean(recent.actual - recent.q05))
                                        if not recent.empty else float("nan"))
                recent_tail_severity = (float(np.maximum(recent.q05 - recent.actual, 0).mean())
                                        if not recent.empty else float("nan"))
                first = profiles[policy] is None
                annual = period.endswith("Q1")
                performance = (len(recent_periods) >= 4 and np.isfinite(recent_coverage)
                               and (recent_coverage < .02 or recent_coverage > .08))
                event_cooldown = (policy in last_fit and
                                  pd.Period(period, freq="Q").ordinal -
                                  pd.Period(last_fit[policy], freq="Q").ordinal >= 2)
                refit = (first or (policy == "annual_retrain" and annual)
                         or (policy == "event_triggered" and event and event_cooldown)
                         or (policy == "performance_triggered" and performance and event_cooldown))
                if policy == "quarterly_recalibration":
                    refit = first
                if refit:
                    profiles[policy] = _train(available)
                    last_fit[policy] = period
                    counts[policy] += 1
                model, pre, training_rows = profiles[policy]
                estimate = model.predict(pre.transform(current))
                shift = 0.0
                if policy == "quarterly_recalibration":
                    static_mature = (matured.loc[matured.policy.eq(policy)]
                                     if not matured.empty else pd.DataFrame())
                    if not static_mature.empty:
                        recent_q = sorted(static_mature.report_period.unique())[-8:]
                        residuals = static_mature.loc[static_mature.report_period.isin(recent_q),
                                                       "actual"].to_numpy() - static_mature.loc[
                            static_mature.report_period.isin(recent_q), "raw_q05"].to_numpy()
                        if len(residuals) >= 100:
                            shift = float(np.quantile(residuals, .05))
                forecast = current[["bank_id", "report_period", "forecast_origin",
                                    "target_availability_date", "deposit_log_growth"]].rename(
                    columns={"deposit_log_growth": "actual"}).copy()
                forecast["policy"] = policy
                forecast["raw_q05"] = estimate
                forecast["q05"] = estimate + shift
                forecasts.append(forecast)
                cash_psi = _psi(reference_cash, current.cash_asset_ratio.to_numpy(dtype=float))
                size_psi = _psi(reference_size, current.log_assets.to_numpy(dtype=float))
                # Project-internal governance thresholds, fixed for this prelock study.
                if np.isfinite(recent_coverage) and recent_coverage > .12:
                    state = "REDEVELOP"
                elif np.isfinite(recent_coverage) and (recent_coverage > .08 or recent_coverage < .02):
                    state = "RECALIBRATE"
                elif event and max(cash_psi, size_psi) > .25:
                    state = "REVIEW"
                elif event or max(cash_psi, size_psi) > .10:
                    state = "WATCH"
                else:
                    state = "GREEN"
                timeline.append({"report_period": period, "policy": policy,
                                 "state": state, "event_regime": event,
                                 "refit": refit, "train_rows": training_rows,
                                 "recent_matured_q05_coverage": recent_coverage,
                                 "recent_matured_q05_pinball": recent_pinball,
                                 "recent_matured_residual_mean": recent_residual_mean,
                                 "recent_matured_tail_severity": recent_tail_severity,
                                 "cash_asset_psi": cash_psi, "log_assets_psi": size_psi,
                                 "calibration_shift": shift, "forecast_rows": len(forecast)})
            print(f"Monitoring {period}: {len(current)} origins", flush=True)
        all_forecasts = pd.concat(forecasts, ignore_index=True)
        all_forecasts.to_parquet(predictions_path, index=False, compression="zstd")
        summary = {}
        for policy, part in all_forecasts.groupby("policy"):
            error = part.actual.to_numpy() - part.q05.to_numpy()
            quarter_loss = part.assign(loss=np.maximum(.05 * error, -.95 * error)).groupby(
                "report_period").loss.mean()
            retrospective_bad = quarter_loss > 1.5 * quarter_loss.median()
            monitored = [row for row in timeline if row["policy"] == policy]
            alerts = {row["report_period"]: row["state"] != "GREEN" for row in monitored}
            false_alerts = sum(alerts[q] and not retrospective_bad.loc[q] for q in quarter_loss.index)
            missed = sum(not alerts[q] and retrospective_bad.loc[q] for q in quarter_loss.index)
            ordered_quarters = list(quarter_loss.index)
            alert_array = np.array([alerts[q] for q in ordered_quarters], dtype=bool)
            bad_array = retrospective_bad.to_numpy(dtype=bool)
            bad_starts = [index for index, bad in enumerate(bad_array)
                          if bad and (index == 0 or not bad_array[index - 1])]
            detection_delays = []
            lead_times = []
            for start in bad_starts:
                stop = next((index for index in range(start + 1, len(bad_array))
                             if not bad_array[index]), len(bad_array))
                detected = np.flatnonzero(alert_array[start:stop])
                if len(detected):
                    detection_delays.append(int(detected[0]))
                prior = np.flatnonzero(alert_array[max(0, start - 4):start])
                if len(prior):
                    lead_times.append(int(start - (max(0, start - 4) + prior[-1])))
            durations = []
            active = 0
            for flagged in alert_array:
                if flagged:
                    active += 1
                elif active:
                    durations.append(active)
                    active = 0
            if active:
                durations.append(active)
            summary[policy] = {
                "q05_pinball": float(np.maximum(.05 * error, -.95 * error).mean()),
                "q05_coverage": float(np.mean(error <= 0)),
                "fit_count": counts[policy], "alert_quarters": sum(alerts.values()),
                "false_alert_quarters_retrospective": int(false_alerts),
                "missed_bad_quarters_retrospective": int(missed),
                "mean_detection_delay_quarters_when_detected": float(np.mean(detection_delays))
                if detection_delays else None,
                "mean_alert_lead_quarters_when_prior_alert": float(np.mean(lead_times))
                if lead_times else None,
                "mean_alert_duration_quarters": float(np.mean(durations)) if durations else 0.0,
                "calibration_update_quarters": sum(row["calibration_shift"] != 0
                                                   for row in monitored),
                "bad_quarter_definition": "quarter Q05 pinball > 1.5 x that policy's 2017-23 median",
            }
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "config_hash": stable_hash(config), "data_snapshot_id": config["snapshot"],
                   "start": periods[0], "end": periods[-1], "forecast_rows": len(all_forecasts),
                   "policy_summary": summary, "timeline": timeline,
                   "limitations": ["Historical backtest; alert thresholds are project-defined research thresholds.",
                                   "The retraining comparison models Q05 only, with the same LightGBM family as B8.",
                                   "False-alert and missed-quarter labels are retrospective, not contemporaneous forecasts."]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("DELETE FROM monitoring WHERE run_id='monitoring_prelock'")
            for item in timeline:
                for metric in ("recent_matured_q05_coverage", "recent_matured_q05_pinball",
                               "recent_matured_residual_mean", "recent_matured_tail_severity",
                               "cash_asset_psi", "log_assets_psi", "calibration_shift", "train_rows"):
                    value = item[metric]
                    if not np.isfinite(value):
                        continue
                    con.execute("INSERT INTO monitoring VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                ["monitoring_prelock", item["report_period"], item["policy"],
                                 item["state"], metric, value, item["state"] != "GREEN",
                                 "artifacts/monitoring_prelock.json"])
            for policy, item in summary.items():
                for metric in ("q05_pinball", "q05_coverage", "fit_count", "alert_quarters",
                               "false_alert_quarters_retrospective", "missed_bad_quarters_retrospective",
                               "mean_detection_delay_quarters_when_detected",
                               "mean_alert_lead_quarters_when_prior_alert",
                               "mean_alert_duration_quarters", "calibration_update_quarters"):
                    if item[metric] is None:
                        continue
                    con.execute("INSERT INTO monitoring VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                ["monitoring_prelock", "2017Q1-2023Q4", policy,
                                 "SUMMARY", metric, item[metric], False,
                                 "artifacts/monitoring_prelock.json"])
            con.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id='E36'",
                        ["artifacts/monitoring_prelock.json"])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print(f"Monitoring complete: {len(all_forecasts)} forecasts", flush=True)

    RunState(root).run("monitoring_prelock", "RUNNING_EXPERIMENTS", config,
                       ["artifacts/monitoring_prelock.json",
                        "artifacts/monitoring_predictions_prelock.parquet"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
