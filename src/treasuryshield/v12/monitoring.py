from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .common import ART, ROOT, pinball, write_json


LEVELS = ["GREEN", "WATCH", "REVIEW", "RECALIBRATE", "REDEVELOP"]


def next_state(previous: str, severity: int, persistent_severe: int,
               structural_ood: int, robust_reversal: int, green_streak: int) -> str:
    """Frozen hysteretic transition rule; arguments are counts through current quarter."""
    if structural_ood >= 2 or robust_reversal >= 2 or (previous == "RECALIBRATE" and persistent_severe >= 2):
        return "REDEVELOP"
    if persistent_severe >= 2:
        return "RECALIBRATE"
    target = LEVELS[min(severity, 2)]
    if previous == "REDEVELOP":
        return "REDEVELOP" if green_streak < 4 else "REVIEW"
    if previous == "RECALIBRATE":
        return "RECALIBRATE" if green_streak < 2 else "WATCH"
    if LEVELS.index(target) > LEVELS.index(previous):
        return target
    if green_streak >= 2:
        return LEVELS[max(0, LEVELS.index(previous) - 1)]
    return previous


def _quarter_metrics(frame: pd.DataFrame, challenger: pd.DataFrame | None = None) -> pd.DataFrame:
    challenger_index = challenger.set_index(["bank_id", "report_period"]) if challenger is not None else None
    rows = []
    for quarter, sub in frame.groupby("report_period", sort=True):
        y, q = sub.actual.to_numpy(float), sub.q05.to_numpy(float)
        selected_loss = pinball(y, q, .05)
        delta = math.nan
        if challenger_index is not None:
            idx = pd.MultiIndex.from_frame(sub[["bank_id", "report_period"]])
            common = idx.isin(challenger_index.index)
            if common.any():
                other_q = challenger_index.loc[idx[common], "q05"].to_numpy(float)
                delta = float((selected_loss[common] - pinball(y[common], other_q, .05)).mean())
        rows.append({"quarter": quarter, "q05_coverage": float((y <= q).mean()),
                     "q05_coverage_deviation": float(abs((y <= q).mean() - .05)),
                     "q05_pinball": float(selected_loss.mean()),
                     "tail_severity": float(np.maximum(q - y, 0).mean()),
                     "model_challenger_delta": delta, "disagreement": float(sub.disagreement.mean()),
                     "ood": float(sub.ood_score.mean()), "missingness": float(sub.missing_count.mean()),
                     "n": len(sub)})
    return pd.DataFrame(rows)


def run_monitoring(pre: pd.DataFrame, final: pd.DataFrame, slice_inference: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    from treasuryshield.v11.common import read_prelock, read_prediction
    challenger_pre = read_prelock("B5_linear_quantile")
    challenger_final = read_prediction("B5_linear_quantile")
    history = _quarter_metrics(pre, challenger_pre)
    locked = _quarter_metrics(final, challenger_final)
    metrics = ["q05_coverage_deviation", "q05_pinball", "tail_severity", "disagreement", "ood", "missingness"]
    thresholds = {}
    for metric in metrics:
        values = history[metric].dropna()
        thresholds[metric] = {"watch": float(values.quantile(.90)), "review": float(values.quantile(.95)),
                              "severe": float(values.quantile(.99)), "source": "pre-2024 quarter distribution"}
    delta_values = history.model_challenger_delta.dropna()
    thresholds["model_challenger_delta"] = {
        "watch": float(delta_values.quantile(.90)), "review": float(delta_values.quantile(.95)),
        "severe": float(delta_values.quantile(.99)), "source": "pre-2024 B8-minus-B5 quarterly loss delta"}
    policy = {
        "schema_version": "1.2", "states": LEVELS, "fit_population": "pre-2024 only",
        "thresholds": thresholds,
        "rules": {
            "GREEN": "all monitored metrics below prelock 90th-percentile limits",
            "WATCH": "one moderate threshold breach",
            "REVIEW": "one review/severe breach or multiple moderate breaches",
            "RECALIBRATE": "review-or-worse condition persists for at least two quarters",
            "REDEVELOP": "RECALIBRATE persists, structural OOD repeats twice, or robust champion reversal repeats twice",
            "deescalation": "two consecutive GREEN-target quarters reduce one state; REDEVELOP requires four before REVIEW",
        },
        "adverse_event_definition": "prelock-frozen: at least one metric exceeds its prelock 95th percentile",
        "limitations": "Research controls; production thresholds require governance approval.",
    }
    write_json(ART / "monitoring_state_machine.json", policy)

    combined = pd.concat([history.assign(window="prelock"), locked.assign(window="locked")], ignore_index=True)
    previous = "GREEN"; severe_streak = structural_streak = reversal_streak = green_streak = 0
    output = []
    robust_reversal_exists = bool((slice_inference.classification == "challenger_wins_robustly").any())
    for row in combined.itertuples(index=False):
        severities = []
        for metric, threshold in thresholds.items():
            value = getattr(row, metric)
            if not np.isfinite(value):
                continue
            severities.append(3 if value > threshold["severe"] else 2 if value > threshold["review"] else 1 if value > threshold["watch"] else 0)
        maximum = max(severities, default=0)
        moderate_count = sum(s >= 1 for s in severities)
        severity = max(maximum, 2 if moderate_count >= 2 else 0)
        adverse = severity >= 2
        severe_streak = severe_streak + 1 if adverse else 0
        structural = row.ood > thresholds["ood"]["severe"] * 1.10
        structural_streak = structural_streak + 1 if structural else 0
        reversal = robust_reversal_exists and row.model_challenger_delta > thresholds["model_challenger_delta"]["review"]
        reversal_streak = reversal_streak + 1 if reversal else 0
        green_streak = green_streak + 1 if severity == 0 else 0
        state = next_state(previous, severity, severe_streak, structural_streak, reversal_streak, green_streak)
        output.append({**row._asdict(), "metric_severity": severity, "adverse_event": adverse,
                       "structural_ood_event": structural, "robust_reversal_event": reversal,
                       "persistent_adverse_quarters": severe_streak, "green_streak": green_streak,
                       "prior_state": previous, "state": state})
        previous = state
    out = pd.DataFrame(output)
    alert = out.state.isin(["REVIEW", "RECALIBRATE", "REDEVELOP"])
    adverse = out.adverse_event.astype(bool)
    policy["backtest"] = {
        "true_alerts": int((alert & adverse).sum()), "false_alerts": int((alert & ~adverse).sum()),
        "missed_deterioration": int((~alert & adverse).sum()),
        "precision": float((alert & adverse).sum() / max(alert.sum(), 1)),
        "recall": float((alert & adverse).sum() / max(adverse.sum(), 1)),
        "time_in_state": out.state.value_counts().to_dict(),
        "average_alert_duration_quarters": float(out.assign(alert=alert).groupby((alert != alert.shift()).cumsum()).alert.agg(["first", "size"]).query("first")["size"].mean()) if alert.any() else 0.0,
        "transitions": out.groupby(["prior_state", "state"]).size().rename("count").reset_index().to_dict("records"),
        "detection_delay": "0 for contemporaneous adverse quarters by construction; persistence delays RECALIBRATE by one quarter",
        "lead_time": "not asserted; thresholds are outcome-monitoring rather than forward failure labels",
    }
    write_json(ART / "monitoring_state_machine.json", policy)
    out.to_parquet(ART / "monitoring_backtest.parquet", index=False)
    return policy, out

