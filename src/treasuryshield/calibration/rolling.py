"""Historical-residual quantile calibration without test-outcome access."""

from __future__ import annotations

import numpy as np
import pandas as pd

from treasuryshield.development.models import peer_group
from treasuryshield.splits.temporal import QUANTILES
from treasuryshield.statistics.metrics import rearrange


def _weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if good.sum() < 10:
        return float(np.nanquantile(values, q))
    order = np.argsort(values[good])
    values = values[good][order]
    weights = weights[good][order]
    cumulative = np.cumsum(weights)
    return float(values[min(np.searchsorted(cumulative, q * cumulative[-1]), len(values) - 1)])


def calibrate(calibration: pd.DataFrame, cal_pred: np.ndarray,
              test: pd.DataFrame, test_pred: np.ndarray, *, kind: str = "global") -> tuple[np.ndarray, dict]:
    if kind in {"adaptive_global", "adaptive_peer", "adaptive_regime"}:
        base_kind = kind.removeprefix("adaptive_")
        output = np.empty_like(test_pred)
        updates = []
        first_origin = test["forecast_origin"].min()
        if calibration["target_availability_date"].ge(first_origin).any():
            raise ValueError("Initial calibration labels not available before first forecast origin")
        for origin in sorted(test["forecast_origin"].unique()):
            current = test["forecast_origin"].eq(origin).to_numpy()
            matured = test["target_availability_date"].lt(origin).to_numpy()
            history = pd.concat([calibration, test.loc[matured]], ignore_index=True)
            historical_prediction = np.vstack([cal_pred, test_pred[matured]])
            last_periods = sorted(history["report_period"].unique())[-8:]
            recent = history["report_period"].isin(last_periods).to_numpy()
            current_prediction, _ = calibrate(
                history.loc[recent], historical_prediction[recent],
                test.loc[current], test_pred[current], kind=base_kind)
            output[current] = current_prediction
            updates.append({"forecast_origin": str(origin),
                            "feedback_rows": int(matured.sum()),
                            "historical_rows": int(recent.sum()),
                            "latest_feedback_availability": (
                                str(test.loc[matured, "target_availability_date"].max())
                                if matured.any() else None)})
        return output, {"kind": kind, "rolling_origin_updates": updates,
                        "recency_quarters": 8, "feedback_rule":
                        "target availability strictly before current forecast origin",
                        "future_evaluation_outcomes_used": False}
    if kind == "none":
        return rearrange(test_pred), {"kind": "none", "calibration_rows": 0}
    y = calibration["deposit_log_growth"].to_numpy(dtype=float)
    if len(y) != len(cal_pred) or len(test) != len(test_pred):
        raise ValueError("Calibration predictions misaligned")
    residual = y[:, None] - cal_pred
    global_shift = np.array([np.quantile(residual[:, j], q) for j, q in enumerate(QUANTILES)])
    if kind == "global":
        shift = np.broadcast_to(global_shift, test_pred.shape)
    elif kind == "peer":
        cal_group = peer_group(calibration)
        test_group = peer_group(test)
        shift = np.broadcast_to(global_shift, test_pred.shape).copy()
        for group in np.unique(test_group):
            which = cal_group == group
            if which.sum() < 200:
                continue
            local = np.array([np.quantile(residual[which, j], q)
                              for j, q in enumerate(QUANTILES)])
            shrinkage = which.sum() / (which.sum() + 300)
            shift[test_group == group] = shrinkage * local + (1 - shrinkage) * global_shift
    elif kind == "regime":
        columns = ["regime_p0", "regime_p1", "regime_p2"]
        if not set(columns).issubset(calibration) or not set(columns).issubset(test):
            raise ValueError("Regime-conditioned calibration requires filtered probabilities")
        cal_p = calibration[columns].to_numpy(dtype=float)
        test_p = test[columns].to_numpy(dtype=float)
        state_shift = np.zeros((3, len(QUANTILES)))
        for state in range(3):
            for j, q in enumerate(QUANTILES):
                state_shift[state, j] = _weighted_quantile(residual[:, j], float(q), cal_p[:, state])
        shift = test_p @ state_shift
    else:
        raise ValueError(kind)
    calibrated = rearrange(test_pred + shift)
    return calibrated, {"kind": kind, "calibration_rows": len(calibration),
                        "global_shift": global_shift.tolist(),
                        "source_max_outcome_availability": str(calibration["target_availability_date"].max()),
                        "evaluation_outcomes_used": False}
