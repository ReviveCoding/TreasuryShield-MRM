"""Distributional and tail metrics with explicit weighting."""

from __future__ import annotations

import numpy as np

from treasuryshield.splits.temporal import QUANTILES


def pinball(y: np.ndarray, prediction: np.ndarray, q: float) -> np.ndarray:
    error = y - prediction
    return np.maximum(q * error, (q - 1) * error)


def crossing_rate(prediction: np.ndarray) -> float:
    if len(prediction) == 0:
        return float("nan")
    return float(np.mean(np.any(np.diff(prediction, axis=1) < 0, axis=1)))


def rearrange(prediction: np.ndarray) -> np.ndarray:
    return np.sort(prediction, axis=1)


def _mean(value: np.ndarray, weight: np.ndarray) -> float:
    good = np.isfinite(value) & np.isfinite(weight) & (weight >= 0)
    if not np.any(good) or weight[good].sum() <= 0:
        return float("nan")
    return float(np.average(value[good], weights=weight[good]))


def score(y: np.ndarray, prediction: np.ndarray, *, assets: np.ndarray | None = None,
          deposits: np.ndarray | None = None) -> list[dict]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(y), len(QUANTILES)):
        raise ValueError(f"Prediction shape {prediction.shape} expected {(len(y), len(QUANTILES))}")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite evaluation actual or prediction")
    weights = {"equal_bank": np.ones(len(y))}
    if deposits is not None:
        weights["deposit_weighted"] = np.maximum(np.asarray(deposits, dtype=float), 0)
    if assets is not None:
        weights["asset_weighted"] = np.maximum(np.asarray(assets, dtype=float), 0)
    out = []
    for weight_name, weight in weights.items():
        for index, q in enumerate(QUANTILES):
            qname = f"Q{round(q * 100):02d}"
            losses = pinball(y, prediction[:, index], float(q))
            out.append({"metric": "pinball", "quantile": qname, "weighting": weight_name,
                        "value": _mean(losses, weight), "n": len(y)})
            out.append({"metric": "coverage", "quantile": qname, "weighting": weight_name,
                        "value": _mean((y <= prediction[:, index]).astype(float), weight), "n": len(y)})
            out.append({"metric": "coverage_error", "quantile": qname, "weighting": weight_name,
                        "value": _mean((y <= prediction[:, index]).astype(float), weight) - float(q), "n": len(y)})
        median = prediction[:, 4]
        out.extend([
            {"metric": "mae", "quantile": "Q50", "weighting": weight_name,
             "value": _mean(np.abs(y - median), weight), "n": len(y)},
            {"metric": "rmse", "quantile": "Q50", "weighting": weight_name,
             "value": np.sqrt(_mean((y - median) ** 2, weight)), "n": len(y)},
            {"metric": "tail_underestimate_frequency", "quantile": "Q05", "weighting": weight_name,
             "value": _mean((y < prediction[:, 1]).astype(float), weight), "n": len(y)},
            {"metric": "tail_underestimate_severity", "quantile": "Q05", "weighting": weight_name,
             "value": _mean(np.maximum(prediction[:, 1] - y, 0), weight), "n": len(y)},
            {"metric": "quantile_crps_approx", "quantile": "all", "weighting": weight_name,
             "value": _mean(2 * np.mean([pinball(y, prediction[:, j], float(q))
                                           for j, q in enumerate(QUANTILES)], axis=0), weight), "n": len(y)},
            {"metric": "crossing_rate", "quantile": "all", "weighting": weight_name,
             "value": _mean(np.any(np.diff(prediction, axis=1) < 0, axis=1).astype(float), weight), "n": len(y)},
        ])
        # Proper weighted interval score for the two available central pairs:
        # 10-90 (80%) and 25-75 (50%), plus the absolute median component.
        wis = .5 * np.abs(y - median)
        for left, right, alpha in [(2, 6, .2), (3, 5, .5)]:
            lo, hi = prediction[:, left], prediction[:, right]
            interval_score = (hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)
            wis += alpha / 2 * interval_score
        wis /= 2.5
        out.append({"metric": "weighted_interval_score", "quantile": "all", "weighting": weight_name,
                    "value": _mean(wis, weight), "n": len(y)})
    return out
