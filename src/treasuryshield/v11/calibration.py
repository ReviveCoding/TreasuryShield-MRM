from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .common import QCOLS, QLEVELS, pinball


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = stats.norm.ppf(0.5 + level / 2)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def calibration_rows(frame: pd.DataFrame, model: str, scope: str = "overall", group: str = "all") -> list[dict]:
    y = frame.actual.to_numpy(float)
    rows = []
    for qcol, alpha in zip(QCOLS, QLEVELS):
        pred = frame[qcol].to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(pred)
        yy, pp = y[mask], pred[mask]
        exceptions = yy < pp
        lo, hi = wilson(int(exceptions.sum()), len(yy))
        severity = np.maximum(pp - yy, 0)
        rows.append({
            "model": model, "scope": scope, "group": str(group), "quantile": qcol,
            "nominal": alpha, "n": len(yy), "empirical_coverage": float(exceptions.mean()),
            "coverage_error": float(exceptions.mean() - alpha), "coverage_ci_low": lo,
            "coverage_ci_high": hi, "pinball": float(pinball(yy, pp, alpha).mean()),
            "tail_underestimate_frequency": float(exceptions.mean()),
            "tail_underestimate_severity": float(severity.mean()),
            "conditional_tail_severity": float(severity[exceptions].mean()) if exceptions.any() else 0.0,
        })
    intervals = [("q01", "q90", 0.89), ("q05", "q90", 0.85), ("q10", "q90", 0.80), ("q25", "q75", 0.50)]
    for low, high, nominal in intervals:
        l, u = frame[low].to_numpy(float), frame[high].to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(l) & np.isfinite(u)
        yy, ll, uu = y[mask], l[mask], u[mask]
        covered = (yy >= ll) & (yy <= uu)
        alpha = 1 - nominal
        interval_score = (uu - ll) + 2 / alpha * (ll - yy) * (yy < ll) + 2 / alpha * (yy - uu) * (yy > uu)
        rows.append({
            "model": model, "scope": scope, "group": str(group), "quantile": f"{low}_{high}",
            "nominal": nominal, "n": len(yy), "empirical_coverage": float(covered.mean()),
            "coverage_error": float(covered.mean() - nominal), "coverage_ci_low": wilson(int(covered.sum()), len(yy))[0],
            "coverage_ci_high": wilson(int(covered.sum()), len(yy))[1], "pinball": math.nan,
            "tail_underestimate_frequency": math.nan, "tail_underestimate_severity": math.nan,
            "conditional_tail_severity": math.nan, "interval_width": float(np.mean(uu - ll)),
            "interval_score": float(np.mean(interval_score)),
        })
    return rows


def _safe_log(value: float) -> float:
    return math.log(max(value, 1e-15))


def exception_tests(exception: np.ndarray, alpha: float) -> dict:
    x = np.asarray(exception, dtype=int)
    n, k = len(x), int(x.sum())
    phat = k / n if n else math.nan
    if not n:
        return {"applicable": False, "reason": "no observations"}
    log_null = k * _safe_log(alpha) + (n - k) * _safe_log(1 - alpha)
    log_alt = k * _safe_log(phat) + (n - k) * _safe_log(1 - phat) if 0 < phat < 1 else 0.0
    lr_uc = max(0.0, -2 * (log_null - log_alt))
    pairs = np.column_stack([x[:-1], x[1:]]) if n > 1 else np.empty((0, 2), int)
    n00 = int(np.sum((pairs[:, 0] == 0) & (pairs[:, 1] == 0))) if len(pairs) else 0
    n01 = int(np.sum((pairs[:, 0] == 0) & (pairs[:, 1] == 1))) if len(pairs) else 0
    n10 = int(np.sum((pairs[:, 0] == 1) & (pairs[:, 1] == 0))) if len(pairs) else 0
    n11 = int(np.sum((pairs[:, 0] == 1) & (pairs[:, 1] == 1))) if len(pairs) else 0
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0
    pi = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    ll_ind_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_ind_alt = n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01) + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    lr_ind = max(0.0, -2 * (ll_ind_null - ll_ind_alt))
    runs, current = [], 0
    for value in x:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    applicable = k >= 5 and n - k >= 5
    return {
        "applicable": applicable,
        "limitation": None if applicable else "Sparse exceptions make asymptotic chi-square p-values unreliable",
        "n": n, "exceptions": k, "rate": phat,
        "unconditional_coverage": {"lr": lr_uc, "p_value": float(stats.chi2.sf(lr_uc, 1)) if applicable else None},
        "independence": {"lr": lr_ind, "p_value": float(stats.chi2.sf(lr_ind, 1)) if applicable else None,
                         "n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "conditional_coverage": {"lr": lr_uc + lr_ind, "p_value": float(stats.chi2.sf(lr_uc + lr_ind, 2)) if applicable else None},
        "runs": {"count": len(runs), "maximum": max(runs, default=0), "mean": float(np.mean(runs)) if runs else 0.0,
                 "clustered_exception_share": float(sum(r for r in runs if r > 1) / k) if k else 0.0},
    }


def pit_values(frame: pd.DataFrame) -> np.ndarray:
    predictions = np.maximum.accumulate(frame[QCOLS].to_numpy(float), axis=1)
    y = frame.actual.to_numpy(float)
    result = np.empty(len(frame), float)
    for i, (yy, qq) in enumerate(zip(y, predictions)):
        span = max(qq[-1] - qq[0], 1e-4)
        xp = np.r_[qq[0] - span, qq, qq[-1] + span]
        fp = np.r_[0.0, QLEVELS, 1.0]
        # Duplicate quantiles are consolidated to preserve a valid monotone interpolant.
        unique, idx = np.unique(xp, return_index=True)
        result[i] = np.interp(yy, unique, fp[idx], left=0.0, right=1.0)
    return result


def pit_summary(frame: pd.DataFrame, scope: str, group: str) -> dict:
    pit = pit_values(frame)
    n = len(pit)
    ks = stats.kstest(pit, "uniform")
    sorted_pit = np.sort(pit)
    expected = (np.arange(n) + 0.5) / n
    cvm = float(np.mean((sorted_pit - expected) ** 2) + 1 / (12 * n))
    return {
        "scope": scope, "group": str(group), "n": n, "pit_mean": float(np.mean(pit)),
        "pit_variance": float(np.var(pit)), "ks_statistic": float(ks.statistic),
        "ks_p_value": float(ks.pvalue), "cvm_style": cvm,
        "lower_tail_share": float(np.mean(pit <= 0.05)), "upper_tail_share": float(np.mean(pit >= 0.95)),
        "lag1_autocorrelation": float(pd.Series(pit).autocorr(1)) if n > 2 else math.nan,
        "interpolation_note": "Approximate monotone piecewise-linear CDF from seven stored quantiles; extrapolated endpoints are diagnostic only.",
    }


def weighted_quantile(values: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, float)
    if weights is None:
        return float(np.quantile(values, q))
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(q, cumulative, values))
