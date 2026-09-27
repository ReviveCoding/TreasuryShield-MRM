from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _cluster_sums(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"value": np.asarray(values, float), "group": groups})
    grouped = frame.groupby("group", sort=True).value.agg(["sum", "size"])
    return grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float)


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    reps: int = 999,
    seed: int = 20260927,
) -> tuple[float, float]:
    """Percentile CI from resampling whole clusters with replacement."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    good = np.isfinite(values) & pd.notna(groups)
    sums, counts = _cluster_sums(values[good], groups[good])
    if len(sums) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for start in range(0, reps, 100):
        n = min(100, reps - start)
        sampled = rng.integers(0, len(sums), size=(n, len(sums)))
        estimates[start:start + n] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def cluster_bootstrap_weighted_mean(
    values: np.ndarray,
    weight: np.ndarray,
    groups: np.ndarray,
    *,
    reps: int = 999,
    seed: int = 20260927,
) -> tuple[float, float]:
    values, weight, groups = np.asarray(values, float), np.asarray(weight, float), np.asarray(groups)
    good = np.isfinite(values) & np.isfinite(weight) & (weight >= 0) & pd.notna(groups)
    frame = pd.DataFrame({"num": values[good] * weight[good], "den": weight[good], "group": groups[good]})
    grouped = frame.groupby("group", sort=True)[["num", "den"]].sum()
    if len(grouped) < 2:
        return float("nan"), float("nan")
    num, den = grouped.num.to_numpy(float), grouped.den.to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for start in range(0, reps, 100):
        n = min(100, reps - start)
        sampled = rng.integers(0, len(num), size=(n, len(num)))
        estimates[start:start + n] = num[sampled].sum(axis=1) / den[sampled].sum(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def two_way_cluster_mean(
    values: np.ndarray, bank: np.ndarray, quarter: np.ndarray
) -> dict[str, float]:
    """Intercept-only CRV1 covariance clustered by bank and quarter.

    This is the Cameron-Gelbach-Miller additive estimator V_bank + V_time -
    V_intersection, with finite-cluster corrections.  It is algebraically the
    intercept-only form of statsmodels ``cov_cluster_2groups``.
    """
    values = np.asarray(values, float)
    bank = np.asarray(bank)
    quarter = np.asarray(quarter)
    good = np.isfinite(values) & pd.notna(bank) & pd.notna(quarter)
    y, b, t = values[good], bank[good], quarter[good]
    n = len(y)
    if n < 2:
        return {"mean": float("nan"), "se": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(y.mean())
    residual = y - mean

    def component(group: np.ndarray) -> tuple[float, int]:
        sums, _ = _cluster_sums(residual, group)
        g = len(sums)
        if g < 2:
            return float("nan"), g
        return float((g / (g - 1)) * np.sum(sums ** 2) / (n ** 2)), g

    vb, gb = component(b)
    vt, gt = component(t)
    intersection = np.array([f"{x}::{z}" for x, z in zip(b, t)], dtype=object)
    vi, _ = component(intersection)
    variance = vb + vt - vi if np.isfinite([vb, vt, vi]).all() else float("nan")
    variance = max(variance, 0.0) if np.isfinite(variance) else variance
    se = math.sqrt(variance) if np.isfinite(variance) else float("nan")
    df = max(min(gb, gt) - 1, 1)
    critical = float(stats.t.ppf(0.975, df))
    return {
        "mean": mean,
        "se": se,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "banks": gb,
        "quarters": gt,
        "df": df,
    }


def clustered_summary(
    values: np.ndarray,
    bank: np.ndarray,
    quarter: np.ndarray,
    *,
    reps: int = 999,
    seed: int = 20260927,
) -> dict[str, float]:
    tw = two_way_cluster_mean(values, bank, quarter)
    time_ci = cluster_bootstrap_mean(values, quarter, reps=reps, seed=seed)
    bank_ci = cluster_bootstrap_mean(values, bank, reps=reps, seed=seed + 1)
    return {
        "estimate": float(np.nanmean(values)),
        "quarter_ci_low": time_ci[0],
        "quarter_ci_high": time_ci[1],
        "bank_ci_low": bank_ci[0],
        "bank_ci_high": bank_ci[1],
        "two_way_se": tw["se"],
        "two_way_ci_low": tw["ci_low"],
        "two_way_ci_high": tw["ci_high"],
        "n": int(np.isfinite(values).sum()),
        "n_banks": int(pd.Series(bank).nunique()),
        "n_quarters": int(pd.Series(quarter).nunique()),
    }


def clustered_weighted_summary(
    values: np.ndarray,
    weight: np.ndarray,
    bank: np.ndarray,
    quarter: np.ndarray,
    *,
    reps: int = 999,
    seed: int = 20260927,
) -> dict[str, float]:
    values, weight = np.asarray(values, float), np.asarray(weight, float)
    good = np.isfinite(values) & np.isfinite(weight) & (weight >= 0)
    estimate = float(np.average(values[good], weights=weight[good]))
    # Weighted influence pseudo-observations retain the estimate as their row mean.
    influence = estimate + weight * (values - estimate) * good.sum() / weight[good].sum()
    tw = two_way_cluster_mean(influence[good], np.asarray(bank)[good], np.asarray(quarter)[good])
    time_ci = cluster_bootstrap_weighted_mean(values, weight, quarter, reps=reps, seed=seed)
    bank_ci = cluster_bootstrap_weighted_mean(values, weight, bank, reps=reps, seed=seed + 1)
    return {
        "estimate": estimate,
        "quarter_ci_low": time_ci[0], "quarter_ci_high": time_ci[1],
        "bank_ci_low": bank_ci[0], "bank_ci_high": bank_ci[1],
        "two_way_se": tw["se"], "two_way_ci_low": tw["ci_low"], "two_way_ci_high": tw["ci_high"],
        "n": int(good.sum()), "n_banks": int(pd.Series(np.asarray(bank)[good]).nunique()),
        "n_quarters": int(pd.Series(np.asarray(quarter)[good]).nunique()),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p[index] * (m - rank)))
        out[index] = running
    return out
