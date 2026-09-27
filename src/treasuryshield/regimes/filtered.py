"""Leakage-safe rule and HMM regime probabilities at forecast origins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from treasuryshield.splits.temporal import Split

VECTOR = ["macro_DFF_revised", "macro_DFF_change_4q_revised",
          "macro_curve_10y2y_revised", "macro_VIXCLS_revised",
          "macro_UNRATE_revised", "macro_CFNAI_revised",
          "macro_DPSACBW027NBOG_qoq_revised"]


@dataclass
class RegimeResult:
    features: pd.DataFrame
    model: object | None
    metadata: dict


def _quarters(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["report_period"] + VECTOR
    out = frame[cols].drop_duplicates("report_period").sort_values("report_period").reset_index(drop=True)
    if out["report_period"].duplicated().any():
        raise ValueError("Duplicate macro origin quarter")
    return out


def _filtered_hmm(model: object, x: np.ndarray) -> np.ndarray:
    emission = model._compute_log_likelihood(x)
    log_transition = np.log(np.maximum(model.transmat_, 1e-100))
    alpha = np.empty_like(emission)
    alpha[0] = np.log(np.maximum(model.startprob_, 1e-100)) + emission[0]
    alpha[0] -= logsumexp(alpha[0])
    for t in range(1, len(x)):
        predictive = logsumexp(alpha[t - 1, :, None] + log_transition, axis=0)
        alpha[t] = predictive + emission[t]
        alpha[t] -= logsumexp(alpha[t])
    probability = np.exp(alpha)
    if not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Filtered HMM probabilities do not sum to one")
    return probability


def build_regimes(frame: pd.DataFrame, split: Split, *, kind: str = "hmm_filtered",
                  states: int = 3, seed: int = 17) -> RegimeResult:
    quarter = _quarters(frame)
    training = quarter["report_period"].le(split.cutoff)
    if kind == "none":
        result = pd.DataFrame({"report_period": quarter["report_period"],
                               "regime_p0": np.zeros(len(quarter)),
                               "regime_p1": np.zeros(len(quarter)),
                               "regime_p2": np.zeros(len(quarter)),
                               "regime_p3": np.zeros(len(quarter)),
                               "regime_hard": -1})
        return RegimeResult(result, None, {"kind": kind})
    if kind == "rule":
        high_vol = quarter["macro_VIXCLS_revised"].gt(30)
        rapid_hike = quarter["macro_DFF_change_4q_revised"].gt(2)
        inverted = quarter["macro_curve_10y2y_revised"].lt(0)
        state = np.where(high_vol, 2, np.where(rapid_hike | inverted, 1, 0))
        result = pd.DataFrame({"report_period": quarter["report_period"],
                               "regime_p0": (state == 0).astype(float),
                               "regime_p1": (state == 1).astype(float),
                               "regime_p2": (state == 2).astype(float),
                               "regime_p3": np.zeros(len(quarter)),
                               "regime_hard": state})
        return RegimeResult(result, None, {"kind": kind, "rule": "VIX>30; else 4Q DFF rise>2pp or curve<0"})
    if kind not in {"hmm_filtered", "hmm_hard", "hmm_smoothed_negative_control"}:
        raise ValueError(kind)
    from hmmlearn.hmm import GaussianHMM
    if training.sum() < states * 6:
        raise ValueError(f"Insufficient training quarters for {states}-state HMM")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(quarter.loc[training, VECTOR]))
    x_full = scaler.transform(imputer.transform(quarter[VECTOR]))
    model = GaussianHMM(n_components=states, covariance_type="diag", n_iter=100,
                        tol=1e-3, min_covar=.05, random_state=seed, implementation="log")
    model.fit(x_train)
    probability = (model.predict_proba(x_full) if kind == "hmm_smoothed_negative_control"
                   else _filtered_hmm(model, x_full))
    if kind == "hmm_hard":
        hard = np.argmax(probability, axis=1)
        probability = np.eye(states)[hard]
    else:
        hard = np.argmax(probability, axis=1)
    result = pd.DataFrame({"report_period": quarter["report_period"],
                           "regime_hard": hard})
    for index in range(max(4, states)):
        result[f"regime_p{index}"] = probability[:, index] if index < states else 0.0
    return RegimeResult(result, {"hmm": model, "imputer": imputer, "scaler": scaler},
                        {"kind": kind, "states": states, "training_quarters": int(training.sum()),
                         "converged": bool(model.monitor_.converged),
                         "last_training_quarter": quarter.loc[training, "report_period"].iloc[-1],
                         "filter": ("future-aware smoothing; INVALID NEGATIVE CONTROL"
                                    if kind == "hmm_smoothed_negative_control"
                                    else "forward P(S_t | Z_<=t), never smoothed")})
