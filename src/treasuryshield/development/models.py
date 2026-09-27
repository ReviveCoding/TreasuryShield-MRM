"""Credible baseline and challenger quantile models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge

from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.splits.temporal import QUANTILES, deterministic_sample


@dataclass
class FitResult:
    calibration: np.ndarray
    test: np.ndarray
    model: Any
    metadata: dict


def peer_group(frame: pd.DataFrame) -> np.ndarray:
    assets = frame["assets"].to_numpy(dtype=float)
    size = np.digitize(assets, [100_000, 1_000_000, 10_000_000]).astype(int)
    residential = frame["residential_loan_share"].fillna(-1).to_numpy(dtype=float)
    brokered = frame["brokered_domestic_ratio"].fillna(-1).to_numpy(dtype=float)
    housing = (residential >= .5).astype(int)
    wholesale = (brokered >= .1).astype(int)
    return size * 4 + housing * 2 + wholesale


def _group_quantiles(train: pd.DataFrame, predict: pd.DataFrame, groups: np.ndarray,
                     predict_groups: np.ndarray, *, recent: int | None = None) -> np.ndarray:
    if recent is not None:
        latest = pd.Period(train["report_period"].max(), freq="Q")
        start = str(latest - recent + 1)
        subset = train.loc[train["report_period"].ge(start)]
        group_values = groups[train["report_period"].ge(start).to_numpy()]
    else:
        subset, group_values = train, groups
    y = subset["deposit_log_growth"].to_numpy(dtype=float)
    global_q = np.quantile(y, QUANTILES)
    quantiles = {}
    for group in np.unique(group_values):
        values = y[group_values == group]
        if len(values) >= 100:
            quantiles[int(group)] = np.quantile(values, QUANTILES)
    size_q = {}
    for size in range(4):
        values = y[group_values // 4 == size]
        if len(values) >= 100:
            size_q[size] = np.quantile(values, QUANTILES)
    output = np.empty((len(predict), len(QUANTILES)), dtype=float)
    for group in np.unique(predict_groups):
        q = quantiles.get(int(group), size_q.get(int(group) // 4, global_q))
        output[predict_groups == group] = q
    return output


def historical_unconditional(train: pd.DataFrame, calibration: pd.DataFrame,
                             test: pd.DataFrame) -> FitResult:
    quantiles = np.quantile(train["deposit_log_growth"].to_numpy(), QUANTILES)
    return FitResult(np.tile(quantiles, (len(calibration), 1)),
                     np.tile(quantiles, (len(test), 1)), quantiles,
                     {"training_rows": len(train)})


def historical_peer(train: pd.DataFrame, calibration: pd.DataFrame,
                    test: pd.DataFrame) -> FitResult:
    groups = peer_group(train)
    cal = _group_quantiles(train, calibration, groups, peer_group(calibration), recent=16)
    pred = _group_quantiles(train, test, groups, peer_group(test), recent=16)
    return FitResult(cal, pred, {"recent_quarters": 16}, {"training_rows": len(train)})


def bank_persistence(train: pd.DataFrame, calibration: pd.DataFrame,
                     test: pd.DataFrame) -> FitResult:
    global_q = np.quantile(train["deposit_log_growth"], QUANTILES)
    ordered = train.sort_values("report_period").groupby("bank_id", sort=False).tail(8)
    by_bank = ordered.groupby("bank_id")["deposit_log_growth"].agg(list)
    def predict(frame: pd.DataFrame) -> np.ndarray:
        out = np.tile(global_q, (len(frame), 1))
        for bank_id, indices in frame.groupby("bank_id").indices.items():
            values = by_bank.get(bank_id)
            if values is None or len(values) < 3:
                continue
            local = np.quantile(np.asarray(values), QUANTILES)
            weight = len(values) / (len(values) + 8)
            out[indices] = weight * local + (1 - weight) * global_q
        return out
    return FitResult(predict(calibration), predict(test), {"bank_history": by_bank},
                     {"training_rows": len(train), "bank_history_max": 8})


def ridge_location(train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame,
                   names: list[str], *, fixed_effects: bool = False) -> FitResult:
    pre = NumericPreprocessor(names, linear=True).fit(train)
    x = pre.transform(train)
    model = (Ridge(alpha=100.0, random_state=17) if fixed_effects
             else LinearRegression(n_jobs=8)).fit(x, train["deposit_log_growth"])
    residual = train["deposit_log_growth"].to_numpy() - model.predict(x)
    bank_effect = None
    if fixed_effects:
        table = pd.DataFrame({"bank_id": train["bank_id"].to_numpy(), "residual": residual})
        grouped = table.groupby("bank_id")["residual"].agg(["count", "mean"])
        grouped["effect"] = grouped["count"] / (grouped["count"] + 8) * grouped["mean"]
        bank_effect = grouped["effect"]
        residual = residual - train["bank_id"].map(bank_effect).fillna(0).to_numpy()
    residual_q = np.quantile(residual, QUANTILES)
    def predict(frame: pd.DataFrame) -> np.ndarray:
        center = model.predict(pre.transform(frame))
        if bank_effect is not None:
            center += frame["bank_id"].map(bank_effect).fillna(0).to_numpy()
        return center[:, None] + residual_q[None, :]
    return FitResult(predict(calibration), predict(test),
                     {"model": model, "preprocessor": pre, "bank_effect": bank_effect},
                     {"training_rows": len(train), "fixed_effects": fixed_effects})


def panel_fixed_effects(train: pd.DataFrame, calibration: pd.DataFrame,
                        test: pd.DataFrame, names: list[str]) -> FitResult:
    """Within-bank OLS slopes with bank intercepts fitted only on training data."""
    pre = NumericPreprocessor(names, linear=True).fit(train)
    x = pre.transform(train).astype(np.float64)
    y = train["deposit_log_growth"].to_numpy(dtype=np.float64)
    banks, codes = np.unique(train["bank_id"].to_numpy(), return_inverse=True)
    counts = np.bincount(codes).astype(float)
    mean_x = np.column_stack([
        np.bincount(codes, weights=x[:, column], minlength=len(banks)) / counts
        for column in range(x.shape[1])])
    mean_y = np.bincount(codes, weights=y, minlength=len(banks)) / counts
    within = LinearRegression(fit_intercept=False, n_jobs=8).fit(
        x - mean_x[codes], y - mean_y[codes])
    effects = pd.Series(mean_y - mean_x @ within.coef_, index=banks)
    fallback = float(y.mean() - x.mean(axis=0) @ within.coef_)

    def center(frame: pd.DataFrame) -> np.ndarray:
        return (within.predict(pre.transform(frame).astype(np.float64)) +
                frame["bank_id"].map(effects).fillna(fallback).to_numpy(dtype=float))

    residual_q = np.quantile(y - center(train), QUANTILES)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        return center(frame)[:, None] + residual_q[None, :]

    return FitResult(predict(calibration), predict(test),
                     {"model": within, "preprocessor": pre, "bank_effect": effects},
                     {"training_rows": len(train), "training_banks": len(banks),
                      "unseen_test_banks": int((~test["bank_id"].isin(banks)).sum()),
                      "method": "within-bank OLS slopes and fixed bank intercepts"})


def bayesian_student_t(train: pd.DataFrame, calibration: pd.DataFrame,
                       test: pd.DataFrame, names: list[str], *, seed: int = 17) -> FitResult:
    sampled = deterministic_sample(train, 180_000, seed)
    pre = NumericPreprocessor(names, linear=True).fit(sampled)
    x = pre.transform(sampled)
    # Gaussian ridge posterior mean with an empirical prior, followed by a
    # bank-level random intercept posterior and a Student-t predictive tail.
    global_model = Ridge(alpha=300.0, random_state=seed).fit(x, sampled["deposit_log_growth"])
    residual = sampled["deposit_log_growth"].to_numpy() - global_model.predict(x)
    sigma = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), 1e-4)
    bank = pd.DataFrame({"bank_id": sampled["bank_id"].to_numpy(), "residual": residual})
    grouped = bank.groupby("bank_id")["residual"].agg(["size", "mean"])
    tau = max(float(grouped.loc[grouped["size"].ge(4), "mean"].std()), sigma / 10)
    prior_n = (sigma / tau) ** 2
    grouped["posterior_mean"] = grouped["size"] / (grouped["size"] + prior_n) * grouped["mean"]
    grouped["posterior_sd"] = sigma / np.sqrt(grouped["size"] + prior_n)
    def predict(frame: pd.DataFrame) -> np.ndarray:
        center = global_model.predict(pre.transform(frame)) + frame["bank_id"].map(
            grouped["posterior_mean"]).fillna(0).to_numpy()
        posterior_sd = frame["bank_id"].map(grouped["posterior_sd"]).fillna(tau).to_numpy()
        scale = np.sqrt(sigma ** 2 + posterior_sd ** 2)
        return center[:, None] + scale[:, None] * stats.t.ppf(QUANTILES, df=4)[None, :]
    return FitResult(predict(calibration), predict(test),
                     {"model": global_model, "preprocessor": pre, "bank_effect": grouped},
                     {"training_rows": len(sampled), "sigma": sigma, "tau": tau,
                      "predictive_student_t_df": 4, "method": "empirical-Bayes hierarchical Student-t"})


def linear_quantile_cuda(train: pd.DataFrame, calibration: pd.DataFrame,
                         test: pd.DataFrame, names: list[str], *, seed: int = 17,
                         epochs: int = 24) -> FitResult:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("B5 linear quantile is configured for CUDA; no GPU available")
    torch.manual_seed(seed)
    pre = NumericPreprocessor(names, linear=True).fit(train)
    x = pre.transform(train)
    y = train["deposit_log_growth"].to_numpy(dtype=np.float32)
    device = torch.device("cuda")
    model = torch.nn.Linear(x.shape[1], len(QUANTILES)).to(device)
    with torch.no_grad():
        model.bias.copy_(torch.tensor(np.quantile(y, QUANTILES), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    q = torch.tensor(QUANTILES, dtype=torch.float32, device=device)[None, :]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        permutation = rng.permutation(len(x))
        for positions in np.array_split(permutation, max(1, int(np.ceil(len(x) / 8192)))):
            xb = torch.as_tensor(x[positions], device=device)
            yb = torch.as_tensor(y[positions], device=device)[:, None]
            error = yb - model(xb)
            loss = torch.maximum(q * error, (q - 1) * error).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    def predict(frame: pd.DataFrame) -> np.ndarray:
        model.eval()
        arrays = []
        with torch.no_grad():
            xx = pre.transform(frame)
            for start in range(0, len(xx), 16384):
                arrays.append(model(torch.as_tensor(xx[start:start + 16384], device=device)).cpu().numpy())
        return np.vstack(arrays) if arrays else np.empty((0, len(QUANTILES)))
    return FitResult(predict(calibration), predict(test),
                     {"model": model, "preprocessor": pre},
                     {"training_rows": len(train), "device": torch.cuda.get_device_name(0), "epochs": epochs})


def quantile_forest(train: pd.DataFrame, calibration: pd.DataFrame,
                    test: pd.DataFrame, names: list[str], *, seed: int = 17) -> FitResult:
    from quantile_forest import RandomForestQuantileRegressor
    sampled = deterministic_sample(train, 100_000, seed)
    pre = NumericPreprocessor(names).fit(sampled)
    model = RandomForestQuantileRegressor(
        n_estimators=80, max_depth=14, min_samples_leaf=25,
        max_features=.8, max_samples=.8, max_samples_leaf=50,
        random_state=seed, n_jobs=8)
    model.fit(pre.transform(sampled), sampled["deposit_log_growth"].to_numpy())
    return FitResult(model.predict(pre.transform(calibration), quantiles=QUANTILES.tolist()),
                     model.predict(pre.transform(test), quantiles=QUANTILES.tolist()),
                     {"model": model, "preprocessor": pre},
                     {"training_rows": len(sampled), "n_estimators": 80})


def xgboost_quantile_cuda(train: pd.DataFrame, calibration: pd.DataFrame,
                          test: pd.DataFrame, names: list[str], *, seed: int = 17,
                          n_estimators: int = 160, max_depth: int = 4,
                          learning_rate: float = .06) -> FitResult:
    import xgboost as xgb
    pre = NumericPreprocessor(names).fit(train)
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=QUANTILES.tolist(),
        device="cuda", tree_method="hist", n_estimators=n_estimators,
        max_depth=max_depth, learning_rate=learning_rate, subsample=.85,
        colsample_bytree=.85, min_child_weight=30, reg_lambda=10,
        max_bin=128, random_state=seed, n_jobs=8)
    model.fit(pre.transform(train), train["deposit_log_growth"].to_numpy(dtype=np.float32),
              verbose=False)
    config = model.get_booster().save_config()
    if '"device":"cuda:' not in config:
        raise RuntimeError("XGBoost quantile model did not train on CUDA")
    return FitResult(np.asarray(model.predict(pre.transform(calibration))),
                     np.asarray(model.predict(pre.transform(test))),
                     {"model": model, "preprocessor": pre},
                     {"training_rows": len(train), "device": "cuda", "estimators": n_estimators,
                      "depth": max_depth, "learning_rate": learning_rate})


def lightgbm_quantile(train: pd.DataFrame, calibration: pd.DataFrame,
                      test: pd.DataFrame, names: list[str], *, seed: int = 17,
                      n_estimators: int = 160) -> FitResult:
    import lightgbm as lgb
    pre = NumericPreprocessor(names).fit(train)
    x = pre.transform(train)
    cal = pre.transform(calibration)
    tst = pre.transform(test)
    models = []
    cal_pred = []
    test_pred = []
    for q in QUANTILES:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=float(q), device_type="cpu", n_estimators=n_estimators,
            num_leaves=31, max_depth=6, min_child_samples=100,
            learning_rate=.06, subsample=.85, colsample_bytree=.85,
            reg_lambda=10, random_state=seed, n_jobs=8, verbosity=-1)
        model.fit(x, train["deposit_log_growth"].to_numpy())
        models.append(model)
        cal_pred.append(model.predict(cal))
        test_pred.append(model.predict(tst))
    return FitResult(np.column_stack(cal_pred), np.column_stack(test_pred),
                     {"models": models, "preprocessor": pre},
                     {"training_rows": len(train), "device": "cpu", "estimators_per_quantile": n_estimators})


MODEL_IDS = ["B0_unconditional", "B1_peer", "B2_persistence", "B3_ols",
             "B4_panel_fe", "B5_linear_quantile", "B6_quantile_forest",
             "B7_xgb_quantile", "B8_lightgbm_quantile", "B9_bayesian_student_t"]


def fit_model(model_id: str, train: pd.DataFrame, calibration: pd.DataFrame,
              test: pd.DataFrame, names: list[str], *, seed: int = 17,
              config: dict | None = None, history: pd.DataFrame | None = None) -> FitResult:
    config = config or {}
    if model_id == "B0_unconditional":
        return historical_unconditional(train, calibration, test)
    if model_id == "B1_peer":
        return historical_peer(train, calibration, test)
    if model_id == "B2_persistence":
        return bank_persistence(train, calibration, test)
    if model_id == "B3_ols":
        return ridge_location(train, calibration, test, names)
    if model_id == "B4_panel_fe":
        return panel_fixed_effects(train, calibration, test, names)
    if model_id == "B5_linear_quantile":
        return linear_quantile_cuda(train, calibration, test, names, seed=seed,
                                    epochs=config.get("epochs", 24))
    if model_id == "B6_quantile_forest":
        return quantile_forest(train, calibration, test, names, seed=seed)
    if model_id == "B7_xgb_quantile":
        return xgboost_quantile_cuda(train, calibration, test, names, seed=seed,
                                     n_estimators=config.get("n_estimators", 160),
                                     max_depth=config.get("max_depth", 4),
                                     learning_rate=config.get("learning_rate", .06))
    if model_id == "B8_lightgbm_quantile":
        return lightgbm_quantile(train, calibration, test, names, seed=seed,
                                 n_estimators=config.get("n_estimators", 160))
    if model_id == "B9_bayesian_student_t":
        return bayesian_student_t(train, calibration, test, names, seed=seed)
    if model_id in {"B10_mlp", "B10_gru", "RAHQ_nn", "RAHQ_C_nn"}:
        from treasuryshield.development.neural import train_neural
        return train_neural(train, calibration, test, names,
                            history=history if history is not None else pd.concat([train, calibration, test]),
                            architecture=config.get("architecture", "gru" if model_id in {
                                "B10_gru", "RAHQ_nn", "RAHQ_C_nn"} else "mlp"),
                            monotonic=config.get("monotonic", model_id == "RAHQ_C_nn"), seed=seed,
                            epochs=config.get("epochs", 18),
                            max_training_rows=config.get("max_training_rows", 200_000),
                            width=config.get("width", 128), batch_size=config.get("batch_size", 4096),
                            tail_weighted=config.get("tail_weighted", False))
    if model_id in {"RAHQ_tree", "RAHQ_C_tree"}:
        return xgboost_quantile_cuda(train, calibration, test, names, seed=seed,
                                     n_estimators=config.get("n_estimators", 200),
                                     max_depth=config.get("max_depth", 5),
                                     learning_rate=config.get("learning_rate", .05))
    raise KeyError(model_id)
