from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd

from treasuryshield.development.models import lightgbm_quantile
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.splits.temporal import SPLITS, load_joined, partition
from treasuryshield.statistics.metrics import rearrange

from .common import ART, QCOLS, ROOT, SEED, write_json


def reproduce_lightgbm(final_enriched: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    import shap

    started = time.perf_counter()
    frame = load_joined(ROOT, "core", 1, mode="B")
    frame = frame.loc[frame.report_period.le(SPLITS["final_locked"].test_end)].copy()
    train, calibration, test = partition(frame, SPLITS["final_locked"], mna_radius=1)
    result = lightgbm_quantile(train, calibration, test, list(CORE_NUMERIC), seed=17, n_estimators=160)
    reproduced = rearrange(result.test)
    frozen = pd.read_parquet(ROOT / "artifacts" / "predictions" / "core_B_1q_final_locked_B8_lightgbm_quantile.parquet")
    keys = ["bank_id", "report_period"]
    expected_keys = test[keys].reset_index(drop=True)
    if not expected_keys.equals(frozen[keys].reset_index(drop=True)):
        raise RuntimeError("Reproduction rows do not align with frozen B8 prediction rows")
    frozen_values = frozen[QCOLS].to_numpy(float)
    difference = reproduced - frozen_values
    tolerance = 1e-10
    per_quantile = []
    for index, quantile in enumerate(QCOLS):
        per_quantile.append({"quantile": quantile.upper(),
                             "max_absolute_difference": float(np.max(np.abs(difference[:, index]))),
                             "mean_absolute_difference": float(np.mean(np.abs(difference[:, index]))),
                             "correlation": float(np.corrcoef(reproduced[:, index], frozen_values[:, index])[0, 1]),
                             "exact_match_rate": float(np.mean(reproduced[:, index] == frozen_values[:, index])),
                             "near_exact_match_rate": float(np.mean(np.abs(difference[:, index]) <= tolerance))})
    parity = float(np.max(np.abs(difference))) <= tolerance
    metadata = {
        "original_fitted_model_artifacts_found": False,
        "reconstruction_basis": "frozen snapshot, feature list, temporal partition, seed 17, and seven 160-tree LightGBM quantile models",
        "strict_tolerance": tolerance, "prediction_parity_passed": parity,
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "near_exact_match_rate": float(np.mean(np.abs(difference) <= tolerance)),
        "per_quantile": per_quantile, "train_rows": len(train), "calibration_rows": len(calibration),
        "test_rows": len(test), "explanation_status": "exact_reproduced_model" if parity else "surrogate_only",
        "runtime_seconds": time.perf_counter() - started,
    }
    error_rows = pd.DataFrame({"row": np.arange(len(frozen)), "bank_id": frozen.bank_id,
                               "report_period": frozen.report_period,
                               **{f"error_{q}": difference[:, i] for i, q in enumerate(QCOLS)}})
    error_rows.to_parquet(ART / "lightgbm_reproduction_errors.parquet", index=False)
    write_json(ART / "lightgbm_reproduction.json", metadata)
    if not parity:
        empty_importance = pd.DataFrame(columns=["quantile", "feature", "gain", "split"])
        empty_shap = pd.DataFrame(columns=["scope", "group", "bank_id", "report_period", "feature", "shap_value", "mean_abs_shap", "exact_label"])
        empty_importance.to_parquet(ART / "lightgbm_exact_importance.parquet", index=False)
        empty_shap.to_parquet(ART / "lightgbm_shap.parquet", index=False)
        return metadata, empty_importance, empty_shap

    importance_rows = []
    for quantile, model in zip(QCOLS, result.model["models"]):
        path = ART / "models" / f"b8_reproduced_{quantile}.txt"
        model.booster_.save_model(str(path))
        for feature, gain, split in zip(result.model["preprocessor"].output_names,
                                        model.booster_.feature_importance("gain"),
                                        model.booster_.feature_importance("split")):
            importance_rows.append({"quantile": quantile.upper(), "feature": feature,
                                    "gain": float(gain), "split": int(split),
                                    "native_model_path": path.relative_to(ROOT).as_posix()})
    importance = pd.DataFrame(importance_rows)
    importance.to_parquet(ART / "lightgbm_exact_importance.parquet", index=False)
    pre = result.model["preprocessor"]
    q05_model = result.model["models"][1]
    sample_indices = np.sort(np.random.default_rng(SEED).choice(len(test), min(6000, len(test)), replace=False))
    x = pre.transform(test.iloc[sample_indices])
    explainer = shap.TreeExplainer(q05_model.booster_, feature_perturbation="tree_path_dependent")
    shap_values = np.asarray(explainer.shap_values(x), float)
    names = pre.output_names
    shap_rows = []
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    for feature, value in zip(names, mean_abs):
        shap_rows.append({"scope": "global", "group": "all", "bank_id": None, "report_period": None,
                          "feature": feature, "shap_value": math.nan, "mean_abs_shap": float(value),
                          "exact_label": True})
    sampled_test = test.iloc[sample_indices].reset_index(drop=True)
    for group_column in ["report_period"]:
        for group, positions in sampled_test.groupby(group_column).groups.items():
            local = np.mean(np.abs(shap_values[np.asarray(list(positions))]), axis=0)
            for feature, value in zip(names, local):
                shap_rows.append({"scope": "quarter", "group": str(group), "bank_id": None,
                                  "report_period": str(group), "feature": feature, "shap_value": math.nan,
                                  "mean_abs_shap": float(value), "exact_label": True})
    # Frozen prediction regime labels provide a descriptive regime stability split.
    regimes = frozen.iloc[sample_indices].regime.reset_index(drop=True)
    for group, positions in regimes.groupby(regimes).groups.items():
        local = np.mean(np.abs(shap_values[np.asarray(list(positions))]), axis=0)
        for feature, value in zip(names, local):
            shap_rows.append({"scope": "regime", "group": str(group), "bank_id": None,
                              "report_period": None, "feature": feature, "shap_value": math.nan,
                              "mean_abs_shap": float(value), "exact_label": True})

    joined = final_enriched.set_index(keys).loc[pd.MultiIndex.from_frame(frozen[keys])].reset_index(drop=True)
    categories = {
        "severe_tail_miss": np.flatnonzero(joined.actual.to_numpy() < joined.q05.to_numpy() - .02),
        "highest_ood": np.argsort(-joined.ood_score.to_numpy())[:10],
        "highest_disagreement": np.argsort(-joined.disagreement.to_numpy())[:10],
        "highest_exposure": np.argsort(-joined.assets.to_numpy())[:10],
    }
    local_indices = np.unique(np.concatenate([positions[:10] for positions in categories.values() if len(positions)]))
    local_x = pre.transform(test.iloc[local_indices])
    local_shap = np.asarray(explainer.shap_values(local_x), float)
    label_lookup = {int(index): ",".join(name for name, positions in categories.items() if index in set(positions[:10]))
                    for index in local_indices}
    for local_position, test_position in enumerate(local_indices):
        top = np.argsort(-np.abs(local_shap[local_position]))[:8]
        for rank, feature_index in enumerate(top, 1):
            shap_rows.append({"scope": "local_reason_code", "group": label_lookup[int(test_position)],
                              "bank_id": test.iloc[test_position].bank_id,
                              "report_period": test.iloc[test_position].report_period,
                              "feature": names[feature_index], "shap_value": float(local_shap[local_position, feature_index]),
                              "mean_abs_shap": math.nan, "rank": rank, "exact_label": True})
    shap_frame = pd.DataFrame(shap_rows)
    shap_frame.to_parquet(ART / "lightgbm_shap.parquet", index=False)
    write_json(ART / "models" / "preprocessor.json", {
        "feature_names": pre.names, "output_names": pre.output_names,
        "medians": pre.medians_.tolist(), "lower": pre.lower_.tolist(), "upper": pre.upper_.tolist(),
        "centers": pre.centers_.tolist(), "scales": pre.scales_.tolist(),
    })
    return metadata, importance, shap_frame
