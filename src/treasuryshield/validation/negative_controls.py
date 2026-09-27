"""Explicitly invalid preprocessing and split controls, confined to prelock data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.splits.temporal import SPLITS, deterministic_sample, load_joined, partition


def _fit_q05(x: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=.05,
        tree_method="hist", device="cuda", n_estimators=100, max_depth=4,
        learning_rate=.06, subsample=.85, colsample_bytree=.85,
        min_child_weight=30, reg_lambda=10, max_bin=128,
        n_jobs=8, random_state=1701)
    model.fit(x, y)
    if '"device":"cuda:' not in model.get_booster().save_config():
        raise RuntimeError("Negative-control XGBoost did not use CUDA")
    return model


def _loss(y: np.ndarray, estimate: np.ndarray) -> float:
    difference = y - estimate
    return float(np.maximum(.05 * difference, -.95 * difference).mean())


def run(root: Path) -> None:
    output = root / "artifacts" / "negative_controls.json"
    config = {"split": "tightening", "mode": "B", "seed": 1701,
              "training_sample_max": 100_000, "source_sha256": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        frame = load_joined(root, "core", 1, mode="B")
        frame = frame.loc[frame.report_period.le("2023Q4")].copy()
        train, calibration, test = partition(frame, SPLITS["tightening"])
        train = deterministic_sample(train, 100_000, 1701)
        names = list(CORE_NUMERIC)
        rng = np.random.default_rng(1701)
        test_selection = rng.random(len(test)) < .20
        held = test.iloc[np.flatnonzero(test_selection)].copy()
        leaked_train = test.iloc[np.flatnonzero(~test_selection)].copy()
        if len(held) < 100 or len(leaked_train) < 100:
            raise ValueError("Negative control test split too small")
        y_train = train.deposit_log_growth.to_numpy(dtype=np.float32)
        y_held = held.deposit_log_growth.to_numpy(dtype=float)
        pre = NumericPreprocessor(names).fit(train)
        legal_model = _fit_q05(pre.transform(train), y_train)
        legal_loss = _loss(y_held, legal_model.predict(pre.transform(held)))

        # NC1 uses outcomes from the same held-out future episode to train.
        invalid_train = pd.concat([train, leaked_train], ignore_index=True)
        random_pre = NumericPreprocessor(names).fit(invalid_train)
        random_model = _fit_q05(random_pre.transform(invalid_train),
                                invalid_train.deposit_log_growth.to_numpy(dtype=np.float32))
        random_loss = _loss(y_held, random_model.predict(random_pre.transform(held)))

        # NC2 fits imputation with test-era inputs, despite training on the
        # original historical labels. The comparison uses identical held rows.
        full_pre = NumericPreprocessor(names).fit(pd.concat([train, calibration, test],
                                                         ignore_index=True))
        full_model = _fit_q05(full_pre.transform(train), y_train)
        full_loss = _loss(y_held, full_model.predict(full_pre.transform(held)))

        checks = {
            "future_dated_bank_features": int(frame.feature_information_date.gt(frame.forecast_origin).sum()),
            "nonfuture_target_availability": int(frame.target_availability_date.le(frame.forecast_origin).sum()),
            "prelock_max_report_period": str(frame.report_period.max()),
            "temporal_train_max_period": str(train.report_period.max()),
            "random_control_future_training_rows": len(leaked_train),
            "sampled_test_rows": len(held),
        }
        if checks["future_dated_bank_features"] or checks["nonfuture_target_availability"]:
            raise AssertionError("Point-in-time validity failed")
        variants = [
            {"experiment_id": "E01", "control": "NC1_random_future_outcomes",
             "q05_pinball": random_loss, "temporal_reference": legal_loss,
             "apparent_gain_reference_minus_control": legal_loss - random_loss,
             "invalid": True},
            {"experiment_id": "E03", "control": "NC2_full_sample_feature_preprocessing",
             "q05_pinball": full_loss, "temporal_reference": legal_loss,
             "apparent_gain_reference_minus_control": legal_loss - full_loss,
             "invalid": True},
        ]
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "config_hash": stable_hash(config), "data_snapshot_id": config["snapshot"],
                   "checks": checks, "valid_temporal_q05_pinball": legal_loss,
                   "negative_controls": variants,
                   "interpretation": "NC1 trains on future episode labels; NC2 fits transforms using future inputs. Neither is a deployable model."}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("DELETE FROM ablations WHERE artifact_path = 'artifacts/negative_controls.json'")
            for item in variants:
                con.execute("INSERT INTO ablations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [item["experiment_id"], item["control"], "tightening", "core", 1,
                             "q05_pinball", item["q05_pinball"], item["temporal_reference"],
                             "artifacts/negative_controls.json"])
                con.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id=?",
                            ["artifacts/negative_controls.json", item["experiment_id"]])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print(f"NC1 {random_loss:.6f}, NC2 {full_loss:.6f}, legal temporal {legal_loss:.6f}", flush=True)

    RunState(root).run("negative_controls_prelock", "RUNNING_EXPERIMENTS", config,
                       ["artifacts/negative_controls.json"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
