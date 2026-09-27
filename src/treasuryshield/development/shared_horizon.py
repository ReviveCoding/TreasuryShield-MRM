"""One CUDA quantile backbone with a horizon input, compared with separate fits."""

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
from treasuryshield.splits.temporal import QUANTILES, SPLITS, load_joined, partition
from treasuryshield.statistics.metrics import rearrange, score


def run(root: Path, split_name: str = "tightening") -> None:
    if split_name not in {"gfc", "pandemic", "tightening"}:
        raise ValueError("Shared-horizon study restricted to prelock episodes")
    path = root / "artifacts" / f"shared_horizon_{split_name}.parquet"
    manifest = path.with_suffix(".json")
    config = {"split": split_name, "mode": "B", "panel": "core",
              "horizons": [1, 2, 4], "seed": 17,
              "model": "XGBoost CUDA 7-quantile with horizon numeric feature",
              "source_sha256": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        training, tests = [], []
        for horizon in config["horizons"]:
            frame = load_joined(root, "core", horizon, mode="B")
            frame = frame.loc[frame.report_period.le(SPLITS[split_name].test_end)].copy()
            train, _, test = partition(frame, SPLITS[split_name])
            train["horizon_input"] = horizon
            test["horizon_input"] = horizon
            training.append(train)
            tests.append(test)
        train = pd.concat(training, ignore_index=True)
        test = pd.concat(tests, ignore_index=True)
        names = list(CORE_NUMERIC) + ["horizon_input"]
        pre = NumericPreprocessor(names).fit(train)
        model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=QUANTILES.tolist(),
            device="cuda", tree_method="hist", n_estimators=160, max_depth=4,
            learning_rate=.06, subsample=.85, colsample_bytree=.85,
            min_child_weight=30, reg_lambda=10, max_bin=128,
            random_state=17, n_jobs=8)
        model.fit(pre.transform(train), train.deposit_log_growth.to_numpy(dtype=np.float32))
        if '"device":"cuda:' not in model.get_booster().save_config():
            raise RuntimeError("Shared-horizon backbone did not train on CUDA")
        pred = rearrange(np.asarray(model.predict(pre.transform(test))))
        result = test[["bank_id", "report_period", "horizon_input", "deposit_log_growth",
                       "assets", "deposits"]].rename(columns={
                           "horizon_input": "horizon", "deposit_log_growth": "actual"}).copy()
        for index, q in enumerate(("q01", "q05", "q10", "q25", "q50", "q75", "q90")):
            result[q] = pred[:, index]
        result.to_parquet(path, index=False, compression="zstd")
        comparisons = []
        for horizon in config["horizons"]:
            sub = result.loc[result.horizon.eq(horizon)]
            separate_path = root / "artifacts" / "predictions" / (
                f"core_B_{horizon}q_{split_name}_B7_xgb_quantile.parquet")
            if separate_path.is_file():
                separate = pd.read_parquet(separate_path, columns=["bank_id", "report_period", "actual", "q05"])
                matched = sub.merge(separate, on=["bank_id", "report_period"],
                                    suffixes=("_shared", "_separate"), validate="one_to_one")
                if len(matched) != len(sub) or not np.allclose(
                        matched.actual_shared, matched.actual_separate, atol=1e-11):
                    raise AssertionError("Shared/separate comparison row or target mismatch")
                y = matched.actual_shared.to_numpy()
                shared_error = y - matched.q05_shared.to_numpy()
                separate_error = y - matched.q05_separate.to_numpy()
                shared_loss = float(np.maximum(.05 * shared_error, -.95 * shared_error).mean())
                separate_loss = float(np.maximum(.05 * separate_error, -.95 * separate_error).mean())
            else:
                matched = sub
                scored = score(sub.actual.to_numpy(), sub[["q01", "q05", "q10", "q25",
                                                            "q50", "q75", "q90"]].to_numpy())
                shared_loss = next(item["value"] for item in scored if item["metric"] == "pinball"
                                   and item["quantile"] == "Q05")
                separate_loss = None
            comparisons.append({"horizon": horizon, "rows": len(matched),
                                "shared_q05_pinball": shared_loss,
                                "separate_q05_pinball": separate_loss})
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                   "training_rows": len(train), "test_rows": len(test),
                   "comparisons": comparisons, "prediction_artifact": str(path.relative_to(root)).replace("\\", "/"),
                   "limitations": "Shared model uses the same hyperparameters across horizons; separate reference fits are independently trained."}
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("DELETE FROM ablations WHERE artifact_path = ?", [payload["prediction_artifact"]])
            for item in comparisons:
                con.execute("INSERT INTO ablations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            ["E16", "shared_horizon_XGBoost_CUDA", split_name, "core",
                             item["horizon"], "q05_pinball", item["shared_q05_pinball"],
                             item["separate_q05_pinball"], payload["prediction_artifact"]])
            con.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id='E16'",
                        [payload["prediction_artifact"]])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print("Shared horizon: " + ", ".join(
            f"{item['horizon']}Q={item['shared_q05_pinball']:.6f}" for item in comparisons), flush=True)

    RunState(root).run(f"shared_horizon_{split_name}", "RUNNING_EXPERIMENTS", config,
                       [str(path.relative_to(root)).replace("\\", "/"),
                        str(manifest.relative_to(root)).replace("\\", "/")], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
