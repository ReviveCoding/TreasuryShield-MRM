"""Training-only LightGBM Q05 gain importance across historical cutoffs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.splits.temporal import SPLITS, deterministic_sample, load_joined, partition


def run(root: Path) -> None:
    output = root / "artifacts" / "feature_importance_prelock.parquet"
    manifest = output.with_suffix(".json")
    config = {"model": "LightGBM Q05", "mode": "B", "folds": ["gfc", "pandemic", "tightening"],
              "training_rows_max": 150_000, "seed": 17,
              "snapshot": RunState(root).data["data_snapshot_id"],
              "source_hash": sha256_file(Path(__file__))}

    def execute() -> None:
        frame = load_joined(root, "core", 1, mode="B")
        pieces = []
        counts = {}
        for split_name in config["folds"]:
            eligible = frame.loc[frame.report_period.le(SPLITS[split_name].test_end)]
            train, _, _ = partition(eligible, SPLITS[split_name])
            sampled = deterministic_sample(train, config["training_rows_max"], 17)
            pre = NumericPreprocessor(list(CORE_NUMERIC)).fit(sampled)
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=.05, device_type="cpu", n_estimators=160,
                num_leaves=31, max_depth=6, min_child_samples=100, learning_rate=.06,
                subsample=.85, colsample_bytree=.85, reg_lambda=10,
                random_state=17, n_jobs=8, verbosity=-1)
            model.fit(pre.transform(sampled), sampled.deposit_log_growth.to_numpy())
            gain = model.booster_.feature_importance(importance_type="gain")
            share = gain / gain.sum() if gain.sum() else np.zeros_like(gain, dtype=float)
            pieces.append(pd.DataFrame({"split": split_name, "feature": pre.output_names,
                                        "gain": gain, "gain_share": share,
                                        "training_rows": len(sampled)}))
            counts[split_name] = len(sampled)
        result = pd.concat(pieces, ignore_index=True)
        result.to_parquet(output, index=False, compression="zstd")
        ranks = result.pivot(index="feature", columns="split", values="gain_share").rank()
        stability = {f"{a}_vs_{b}": float(ranks[a].corr(ranks[b], method="spearman"))
                     for a, b in (("gfc", "pandemic"), ("gfc", "tightening"),
                                  ("pandemic", "tightening"))}
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "config_hash": stable_hash(config), "data_snapshot_id": config["snapshot"],
                   "training_rows": counts, "rank_stability": stability,
                   "importance_method": "LightGBM gain, training rows only, Q05 objective",
                   "limitation": "Gain importance is associational and can split credit among correlated variables."}
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("CREATE TABLE IF NOT EXISTS feature_importance (split VARCHAR, feature VARCHAR, gain DOUBLE, gain_share DOUBLE, training_rows BIGINT, artifact_path VARCHAR)")
            con.execute("DELETE FROM feature_importance WHERE artifact_path='artifacts/feature_importance_prelock.parquet'")
            incoming = result.copy()
            incoming["artifact_path"] = "artifacts/feature_importance_prelock.parquet"
            con.register("incoming_importance", incoming)
            con.execute("INSERT INTO feature_importance SELECT split, feature, gain, gain_share, training_rows, artifact_path FROM incoming_importance")
            con.unregister("incoming_importance")
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print(f"Feature importance complete: {len(result)} fold-feature rows", flush=True)

    RunState(root).run("feature_importance_prelock", "RUNNING_EXPERIMENTS", config,
                       ["artifacts/feature_importance_prelock.parquet",
                        "artifacts/feature_importance_prelock.json"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
