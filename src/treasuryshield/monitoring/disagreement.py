"""Independent challenger-disagreement diagnostic on prelock episodes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect


MODELS = ("B1_peer", "B5_linear_quantile", "B7_xgb_quantile", "B8_lightgbm_quantile")
SPLITS = ("gfc", "pandemic", "tightening")


def run(root: Path) -> None:
    output = root / "artifacts" / "disagreement_prelock.json"
    input_paths = [root / "artifacts" / "predictions" / f"core_A_1q_{split}_{model}.parquet"
                   for split in SPLITS for model in MODELS]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    config = {"run_ids": [path.stem for path in input_paths],
              "input_hashes": {path.stem: sha256_file(path) for path in input_paths},
              "source_hash": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        episodes = []
        for split in SPLITS:
            frames = []
            for model in MODELS:
                path = root / "artifacts" / "predictions" / f"core_A_1q_{split}_{model}.parquet"
                frame = pd.read_parquet(path, columns=["bank_id", "report_period", "actual", "q05"])
                frame = frame.rename(columns={"q05": model, "actual": f"actual_{model}"})
                frames.append(frame)
            paired = frames[0]
            for other in frames[1:]:
                paired = paired.merge(other, on=["bank_id", "report_period"], validate="one_to_one")
            actuals = paired[[f"actual_{model}" for model in MODELS]].to_numpy()
            if not np.allclose(actuals, actuals[:, :1], atol=1e-11):
                raise AssertionError("Disagreement outcomes do not reconcile")
            q = paired[list(MODELS)].to_numpy(dtype=float)
            spread = q.max(axis=1) - q.min(axis=1)
            y = actuals[:, 0]
            selected = paired["B8_lightgbm_quantile"].to_numpy(dtype=float)
            tail_miss = y < selected - .02
            abs_error = np.abs(y - selected)
            threshold = np.quantile(spread, .9)
            top = spread >= threshold
            prevalence = float(tail_miss.mean())
            episode = {"split": split, "rows": len(paired),
                       "spread_median": float(np.median(spread)),
                       "spread_p90": float(threshold),
                       "tail_miss_definition": "actual < B8 Q05 minus 0.02 log points",
                       "tail_miss_prevalence": prevalence,
                       "tail_miss_rate_high_disagreement": float(tail_miss[top].mean()),
                       "tail_miss_lift_at_top_decile": float(tail_miss[top].mean() / prevalence)
                       if prevalence > 0 else float("nan"),
                       "spearman_spread_abs_error": float(spearmanr(spread, abs_error).statistic),
                       "roc_auc_tail_miss": float(roc_auc_score(tail_miss, spread))
                       if 0 < tail_miss.sum() < len(tail_miss) else float("nan"),
                       "pr_auc_tail_miss": float(average_precision_score(tail_miss, spread))
                       if tail_miss.any() else float("nan")}
            episodes.append(episode)
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "config_hash": stable_hash(config), "data_snapshot_id": config["snapshot"],
                   "candidate_models": list(MODELS), "episodes": episodes,
                   "use_gate": "Escalation signal only if lift is stable across all three prelock episodes; no final-holdout tuning."}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("DELETE FROM monitoring WHERE run_id='disagreement_prelock'")
            for episode in episodes:
                for metric in ("tail_miss_prevalence", "tail_miss_rate_high_disagreement",
                               "tail_miss_lift_at_top_decile", "spearman_spread_abs_error",
                               "roc_auc_tail_miss", "pr_auc_tail_miss"):
                    con.execute("INSERT INTO monitoring VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                ["disagreement_prelock", episode["split"], "four_challenger_spread",
                                 "RESEARCH", metric, episode[metric], False,
                                 "artifacts/disagreement_prelock.json"])
            con.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id='E37'",
                        ["artifacts/disagreement_prelock.json"])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()
        print("Disagreement diagnostic: " + ", ".join(
            f"{item['split']} lift={item['tail_miss_lift_at_top_decile']:.2f}"
            for item in episodes), flush=True)

    RunState(root).run("disagreement_prelock", "RUNNING_EXPERIMENTS", config,
                       ["artifacts/disagreement_prelock.json"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
