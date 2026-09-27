"""Secondary external validity against explicitly sourced FDIC failures."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from treasuryshield.orchestration.state import RunState, sha256_file


def evaluate(root: Path, run_ids: list[str]) -> None:
    events = pd.read_parquet(root / "data" / "processed" / "entity_events.parquet",
                             columns=["bank_id", "event_date", "event_type", "source", "mapping_confidence"])
    failures = events.loc[events["event_type"].eq("failed") & events["bank_id"].notna()
                          & events["mapping_confidence"].isin(["high", "medium"])].copy()
    failures = failures.loc[failures["event_date"].dt.year.ge(1984)]
    failures["bank_id"] = failures["bank_id"].astype("int64")
    # Both FDIC history and failure listing may record the same resolution.
    failures = failures.drop_duplicates(["bank_id", "event_date"])
    output = []
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    con.execute("DELETE FROM metrics WHERE experiment_id='E38'")
    for run_id in run_ids:
        prediction = pd.read_parquet(root / "artifacts" / "predictions" / f"{run_id}.parquet",
                                     columns=["bank_id", "report_period", "forecast_origin", "q05"])
        prediction["horizon_end"] = prediction["forecast_origin"] + pd.DateOffset(years=1)
        matching = prediction[["bank_id", "forecast_origin", "horizon_end"]].reset_index(names="row_number").merge(
            failures[["bank_id", "event_date"]], on="bank_id", how="left")
        matching["is_future_failure"] = matching["event_date"].gt(matching["forecast_origin"]) & matching["event_date"].le(matching["horizon_end"])
        label = matching.groupby("row_number")["is_future_failure"].any().reindex(range(len(prediction)), fill_value=False).to_numpy()
        risk = -prediction["q05"].to_numpy(dtype=float)
        positives = int(label.sum())
        prevalence = float(label.mean())
        if positives < 5:
            result = {"run_id": run_id, "n": len(prediction), "future_failures": positives,
                      "prevalence": prevalence, "pr_auc": None, "roc_auc": None,
                      "status": "insufficient_positive_events"}
        else:
            result = {"run_id": run_id, "n": len(prediction), "future_failures": positives,
                      "prevalence": prevalence,
                      "pr_auc": float(average_precision_score(label, risk)),
                      "roc_auc": float(roc_auc_score(label, risk)), "status": "evaluated"}
            details = con.execute("SELECT panel,mode,split,horizon,model_id,config_hash,snapshot_id FROM experiments WHERE run_id=?", [run_id]).fetchone()
            for metric, value in [("fdic_failure_pr_auc", result["pr_auc"]),
                                  ("fdic_failure_roc_auc", result["roc_auc"]),
                                  ("fdic_failure_prevalence", prevalence)]:
                con.execute("INSERT INTO metrics VALUES (?, 'E38', ?, ?, ?, ?, ?, ?, 'none', 'equal_bank', ?, ?, ?, ?)",
                            [run_id, details[0], details[1], details[2], details[3], details[4],
                             metric, value, len(prediction), details[6], details[5]])
        output.append(result)
        print(run_id, result, flush=True)
    con.execute("UPDATE experiment_registry SET status='completed', evidence_paths='artifacts/fdic_external_validity.json', updated_at=current_timestamp WHERE experiment_id='E38'")
    con.close()
    (root / "artifacts" / "fdic_external_validity.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "runs": output,
        "failure_label": "Explicit FDIC history or failures event within one year; high/medium RSSD mapping only",
        "use_limit": "Ranking external validity only, not calibrated probability of failure",
    }, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    run_ids = ["core_A_1q_gfc_B1_peer", "core_A_1q_gfc_B7_xgb_quantile",
               "core_A_1q_gfc_B8_lightgbm_quantile", "core_B_1q_gfc_B8_lightgbm_quantile"]
    selection_path = root / "artifacts" / "model_selection.json"
    if selection_path.is_file():
        selected = json.loads(selection_path.read_text(encoding="utf-8"))["final_run_id"]
        if (root / "artifacts" / "predictions" / f"{selected}.parquet").is_file():
            run_ids.append(selected)
    RunState(root).run(
        "fdic_external_validity", "VALIDATING",
        {"event_sha256": sha256_file(root / "data" / "processed" / "entity_events.parquet"),
         "runs": run_ids}, ["artifacts/fdic_external_validity.json"],
        lambda: evaluate(root, run_ids), resume=True)


if __name__ == "__main__":
    main()
