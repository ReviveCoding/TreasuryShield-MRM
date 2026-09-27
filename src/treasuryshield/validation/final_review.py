"""Independent final-holdout review with source rejoin and paired uncertainty."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.validation.independent import (
    paired_block_bootstrap, paired_cluster_sensitivity, verify_run)


def _feature_clock(root: Path, run_id: str) -> dict:
    prediction = pd.read_parquet(root / "artifacts" / "predictions" / f"{run_id}.parquet",
                                 columns=["bank_id", "report_period", "forecast_origin"])
    columns = ["bank_id", "report_period", "feature_information_date", "forecast_origin",
               "macro_UNRATE_vintage_date", "macro_CFNAI_vintage_date",
               "macro_CPIAUCSL_vintage_date"]
    features = pd.read_parquet(root / "data" / "features" / "core_features.parquet",
                               columns=columns)
    merged = prediction.merge(features, on=["bank_id", "report_period"],
                              suffixes=("_prediction", "_source"), validate="one_to_one")
    if len(merged) != len(prediction):
        raise AssertionError("Final predictions cannot be joined to source-derived features")
    if not (merged.forecast_origin_prediction == merged.forecast_origin_source).all():
        raise AssertionError("Prediction origin differs from feature origin")
    if merged.feature_information_date.gt(merged.forecast_origin_prediction).any():
        raise AssertionError("Future bank feature in final holdout")
    future_vintages = {}
    for series in ("UNRATE", "CFNAI", "CPIAUCSL"):
        col = f"macro_{series}_vintage_date"
        count = int(merged[col].gt(merged.forecast_origin_prediction).sum())
        future_vintages[series] = count
        if count:
            raise AssertionError(f"Future ALFRED vintage for {series}")
    return {"rows": len(merged), "future_bank_features": 0,
            "future_macro_vintages": future_vintages}


def _final_subgroups(root: Path, run_id: str) -> list[dict]:
    prediction = pd.read_parquet(root / "artifacts" / "predictions" / f"{run_id}.parquet",
                                 columns=["bank_id", "report_period", "actual", "q05",
                                          "assets", "deposits"])
    names = ["bank_id", "report_period", "cash_asset_ratio",
             "uninsured_proxy_domestic_share", "brokered_domestic_ratio",
             "residential_loan_share", "state"]
    features = pd.read_parquet(root / "data" / "features" / "core_features.parquet", columns=names)
    historical = features.loc[features.report_period.between("2001Q1", "2023Q4")]
    cash_cutoff = historical.cash_asset_ratio.median()
    uninsured_cutoff = historical.uninsured_proxy_domestic_share.median()
    merged = prediction.merge(features, on=["bank_id", "report_period"], validate="one_to_one")
    if len(merged) != len(prediction):
        raise AssertionError("Final subgroup feature join incomplete")
    size = pd.cut(merged.assets, [-np.inf, 100_000, 1_000_000, 10_000_000, np.inf],
                  labels=["micro", "small", "medium", "large"]).astype(str)
    business = np.select([merged.brokered_domestic_ratio.ge(.1),
                          merged.residential_loan_share.ge(.5)],
                         ["brokered_funding", "residential_lending"], default="other")
    labels = {
        "bank_size": size,
        "business_model": business,
        "cash_to_assets": np.where(merged.cash_asset_ratio.ge(cash_cutoff), "high", "low"),
        "uninsured_proxy": np.where(merged.uninsured_proxy_domestic_share.ge(uninsured_cutoff),
                                    "high", "low"),
    }
    y = merged.actual.to_numpy(dtype=float)
    q = merged.q05.to_numpy(dtype=float)
    residual = y - q
    loss = np.maximum(.05 * residual, -.95 * residual)
    breach = (residual <= 0).astype(float)
    out = []
    for family, values in labels.items():
        for group in sorted(pd.unique(values)):
            which = np.asarray(values == group)
            if which.sum() < 20:
                continue
            for weighting, weight in (("equal_bank", np.ones(which.sum())),
                                      ("deposit_weighted", merged.deposits.to_numpy()[which]),
                                      ("asset_weighted", merged.assets.to_numpy()[which])):
                out.append({"family": family, "group": str(group), "weighting": weighting,
                            "n": int(which.sum()),
                            "q05_pinball": float(np.average(loss[which], weights=weight)),
                            "q05_coverage": float(np.average(breach[which], weights=weight))})
    return out


def run(root: Path) -> None:
    selection_path = root / "artifacts" / "model_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if sha256_file(selection_path) != (root / "artifacts" / "model_selection.sha256").read_text(
            encoding="ascii").split()[0]:
        raise RuntimeError("Frozen selection was modified")
    run_ids = [item["run_id"] for item in selection["final_comparison_roster"]]
    for run_id in run_ids:
        if not (root / "artifacts" / "predictions" / f"{run_id}.parquet").is_file():
            raise FileNotFoundError(run_id)
    output = root / "artifacts" / "independent_final_review.json"
    config = {"run_ids": run_ids, "selection_sha256": sha256_file(selection_path),
              "prediction_hashes": {run_id: sha256_file(root / "artifacts" / "predictions" /
                                                         f"{run_id}.parquet") for run_id in run_ids},
              "source_hash": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        reconciled = [verify_run(root, run_id) for run_id in run_ids]
        clock = _feature_clock(root, selection["final_run_id"])
        selected = selection["final_run_id"]
        subgroups = _final_subgroups(root, selected)
        comparisons = [paired_block_bootstrap(root, selected, run_id)
                       for run_id in run_ids if run_id != selected]
        sensitivity = []
        for item in selection["final_comparison_roster"]:
            if item["run_id"] == selected or item["model_id"] not in {
                    "B1_peer", "B7_xgb_quantile", "RAHQ_C_nn"}:
                continue
            sensitivity.extend(paired_cluster_sensitivity(root, selected, item["run_id"]))
        prediction = pd.read_parquet(root / "artifacts" / "predictions" / f"{selected}.parquet",
                                     columns=["report_period", "actual", "q05", "assets", "deposits"])
        prediction["breach"] = prediction.actual.lt(prediction.q05)
        prediction["severity"] = np.maximum(prediction.q05 - prediction.actual, 0)
        by_quarter = prediction.groupby("report_period").agg(
            rows=("breach", "size"), q05_breach_frequency=("breach", "mean"),
            q05_tail_underestimate_severity=("severity", "mean")).reset_index().to_dict("records")
        con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
        metric = con.execute("""SELECT metric,quantile,weighting,value,n FROM metrics
            WHERE run_id=? AND metric IN ('pinball','coverage','mae','tail_underestimate_severity')
            AND weighting IN ('equal_bank','deposit_weighted','asset_weighted')""", [selected]).df()
        con.close()
        payload = {"reviewed_at": datetime.now(timezone.utc).isoformat(),
                   "selection_sha256": config["selection_sha256"],
                   "data_snapshot_id": config["snapshot"],
                   "selected_run_id": selected, "reconciled_runs": reconciled,
                   "feature_clock": clock, "paired_time_block_bootstrap": comparisons,
                   "cluster_sensitivity": sensitivity,
                   "selected_final_metrics": metric.to_dict("records"),
                   "selected_quarterly_failures": by_quarter,
                   "selected_final_subgroups": subgroups,
                   "validator_scope": "Independent joins, metric equations, paired uncertainty and timing checks; development model code is not imported.",
                   "limitations": ["Eight or fewer final origin quarters make time-block intervals coarse.",
                                   "Raw Call Reports are harmonized restatements and exact amendment timestamps are absent."]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        writer = connect(root)
        try:
            writer.begin()
            for item in comparisons + sensitivity:
                method = item.get("method", "time_block")
                suffix = "time_block" if method.startswith("paired circular moving") else method
                cid = item["first"] + "__vs__" + item["second"] + "__" + suffix
                writer.execute("DELETE FROM bootstrap_results WHERE comparison_id=?", [cid])
                first = writer.execute("SELECT split,panel,horizon,model_id FROM experiments WHERE run_id=?",
                                       [item["first"]]).fetchone()
                second = writer.execute("SELECT model_id FROM experiments WHERE run_id=?",
                                        [item["second"]]).fetchone()
                writer.execute("INSERT INTO bootstrap_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               [cid, first[0], first[1], first[2], first[3], second[0],
                                "pinball", "Q05", item["estimate"], item["ci_low"],
                                 item["ci_high"], item["replicates"], item["seed"]])
            writer.execute("DELETE FROM subgroup_metrics WHERE run_id=?", [selected])
            for item in subgroups:
                for metric, value in (("pinball", item["q05_pinball"]),
                                      ("coverage", item["q05_coverage"])):
                    writer.execute("INSERT INTO subgroup_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                   [selected, item["family"], item["group"], metric, "Q05",
                                    item["weighting"], value, item["n"]])
            writer.execute("UPDATE experiment_registry SET status='completed', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id='E23'",
                           ["artifacts/independent_final_review.json"])
            writer.commit()
        except Exception:
            writer.rollback()
            raise
        finally:
            writer.close()
        print(f"Independent final review reconciled {len(run_ids)} frozen runs", flush=True)

    RunState(root).run("independent_final_review", "VALIDATION", config,
                       ["artifacts/independent_final_review.json"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
