"""Independent outcome and implementation checks on frozen prediction artifacts.

This module does not call development metric or model functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file


def independent_pinball(actual: np.ndarray, estimate: np.ndarray, quantile: float) -> np.ndarray:
    residual = actual - estimate
    return np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)


def verify_run(root: Path, run_id: str) -> dict:
    base = root / "artifacts" / "predictions"
    path = base / f"{run_id}.parquet"
    info = json.loads((base / f"{run_id}.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(path)
    if len(frame) != info["test_rows"]:
        raise AssertionError("Prediction count differs from recorded test count")
    if frame.duplicated(["bank_id", "report_period", "horizon"]).any():
        raise AssertionError("Duplicate bank-origin-horizon prediction")
    if frame["snapshot_id"].nunique() != 1 or frame["snapshot_id"].iloc[0] != info["snapshot_id"]:
        raise AssertionError("Snapshot mismatch")
    if frame["config_hash"].nunique() != 1 or frame["config_hash"].iloc[0] != info["config_hash"]:
        raise AssertionError("Config hash mismatch")
    if frame[["q01", "q05", "q10", "q25", "q50", "q75", "q90"]].isna().any().any():
        raise AssertionError("Missing predicted quantile")
    if np.any(np.diff(frame[["q01", "q05", "q10", "q25", "q50", "q75", "q90"]].to_numpy(), axis=1) < -1e-10):
        raise AssertionError("Postprocessed quantile crossing")
    labels = pd.read_parquet(root / "data" / "features" / "targets.parquet",
                             columns=["bank_id", "report_period", "horizon_quarters",
                                      "deposit_log_growth", "target_availability_date"])
    labels = labels.loc[labels["horizon_quarters"].eq(info["horizon"])].drop(columns="horizon_quarters")
    audited = frame.merge(labels, on=["bank_id", "report_period"], validate="one_to_one")
    if len(audited) != len(frame):
        raise AssertionError("Prediction cannot be reconciled to official-source target")
    if not np.allclose(audited["actual"], audited["deposit_log_growth"], atol=1e-11):
        raise AssertionError("Stored actual differs from independently rejoined target")
    if audited["target_availability_date"].le(audited["forecast_origin"]).any():
        raise AssertionError("Outcome available at or before forecast origin")
    y = audited["actual"].to_numpy(dtype=float)
    q05 = audited["q05"].to_numpy(dtype=float)
    q05_loss = float(independent_pinball(y, q05, .05).mean())
    coverage = float(np.mean(y <= q05))
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"), read_only=True)
    stored = con.execute("""SELECT value FROM metrics WHERE run_id = ? AND metric='pinball'
        AND quantile='Q05' AND weighting='equal_bank'""", [run_id]).fetchone()
    stored_coverage = con.execute("""SELECT value FROM metrics WHERE run_id = ? AND metric='coverage'
        AND quantile='Q05' AND weighting='equal_bank'""", [run_id]).fetchone()
    hashed = con.execute("SELECT artifact_sha256 FROM experiments WHERE run_id = ?", [run_id]).fetchone()
    con.close()
    if stored is None or stored_coverage is None or hashed is None:
        raise AssertionError("Warehouse metric or experiment record missing")
    if abs(q05_loss - stored[0]) > 1e-10 or abs(coverage - stored_coverage[0]) > 1e-10:
        raise AssertionError("Independent metric differs from development warehouse")
    if hashed[0] != sha256_file(path):
        raise AssertionError("Prediction artifact hash changed")
    result = {"run_id": run_id, "rows": len(frame), "q05_pinball_independent": q05_loss,
              "q05_coverage_independent": coverage, "predictions_sha256": hashed[0],
              "metric_reconciliation": "passed", "target_rejoin": "passed",
              "postprocessed_quantile_crossing": 0}
    return result


def paired_block_bootstrap(root: Path, first_run: str, second_run: str,
                           *, replicates: int = 1000, seed: int = 1701) -> dict:
    base = root / "artifacts" / "predictions"
    first = pd.read_parquet(base / f"{first_run}.parquet",
                            columns=["bank_id", "report_period", "actual", "q05"])
    second = pd.read_parquet(base / f"{second_run}.parquet",
                             columns=["bank_id", "report_period", "actual", "q05"])
    paired = first.merge(second, on=["bank_id", "report_period"], suffixes=("_first", "_second"), validate="one_to_one")
    if len(paired) != len(first) or len(paired) != len(second):
        raise AssertionError("Bootstrap comparison requires identical evaluated rows")
    if not np.allclose(paired["actual_first"], paired["actual_second"], atol=1e-11):
        raise AssertionError("Paired outcomes differ")
    y = paired["actual_first"].to_numpy()
    difference = independent_pinball(y, paired["q05_first"].to_numpy(), .05) - independent_pinball(
        y, paired["q05_second"].to_numpy(), .05)
    paired["difference"] = difference
    quarter = paired.groupby("report_period")["difference"].agg(["sum", "size"])
    periods = quarter.index.to_list()
    sums = quarter["sum"].to_numpy()
    counts = quarter["size"].to_numpy()
    estimate = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    block_length = min(4, len(periods))
    blocks = [np.mod(np.arange(start, start + block_length), len(periods))
              for start in range(len(periods))]
    for iteration in range(replicates):
        selected = rng.integers(0, len(blocks), size=int(np.ceil(len(periods) / block_length)))
        indices = np.concatenate([blocks[index] for index in selected])[: len(periods)]
        draws[iteration] = sums[indices].sum() / counts[indices].sum()
    low, high = np.quantile(draws, [.025, .975])
    return {"first": first_run, "second": second_run,
            "metric": "Q05_pinball_first_minus_second", "estimate": estimate,
            "ci_low": float(low), "ci_high": float(high),
            "replicates": replicates, "seed": seed,
            "method": "paired circular moving 4-quarter block bootstrap, all banks within selected blocks",
            "evaluated_rows": len(paired), "quarters": len(periods)}


def paired_cluster_sensitivity(root: Path, first_run: str, second_run: str,
                               *, replicates: int = 1000, seed: int = 1701) -> list[dict]:
    """Bank-cluster and crossed bank/time-block uncertainty sensitivity."""
    base = root / "artifacts" / "predictions"
    first = pd.read_parquet(base / f"{first_run}.parquet",
                            columns=["bank_id", "report_period", "actual", "q05"])
    second = pd.read_parquet(base / f"{second_run}.parquet",
                             columns=["bank_id", "report_period", "actual", "q05"])
    pair = first.merge(second, on=["bank_id", "report_period"],
                       suffixes=("_first", "_second"), validate="one_to_one")
    if len(pair) != len(first) or len(pair) != len(second):
        raise AssertionError("Cluster sensitivity requires identical evaluation rows")
    if not np.allclose(pair.actual_first, pair.actual_second, atol=1e-11):
        raise AssertionError("Paired actuals differ")
    y = pair.actual_first.to_numpy()
    pair["difference"] = (independent_pinball(y, pair.q05_first.to_numpy(), .05) -
                          independent_pinball(y, pair.q05_second.to_numpy(), .05))
    matrix = pair.pivot(index="bank_id", columns="report_period", values="difference")
    values = matrix.to_numpy(dtype=float)
    present = np.isfinite(values).astype(float)
    values = np.nan_to_num(values)
    banks, quarters = values.shape
    estimate = float(values.sum() / present.sum())
    rng = np.random.default_rng(seed)
    bank_draws = np.empty(replicates)
    two_way_draws = np.empty(replicates)
    block_len = min(4, quarters)
    blocks = [np.mod(np.arange(start, start + block_len), quarters)
              for start in range(quarters)]
    for iteration in range(replicates):
        bank_count = np.bincount(rng.integers(0, banks, size=banks), minlength=banks)
        bank_draws[iteration] = (bank_count @ values.sum(axis=1)) / (bank_count @ present.sum(axis=1))
        selected = rng.integers(0, quarters, size=int(np.ceil(quarters / block_len)))
        indices = np.concatenate([blocks[index] for index in selected])[:quarters]
        quarter_count = np.bincount(indices, minlength=quarters)
        weighted = bank_count[:, None] * quarter_count[None, :]
        denominator = np.sum(weighted * present)
        two_way_draws[iteration] = np.sum(weighted * values) / denominator
    results = []
    for method, draws in (("bank_cluster", bank_draws),
                          ("crossed_bank_and_circular_4q_block", two_way_draws)):
        low, high = np.quantile(draws, [.025, .975])
        results.append({"first": first_run, "second": second_run, "method": method,
                        "estimate": estimate, "ci_low": float(low), "ci_high": float(high),
                        "replicates": replicates, "seed": seed, "banks": banks,
                        "quarters": quarters, "evaluated_rows": len(pair)})
    return results


def write_validation(root: Path, run_ids: list[str], comparisons: list[tuple[str, str]]) -> None:
    reconciled = [verify_run(root, run_id) for run_id in run_ids]
    bootstrap = [paired_block_bootstrap(root, a, b) for a, b in comparisons]
    out = root / "artifacts" / "independent_validation_checks.json"
    out.write_text(json.dumps({"runs": reconciled, "paired_bootstrap": bootstrap}, indent=2), encoding="utf-8")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    for item in bootstrap:
        first = con.execute("SELECT split,panel,horizon,model_id FROM experiments WHERE run_id=?", [item["first"]]).fetchone()
        second = con.execute("SELECT model_id FROM experiments WHERE run_id=?", [item["second"]]).fetchone()
        comparison_id = item["first"] + "__vs__" + item["second"]
        con.execute("DELETE FROM bootstrap_results WHERE comparison_id=?", [comparison_id])
        con.execute("INSERT INTO bootstrap_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [comparison_id, first[0], first[1], first[2], first[3], second[0],
                     "pinball", "Q05", item["estimate"], item["ci_low"], item["ci_high"],
                     item["replicates"], item["seed"]])
    con.close()
    print(f"Independent validator reconciled {len(reconciled)} runs and {len(bootstrap)} paired comparisons", flush=True)
