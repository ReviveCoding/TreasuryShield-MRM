"""Checkpointed distributional-model run and warehouse persistence."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from treasuryshield.development.models import fit_model, peer_group
from treasuryshield.development.context import enrich
from treasuryshield.calibration.rolling import calibrate
from treasuryshield.features.build import CORE_NUMERIC, ENHANCED_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.splits.temporal import SPLITS, load_joined, partition
from treasuryshield.statistics.metrics import crossing_rate, rearrange, score

QCOLS = ["q01", "q05", "q10", "q25", "q50", "q75", "q90"]
QVALUES = [.01, .05, .10, .25, .50, .75, .90]
SPLIT_GROUP = {"gfc": "E20", "routine_2018_19": "E02", "pandemic": "E21",
               "tightening": "E22", "final_locked": "E23"}


class ResourceSampler:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak_ram = 0
        self.peak_gpu_memory_mb = 0
        self.peak_gpu_utilization = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_ram = max(self.peak_ram, process.memory_info().rss)
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    timeout=2, text=True, stderr=subprocess.DEVNULL)
                memory, utilization = [int(x.strip()) for x in output.splitlines()[0].split(",")]
                self.peak_gpu_memory_mb = max(self.peak_gpu_memory_mb, memory)
                self.peak_gpu_utilization = max(self.peak_gpu_utilization, utilization)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop.wait(.7)

    def __enter__(self) -> "ResourceSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=3)


def _write_warehouse(root: Path, run_id: str, experiment_id: str,
                     info: dict, predictions: pd.DataFrame, metrics: list[dict],
                     prediction_path: Path) -> None:
    con = connect(root)
    relpath = str(prediction_path.relative_to(root)).replace("\\", "/")
    con.begin()
    try:
        con.execute("DELETE FROM experiments WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM predictions WHERE run_id = ?", [run_id])
        con.execute("DELETE FROM metrics WHERE run_id = ?", [run_id])
        con.execute("""INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)""", [
            run_id, experiment_id, info["panel"], info["mode"], "deposit_log_growth",
            info["horizon"], info["split"], info["model_id"], info["seed"],
            json.dumps(info["config"], sort_keys=True), info["config_hash"],
            info["snapshot_id"], "completed", info["runtime_seconds"],
            info["peak_ram_bytes"], info["peak_vram_bytes"], relpath,
            sha256_file(prediction_path), None,
        ])
        tall = predictions.melt(
            id_vars=["run_id", "bank_id", "report_period", "forecast_origin", "horizon",
                     "panel", "model_id", "actual", "assets", "deposits", "peer_group",
                     "regime", "snapshot_id", "config_hash", "seed"],
            value_vars=QCOLS, var_name="quantile_name", value_name="prediction")
        tall["quantile"] = tall["quantile_name"].map(dict(zip(QCOLS, QVALUES)))
        con.register("incoming_predictions", tall)
        con.execute("""INSERT INTO predictions SELECT run_id, bank_id, report_period,
            forecast_origin, horizon, panel, model_id, quantile, prediction, actual,
            assets, deposits, peer_group, regime, snapshot_id, config_hash, seed
            FROM incoming_predictions""")
        con.unregister("incoming_predictions")
        metric_table = pd.DataFrame(metrics)
        metric_table["run_id"] = run_id
        metric_table["experiment_id"] = experiment_id
        metric_table["panel"] = info["panel"]
        metric_table["mode"] = info["mode"]
        metric_table["split"] = info["split"]
        metric_table["horizon"] = info["horizon"]
        metric_table["model_id"] = info["model_id"]
        metric_table["snapshot_id"] = info["snapshot_id"]
        metric_table["config_hash"] = info["config_hash"]
        con.register("incoming_metrics", metric_table)
        con.execute("""INSERT INTO metrics SELECT run_id, experiment_id, panel, mode,
            split, horizon, model_id, metric, quantile, weighting, value, n,
            snapshot_id, config_hash FROM incoming_metrics""")
        con.unregister("incoming_metrics")
        con.execute("UPDATE experiment_registry SET status='partial', evidence_paths=?, updated_at=current_timestamp WHERE experiment_id=?",
                    [relpath, experiment_id])
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def run(root: Path, model_id: str, split_name: str, panel: str, horizon: int,
        *, mode: str = "A", seed: int = 17, model_config: dict | None = None,
        frame_override: pd.DataFrame | None = None,
        mna_radius: int | None = 1) -> str:
    if split_name not in SPLITS:
        raise KeyError(split_name)
    if panel == "enhanced" and split_name == "gfc":
        raise ValueError("Enhanced data begin in 2017 and cannot support the GFC split")
    model_config = model_config or {}
    run_id = f"{panel}_{mode}_{horizon}q_{split_name}_{model_id}"
    if mna_radius != 1:
        run_id += f"_mna{mna_radius if mna_radius is not None else 'none'}"
    if model_config:
        run_id += "_" + stable_hash(model_config)[:8]
    output_dir = root / "artifacts" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{run_id}.parquet"
    info_path = output_dir / f"{run_id}.json"
    metrics_path = output_dir / f"{run_id}_metrics.json"
    relative_outputs = [str(p.relative_to(root)).replace("\\", "/")
                        for p in [prediction_path, info_path, metrics_path]]
    names = CORE_NUMERIC if panel == "core" else ENHANCED_NUMERIC
    code_paths = ["development/models.py", "development/preprocess.py",
                  "splits/temporal.py", "statistics/metrics.py"]
    if model_id in {"B10_mlp", "B10_gru"}:
        code_paths += ["development/neural.py"]
    if model_id.startswith("RAHQ"):
        code_paths += ["development/context.py", "peers/context.py",
                       "regimes/filtered.py", "calibration/rolling.py", "development/neural.py"]
    elif model_config.get("calibration_kind", "none") != "none":
        code_paths += ["calibration/rolling.py"]
    config = {
        "model_id": model_id, "split": split_name, "panel": panel,
        "horizon": horizon, "mode": mode, "seed": seed,
        "mna_radius": mna_radius,
        "model_config": model_config, "features": names,
        "snapshot_id": RunState(root).data["data_snapshot_id"],
        "preregistration_sha256": sha256_file(root / "reports" / "preregistration.md"),
        "source_sha256": {name: sha256_file(root / "src" / "treasuryshield" / name)
                          for name in code_paths},
    }
    config_hash = stable_hash(config)
    if split_name == "final_locked":
        selection_path = root / "artifacts" / "model_selection.json"
        hash_path = root / "artifacts" / "model_selection.sha256"
        if not selection_path.is_file() or not hash_path.is_file():
            raise RuntimeError("Final locked evaluation requires frozen model selection")
        if sha256_file(selection_path) != hash_path.read_text(encoding="ascii").split()[0]:
            raise RuntimeError("Frozen model selection hash mismatch")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if run_id not in {item["run_id"] for item in selection["final_comparison_roster"]}:
            raise RuntimeError(f"Final run {run_id} was not frozen in the comparison roster")
        if sha256_file(Path(__file__)) != selection["source_hashes_at_freeze"][
                "src/treasuryshield/development/run.py"]:
            raise RuntimeError("Final model runner changed after selection")
        for name in code_paths:
            rel = f"src/treasuryshield/{name}"
            if config["source_sha256"][name] != selection["source_hashes_at_freeze"][rel]:
                raise RuntimeError(f"Final model implementation changed after selection: {rel}")

    def execute() -> None:
        started = time.perf_counter()
        frame = frame_override if frame_override is not None else load_joined(root, panel, horizon, mode=mode)
        frame = frame.loc[frame["report_period"].le(SPLITS[split_name].test_end)].copy()
        context_metadata = {}
        run_names = list(names)
        if model_id.startswith("RAHQ"):
            frame, context_names, context_metadata = enrich(
                frame, root, panel, SPLITS[split_name],
                peer_kind=model_config.get("peer_kind", "rule"),
                regime_kind=model_config.get("regime_kind", "hmm_filtered"),
                states=model_config.get("states", 3), seed=seed)
            run_names += context_names
            removals = set(model_config.get("remove_families", []))
            if "peer" in removals:
                run_names = [name for name in run_names if not name.startswith(("peer_", "own_minus_peer_"))]
            if "macro" in removals:
                run_names = [name for name in run_names if not name.startswith("macro_")]
            if "regime" in removals:
                run_names = [name for name in run_names if not name.startswith("regime_")]
            if "dynamics" in removals:
                run_names = [name for name in run_names if name not in {
                    "deposit_growth_qoq_lagged", "deposit_growth_yoy_lagged",
                    "asset_growth_qoq_lagged", "cash_ratio_change_qoq_lagged",
                    "borrowings_change_asset_lagged", "deposit_growth_rolling_vol_4q",
                    "deposit_growth_acceleration"}]
            if "deposit_composition" in removals:
                run_names = [name for name in run_names if name not in {
                    "brokered_domestic_ratio", "uninsured_proxy_domestic_share",
                    "uninsured_time_deposit_ratio", "uninsured_domestic_share",
                    "brokered_domestic_share", "transaction_domestic_share",
                    "nontransaction_domestic_share", "large_time_domestic_share"}]
            if "rate_sensitivity" in removals:
                run_names = [name for name in run_names if name not in {
                    "deposit_beta_4q", "deposit_cost_annualized",
                    "macro_DFF_revised", "macro_DFF_change_4q_revised"}]
        train, calibration, test = partition(frame, SPLITS[split_name], mna_radius=mna_radius)
        if split_name == "final_locked" and sha256_file(root / "reports" / "preregistration.md") != (
                root / "reports" / "preregistration.sha256").read_text(encoding="ascii").split()[0]:
            raise RuntimeError("Preregistration hash changed before final locked scoring")
        with ResourceSampler() as resources:
            result = fit_model(model_id, train, calibration, test, run_names, seed=seed,
                               config=model_config, history=frame)
        raw_crossing = crossing_rate(result.test)
        calibration_kind = model_config.get(
            "calibration_kind", "global" if model_id.startswith("RAHQ_C") else "none")
        final_prediction, calibration_metadata = calibrate(
            calibration, result.calibration, test, result.test, kind=calibration_kind)
        metrics = score(test["deposit_log_growth"].to_numpy(), final_prediction,
                        assets=test["assets"].to_numpy(), deposits=test["deposits"].to_numpy())
        metrics.append({"metric": "raw_crossing_rate", "quantile": "all", "weighting": "equal_bank",
                        "value": raw_crossing, "n": len(test)})
        metrics.append({"metric": "training_rows", "quantile": "all", "weighting": "equal_bank",
                        "value": float(result.metadata.get("training_rows", len(train))), "n": len(train)})
        predictions = test[["bank_id", "report_period", "forecast_origin", "assets", "deposits"]].copy()
        predictions["actual"] = test["deposit_log_growth"].to_numpy()
        predictions["peer_group"] = peer_group(test)
        predictions["regime"] = (test["regime_hard"].astype(str).to_numpy()
                                 if "regime_hard" in test else "unassigned")
        for index, qname in enumerate(QCOLS):
            predictions[qname] = final_prediction[:, index]
        predictions["run_id"] = run_id
        predictions["horizon"] = horizon
        predictions["panel"] = panel
        predictions["model_id"] = model_id
        predictions["snapshot_id"] = config["snapshot_id"]
        predictions["config_hash"] = config_hash
        predictions["seed"] = seed
        predictions.to_parquet(prediction_path, index=False, compression="zstd")
        info = {
            "run_id": run_id, "model_id": model_id, "split": split_name, "panel": panel,
            "horizon": horizon, "mode": mode, "seed": seed, "config": config,
            "config_hash": config_hash, "snapshot_id": config["snapshot_id"],
            "train_rows": len(train), "calibration_rows": len(calibration), "test_rows": len(test),
            "training_metadata": result.metadata, "context_metadata": context_metadata,
            "calibration_metadata": calibration_metadata,
            "feature_names": run_names, "raw_crossing_rate": raw_crossing,
            "runtime_seconds": time.perf_counter() - started,
            "peak_ram_bytes": resources.peak_ram,
            "peak_vram_bytes": resources.peak_gpu_memory_mb * 1024 * 1024,
            "peak_gpu_utilization_pct": resources.peak_gpu_utilization,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        info_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_warehouse(root, run_id, SPLIT_GROUP[split_name], info, predictions, metrics, prediction_path)
        primary = next(x for x in metrics if x["metric"] == "pinball" and x["quantile"] == "Q05" and x["weighting"] == "equal_bank")
        print(f"{run_id}: n={len(test)} Q05 pinball={primary['value']:.6f} runtime={info['runtime_seconds']:.1f}s", flush=True)

    RunState(root).run(
        f"model_{run_id}", "RUNNING_EXPERIMENTS", config, relative_outputs,
        execute, resume=True)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--panel", choices=["core", "enhanced"], default="core")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--mode", choices=["A", "B"], default="A")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-config", default="{}")
    parser.add_argument("--mna-radius", default="1", choices=["0", "1", "2", "none"])
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[3], args.model, args.split, args.panel,
        args.horizon, mode=args.mode, seed=args.seed,
        model_config=json.loads(args.model_config),
        mna_radius=None if args.mna_radius == "none" else int(args.mna_radius))


if __name__ == "__main__":
    main()
