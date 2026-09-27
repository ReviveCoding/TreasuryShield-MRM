"""Matched CPU/GPU benchmarks on actual bank training rows."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import torch
import xgboost as xgb

from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file
from treasuryshield.splits.temporal import SPLITS, deterministic_sample, load_joined, partition


def build(root: Path) -> None:
    frame = load_joined(root, "core", 1, mode="A")
    train, _, test = partition(frame, SPLITS["gfc"])
    train = deterministic_sample(train, 100_000, 17)
    test = deterministic_sample(test, 25_000, 71)
    pre = NumericPreprocessor(CORE_NUMERIC).fit(train)
    x_train = pre.transform(train)
    y_train = train["deposit_log_growth"].to_numpy(dtype=np.float32)
    x_test = pre.transform(test)
    y_test = test["deposit_log_growth"].to_numpy(dtype=np.float32)
    records = []
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for required representative benchmark")
        model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=[.01, .05, .10, .25, .5, .75, .9],
            tree_method="hist", device=device, n_estimators=80, max_depth=4,
            learning_rate=.06, subsample=.85, colsample_bytree=.85,
            min_child_weight=30, reg_lambda=10, max_bin=128, random_state=17, n_jobs=8)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predicted = model.predict(x_test)
        predict_seconds = time.perf_counter() - start
        config = model.get_booster().save_config()
        if f'"device":"{device}' not in config:
            raise RuntimeError(f"XGBoost benchmark device mismatch: {device}")
        records.append({"workload": "xgboost_7quantile_100k_bank_rows_80trees", "device": device,
                        "training_seconds": train_seconds, "prediction_seconds": predict_seconds,
                        "input_rows": len(train), "test_rows": len(test),
                        "q05_pinball": float(np.maximum(.05 * (y_test - predicted[:, 1]),
                                                        -.95 * (y_test - predicted[:, 1])).mean()),
                        "verified_booster_device": device})
        print(f"XGBoost benchmark {device}: train {train_seconds:.2f}s predict {predict_seconds:.2f}s", flush=True)
    # Identical small MLP architecture, data, batches, and epochs on CPU/CUDA.
    linear_pre = NumericPreprocessor(CORE_NUMERIC, linear=True).fit(train)
    neural_x = linear_pre.transform(train.iloc[:40_000])
    neural_y = y_train[:40_000]
    for device_name in ["cpu", "cuda"]:
        device = torch.device(device_name)
        torch.manual_seed(17)
        network = torch.nn.Sequential(torch.nn.Linear(neural_x.shape[1], 128),
                                      torch.nn.ReLU(), torch.nn.Linear(128, 7)).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=.001)
        quantiles = torch.tensor([.01, .05, .10, .25, .5, .75, .9], device=device)[None, :]
        if device_name == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for epoch in range(4):
            permutation = np.random.default_rng(17 + epoch).permutation(len(neural_x))
            for positions in np.array_split(permutation, 20):
                xb = torch.as_tensor(neural_x[positions], device=device)
                yb = torch.as_tensor(neural_y[positions], device=device)[:, None]
                error = yb - network(xb)
                loss = torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        if device_name == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        records.append({"workload": "pytorch_linear_mlp_40k_bank_rows_4epochs", "device": device_name,
                        "training_seconds": seconds, "input_rows": len(neural_x),
                        "final_loss": float(loss.detach().cpu()),
                        "peak_torch_vram_bytes": int(torch.cuda.max_memory_allocated()) if device_name == "cuda" else 0,
                        "actual_tensor_device": str(next(network.parameters()).device)})
        print(f"PyTorch benchmark {device_name}: train {seconds:.2f}s", flush=True)
    out = root / "artifacts" / "hardware_benchmarks.json"
    out.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
                               "data_snapshot_id": RunState(root).data["data_snapshot_id"],
                               "matched_workloads": records}, indent=2), encoding="utf-8")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    con.execute("DELETE FROM hardware_benchmarks WHERE artifact_path = 'artifacts/hardware_benchmarks.json'")
    for record in records:
        con.execute("INSERT INTO hardware_benchmarks VALUES (?, ?, ?, ?, ?, ?)",
                    [record["workload"], record["device"], record["training_seconds"],
                     record["input_rows"], json.dumps(record), "artifacts/hardware_benchmarks.json"])
    con.execute("UPDATE experiment_registry SET status='completed', evidence_paths='artifacts/hardware_benchmarks.json', updated_at=current_timestamp WHERE experiment_id='E40'")
    con.close()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    RunState(root).run(
        "cpu_gpu_benchmark", "RUNNING_EXPERIMENTS",
        {"data_sha256": sha256_file(root / "data" / "features" / "core_features.parquet"),
         "target_sha256": sha256_file(root / "data" / "features" / "targets.parquet"),
         "rows_xgb": 100000, "trees": 80, "rows_torch": 40000, "epochs_torch": 4},
        ["artifacts/hardware_benchmarks.json"], lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
