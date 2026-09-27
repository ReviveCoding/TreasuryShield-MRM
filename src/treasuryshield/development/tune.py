"""Bounded CUDA Optuna search with inner temporal folds and pruning."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import optuna
import pandas as pd

from treasuryshield.development.context import enrich
from treasuryshield.development.models import xgboost_quantile_cuda
from treasuryshield.features.build import CORE_NUMERIC, ENHANCED_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.splits.temporal import Split, deterministic_sample, load_joined, partition

FOLDS = [
    Split("inner_2016_17", "2015Q4", "2016Q1", "2017Q4"),
    Split("inner_2018_19", "2017Q4", "2018Q1", "2019Q4"),
]


def tune(root: Path, panel: str = "core", mode: str = "B",
         model_family: str = "xgb", trials: int = 8) -> None:
    if model_family not in {"xgb", "rahq_tree"}:
        raise ValueError(model_family)
    frame = load_joined(root, panel, 1, mode=mode)
    names_base = CORE_NUMERIC if panel == "core" else ENHANCED_NUMERIC
    fold_data = []
    for fold in FOLDS:
        if panel == "enhanced" and fold.cutoff < "2017Q1":
            continue
        sub = frame.loc[frame["report_period"].le(fold.test_end)].copy()
        names = list(names_base)
        if model_family == "rahq_tree":
            sub, context_names, _ = enrich(sub, root, panel, fold, seed=17)
            names += context_names
        train, _, validation = partition(sub, fold, calibration_quarters=4)
        train = deterministic_sample(train, 150_000, 17)
        fold_data.append((fold.name, train, validation, names))
    if not fold_data:
        raise ValueError("No feasible inner temporal folds")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=17),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1))
    trial_records = []

    def objective(trial: optuna.trial.Trial) -> float:
        config = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 220, step=20),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", .03, .10, log=True),
        }
        started = time.perf_counter()
        losses = []
        try:
            for step, (name, train, validation, features) in enumerate(fold_data, 1):
                fit = xgboost_quantile_cuda(train, validation.iloc[:10], validation,
                                             features, seed=17,
                                             n_estimators=config["n_estimators"],
                                             max_depth=config["max_depth"],
                                             learning_rate=config["learning_rate"])
                residual = validation["deposit_log_growth"].to_numpy() - fit.test[:, 1]
                loss = float(np.maximum(.05 * residual, -.95 * residual).mean())
                losses.append(loss)
                trial.report(float(np.mean(losses)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(losses))
        finally:
            trial_records.append({"number": trial.number, "config": config,
                                  "fold_losses": losses, "runtime_seconds": time.perf_counter() - started})

    study.optimize(objective, n_trials=trials, gc_after_trial=True, show_progress_bar=False)
    records = []
    for trial in study.trials:
        details = next(x for x in trial_records if x["number"] == trial.number)
        records.append({**details, "state": trial.state.name,
                        "value": trial.value if trial.value is not None else None})
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "panel": panel, "mode": mode, "model_family": model_family,
        "data_snapshot_id": RunState(root).data["data_snapshot_id"],
        "folds": [fold.name for fold in FOLDS], "seed": 17,
        "best_config": study.best_params, "best_q05_pinball": study.best_value,
        "trials": records,
        "holdout_policy": "All inner validation periods end before 2020; 2024+ locked outcomes not accessed",
    }
    path = root / "artifacts" / f"hpo_{panel}_{mode}_{model_family}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    run_id = f"hpo_{panel}_{mode}_{model_family}"
    con.execute("DELETE FROM trials WHERE run_id=?", [run_id])
    for record in records:
        con.execute("INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?)", [
            run_id, record["number"], json.dumps(record["config"]), record["value"],
            record["state"], record["runtime_seconds"], 17])
    con.close()
    print(f"{run_id}: {len(records)} trials, best Q05={study.best_value:.6f}, config={study.best_params}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="core", choices=["core", "enhanced"])
    parser.add_argument("--mode", default="B", choices=["A", "B"])
    parser.add_argument("--family", default="xgb", choices=["xgb", "rahq_tree"])
    parser.add_argument("--trials", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = f"artifacts/hpo_{args.panel}_{args.mode}_{args.family}.json"
    RunState(root).run(
        f"hpo_{args.panel}_{args.mode}_{args.family}", "TUNING",
        {"panel": args.panel, "mode": args.mode, "family": args.family,
         "trials": args.trials, "folds": [vars(x) for x in FOLDS],
         "feature_sha256": sha256_file(root / "data" / "features" / f"{args.panel}_features.parquet")},
        [output], lambda: tune(root, args.panel, args.mode, args.family, args.trials), resume=True)


if __name__ == "__main__":
    main()
