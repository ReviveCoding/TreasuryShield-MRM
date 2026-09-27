"""Training-frozen peer-relative funding stress-event classification study."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from treasuryshield.development.models import peer_group
from treasuryshield.development.preprocess import NumericPreprocessor
from treasuryshield.features.build import CORE_NUMERIC
from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.splits.temporal import SPLITS, deterministic_sample, load_joined, partition


FIELDS = ("deposit_log_growth", "borrowings_change_asset_scaled", "liquid_broad_change")


def _thresholds(train: pd.DataFrame, q: float) -> dict:
    grouped = peer_group(train)
    values = train[list(FIELDS)].to_numpy(dtype=float)
    global_t = np.nanquantile(values, [q, 1 - q, q], axis=0).diagonal()
    thresholds = {}
    for group in np.unique(grouped):
        sub = values[grouped == group]
        if len(sub) < 100:
            continue
        thresholds[int(group)] = [float(np.nanquantile(sub[:, 0], q)),
                                  float(np.nanquantile(sub[:, 1], 1 - q)),
                                  float(np.nanquantile(sub[:, 2], q))]
    return {"global": global_t.tolist(), "peer": thresholds, "q": q}


def _labels(frame: pd.DataFrame, thresholds: dict, definition: str) -> np.ndarray:
    groups = peer_group(frame)
    cut = np.array([thresholds["peer"].get(int(group), thresholds["global"])
                    for group in groups])
    deposit = frame.deposit_log_growth.to_numpy(dtype=float) < cut[:, 0]
    borrowing = frame.borrowings_change_asset_scaled.to_numpy(dtype=float) > cut[:, 1]
    liquid = frame.liquid_broad_change.to_numpy(dtype=float) < cut[:, 2]
    if definition == "co_stress":
        return deposit & (borrowing | liquid)
    if definition == "union":
        return deposit | borrowing | liquid
    raise ValueError(definition)


def _statistics(y: np.ndarray, risk: np.ndarray, weights: np.ndarray | None = None) -> dict:
    if weights is None:
        weights = np.ones(len(y), dtype=float)
    weights = np.maximum(np.nan_to_num(weights.astype(float)), 0)
    if y.sum() < 5 or (~y).sum() < 5:
        return {"status": "insufficient_events", "n": len(y), "events": int(y.sum())}
    top = np.argsort(-risk)[: max(1, int(np.ceil(.05 * len(y))))]
    if np.ptp(risk) < 1e-12:
        recall = float(len(top) / len(y))
        precision = float(np.average(y, weights=weights))
    else:
        recall = float(np.sum(weights[top] * y[top]) / np.sum(weights * y))
        precision = float(np.sum(weights[top] * y[top]) / np.sum(weights[top]))
    return {"status": "evaluated", "n": len(y), "events": int(y.sum()),
            "prevalence": float(np.average(y, weights=weights)),
            "pr_auc": float(average_precision_score(y, risk, sample_weight=weights)),
            "brier": float(brier_score_loss(y, risk, sample_weight=weights)),
            "recall_at_top5pct": recall, "precision_at_top5pct": precision}


def run(root: Path) -> None:
    selection_path = root / "artifacts" / "model_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Freeze model selection before final classification study")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    output = root / "artifacts" / "stress_event_classification.json"
    predictions_path = root / "artifacts" / "stress_event_classification.parquet"
    config = {"mode": "B", "horizon": 1,
              "folds": ["gfc", "pandemic", "tightening", "final_locked"],
              "definitions": {"co_stress": "peer Q10 deposit drop AND (peer Q90 borrowing rise OR peer Q10 liquid-buffer decline)",
                              "union": "any peer Q05 deposit drop, peer Q95 borrowing rise, or peer Q05 liquid-buffer decline"},
              "train_sample_max": 150_000, "seed": 1701,
              "source_sha256": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"],
              "selection_sha256": sha256_file(selection_path)}

    def execute() -> None:
        frame = load_joined(root, "core", 1, mode="B")
        results = []
        predictions = []
        bins = []
        for split in config["folds"]:
            subset = frame.loc[frame.report_period.le(SPLITS[split].test_end)]
            train, _, test = partition(subset, SPLITS[split])
            train = train.dropna(subset=list(FIELDS)).copy()
            test = test.dropna(subset=list(FIELDS)).copy()
            sampled = deterministic_sample(train, config["train_sample_max"], 1701)
            if len(test) < 100:
                raise ValueError(f"Insufficient stress-event evaluation rows {split}")
            for definition, threshold_q in (("co_stress", .10), ("union", .05)):
                thresholds = _thresholds(train, threshold_q)
                train_label = _labels(sampled, thresholds, definition)
                test_label = _labels(test, thresholds, definition)
                if train_label.sum() < 30:
                    raise ValueError(f"Too few training events {split} {definition}")
                pre = NumericPreprocessor(list(CORE_NUMERIC), linear=True).fit(sampled)
                classifier = LogisticRegression(max_iter=300, C=.3, random_state=1701)
                classifier.fit(pre.transform(sampled), train_label.astype(int))
                probability = classifier.predict_proba(pre.transform(test))[:, 1]
                probability = np.clip(probability, 1e-6, 1 - 1e-6)
                training_prevalence = float(train_label.mean())
                for model_name, model_risk in (("logistic", probability),
                                               ("training_prevalence", np.full(len(test), training_prevalence))):
                    for weighting, weights in (
                        ("equal_bank", None), ("deposit_weighted", test.deposits.to_numpy()),
                        ("asset_weighted", test.assets.to_numpy()),
                    ):
                        stat = _statistics(test_label, model_risk, weights)
                        results.append({"split": split, "definition": definition,
                                        "model": model_name, "weighting": weighting,
                                        "training_rows": len(sampled), "test_rows": len(test),
                                        "thresholds": thresholds if weighting == "equal_bank"
                                        and model_name == "logistic" else None, **stat})
                forecast = test[["bank_id", "report_period", "assets", "deposits"]].copy()
                forecast["split"] = split
                forecast["definition"] = definition
                forecast["event"] = test_label
                forecast["probability"] = probability
                predictions.append(forecast)
                binned = forecast.assign(bin=pd.qcut(forecast.probability.rank(method="first"),
                                                      10, labels=False)).groupby("bin").agg(
                    mean_probability=("probability", "mean"), event_rate=("event", "mean"),
                    n=("event", "size")).reset_index()
                binned["split"] = split
                binned["definition"] = definition
                bins.extend(binned.to_dict("records"))
            print(f"Stress-event classification {split}: {len(test)} evaluated rows", flush=True)
        pd.concat(predictions, ignore_index=True).to_parquet(predictions_path, index=False, compression="zstd")
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                   "definition_notes": config["definitions"],
                   "results": results, "calibration_bins": bins,
                   "limitations": ["Peer thresholds are fitted on training labels only and frozen by fold.",
                                   "Public balance-sheet stress labels are predictive proxies, not observed liquidity failures.",
                                   "The final fold was scored only after the classification protocol and selection were frozen."]}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.begin()
            con.execute("CREATE TABLE IF NOT EXISTS classification_results (split VARCHAR, definition VARCHAR, model VARCHAR, weighting VARCHAR, metric VARCHAR, value DOUBLE, n BIGINT, artifact_path VARCHAR)")
            con.execute("CREATE TABLE IF NOT EXISTS classification_calibration (split VARCHAR, definition VARCHAR, bin INTEGER, mean_probability DOUBLE, event_rate DOUBLE, n BIGINT, artifact_path VARCHAR)")
            con.execute("DELETE FROM classification_results WHERE artifact_path='artifacts/stress_event_classification.json'")
            con.execute("DELETE FROM classification_calibration WHERE artifact_path='artifacts/stress_event_classification.json'")
            for item in results:
                if item["status"] != "evaluated":
                    continue
                for metric in ("prevalence", "pr_auc", "brier", "recall_at_top5pct", "precision_at_top5pct"):
                    con.execute("INSERT INTO classification_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                [item["split"], item["definition"], item["model"], item["weighting"], metric,
                                 item[metric], item["n"], "artifacts/stress_event_classification.json"])
            for item in bins:
                con.execute("INSERT INTO classification_calibration VALUES (?, ?, ?, ?, ?, ?, ?)",
                            [item["split"], item["definition"], item["bin"],
                             item["mean_probability"], item["event_rate"], item["n"],
                             "artifacts/stress_event_classification.json"])
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    RunState(root).run("stress_event_classification", "VALIDATION", config,
                       ["artifacts/stress_event_classification.json",
                        "artifacts/stress_event_classification.parquet"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
