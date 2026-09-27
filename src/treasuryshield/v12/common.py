from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "v1_2_hardening"
FIG = ROOT / "figures" / "v1_2"
REP = ROOT / "reports" / "v1_2"
DB = ART / "results_v1_2.duckdb"
QCOLS = ["q01", "q05", "q10", "q25", "q50", "q75", "q90"]
QLEVELS = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
SEED = 20260927


def ensure_dirs() -> None:
    for path in (ART, FIG, REP, ART / "models"):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    def clean(item: object) -> object:
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, np.floating):
            item = float(item)
        if isinstance(item, float) and not np.isfinite(item):
            return "Infinity" if item > 0 else "-Infinity" if item < 0 else None
        return item
    path.write_text(
        json.dumps(clean(value), indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    error = np.asarray(y, float) - np.asarray(q, float)
    return np.maximum(alpha * error, (alpha - 1.0) * error)


def weights(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "equal_bank":
        counts = frame.groupby("bank_id")["bank_id"].transform("size").to_numpy(float)
        return 1.0 / counts
    if kind == "asset_weighted":
        return np.maximum(frame["assets"].to_numpy(float), 0)
    if kind == "deposit_weighted":
        return np.maximum(frame["deposits"].to_numpy(float), 0)
    raise KeyError(kind)


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    values = np.asarray(values, float)
    weight = np.asarray(weight, float)
    good = np.isfinite(values) & np.isfinite(weight) & (weight >= 0)
    if not good.any() or weight[good].sum() <= 0:
        return float("nan")
    return float(np.average(values[good], weights=weight[good]))

