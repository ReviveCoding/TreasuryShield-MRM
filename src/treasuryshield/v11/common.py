from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts" / "v1_1_analysis"
FIG = ROOT / "figures" / "v1_1"
REP = ROOT / "reports" / "v1_1"
DB = ART / "results_v1_1.duckdb"
QCOLS = ["q01", "q05", "q10", "q25", "q50", "q75", "q90"]
QLEVELS = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
FINAL_PREFIX = "core_B_1q_final_locked_"
PRELOCK_SPLITS = ("gfc", "pandemic", "tightening")
KEY_MODELS = (
    "B1_peer", "B5_linear_quantile", "B7_xgb_quantile",
    "B8_lightgbm_quantile", "B10_gru", "RAHQ_tree", "RAHQ_C_tree", "RAHQ_C_nn",
)


def ensure_dirs() -> None:
    for path in (ART, FIG, REP):
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
    path.write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def model_path(model: str, split: str = "final_locked") -> Path:
    return ROOT / "artifacts" / "predictions" / f"core_B_1q_{split}_{model}.parquet"


def read_prediction(model: str, split: str = "final_locked") -> pd.DataFrame:
    frame = pd.read_parquet(model_path(model, split))
    frame["report_period"] = frame["report_period"].astype(str)
    frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    frame["split"] = split
    return frame


def read_prelock(model: str) -> pd.DataFrame:
    frames = [read_prediction(model, split) for split in PRELOCK_SPLITS if model_path(model, split).exists()]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Episode test windows are disjoint by design; fail loudly if that ever changes.
    if out.duplicated(["bank_id", "report_period"]).any():
        raise ValueError(f"Overlapping prelock prediction episodes for {model}")
    return out.sort_values(["forecast_origin", "bank_id"]).reset_index(drop=True)


def available_feature_columns() -> list[str]:
    import pyarrow.parquet as pq
    return pq.ParquetFile(ROOT / "data" / "features" / "core_features.parquet").schema_arrow.names


def context_columns() -> list[str]:
    wanted = [
        "bank_id", "report_period", "state", "entity_type", "panel_mode", "source_vintage",
        "assets", "deposits", "cash_asset_ratio", "liquid_broad_asset_ratio",
        "loan_deposit_ratio", "uninsured_proxy_domestic_share", "brokered_domestic_ratio",
        "uninsured_time_deposit_ratio", "borrowings_asset_ratio", "capital_asset_ratio",
        "npl_loan_ratio", "loan_loss_reserve_ratio", "profit_asset_ratio",
        "deposit_cost_annualized", "deposit_beta_4q", "deposit_growth_qoq_lagged",
        "deposit_growth_rolling_vol_4q", "deposit_growth_acceleration",
        "macro_DFF_change_4q_revised", "macro_curve_10y2y_revised", "macro_VIXCLS_revised",
        "macro_DPSACBW027NBOG_qoq_revised", "macro_source_vintage",
    ]
    present = set(available_feature_columns())
    return [name for name in wanted if name in present]


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_parquet(ROOT / "data" / "features" / "core_features.parquet", columns=context_columns())
    features["report_period"] = features["report_period"].astype(str)
    duplicate = features.duplicated(["bank_id", "report_period"])
    if duplicate.any():
        raise ValueError("Feature join keys are not unique")
    left = frame.drop(columns=[c for c in context_columns() if c in frame and c not in {"bank_id", "report_period"}])
    out = left.merge(features, on=["bank_id", "report_period"], how="left", validate="many_to_one")
    targets = pd.read_parquet(
        ROOT / "data" / "features" / "targets.parquet",
        columns=["bank_id", "report_period", "horizon_quarters", "target_availability_date", "mna_contaminated_radius_1"],
    )
    targets["report_period"] = targets["report_period"].astype(str)
    targets = targets.loc[targets.horizon_quarters.eq(1)].drop(columns="horizon_quarters")
    out = out.merge(targets, on=["bank_id", "report_period"], how="left", validate="many_to_one")
    out["target_availability_date"] = pd.to_datetime(out["target_availability_date"])
    return out


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    error = y - q
    return np.maximum(alpha * error, (alpha - 1.0) * error)


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    mask = np.isfinite(values)
    if weights is None:
        return float(np.mean(values[mask])) if mask.any() else float("nan")
    weights = np.asarray(weights, dtype=float)
    mask &= np.isfinite(weights) & (weights >= 0)
    return float(np.average(values[mask], weights=weights[mask])) if mask.any() and weights[mask].sum() else float("nan")


def period_ordinal(values: Iterable[str]) -> np.ndarray:
    return pd.PeriodIndex(list(values), freq="Q").astype(int).to_numpy()


def stable_edges(values: pd.Series, bins: int = 5) -> list[float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return []
    edges = np.unique(np.nanquantile(clean, np.linspace(0, 1, bins + 1))).astype(float)
    if len(edges) < 3:
        return []
    edges[0], edges[-1] = -np.inf, np.inf
    return edges.tolist()


def apply_edges(values: pd.Series, edges: list[float], prefix: str = "Q") -> pd.Series:
    if len(edges) < 3:
        return pd.Series("unavailable", index=values.index, dtype="string")
    labels = [f"{prefix}{i + 1}" for i in range(len(edges) - 1)]
    return pd.cut(values, bins=edges, labels=labels, include_lowest=True).astype("string").fillna("missing")


def aligned_predictions(models: Iterable[str], split: str = "final_locked") -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    keys = ["bank_id", "report_period"]
    for model in models:
        path = model_path(model, split)
        if not path.exists():
            continue
        part = pd.read_parquet(path, columns=keys + ["actual", *QCOLS])
        part["report_period"] = part["report_period"].astype(str)
        part = part.rename(columns={q: f"{q}__{model}" for q in QCOLS})
        if merged is None:
            merged = part.rename(columns={"actual": "actual_reference"})
        else:
            merged = merged.merge(part.drop(columns="actual"), on=keys, how="inner", validate="one_to_one")
    if merged is None:
        raise FileNotFoundError("No prediction artifacts found")
    return merged
