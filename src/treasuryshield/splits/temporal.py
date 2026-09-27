"""Temporal partitions enforce outcome availability before each fit cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.features.build import CORE_NUMERIC, ENHANCED_NUMERIC

QUANTILES = np.array([.01, .05, .10, .25, .50, .75, .90], dtype=float)


@dataclass(frozen=True)
class Split:
    name: str
    cutoff: str
    test_start: str
    test_end: str
    train_start: str = "2001Q1"


SPLITS = {
    "gfc": Split("gfc", "2006Q4", "2007Q3", "2009Q4"),
    "routine_2018_19": Split("routine_2018_19", "2017Q4", "2018Q1", "2019Q4"),
    "pandemic": Split("pandemic", "2019Q4", "2020Q1", "2021Q4"),
    "tightening": Split("tightening", "2021Q4", "2022Q1", "2023Q4"),
    "final_locked": Split("final_locked", "2023Q4", "2024Q1", "2026Q1"),
}


def load_joined(root: Path, panel: str, horizon: int, *, mode: str = "A") -> pd.DataFrame:
    if panel not in {"core", "enhanced"}:
        raise ValueError(panel)
    if horizon not in {1, 2, 4}:
        raise ValueError(horizon)
    path = root / "data" / "features" / f"{panel}_features.parquet"
    features = pd.read_parquet(path)
    features = features.loc[features["report_period"].ge("2001Q1")].copy()
    targets = pd.read_parquet(root / "data" / "features" / "targets.parquet")
    targets = targets.loc[targets["horizon_quarters"].eq(horizon), [
        "bank_id", "report_period", "target_quarter_end", "target_availability_date",
        "deposit_log_growth", "borrowings_change_asset_scaled", "liquid_narrow_change",
        "liquid_broad_change", "mna_contaminated_radius_0",
        "mna_contaminated_radius_1", "mna_contaminated_radius_2",
    ]]
    frame = features.merge(targets, on=["bank_id", "report_period"], how="inner", validate="one_to_one")
    frame = frame.loc[frame["core_eligible"]].copy()
    if panel == "enhanced":
        frame = frame.loc[frame["enhanced_eligible"]].copy()
    frame["panel"] = panel
    frame["mode"] = mode
    if mode == "B":
        for series in ["UNRATE", "CFNAI", "CPIAUCSL"]:
            frame[f"macro_{series}_revised"] = frame[f"macro_{series}_pit"]
    elif mode != "A":
        raise ValueError(mode)
    feature_names = CORE_NUMERIC if panel == "core" else ENHANCED_NUMERIC
    missing = [name for name in feature_names if name not in frame]
    if missing:
        raise ValueError(f"Feature columns absent: {missing}")
    if frame["feature_information_date"].gt(frame["forecast_origin"]).any():
        raise ValueError("Future-dated bank feature")
    if frame["target_availability_date"].le(frame["forecast_origin"]).any():
        raise ValueError("Nonfuture target")
    return frame


def partition(frame: pd.DataFrame, split: Split, *, mna_radius: int | None = 1,
              calibration_quarters: int = 8) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if mna_radius is None:
        eligible = pd.Series(True, index=frame.index)
    else:
        eligible = frame[f"mna_contaminated_radius_{mna_radius}"].eq(False)
    cutoff_end = pd.Period(split.cutoff, freq="Q").to_timestamp(how="end").normalize()
    allowed = eligible & frame["report_period"].ge(split.train_start)
    training = frame.loc[allowed & frame["target_quarter_end"].le(cutoff_end)].copy()
    # The calibration partition is the last available outcome window; the
    # backbone is fit without these rows.
    calibration_start = str(pd.Period(split.cutoff, freq="Q") - calibration_quarters + 1)
    calibration = training.loc[training["target_quarter_end"].ge(
        pd.Period(calibration_start, freq="Q").to_timestamp(how="start"))].copy()
    train = training.loc[~training.index.isin(calibration.index)].copy()
    test = frame.loc[eligible & frame["report_period"].between(split.test_start, split.test_end)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Empty split {split.name}: train={len(train)}, test={len(test)}")
    first_test_origin = test["forecast_origin"].min()
    if training["target_availability_date"].gt(first_test_origin).any():
        raise ValueError(f"Training target not publicly available by first {split.name} origin")
    if calibration["target_availability_date"].gt(first_test_origin).any():
        raise ValueError(f"Calibration target not available by first {split.name} origin")
    return train, calibration, test


def deterministic_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    # Quarter-stratified deterministic allocation preserves severe historical
    # periods that would be swamped by earlier bank-count differences.
    rng = np.random.default_rng(seed)
    pieces = []
    counts = frame.groupby("report_period").size()
    allocations = np.maximum(1, np.floor(maximum * counts / len(frame)).astype(int))
    for period, group in frame.groupby("report_period", sort=True):
        size = min(len(group), int(allocations.loc[period]))
        positions = rng.choice(len(group), size=size, replace=False)
        pieces.append(group.iloc[np.sort(positions)])
    sample = pd.concat(pieces)
    if len(sample) > maximum:
        sample = sample.sample(n=maximum, random_state=seed)
    return sample.sort_values(["report_period", "bank_id"])
