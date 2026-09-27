"""Train-partition-only numerical preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NumericPreprocessor:
    names: list[str]
    linear: bool = False
    clip_neural: bool = False

    def fit(self, frame: pd.DataFrame) -> "NumericPreprocessor":
        x = frame[self.names].to_numpy(dtype=np.float64, copy=True)
        x[~np.isfinite(x)] = np.nan
        with np.errstate(all="ignore"):
            self.medians_ = np.nanmedian(x, axis=0)
            self.lower_ = np.nanpercentile(x, .5, axis=0)
            self.upper_ = np.nanpercentile(x, 99.5, axis=0)
        self.medians_ = np.nan_to_num(self.medians_, nan=0, posinf=0, neginf=0)
        self.lower_ = np.nan_to_num(self.lower_, nan=0, posinf=0, neginf=0)
        self.upper_ = np.nan_to_num(self.upper_, nan=0, posinf=0, neginf=0)
        filled = np.where(np.isfinite(x), x, self.medians_)
        if self.linear or self.clip_neural:
            filled = np.clip(filled, self.lower_, self.upper_)
        self.centers_ = np.mean(filled, axis=0)
        self.scales_ = np.std(filled, axis=0)
        self.scales_ = np.where(self.scales_ > 1e-8, self.scales_, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.names].to_numpy(dtype=np.float64, copy=True)
        missing = ~np.isfinite(x)
        x[missing] = np.broadcast_to(self.medians_, x.shape)[missing]
        if self.linear or self.clip_neural:
            x = np.clip(x, self.lower_, self.upper_)
            x = (x - self.centers_) / self.scales_
        return np.column_stack([x.astype(np.float32), missing.astype(np.float32)])

    @property
    def output_names(self) -> list[str]:
        return self.names + [f"{name}__missing" for name in self.names]
