"""Point-in-time peer context from contemporaneously available reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from treasuryshield.development.models import peer_group
from treasuryshield.splits.temporal import Split, deterministic_sample

PEER_VALUES = ["cash_asset_ratio", "liquid_broad_asset_ratio", "loan_deposit_ratio",
               "brokered_domestic_ratio", "capital_asset_ratio",
               "deposit_growth_qoq_lagged", "uninsured_proxy_domestic_share"]
CLUSTER_VALUES = ["log_assets", "residential_loan_share", "commercial_real_estate_loan_share",
                  "ci_loan_share", "brokered_domestic_ratio", "loan_deposit_ratio",
                  "cash_asset_ratio"]


@dataclass
class PeerResult:
    features: pd.DataFrame
    model: object | None
    metadata: dict


def build_peers(root: Path, panel: str, split: Split, *, kind: str = "rule",
                seed: int = 17) -> PeerResult:
    path = root / "data" / "features" / f"{panel}_features.parquet"
    use = list(dict.fromkeys(["bank_id", "report_period", "assets"] + PEER_VALUES + CLUSTER_VALUES))
    frame = pd.read_parquet(path, columns=use)
    frame = frame.loc[frame["report_period"].le(split.test_end)].copy()
    if kind == "none":
        return PeerResult(frame[["bank_id", "report_period"]].copy(), None, {"kind": kind})
    if kind == "rule":
        frame["peer_group"] = peer_group(frame)
        model = None
    elif kind == "learned":
        train = frame.loc[frame["report_period"].le(split.cutoff)]
        sampled = deterministic_sample(train, 100_000, seed)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(imputer.fit_transform(sampled[CLUSTER_VALUES]))
        model = MiniBatchKMeans(n_clusters=12, batch_size=4096, n_init=5,
                                random_state=seed).fit(fit_x)
        frame["peer_group"] = model.predict(scaler.transform(imputer.transform(frame[CLUSTER_VALUES])))
        model = {"cluster": model, "imputer": imputer, "scaler": scaler}
    else:
        raise ValueError(kind)
    result = frame[["bank_id", "report_period", "peer_group"]].copy()
    group = frame.groupby(["report_period", "peer_group"], observed=True)
    group_size = group["bank_id"].transform("size")
    result["peer_group_size"] = group_size
    size_only = np.digitize(frame["assets"].to_numpy(dtype=float),
                            [100_000, 1_000_000, 10_000_000]).astype(int)
    frame["peer_size_band"] = size_only
    size_group = frame.groupby(["report_period", "peer_size_band"], observed=True)
    for column in PEER_VALUES:
        peer_median = group[column].transform("median")
        fallback = size_group[column].transform("median")
        peer_median = peer_median.where(group_size >= 20, fallback)
        result[f"peer_{column}_median"] = peer_median
        result[f"own_minus_peer_{column}"] = frame[column] - peer_median
        result[f"peer_{column}_percentile"] = group[column].rank(pct=True)
    if result.duplicated(["bank_id", "report_period"]).any():
        raise ValueError("Duplicate peer context bank-quarter")
    return PeerResult(result, model,
                      {"kind": kind, "training_cutoff": split.cutoff,
                       "reference_rows": len(frame), "small_group_fallback": "size-band median if fewer than 20 peers",
                       "peer_input_scope": "all available source-report bank rows; no future target eligibility filter"})
