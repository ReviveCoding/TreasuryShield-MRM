"""Fold-fitted peer and regime inputs for RAHQ candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from treasuryshield.peers.context import PEER_VALUES, build_peers
from treasuryshield.regimes.filtered import VECTOR, build_regimes
from treasuryshield.splits.temporal import Split


def enrich(frame: pd.DataFrame, root: Path, panel: str, split: Split,
           *, peer_kind: str = "rule", regime_kind: str = "hmm_filtered",
           states: int = 3, seed: int = 17) -> tuple[pd.DataFrame, list[str], dict]:
    out = frame
    names = []
    metadata = {}
    if peer_kind != "none":
        peer = build_peers(root, panel, split, kind=peer_kind, seed=seed)
        out = out.merge(peer.features, on=["bank_id", "report_period"], how="left", validate="one_to_one")
        if out["peer_group"].isna().any():
            raise ValueError("Peer context missing for eligible bank-quarter")
        names = ["peer_group_size"]
        for value in PEER_VALUES:
            names.extend([f"peer_{value}_median", f"own_minus_peer_{value}",
                          f"peer_{value}_percentile"])
        metadata["peer"] = peer.metadata
    regime_input = out
    if panel == "enhanced" and regime_kind.startswith("hmm_"):
        # The FFIEC-enhanced panel begins in 2017. The macro regime has a
        # longer, independently available Core history; fit its state model
        # on that earlier macro history plus the Enhanced-origin quarters.
        earliest = out.report_period.min()
        pit_columns = [f"macro_{name}_pit" for name in ("UNRATE", "CFNAI", "CPIAUCSL")]
        history = pd.read_parquet(root / "data" / "features" / "core_features.parquet",
                                  columns=["report_period", *VECTOR, *pit_columns])
        history = history.loc[history.report_period.lt(earliest)].drop_duplicates("report_period")
        if out["mode"].iloc[0] == "B":
            for name in ("UNRATE", "CFNAI", "CPIAUCSL"):
                history[f"macro_{name}_revised"] = history[f"macro_{name}_pit"]
        regime_input = pd.concat([history[["report_period", *VECTOR]],
                                  out[["report_period", *VECTOR]].drop_duplicates("report_period")],
                                 ignore_index=True)
        metadata["regime_history_extension"] = {
            "source": "Core macro forecast-origin features before Enhanced panel starts",
            "prior_quarters": len(history), "last_prior_quarter": history.report_period.max()}
    regime = build_regimes(regime_input, split, kind=regime_kind, states=states, seed=seed)
    out = out.merge(regime.features, on="report_period", how="left", validate="many_to_one")
    if regime_kind != "none":
        names.extend([f"regime_p{i}" for i in range(4)])
    metadata["regime"] = regime.metadata
    return out, names, metadata
