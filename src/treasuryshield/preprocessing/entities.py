"""Resolve official FDIC institution events to commercial-bank RSSD IDs."""

from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

FAILURE_CODES = {211, 215, 216, 217, 230, 235}
MERGER_CODES = {213, 221, 222, 223, 224, 225, 810, 811, 812, 820}
CLOSED_CODES = {240, 260}
CHARTER_CODES = {150, 420, 430, 440, 470}
BRANCH_TRANSFER_CODES = {712}


def _classify(code: int) -> str:
    if code in FAILURE_CODES:
        return "failed"
    if code in MERGER_CODES:
        return "merged_or_acquired"
    if code in CLOSED_CODES:
        return "closed"
    if code in CHARTER_CODES:
        return "charter_or_regulator_change"
    if code in BRANCH_TRANSFER_CODES:
        return "branch_transfer"
    if code == 110:
        return "new_institution"
    return "other_change"


def _resolve(events: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    mapping = core[["fdic_cert", "stata_quarter", "bank_id"]].copy()
    mapping["cert_key"] = pd.to_numeric(mapping["fdic_cert"], errors="coerce").astype("Int64")
    mapping = mapping.loc[mapping["cert_key"].gt(0)].copy()
    grouped = mapping.groupby(["cert_key", "stata_quarter"], observed=True)["bank_id"].agg(["nunique", "first"]).reset_index()
    mapping = grouped.loc[grouped["nunique"].eq(1), ["cert_key", "stata_quarter", "first"]]
    mapping = mapping.rename(columns={"first": "candidate_bank_id", "stata_quarter": "lookup_quarter"})
    events = events.copy()
    events["cert_key"] = pd.to_numeric(events["fdic_cert"], errors="coerce").astype("Int64")
    date = pd.to_datetime(events["event_date"], errors="coerce")
    events["event_quarter"] = ((date.dt.year - 1960) * 4 + (date.dt.month - 1) // 3).astype("Int64")
    events["bank_id"] = np.nan
    events["resolution_distance_quarters"] = np.nan
    for offset in [0, -1, 1, -2, 2]:
        unresolved = events["bank_id"].isna() & events["cert_key"].notna() & events["event_quarter"].notna()
        if not unresolved.any():
            break
        subset = events.loc[unresolved, ["cert_key", "event_quarter"]].copy()
        subset["lookup_quarter"] = (subset["event_quarter"] + offset).astype("int64")
        match = subset.merge(mapping, on=["cert_key", "lookup_quarter"], how="left", sort=False)
        if len(match) != len(subset):
            raise ValueError("Nonunique certificate-quarter entity mapping")
        good = match["candidate_bank_id"].notna().to_numpy()
        indices = subset.index.to_numpy()[good]
        events.loc[indices, "bank_id"] = match.loc[good, "candidate_bank_id"].to_numpy()
        events.loc[indices, "resolution_distance_quarters"] = abs(offset)
    events["mapping_confidence"] = np.select(
        [events["resolution_distance_quarters"].eq(0),
         events["resolution_distance_quarters"].eq(1),
         events["resolution_distance_quarters"].eq(2)],
        ["high", "medium", "low"], default="unmatched")
    return events


def build(root: Path) -> None:
    core = pd.read_parquet(root / "data" / "processed" / "core_panel.parquet",
                           columns=["bank_id", "fdic_cert", "stata_quarter"])
    paths = sorted((root / "data" / "raw" / "fdic").glob("history_fi_[0-9]*.csv"))
    history = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    if history["ID"].duplicated().any():
        raise ValueError("Duplicate FDIC history event ID")
    codes = pd.to_numeric(history["CHANGECODE"], errors="coerce").fillna(-1).astype(int)
    events = pd.DataFrame({
        "fdic_cert": history["CERT"],
        "event_date": pd.to_datetime(history["EFFDATE"], errors="coerce").dt.normalize(),
        "event_type": codes.map(_classify),
        "source_event_code": codes,
        "source_event_description": history["CHANGECODE_DESC"],
        "source_record_id": history["ID"],
        "predecessor": history.get("OUT_UNINUM", pd.Series(index=history.index)).apply(
            lambda x: f"FDIC_UNINUM:{int(x)}" if pd.notna(x) else None),
        "successor": history.get("ACQ_UNINUM", pd.Series(index=history.index)).apply(
            lambda x: f"FDIC_UNINUM:{int(x)}" if pd.notna(x) else None),
        "fdic_uninum": history["UNINUM"],
        "source": "FDIC BankFind history",
    })
    failures = pd.read_csv(root / "data" / "raw" / "fdic" / "failures.csv", low_memory=False)
    failure_events = pd.DataFrame({
        "fdic_cert": failures["CERT"],
        "event_date": pd.to_datetime(failures["FAILDATE"], errors="coerce").dt.normalize(),
        "event_type": "failed",
        "source_event_code": -1,
        "source_event_description": failures["RESTYPE"].fillna("FDIC failure listing"),
        "source_record_id": "failure:" + failures["ID"].astype(str),
        "predecessor": None, "successor": None, "fdic_uninum": np.nan,
        "source": "FDIC BankFind failures",
    })
    combined = pd.concat([events, failure_events], ignore_index=True)
    combined = _resolve(combined, core)
    combined["structural_contamination"] = combined["event_type"].isin(
        ["merged_or_acquired", "branch_transfer", "closed", "failed", "charter_or_regulator_change"])
    out = root / "data" / "processed" / "entity_events.parquet"
    combined.to_parquet(out, index=False, compression="zstd")
    quality = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "history_events": len(events), "failure_rows": len(failure_events),
        "matched_events_1984_plus": int(combined.loc[combined["event_date"].dt.year.ge(1984), "bank_id"].notna().sum()),
        "event_counts": combined["event_type"].value_counts().to_dict(),
        "mapping_confidence": combined["mapping_confidence"].value_counts().to_dict(),
        "structural_matched_1984_plus": int((combined["structural_contamination"] & combined["bank_id"].notna() & combined["event_date"].dt.year.ge(1984)).sum()),
        "predecessor_successor_namespace": "FDIC_UNINUM, not RSSD; null where absent",
        "disappearance_policy": "Never infer failure or merger solely from Call Report disappearance",
    }
    (root / "data" / "manifests" / "entity_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "data" / "raw" / "fdic").glob("history_fi_[0-9]*.csv"))
    paths += [root / "data" / "raw" / "fdic" / "failures.csv"]
    RunState(root).run(
        "entity_events", "BUILDING_DATA",
        {"fdic_sha256": {p.name: sha256_file(p) for p in paths},
         "core_sha256": sha256_file(root / "data" / "processed" / "core_panel.parquet")},
        ["data/processed/entity_events.parquet", "data/manifests/entity_quality.json"],
        lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
