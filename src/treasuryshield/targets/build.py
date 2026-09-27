"""Create future funding and liquidity outcomes without future-derived features."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

HORIZONS = (1, 2, 4)


def _event_mask(frame: pd.DataFrame, events: pd.DataFrame, horizon: int, radius: int) -> np.ndarray:
    # Event quarter +/- radius contaminates a forecast path [t,t+h].
    keys = []
    for delta in range(-horizon - radius, radius + 1):
        keys.append(pd.DataFrame({
            "bank_id": events["bank_id"].to_numpy(),
            "stata_quarter": events["event_quarter"].to_numpy() + delta,
        }))
    flagged = pd.concat(keys, ignore_index=True).drop_duplicates()
    observed = pd.MultiIndex.from_frame(frame[["bank_id", "stata_quarter"]])
    flagged_index = pd.MultiIndex.from_frame(flagged)
    return observed.isin(flagged_index)


def build(root: Path) -> None:
    columns = ["bank_id", "fdic_cert", "stata_quarter", "report_period", "quarter_end",
               "forecast_origin", "public_availability_date", "year", "assets", "deposits",
               "borrowings", "cash_asset_ratio", "liquid_broad_asset_ratio"]
    core = pd.read_parquet(root / "data" / "processed" / "core_panel.parquet", columns=columns)
    core = core.loc[core["assets"].gt(0) & core["deposits"].gt(0)].copy()
    if core.duplicated(["bank_id", "stata_quarter"]).any():
        raise ValueError("Duplicate Core Panel bank-quarter")
    entity = pd.read_parquet(root / "data" / "processed" / "entity_events.parquet",
                             columns=["bank_id", "event_quarter", "event_type"])
    mna = entity.loc[entity["event_type"].isin(["merged_or_acquired", "branch_transfer"])
                     & entity["bank_id"].notna() & entity["event_quarter"].notna(),
                     ["bank_id", "event_quarter"]].drop_duplicates().copy()
    mna["bank_id"] = mna["bank_id"].astype("int64")
    mna["event_quarter"] = mna["event_quarter"].astype("int64")
    future_columns = ["bank_id", "stata_quarter", "quarter_end", "public_availability_date",
                      "deposits", "borrowings", "cash_asset_ratio", "liquid_broad_asset_ratio"]
    outputs = []
    diagnostics = {}
    for horizon in HORIZONS:
        future = core[future_columns].copy()
        future["stata_quarter"] -= horizon
        future = future.rename(columns={
            "quarter_end": "target_quarter_end",
            "public_availability_date": "target_availability_date",
            "deposits": "future_deposits", "borrowings": "future_borrowings",
            "cash_asset_ratio": "future_cash_asset_ratio",
            "liquid_broad_asset_ratio": "future_liquid_broad_asset_ratio",
        })
        joined = core.merge(future, on=["bank_id", "stata_quarter"], how="left", validate="one_to_one")
        observed = joined["future_deposits"].gt(0)
        joined = joined.loc[observed].copy()
        if not joined["target_availability_date"].gt(joined["forecast_origin"]).all():
            raise ValueError(f"Target is not after origin for {horizon}Q")
        joined["horizon_quarters"] = horizon
        joined["deposit_log_growth"] = np.log(joined["future_deposits"] / joined["deposits"])
        joined["borrowings_change_asset_scaled"] = (
            joined["future_borrowings"] - joined["borrowings"]) / joined["assets"]
        joined["liquid_narrow_change"] = joined["future_cash_asset_ratio"] - joined["cash_asset_ratio"]
        joined["liquid_broad_change"] = joined["future_liquid_broad_asset_ratio"] - joined["liquid_broad_asset_ratio"]
        for radius in (0, 1, 2):
            joined[f"mna_contaminated_radius_{radius}"] = _event_mask(joined, mna, horizon, radius)
        diagnostics[str(horizon)] = {
            "origin_rows": len(core), "observed_future_rows": len(joined),
            "missing_future_rows": int(len(core) - len(joined)),
            "mna_contaminated_radius_0": int(joined["mna_contaminated_radius_0"].sum()),
            "mna_contaminated_radius_1": int(joined["mna_contaminated_radius_1"].sum()),
            "mna_contaminated_radius_2": int(joined["mna_contaminated_radius_2"].sum()),
            "prelock_observed_rows": int(joined["year"].le(2023).sum()),
        }
        outputs.append(joined[[
            "bank_id", "fdic_cert", "stata_quarter", "report_period", "quarter_end",
            "forecast_origin", "target_quarter_end", "target_availability_date",
            "year", "horizon_quarters", "assets", "deposits", "future_deposits",
            "deposit_log_growth", "borrowings_change_asset_scaled",
            "liquid_narrow_change", "liquid_broad_change",
            "mna_contaminated_radius_0", "mna_contaminated_radius_1", "mna_contaminated_radius_2",
        ]])
        print(f"{horizon}Q target: {len(joined)} observable bank origins", flush=True)
    combined = pd.concat(outputs, ignore_index=True)
    out = root / "data" / "features" / "targets.parquet"
    combined.to_parquet(out, index=False, compression="zstd")
    (root / "data" / "manifests" / "target_quality.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": len(combined), "by_horizon": diagnostics,
        "event_exclusion_definition": "M&A or branch-transfer event quarter within forecast path plus selected radius; primary radius=1",
        "failure_policy": "Failure comes from FDIC only; no failure inference from disappearance",
        "outcome_availability_rule": "Future report quarter end plus 45 days; all outcome dates after origin",
        "locked_holdout_policy": "Only eligibility counts before preregistration; do not inspect 2024+ target values or performance",
    }, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    inputs = [root / "data" / "processed" / f"{x}.parquet" for x in ["core_panel", "entity_events"]]
    RunState(root).run(
        "targets", "BUILDING_DATA",
        {"input_sha256": {p.name: sha256_file(p) for p in inputs},
         "horizons": HORIZONS, "mna_radii": [0, 1, 2]},
        ["data/features/targets.parquet", "data/manifests/target_quality.json"],
        lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
