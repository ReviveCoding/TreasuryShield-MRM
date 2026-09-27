"""Validate matching units and key accounting levels across official sources."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file


def build(root: Path) -> None:
    core = pd.read_parquet(root / "data" / "processed" / "core_panel.parquet",
                           columns=["bank_id", "report_period", "assets", "domestic_dep"])
    ffiec = pd.read_parquet(root / "data" / "processed" / "enhanced_panel.parquet",
                            columns=["bank_id", "report_period", "consolidated_assets_ffiec",
                                     "domestic_deposits_ffiec"])
    joined = core.merge(ffiec, on=["bank_id", "report_period"], how="inner", validate="one_to_one")
    joined["asset_relative_gap"] = (joined["consolidated_assets_ffiec"] - joined["assets"]).abs() / joined["assets"].where(joined["assets"] > 0)
    joined["deposit_relative_gap"] = (joined["domestic_deposits_ffiec"] - joined["domestic_dep"]).abs() / joined["domestic_dep"].where(joined["domestic_dep"] > 0)
    discrepant = joined.loc[joined["asset_relative_gap"].gt(.01) | joined["deposit_relative_gap"].gt(.01),
                            ["bank_id", "report_period", "assets", "consolidated_assets_ffiec",
                             "domestic_dep", "domestic_deposits_ffiec",
                             "asset_relative_gap", "deposit_relative_gap"]]
    discrepant.to_parquet(root / "artifacts" / "cross_source_discrepancies.parquet", index=False)
    quality = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "matched_bank_quarters": len(joined),
        "median_ffiec_to_ny_assets_ratio": float(np.nanmedian(joined["consolidated_assets_ffiec"] / joined["assets"])),
        "median_ffiec_to_ny_domestic_deposits_ratio": float(np.nanmedian(joined["domestic_deposits_ffiec"] / joined["domestic_dep"])),
        "asset_relative_gap_gt_1pct": int(joined["asset_relative_gap"].gt(.01).sum()),
        "deposit_relative_gap_gt_1pct": int(joined["deposit_relative_gap"].gt(.01).sum()),
        "comparison_scope": "Consolidated assets to consolidated assets; domestic deposits to domestic deposits, both USD thousands",
        "discrepancy_artifact": "artifacts/cross_source_discrepancies.parquet",
    }
    (root / "data" / "manifests" / "cross_source_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    con.execute("DELETE FROM data_quality WHERE scope='cross_source'")
    for key, value in quality.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            con.execute("INSERT INTO data_quality VALUES ('cross_source', ?, ?, ?, ?)",
                        [key, float(value), quality["comparison_scope"], quality["discrepancy_artifact"]])
    con.close()
    print(json.dumps(quality, indent=2), flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    inputs = [root / "data" / "processed" / f"{name}.parquet" for name in ["core_panel", "enhanced_panel"]]
    RunState(root).run("cross_source_quality", "VALIDATING",
                       {"sha256": {p.name: sha256_file(p) for p in inputs}},
                       ["data/manifests/cross_source_quality.json", "artifacts/cross_source_discrepancies.parquet"],
                       lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
