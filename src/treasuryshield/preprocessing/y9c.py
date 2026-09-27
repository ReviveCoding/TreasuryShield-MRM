"""Separate consolidated holding-company panel from Chicago Fed FR Y-9C files."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

FIELDS = ["RSSD9001", "BHCK2170", "BHCK3210", "BHCK3300", "BHCK2122", "BHCK0081",
          "BHCK1773", "BHCKJJ34", "BHCKJA22", "BHDM6631", "BHDM6636", "BHFN6631", "BHFN6636"]


def build(root: Path) -> None:
    paths = sorted((root / "data" / "raw" / "y9c").glob("bhcf*.csv"))
    if len(paths) != 81:
        raise ValueError(f"Expected 81 FR Y-9C files, got {len(paths)}")
    frames = []
    diagnostics = []
    for number, path in enumerate(paths, 1):
        match = re.fullmatch(r"bhcf(\d{2})(\d{2})\.csv", path.name, re.I)
        if not match:
            raise ValueError(path.name)
        yy, month = map(int, match.groups())
        year = 2000 + yy
        quarter = (month - 1) // 3 + 1
        with path.open("r", encoding="latin-1", newline="") as stream:
            present = set(next(csv.reader(stream)))
        if "RSSD9001" not in present or "BHCK2170" not in present:
            raise ValueError(f"Missing BHC identifier or assets: {path}")
        frame = pd.read_csv(path, usecols=lambda name: name in FIELDS, skiprows=[1],
                            encoding="latin-1", low_memory=False)
        for name in FIELDS:
            if name not in frame:
                frame[name] = np.nan
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame = frame.loc[frame["BHCK2170"].gt(0) & frame["RSSD9001"].gt(0)].copy()
        frame = frame.rename(columns={"RSSD9001": "bhc_rssd", "BHCK2170": "assets",
                                      "BHCK3210": "equity", "BHCK3300": "liabilities_plus_equity",
                                      "BHCK2122": "loans", "BHCK0081": "interest_bearing_balances"})
        frame["bhc_rssd"] = frame["bhc_rssd"].astype("int64")
        deposit_components = ["BHDM6631", "BHDM6636", "BHFN6631", "BHFN6636"]
        frame["deposits_domestic"] = frame[["BHDM6631", "BHDM6636"]].sum(axis=1, min_count=2)
        frame["deposits_foreign"] = frame[["BHFN6631", "BHFN6636"]].sum(axis=1, min_count=2)
        frame["deposits_consolidated"] = frame[deposit_components].sum(axis=1, min_count=4)
        frame["securities_broad"] = frame[["BHCK1773", "BHCKJJ34", "BHCKJA22"]].sum(axis=1, min_count=3)
        frame["equity_asset_ratio"] = frame["equity"] / frame["assets"]
        frame["loan_deposit_ratio"] = frame["loans"] / frame["deposits_consolidated"].where(frame["deposits_consolidated"] > 0)
        frame["identity_relative_gap"] = (frame["assets"] - frame["liabilities_plus_equity"]).abs() / frame["assets"]
        frame["report_period"] = f"{year}Q{quarter}"
        frame["quarter_end"] = pd.Period(f"{year}Q{quarter}", freq="Q").to_timestamp(how="end").normalize()
        frame["forecast_origin"] = frame["quarter_end"] + pd.Timedelta(days=45)
        frame["source_vintage"] = "CHICAGO_FED_RESTATED_2026-09-27"
        frame["unit_of_observation"] = "consolidated_holding_company"
        diagnostics.append({"quarter": f"{year}Q{quarter}", "rows": len(frame),
                            "consolidated_deposit_coverage": float(frame["deposits_consolidated"].notna().mean()),
                            "identity_gap_gt_1pct": int(frame["identity_relative_gap"].gt(.01).sum())})
        frames.append(frame)
        if number % 20 == 0 or number == len(paths):
            print(f"FR Y-9C parsed {number}/{len(paths)}", flush=True)
    panel = pd.concat(frames, ignore_index=True).sort_values(["bhc_rssd", "quarter_end"])
    if panel.duplicated(["bhc_rssd", "report_period"]).any():
        raise ValueError("Duplicate FR Y-9C BHC-quarter")
    panel.to_parquet(root / "data" / "processed" / "y9c_bhc_panel.parquet", index=False, compression="zstd")
    (root / "data" / "manifests" / "y9c_quality.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(panel), "unique_bhcs": int(panel["bhc_rssd"].nunique()),
        "first_quarter": panel["report_period"].min(), "last_quarter": panel["report_period"].max(),
        "quarterly": diagnostics,
        "scope": "FR Y-9C consolidated holding companies only; never pooled with bank Call Reports",
        "modern_data_limit": "NIC current endpoint presented CAPTCHA; historical Chicago files end 2021Q1",
    }, indent=2), encoding="utf-8")
    print(f"Separate FR Y-9C panel: {len(panel)} BHC-quarters", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "data" / "raw" / "y9c").glob("bhcf*.csv"))
    RunState(root).run(
        "y9c_bhc_panel", "BUILDING_DATA",
        {"raw_sha256": {p.name: sha256_file(p) for p in paths}, "fields": FIELDS},
        ["data/processed/y9c_bhc_panel.parquet", "data/manifests/y9c_quality.json"],
        lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
