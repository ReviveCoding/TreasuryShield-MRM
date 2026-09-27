"""Quarterly forecast-origin macro data with explicit historical vintage columns."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

MARKET_SERIES = ["DFF", "DGS2", "DGS10", "T10Y2Y", "VIXCLS", "SOFR"]
MONTHLY_SERIES = ["UNRATE", "CFNAI", "CPIAUCSL"]
H8_SERIES = ["DPSACBW027NBOG", "TLAACBW027NBOG", "CASACBW027NBOG", "SBCACBW027NBOG"]


def _read_series(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame.columns) != 2 or frame.columns[0] != "observation_date":
        raise ValueError(f"Unexpected FRED/ALFRED CSV schema: {path}")
    frame.columns = ["observation_date", "value"]
    frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna().sort_values("observation_date")


def _asof(frame: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[float, pd.Timestamp | None]:
    eligible = frame.loc[frame["observation_date"].le(cutoff)]
    if eligible.empty:
        return np.nan, None
    row = eligible.iloc[-1]
    return float(row["value"]), row["observation_date"]


def build(root: Path) -> None:
    fred = {name: _read_series(root / "data" / "raw" / "fred" / f"{name}.csv")
            for name in MARKET_SERIES + MONTHLY_SERIES + H8_SERIES}
    vintage_manifest = json.loads((root / "data" / "manifests" / "alfred_vintages.json").read_text(encoding="utf-8"))
    snapshots = {(entry["quarter"], entry["series"]): root / entry["file"]
                 for entry in vintage_manifest["rows"]}
    quarters = pd.period_range("1984Q1", "2026Q2", freq="Q")
    rows = []
    for period in quarters:
        quarter_end = period.to_timestamp(how="end").normalize()
        origin = quarter_end + pd.Timedelta(days=45)
        record: dict[str, object] = {
            "report_period": str(period), "quarter_end": quarter_end,
            "forecast_origin": origin,
            "macro_source_vintage": "FRED_REVISED_RETRIEVED_2026-09-27",
        }
        for name in MARKET_SERIES + MONTHLY_SERIES + H8_SERIES:
            # Conservative source-specific release proxy. H.8 Wednesday reference
            # values are released after the reference week; do not use that week
            # at the exact observation date.
            lag = 9 if name in H8_SERIES else 2 if name in MARKET_SERIES else 30
            value, obs_date = _asof(fred[name], origin - pd.Timedelta(days=lag))
            record[f"macro_{name}_revised"] = value
            record[f"macro_{name}_observation_date"] = obs_date
            record[f"macro_{name}_assumed_availability_date"] = (
                obs_date + pd.Timedelta(days=lag) if obs_date is not None else pd.NaT)
        for name in MONTHLY_SERIES:
            path = snapshots.get((str(period), name))
            if path is None:
                record[f"macro_{name}_pit"] = np.nan
                record[f"macro_{name}_pit_observation_date"] = pd.NaT
                record[f"macro_{name}_vintage_date"] = pd.NaT
                continue
            vintage = _read_series(path)
            value, obs_date = _asof(vintage, origin)
            record[f"macro_{name}_pit"] = value
            record[f"macro_{name}_pit_observation_date"] = obs_date
            record[f"macro_{name}_vintage_date"] = origin
        rows.append(record)
    macro = pd.DataFrame(rows)
    # Historical H.8 is an aggregate only and may itself be revised. It is never
    # represented as observed weekly bank-level deposits.
    for name in H8_SERIES:
        macro[f"macro_{name}_qoq_revised"] = macro[f"macro_{name}_revised"].pct_change(fill_method=None)
    macro["macro_curve_10y2y_revised"] = macro["macro_DGS10_revised"] - macro["macro_DGS2_revised"]
    macro["macro_DFF_change_4q_revised"] = macro["macro_DFF_revised"].diff(4)
    out = root / "data" / "processed" / "macro_origins.parquet"
    macro.to_parquet(out, index=False, compression="zstd")
    quality = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quarter_count": len(macro), "first_quarter": str(quarters[0]),
        "last_quarter": str(quarters[-1]),
        "historical_vintage_snapshots": len(snapshots),
        "pit_coverage": {name: int(macro[f"macro_{name}_pit"].notna().sum()) for name in MONTHLY_SERIES},
        "revised_coverage": {name: int(macro[f"macro_{name}_revised"].notna().sum())
                             for name in MARKET_SERIES + MONTHLY_SERIES + H8_SERIES},
        "h8_scope": "system aggregate, not bank specific",
        "availability_assumptions_days": {"market": 2, "monthly_revised": 30, "h8_weekly": 9},
        "pit_caveat": "ALFRED vintage snapshots at forecast origin; market and H.8 series have conservative lags but no archived vintages",
    }
    (root / "data" / "manifests" / "macro_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2), flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    fred = sorted((root / "data" / "raw" / "fred").glob("*.csv"))
    RunState(root).run(
        "macro_origins", "BUILDING_DATA",
        {"fred_sha256": {p.name: sha256_file(p) for p in fred},
         "alfred_manifest_sha256": sha256_file(root / "data" / "manifests" / "alfred_vintages.json")},
        ["data/processed/macro_origins.parquet", "data/manifests/macro_quality.json"],
        lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
