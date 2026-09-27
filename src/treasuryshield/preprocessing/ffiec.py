"""Parse official FFIEC quarterly bulk schedules into a separate Enhanced Panel.

The raw ZIPs stay intact. Missing schedule items remain missing, including items
that did not exist on a form in a particular quarter.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

FIELDS = {
    "RCON5597": "uninsured_domestic_deposits",
    "RCONL197": "uninsured_consolidated_deposits",
    "RCON2365": "brokered_domestic_deposits",
    "RCON2215": "transaction_domestic_deposits",
    "RCON2385": "nontransaction_domestic_deposits",
    "RCONJ474": "large_time_domestic_deposits",
    "RCONG803": "reciprocal_brokered_deposits",
    "RCON3190": "other_borrowed_money_domestic",
    "RCON2200": "domestic_deposits_ffiec",
    "RCFD2170": "consolidated_assets_ffiec",
    "RCON2170": "domestic_assets_ffiec",
    "RCFD3210": "consolidated_equity_ffiec",
    "RCON3210": "domestic_equity_ffiec",
    "RCFD0071": "cash_due_banks_noninterest_ffiec",
    "RCFD0081": "cash_due_banks_interest_ffiec",
    "RCONB993": "fed_funds_purchased_domestic",
    "RCONB995": "repurchase_agreements_domestic",
    "RCFD3123": "loan_loss_allowance_ffiec",
}
DATE_RE = re.compile(r"call_(\d{2})(\d{2})(\d{4})\.zip$", re.I)


def _number(value: str) -> float:
    if value is None or value.strip() == "":
        return np.nan
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return np.nan


def parse_zip(path: Path) -> tuple[pd.DataFrame, dict]:
    match = DATE_RE.fullmatch(path.name)
    if not match:
        raise ValueError(path.name)
    month, day, year = map(int, match.groups())
    quarter = (month - 1) // 3 + 1
    period = f"{year}Q{quarter}"
    records: dict[int, dict] = {}
    presence = Counter()
    field_schedules: dict[str, list[str]] = defaultdict(list)
    conflicts = Counter()
    with zipfile.ZipFile(path) as archive:
        por = next((name for name in archive.namelist() if " Bulk POR " in name), None)
        if por is None:
            raise ValueError(f"Missing POR schedule: {path}")
        with io.TextIOWrapper(archive.open(por), encoding="utf-8-sig", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            for row in reader:
                raw_id = row.get("IDRSSD", "")
                if not raw_id.isdigit():
                    continue
                bank_id = int(raw_id)
                if bank_id in records:
                    raise ValueError(f"Duplicate RSSD in POR: {path}: {bank_id}")
                records[bank_id] = {
                    "bank_id": bank_id, "report_period": period,
                    "ffiec_form": row.get("Financial Institution Filing Type", ""),
                    "fdic_cert_ffiec": _number(row.get("FDIC Certificate Number", "")),
                    "institution_name_ffiec": row.get("Financial Institution Name", ""),
                    "submission_updated_at": row.get("Last Date/Time Submission Updated On", ""),
                }
        for name in archive.namelist():
            if name == por or "Schedule" not in name:
                continue
            with io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", errors="replace", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                selected = [(mnemonic, FIELDS[mnemonic]) for mnemonic in FIELDS if mnemonic in (reader.fieldnames or [])]
                if not selected:
                    continue
                for mnemonic, _ in selected:
                    field_schedules[mnemonic].append(name)
                for row in reader:
                    raw_id = row.get("IDRSSD", "")
                    if not raw_id.isdigit():
                        continue  # Schedule label row.
                    bank_id = int(raw_id)
                    if bank_id not in records:
                        raise ValueError(f"Schedule RSSD absent from POR: {path}: {bank_id}")
                    dest = records[bank_id]
                    for mnemonic, canonical in selected:
                        value = _number(row.get(mnemonic, ""))
                        if np.isnan(value):
                            continue
                        old = dest.get(canonical)
                        if old is not None and not np.isnan(old) and old != value:
                            conflicts[mnemonic] += 1
                            continue
                        dest[canonical] = value
                        presence[mnemonic] += 1
    frame = pd.DataFrame.from_records(list(records.values()))
    for canonical in FIELDS.values():
        if canonical not in frame:
            frame[canonical] = np.nan
    quarter_end = pd.Period(period, freq="Q").to_timestamp(how="end").normalize()
    frame["quarter_end"] = quarter_end
    frame["public_availability_date"] = quarter_end + pd.Timedelta(days=45)
    frame["source_vintage_ffiec"] = "retrieval_snapshot_2026-09-27"
    frame["panel_mode"] = "A_RESTATED"
    denominator = frame["domestic_deposits_ffiec"].where(frame["domestic_deposits_ffiec"] > 0)
    for value, ratio in [
        ("uninsured_domestic_deposits", "uninsured_domestic_share"),
        ("brokered_domestic_deposits", "brokered_domestic_share"),
        ("transaction_domestic_deposits", "transaction_domestic_share"),
        ("nontransaction_domestic_deposits", "nontransaction_domestic_share"),
        ("large_time_domestic_deposits", "large_time_domestic_share"),
    ]:
        frame[ratio] = frame[value] / denominator
    meta = {
        "report_period": period, "por_rows": len(frame),
        "filing_types": frame["ffiec_form"].value_counts(dropna=False).to_dict(),
        "present_mnemonics": dict(presence),
        "field_schedules": dict(field_schedules),
        "conflicting_duplicate_values": dict(conflicts),
    }
    return frame, meta


def build(root: Path) -> None:
    paths = sorted((root / "data" / "raw" / "ffiec").glob("call_*.zip"))
    if len(paths) < 35:
        raise ValueError(f"Expected 2017-2026 FFIEC quarterly files, found {len(paths)}")
    frames = []
    diagnostics = []
    for number, path in enumerate(paths, 1):
        frame, meta = parse_zip(path)
        frames.append(frame)
        diagnostics.append(meta)
        if number % 4 == 0 or number == len(paths):
            print(f"FFIEC Enhanced Panel: {number}/{len(paths)} ({meta['report_period']}, {len(frame)} filers)", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["bank_id", "report_period"]).any():
        raise ValueError("Duplicate bank-quarter after FFIEC parsing")
    out = root / "data" / "processed" / "enhanced_panel.parquet"
    combined.to_parquet(out, index=False, compression="zstd")
    diagnostics_path = root / "data" / "manifests" / "ffiec_quality.json"
    diagnostics_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": diagnostics, "rows": len(combined),
        "unique_banks": int(combined["bank_id"].nunique()),
        "first_period": combined["report_period"].min(),
        "last_period": combined["report_period"].max(),
        "uninsured_coverage": float(combined["uninsured_domestic_deposits"].notna().mean()),
        "domestic_deposit_coverage": float(combined["domestic_deposits_ffiec"].notna().mean()),
    }, indent=2), encoding="utf-8")
    print(f"Parsed {len(combined)} Enhanced Panel bank-quarters", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "data" / "raw" / "ffiec").glob("call_*.zip"))
    RunState(root).run(
        "enhanced_panel", "BUILDING_DATA",
        {"raw_sha256": {p.name: sha256_file(p) for p in paths}, "fields": FIELDS,
         "availability_lag_days": 45},
        ["data/processed/enhanced_panel.parquet", "data/manifests/ffiec_quality.json"],
        lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
