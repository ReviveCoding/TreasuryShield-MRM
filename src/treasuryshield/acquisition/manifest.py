"""Verify retained raw files and create a single immutable snapshot manifest."""

from __future__ import annotations

import csv
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import re

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash

NY = "https://www.newyorkfed.org/medialibrary/media/research/banking_research/balance-sheets-income-statements/"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
ALFRED = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id="
Y9C = "https://www.chicagofed.org/~/media/others/banking/financial-institution-reports/bhc-data/"


def source_metadata(path: Path) -> tuple[str, str, str | None]:
    name = path.name
    parent = path.parent.name
    if parent == "newyorkfed":
        item = {
            "call-reports-balance-sheets-Jan2026.zip": "call-reports-balance-sheets-Jan2026",
            "call-reports-income-statements-Jan2026.zip": "call-reports-income-statements-Jan2026",
            "historical_call_data_dictionary.xlsx": "historical_call_data_dictionary",
            "README.html": "README",
        }.get(name, name)
        return "Federal Reserve Bank of New York", NY + item, "Jun2026" if name.endswith(".zip") else None
    if parent == "fred":
        return "Federal Reserve Bank of St. Louis / source agency", FRED + path.stem, None
    if parent == "alfred":
        match = re.fullmatch(r"([A-Z0-9]+)_(\d{4}-\d{2}-\d{2})\.csv", name)
        if not match:
            raise ValueError(f"Unexpected ALFRED raw filename: {name}")
        series, vintage = match.groups()
        return "Federal Reserve Bank of St. Louis / source agency", ALFRED + series + "&vintage_date=" + vintage, vintage
    if parent == "y9c":
        return "Federal Reserve Bank of Chicago", Y9C + name, None
    if parent == "fdic":
        endpoint = "history" if "history" in name else "failures" if "failure" in name else "institutions"
        return "FDIC", "https://api.fdic.gov/banks/" + endpoint, None
    if parent == "ffiec":
        return "FFIEC", "https://cdr.ffiec.gov/public/PWS/DownloadBulkData.aspx", None
    return "unknown", "", None


def validate_and_write(root: Path, *, freeze: bool = True) -> str:
    raw = root / "data" / "raw"
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError("No official raw files acquired")
    fields = [
        "organization", "source_url", "retrieved_at", "source_release_date",
        "reporting_period", "availability_date", "source_vintage", "local_filename",
        "sha256", "compressed_size", "uncompressed_size", "parser_version",
    ]
    records: list[dict[str, object]] = []
    for path in files:
        if path.stat().st_size == 0:
            raise ValueError(f"Empty raw file: {path}")
        uncompressed = path.stat().st_size
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f"ZIP CRC failed for {path}: {bad}")
                uncompressed = sum(e.file_size for e in archive.infolist() if not e.is_dir())
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
                if not stream.readline().strip():
                    raise ValueError(f"Missing CSV header: {path}")
        org, url, vintage = source_metadata(path)
        reporting_period = ""
        availability_date = ""
        if path.parent.name == "ffiec":
            match = re.fullmatch(r"call_(\d{2})(\d{2})(\d{4})\.zip", path.name, re.I)
            if match:
                month, _, year = map(int, match.groups())
                reporting_period = f"{year}Q{(month-1)//3+1}"
        elif path.parent.name == "y9c":
            match = re.fullmatch(r"bhcf(\d{2})(\d{2})\.csv", path.name, re.I)
            if match:
                year, month = map(int, match.groups())
                reporting_period = f"20{year:02d}Q{(month-1)//3+1}"
        elif path.parent.name == "alfred":
            availability_date = vintage or ""
        records.append({
            "organization": org,
            "source_url": url,
            "retrieved_at": datetime.fromtimestamp(path.stat().st_ctime, timezone.utc).isoformat(),
            "source_release_date": "",
            "reporting_period": reporting_period,
            "availability_date": availability_date,
            "source_vintage": vintage or "retrieval_snapshot_2026-09-27",
            "local_filename": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "compressed_size": path.stat().st_size,
            "uncompressed_size": uncompressed,
            "parser_version": "treasuryshield-0.1.0",
        })
    snapshot = stable_hash([(r["local_filename"], r["sha256"]) for r in records])[:20]
    manifest_path = root / "data" / "manifests" / "raw_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "data_snapshot_id": snapshot,
        "raw_file_count": len(records),
        "total_compressed_bytes": sum(int(r["compressed_size"]) for r in records),
        "total_uncompressed_bytes": sum(int(r["uncompressed_size"]) for r in records),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "data" / "manifests" / "snapshot.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if freeze:
        for path in files:
            os.chmod(path, stat.S_IREAD)
    RunState(root).set_snapshot(snapshot)
    return snapshot
