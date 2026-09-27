"""Acquire official Chicago Fed historical FR Y-9C holding-company files.

These observations are never pooled with commercial-bank Call Reports.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import requests

from treasuryshield.orchestration.state import RunState

BASE = "https://www.chicagofed.org/~/media/others/banking/financial-institution-reports/bhc-data/"


def acquire(root: Path) -> None:
    out = root / "data" / "raw" / "y9c"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    session = requests.Session()
    periods = [(year, q) for year in range(2001, 2022)
               for q in range(1, 5) if year < 2021 or q == 1]
    for number, (year, quarter) in enumerate(periods, 1):
        month = quarter * 3
        filename = f"bhcf{year % 100:02d}{month:02d}.csv"
        path = out / filename
        url = BASE + filename
        if not path.exists() or path.stat().st_size < 1000:
            for attempt in range(4):
                try:
                    response = session.get(url, timeout=75)
                    response.raise_for_status()
                    if not response.content.startswith(b"RSSD"):
                        raise ValueError(f"Unexpected file header: {url}")
                    path.write_bytes(response.content)
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            header = next(csv.reader(stream))
        if "RSSD9001" not in header or len(header) < 100:
            raise ValueError(f"Invalid FR Y-9C CSV: {path}")
        records.append({"period": f"{year}Q{quarter}", "path": str(path.relative_to(root)),
                        "source_url": url, "bytes": path.stat().st_size,
                        "field_count": len(header)})
        if number % 10 == 0 or number == len(periods):
            print(f"FR Y-9C {number}/{len(periods)}: {year}Q{quarter}", flush=True)
    (root / "data" / "manifests" / "y9c_files.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    RunState(root).run(
        "y9c_chicago_2001_2021q1", "ACQUIRING_DATA",
        {"source": BASE, "start": "2001Q1", "end": "2021Q1"},
        ["data/manifests/y9c_files.json"], lambda: acquire(root), resume=True)


if __name__ == "__main__":
    main()
