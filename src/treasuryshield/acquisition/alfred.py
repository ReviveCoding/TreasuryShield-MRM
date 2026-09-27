"""Official no-key ALFRED graph vintages at historical forecast origins."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from treasuryshield.orchestration.state import RunState

SERIES = ("UNRATE", "CFNAI", "CPIAUCSL")
FIRST_TESTED_VINTAGE = {"UNRATE": date(2001, 5, 15),
                        "CPIAUCSL": date(2001, 5, 15),
                        "CFNAI": date(2012, 5, 15)}
URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def quarter_ends(start_year: int, end_year: int) -> list[tuple[int, int, date]]:
    result = []
    for year in range(start_year, end_year + 1):
        for quarter, month, day in [(1, 3, 31), (2, 6, 30), (3, 9, 30), (4, 12, 31)]:
            result.append((year, quarter, date(year, month, day)))
    return result


def download_vintages(root: Path, start_year: int = 2001, end_year: int = 2026) -> None:
    today = datetime.now(timezone.utc).date()
    origins = [(year, q, end + timedelta(days=45))
               for year, q, end in quarter_ends(start_year, end_year)
               if end + timedelta(days=45) <= today]
    raw = root / "data" / "raw" / "alfred"
    raw.mkdir(parents=True, exist_ok=True)
    selections = [(year, q, origin, series) for year, q, origin in origins for series in SERIES
                  if origin >= FIRST_TESTED_VINTAGE[series]]
    outputs = [str((raw / f"{series}_{origin.isoformat()}.csv").relative_to(root)).replace("\\", "/")
               for _, _, origin, series in selections]
    session = requests.Session()
    session.headers["User-Agent"] = "TreasuryShield-MRM public-data research/0.1"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]
    )))

    def acquire() -> None:
        rows = []
        for year, quarter, origin in origins:
            for series in SERIES:
                if origin < FIRST_TESTED_VINTAGE[series]:
                    continue
                path = raw / f"{series}_{origin.isoformat()}.csv"
                url = f"{URL}?id={series}&vintage_date={origin.isoformat()}"
                if path.exists():
                    payload = path.read_bytes()
                else:
                    response = session.get(URL, params={"id": series, "vintage_date": origin.isoformat()}, timeout=60)
                    response.raise_for_status()
                    payload = response.content
                reader = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
                header = next(reader, [])
                expected = f"{series}_{origin:%Y%m%d}"
                if header != ["observation_date", expected]:
                    raise ValueError(f"ALFRED vintage invalid for {series} {origin}: {header}")
                observations = sum(1 for _ in reader)
                if observations == 0:
                    raise ValueError(f"ALFRED vintage empty for {series} {origin}")
                if not path.exists():
                    path.write_bytes(payload)
                    time.sleep(0.05)
                rows.append({"quarter": f"{year}Q{quarter}", "forecast_origin": origin.isoformat(),
                             "series": series, "observations": observations,
                             "file": str(path.relative_to(root)).replace("\\", "/"), "url": url})
            print(f"ALFRED {year}Q{quarter}: {len(rows)}/{len(outputs)} series snapshots", flush=True)
        out = root / "data" / "manifests" / "alfred_vintages.json"
        out.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(),
                                   "rows": rows}, indent=2), encoding="utf-8")

    RunState(root).run(
        "alfred_vintages", "ACQUIRING_DATA",
        {"series": SERIES, "series_first_vintage": FIRST_TESTED_VINTAGE,
         "start_year": start_year, "end_year": end_year,
         "origins": [o.isoformat() for _, _, o in origins]},
        outputs + ["data/manifests/alfred_vintages.json"], acquire, resume=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2026)
    args = parser.parse_args()
    download_vintages(Path(__file__).resolve().parents[3], args.start_year, args.end_year)
