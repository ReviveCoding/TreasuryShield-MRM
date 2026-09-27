"""Paginated public FDIC BankFind acquisition with count reconciliation."""

from __future__ import annotations

import csv
import io
import json
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://api.fdic.gov/banks"


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "TreasuryShield-MRM public-data research/0.1"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )))
    return session


def download_history(root: Path, *, page_size: int = 10000) -> dict[str, object]:
    session = _session()
    raw = root / "data" / "raw" / "fdic"
    raw.mkdir(parents=True, exist_ok=True)
    filters = "ORG_ROLE_CDE:FI"
    fields = "CERT,CHANGECODE,CHANGECODE_DESC,EFFDATE,ACQ_UNINUM,OUT_UNINUM,UNINUM,ORG_ROLE_CDE,TRANSNUM"
    probe = session.get(f"{BASE}/history", params={"filters": filters, "limit": 1, "format": "json"}, timeout=45)
    probe.raise_for_status()
    total = int(probe.json()["meta"]["total"])
    pages: list[dict[str, object]] = []
    sum_rows = 0
    seen_ids: set[str] = set()
    for offset in range(0, total, page_size):
        path = raw / f"history_fi_{offset:06d}.csv"
        params = {
            "filters": filters, "fields": fields, "format": "csv",
            "limit": page_size, "offset": offset, "sort_by": "TRANSNUM", "sort_order": "ASC",
        }
        if path.exists():
            payload = path.read_bytes()
        else:
            response = session.get(f"{BASE}/history", params=params, timeout=180)
            response.raise_for_status()
            payload = response.content
            if not payload.startswith(b'"') and not payload.startswith(b"CERT"):
                raise ValueError(f"Unexpected FDIC CSV at offset {offset}: {payload[:100]!r}")
            path.write_bytes(payload)
            time.sleep(0.2)
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        count = len(rows)
        if count == 0:
            raise ValueError(f"Empty FDIC history page at offset {offset}")
        ids = {str(row["ID"]) for row in rows}
        if len(ids) != count or seen_ids.intersection(ids):
            raise ValueError(f"Duplicate FDIC history IDs in or across page {offset}")
        seen_ids.update(ids)
        pages.append({"file": str(path.relative_to(root)).replace("\\", "/"), "offset": offset, "rows": count})
        sum_rows += count
        print(f"FDIC history: {sum_rows}/{total}", flush=True)
    if sum_rows != total:
        raise ValueError(f"FDIC history count mismatch: got {sum_rows}, API reported {total}")
    summary = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{BASE}/history", "filters": filters, "fields": fields,
        "expected_rows": total, "actual_rows": sum_rows, "pages": pages,
    }
    out = root / "data" / "manifests" / "fdic_history_pages.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def download_institutions(root: Path, *, page_size: int = 10000) -> dict[str, object]:
    session = _session()
    raw = root / "data" / "raw" / "fdic"
    raw.mkdir(parents=True, exist_ok=True)
    fields = "CERT,FED_RSSD,ACTIVE,INACTIVE,NAME,CITY,STNAME,EFFDATE,INSDATE,DATEUPDT,NEWCERT,ULTCERT,BKCLASS,CHARTER"
    probe = session.get(f"{BASE}/institutions", params={"limit": 1, "format": "json"}, timeout=45)
    probe.raise_for_status()
    total = int(probe.json()["meta"]["total"])
    pages = []
    sum_rows = 0
    for offset in range(0, total, page_size):
        path = raw / f"institutions_{offset:06d}.csv"
        if path.exists():
            payload = path.read_bytes()
        else:
            response = session.get(f"{BASE}/institutions", params={
                "fields": fields, "format": "csv", "limit": page_size,
                "offset": offset, "sort_by": "CERT", "sort_order": "ASC",
            }, timeout=120)
            response.raise_for_status()
            payload = response.content
            path.write_bytes(payload)
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
        if not rows:
            raise ValueError(f"Empty FDIC institution page at offset {offset}")
        pages.append({"file": str(path.relative_to(root)).replace("\\", "/"), "offset": offset, "rows": len(rows)})
        sum_rows += len(rows)
        print(f"FDIC institutions: {sum_rows}/{total}", flush=True)
    if sum_rows != total:
        raise ValueError(f"FDIC institution count mismatch: {sum_rows} != {total}")
    summary = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{BASE}/institutions", "fields": fields,
        "expected_rows": total, "actual_rows": sum_rows, "pages": pages,
    }
    out = root / "data" / "manifests" / "fdic_institutions_pages.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--institutions", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    download_institutions(root) if args.institutions else download_history(root)
