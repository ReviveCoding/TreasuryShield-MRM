"""Inspect public FFIEC ZIP schedule names and raw mnemonic headers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "data" / "raw" / "ffiec" / "call_12312023.zip"
    summary: dict[str, object] = {}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            with archive.open(entry) as stream:
                header = stream.readline().decode("utf-8-sig", errors="replace").rstrip("\r\n").split("\t")
                second = stream.readline().decode("utf-8-sig", errors="replace").rstrip("\r\n").split("\t")
            summary[entry.filename] = {
                "uncompressed_bytes": entry.file_size,
                "column_count": len(header),
                "header": header,
                "second_row": second,
            }
    out = root / "data" / "manifests" / "ffiec_schema_2023q4.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, info in summary.items():
        print(name, info["column_count"], info["uncompressed_bytes"])
        if any(x in name for x in ["Schedule RC ", "Schedule RCO ", "Schedule RCE ", "Schedule RI ", "Schedule ENT "]):
            print(" ", info["header"][:25])
            print(" ", info["second_row"][:10])


if __name__ == "__main__":
    main()
