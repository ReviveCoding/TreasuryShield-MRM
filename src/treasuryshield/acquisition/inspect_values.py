"""Check actual temporal and entity coverage of the harmonized bank panel."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyreadstat


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    path = root / "data" / "interim" / "newyorkfed" / "call-reports-balance-sheets-Jun2026.dta"
    columns = ["id_rssd", "date", "assets", "deposits", "id_fdic_cert", "entity_type", "reason_term_cd"]
    frame, _ = pyreadstat.read_dta(str(path), usecols=columns)
    # The source variable is a Stata quarterly integer: 0 means 1960Q1.
    # Treating it as a Unix/Pandas timestamp would silently collapse all years.
    stata_quarter = frame["date"].astype("int64")
    q = pd.PeriodIndex.from_fields(
        year=(1960 + stata_quarter // 4).to_numpy(),
        quarter=(stata_quarter % 4 + 1).to_numpy(), freq="Q",
    )
    summary = {
        "rows": len(frame),
        "unique_rssd": int(frame["id_rssd"].nunique()),
        "min_stata_quarter": int(stata_quarter.min()),
        "max_stata_quarter": int(stata_quarter.max()),
        "min_date": str(q.min().end_time.date()),
        "max_date": str(q.max().end_time.date()),
        "min_quarter": str(q.min()),
        "max_quarter": str(q.max()),
        "assets_nonmissing": int(frame["assets"].notna().sum()),
        "deposits_nonmissing": int(frame["deposits"].notna().sum()),
        "fdic_cert_nonmissing": int(frame["id_fdic_cert"].notna().sum()),
        "duplicate_rssd_date": int(frame.duplicated(["id_rssd", "date"]).sum()),
        "latest_quarters": {str(k): int(v) for k, v in q.value_counts().sort_index().tail(12).items()},
        "entity_types": {str(k): int(v) for k, v in frame["entity_type"].value_counts(dropna=False).head(12).items()},
        "sample": frame.head(3).astype(str).to_dict(orient="records"),
    }
    out = root / "data" / "manifests" / "ny_coverage.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
