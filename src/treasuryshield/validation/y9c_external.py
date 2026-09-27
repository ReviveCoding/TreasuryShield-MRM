"""Separate BHC-level transfer check using official historical FR Y-9C data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file

QUANTILES = [.01, .05, .10, .25, .5, .75, .9]


def _pinball(y: np.ndarray, prediction: float, q: float) -> float:
    error = y - prediction
    return float(np.maximum(q * error, (q - 1) * error).mean())


def build(root: Path) -> None:
    bhc = pd.read_parquet(root / "data" / "processed" / "y9c_bhc_panel.parquet",
                          columns=["bhc_rssd", "report_period", "quarter_end", "deposits_consolidated",
                                   "assets", "equity_asset_ratio"])
    bhc["quarter_code"] = bhc["report_period"].str[:4].astype(int) * 4 + bhc["report_period"].str[-1].astype(int) - 1
    future = bhc[["bhc_rssd", "quarter_code", "deposits_consolidated"]].copy()
    future["quarter_code"] -= 1
    future = future.rename(columns={"deposits_consolidated": "future_deposits"})
    panel = bhc.merge(future, on=["bhc_rssd", "quarter_code"], how="left", validate="one_to_one")
    panel = panel.loc[panel["deposits_consolidated"].gt(0) & panel["future_deposits"].gt(0)].copy()
    panel["deposit_log_growth"] = np.log(panel["future_deposits"] / panel["deposits_consolidated"])
    # 2019Q4 origins have 2020Q1 outcomes and are not admissible when the
    # training outcome cutoff is 2019Q4.
    train = panel.loc[panel["report_period"].between("2001Q1", "2019Q3")]
    test = panel.loc[panel["report_period"].between("2020Q1", "2020Q4")]
    if len(train) < 1000 or len(test) < 100:
        raise ValueError("Insufficient BHC historical validation data")
    bank_targets = pd.read_parquet(root / "data" / "features" / "targets.parquet",
                                   columns=["report_period", "horizon_quarters", "deposit_log_growth",
                                            "mna_contaminated_radius_1"])
    bank_train = bank_targets.loc[bank_targets["report_period"].between("2001Q1", "2019Q3")
                                  & bank_targets["horizon_quarters"].eq(1)
                                  & bank_targets["mna_contaminated_radius_1"].eq(False)]
    bank_q = np.quantile(bank_train["deposit_log_growth"], QUANTILES)
    bhc_q = np.quantile(train["deposit_log_growth"], QUANTILES)
    actual = test["deposit_log_growth"].to_numpy()
    rows = []
    for name, quantiles in [("bank_trained_unconditional_transfer", bank_q),
                            ("bhc_historical_unconditional", bhc_q)]:
        for q, prediction in zip(QUANTILES, quantiles):
            rows.append({"model_id": name, "quantile": q, "value": float(prediction),
                         "pinball": _pinball(actual, float(prediction), float(q)),
                         "coverage": float(np.mean(actual <= prediction))})
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "unit": "consolidated holding company, not bank charter",
        "training_bhc_rows": len(train), "evaluation_bhc_rows": len(test),
        "evaluation_origins": "2020Q1-2020Q4; one-quarter outcomes through 2021Q1",
        "bank_training_rows": len(bank_train),
        "models": rows,
        "limitation": "Current NIC bulk endpoint returned CAPTCHA; extension ends 2021Q1. Transfer check uses unconditional quantiles, not a bank-trained RAHQ feature mapping.",
    }
    path = root / "artifacts" / "y9c_external_validity.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    con = duckdb.connect(str(root / "artifacts" / "results.duckdb"))
    con.execute("DELETE FROM metrics WHERE experiment_id='E39'")
    for row in rows:
        quantile = f"Q{round(row['quantile']*100):02d}"
        for metric in ["pinball", "coverage"]:
            con.execute("""INSERT INTO metrics VALUES ('E39_y9c_transfer', 'E39', 'y9c_bhc',
                'A', 'pandemic', 1, ?, ?, ?, 'equal_bhc', ?, ?, ?, 'bhc_transfer')""",
                [row["model_id"], metric, quantile, row[metric], len(test),
                 RunState(root).data["data_snapshot_id"]])
    con.execute("UPDATE experiment_registry SET status='completed', evidence_paths='artifacts/y9c_external_validity.json', updated_at=current_timestamp WHERE experiment_id='E39'")
    con.close()
    print(f"FR Y-9C external check: {len(test)} separate BHC test observations", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    RunState(root).run(
        "y9c_external_validity", "VALIDATING",
        {"bhc_sha256": sha256_file(root / "data" / "processed" / "y9c_bhc_panel.parquet"),
         "bank_target_sha256": sha256_file(root / "data" / "features" / "targets.parquet")},
        ["artifacts/y9c_external_validity.json"], lambda: build(root), resume=True)


if __name__ == "__main__":
    main()
