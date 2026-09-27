"""Persist fold-fitted forward-filtered probabilities for visual verification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from treasuryshield.orchestration.state import RunState, sha256_file, stable_hash
from treasuryshield.orchestration.warehouse import connect
from treasuryshield.regimes.filtered import VECTOR, build_regimes
from treasuryshield.splits.temporal import SPLITS


def run(root: Path) -> None:
    output = root / "artifacts" / "regime_probabilities_prelock.parquet"
    manifest = output.with_suffix(".json")
    config = {"split": "tightening", "states": 3, "seed": 17,
              "mode": "B", "source_hash": sha256_file(Path(__file__)),
              "snapshot": RunState(root).data["data_snapshot_id"]}

    def execute() -> None:
        columns = ["report_period"] + list(VECTOR) + [
            "macro_UNRATE_pit", "macro_CFNAI_pit"]
        frame = pd.read_parquet(root / "data" / "features" / "core_features.parquet",
                                columns=columns)
        frame = frame.loc[frame.report_period.between("2001Q1", "2023Q4")].copy()
        frame["macro_UNRATE_revised"] = frame["macro_UNRATE_pit"]
        frame["macro_CFNAI_revised"] = frame["macro_CFNAI_pit"]
        result = build_regimes(frame, SPLITS["tightening"], kind="hmm_filtered", states=3)
        result.features.to_parquet(output, index=False, compression="zstd")
        likelihood_history = list(result.model["hmm"].monitor_.history)
        likelihood_delta = (float(likelihood_history[-1] - likelihood_history[-2])
                            if len(likelihood_history) > 1 else None)
        payload = {"created_at": datetime.now(timezone.utc).isoformat(),
                   "data_snapshot_id": config["snapshot"], "config_hash": stable_hash(config),
                   "regime_metadata": result.metadata, "rows": len(result.features),
                   "last_training_log_likelihood_delta": likelihood_delta,
                   "monotone_last_step": likelihood_delta is None or likelihood_delta >= 0,
                   "method": "HMM parameters fitted through 2021Q4; forward filtering over 2001-2023, with no future smoothing."}
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        con = connect(root)
        try:
            con.execute("DELETE FROM monitoring WHERE run_id='regime_trace_prelock'")
            if likelihood_delta is not None:
                con.execute("INSERT INTO monitoring VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            ["regime_trace_prelock", result.metadata["last_training_quarter"],
                             "hmm_filtered_3_state", "DIAGNOSTIC",
                             "last_training_log_likelihood_delta", likelihood_delta,
                             likelihood_delta < 0,
                             "artifacts/regime_probabilities_prelock.json"])
        finally:
            con.close()
        print(f"Filtered regime trace: {len(result.features)} quarters", flush=True)

    RunState(root).run("regime_trace_prelock", "RUNNING_EXPERIMENTS", config,
                       ["artifacts/regime_probabilities_prelock.parquet",
                        "artifacts/regime_probabilities_prelock.json"], execute, resume=True)


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[3])
