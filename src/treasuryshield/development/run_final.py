"""Execute the frozen final-holdout roster exactly as selected prelock."""

from __future__ import annotations

import json
from pathlib import Path

from treasuryshield.development.run import run
from treasuryshield.orchestration.state import sha256_file
from treasuryshield.splits.temporal import load_joined


def execute(root: Path) -> None:
    selection_path = root / "artifacts" / "model_selection.json"
    digest = (root / "artifacts" / "model_selection.sha256").read_text(
        encoding="ascii").split()[0]
    if sha256_file(selection_path) != digest:
        raise RuntimeError("Frozen selection hash mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    frame = load_joined(root, "core", 1, mode="B")
    roster = selection["final_comparison_roster"]
    # Selected run is first; all fixed comparators receive exactly the same
    # official-source forecast origins and locked 2024+ target rows.
    for index, item in enumerate(roster, 1):
        print(f"FINAL {index}/{len(roster)} {item['run_id']}", flush=True)
        observed = run(root, item["model_id"], "final_locked", "core", 1,
                       mode="B", model_config=item["model_config"],
                       frame_override=frame)
        if observed != item["run_id"]:
            raise AssertionError("Frozen run identifier changed")


if __name__ == "__main__":
    execute(Path(__file__).resolve().parents[3])
