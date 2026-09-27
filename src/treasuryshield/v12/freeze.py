from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "v1_2_hardening"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def material_v11_paths() -> list[Path]:
    roots = [
        ROOT / "artifacts" / "v1_1_analysis",
        ROOT / "figures" / "v1_1",
        ROOT / "reports" / "v1_1",
    ]
    paths = [ROOT / ".treasuryshield" / "v1_1_run_state.json"]
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(set(paths))


def main() -> None:
    v1_state = json.loads((ROOT / ".treasuryshield" / "run_state.json").read_text())
    v11_state = json.loads(
        (ROOT / ".treasuryshield" / "v1_1_run_state.json").read_text()
    )
    v1_audit = json.loads((ROOT / "artifacts" / "completion_audit.json").read_text())
    v11_audit = json.loads(
        (ROOT / "artifacts" / "v1_1_analysis" / "v1_1_audit.json").read_text()
    )
    prior_freeze = json.loads(
        (ROOT / "artifacts" / "v1_1_analysis" / "v1_freeze_manifest.json").read_text()
    )
    if v1_state.get("status") != "COMPLETE" or not v1_audit.get("verified_complete"):
        raise RuntimeError("v1 is not in a verified COMPLETE state")
    if v11_state.get("status") != "COMPLETE" or not v11_audit.get("verified_complete"):
        raise RuntimeError("v1.1 is not in a verified COMPLETE state")

    changed_before_freeze: list[str] = []
    for relative, metadata in prior_freeze["frozen_files"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != metadata["sha256"]:
            changed_before_freeze.append(relative)
    if changed_before_freeze:
        raise RuntimeError(
            "v1 evidence differs from the v1.1 freeze: "
            + ", ".join(changed_before_freeze)
        )

    v1_paths = [ROOT / relative for relative in prior_freeze["frozen_files"]]
    v11_paths = material_v11_paths()
    missing = [str(path.relative_to(ROOT)) for path in v11_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v1.1 evidence: {missing}")

    source_v1 = [
        path
        for path in (ROOT / "src" / "treasuryshield").rglob("*.py")
        if "v11" not in path.parts and "v12" not in path.parts
    ]
    source_v11 = list((ROOT / "src" / "treasuryshield" / "v11").rglob("*.py"))
    manifest = {
        "schema_version": "1.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Byte-level freeze of material v1 and v1.1 evidence before v1.2 hardening",
        "v1_complete": True,
        "v1_1_complete": True,
        "prior_v1_freeze_verified": True,
        "git_commit": git_value("rev-parse", "HEAD") or "NO_COMMIT_REPOSITORY",
        "git_status_porcelain": (git_value("status", "--porcelain") or "").splitlines(),
        "source_tree_hashes": {
            "v1_sha256": tree_hash(source_v1),
            "v1_1_sha256": tree_hash(source_v11),
        },
        "frozen_v1_files": {
            path.relative_to(ROOT).as_posix(): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in v1_paths
        },
        "frozen_v1_1_files": {
            path.relative_to(ROOT).as_posix(): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in v11_paths
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "freeze_manifest_v1_v11.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": output.relative_to(ROOT).as_posix(),
                "v1_files": len(v1_paths),
                "v1_1_files": len(v11_paths),
                "v1_source_hash": manifest["source_tree_hashes"]["v1_sha256"],
                "v1_1_source_hash": manifest["source_tree_hashes"]["v1_1_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
