from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "v1_1_analysis"


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


def source_tree_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        # v1.1 is intentionally excluded: this identifies the frozen v1 implementation.
        if "v11" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        digest.update(rel.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def frozen_paths() -> list[Path]:
    fixed = [
        ROOT / "artifacts" / "model_selection.json",
        ROOT / "artifacts" / "model_selection.sha256",
        ROOT / "artifacts" / "completion_audit.json",
        ROOT / "artifacts" / "results.duckdb",
        ROOT / "artifacts" / "predictions" / "core_B_1q_final_locked_B8_lightgbm_quantile.parquet",
        ROOT / "artifacts" / "predictions" / "core_B_1q_final_locked_B8_lightgbm_quantile.json",
        ROOT / "artifacts" / "predictions" / "core_B_1q_final_locked_B8_lightgbm_quantile_metrics.json",
        ROOT / "reports" / "preregistration.md",
        ROOT / "reports" / "preregistration.sha256",
    ]
    reports = [
        path
        for path in (ROOT / "reports").iterdir()
        if path.is_file() and path.suffix.lower() in {".html", ".md"}
    ]
    return sorted(set(fixed + reports))


def package_versions() -> dict[str, str]:
    names = [
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "duckdb",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "lightgbm",
        "xgboost",
        "torch",
        "shap",
    ]
    versions = {"python": platform.python_version()}
    for name in names[1:]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT / "artifacts" / "model_selection.json").read_text())
    audit = json.loads((ROOT / "artifacts" / "completion_audit.json").read_text())
    paths = frozen_paths()
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing v1 evidence: {missing}")
    expected = "core_B_1q_final_locked_B8_lightgbm_quantile"
    if selection.get("final_run_id") != expected:
        raise RuntimeError(
            f"Repository evidence selected {selection.get('final_run_id')!r}, expected {expected!r}"
        )
    manifest = {
        "schema_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Byte-level freeze of v1 evidence before post-model diagnostics",
        "v1_completion_verified": bool(audit.get("verified_complete")),
        "selected_v1_model": selection["selected_model_id"],
        "selected_v1_run_id": selection["final_run_id"],
        "data_snapshot_id": selection["data_snapshot_id"],
        "git_commit": git_value("rev-parse", "HEAD") or "NO_COMMIT_REPOSITORY",
        "git_status_porcelain": (git_value("status", "--porcelain") or "").splitlines(),
        "v1_source_tree_sha256": source_tree_hash(),
        "runtime": {
            "platform": platform.platform(),
            "packages": package_versions(),
            "gpu": audit.get("checks", {}).get("verified_gpu_name"),
        },
        "frozen_files": {
            path.relative_to(ROOT).as_posix(): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        },
    }
    output = OUT / "v1_freeze_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(output.relative_to(ROOT)),
        "frozen_files": len(paths),
        "selected_run": selection["final_run_id"],
        "snapshot": selection["data_snapshot_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
