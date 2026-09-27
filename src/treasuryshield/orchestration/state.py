"""Atomic, checksum-based stage checkpointing. A cache hit is verified, not assumed."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunState:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / ".treasuryshield" / "run_state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        else:
            self.data = {
                "project": "TreasuryShield-MRM",
                "status": "INITIALIZING",
                "updated_at": utcnow(),
                "data_snapshot_id": None,
                "stages": {},
                "blocked_optional": [],
                "completion_verified": False,
            }
            self.save()

    def save(self) -> None:
        self.data["updated_at"] = utcnow()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def set_status(self, status: str) -> None:
        self.data["status"] = status
        self.save()

    def set_snapshot(self, snapshot_id: str) -> None:
        self.data["data_snapshot_id"] = snapshot_id
        self.save()

    def _checksums(self, outputs: Iterable[str]) -> dict[str, str]:
        return {
            name: sha256_file(self.root / name)
            for name in outputs
            if (self.root / name).is_file()
        }

    def cached(self, name: str, config_hash: str, outputs: list[str]) -> bool:
        stage = self.data.get("stages", {}).get(name, {})
        if stage.get("status") != "completed" or stage.get("config_hash") != config_hash:
            return False
        if set(stage.get("actual_outputs", [])) != set(outputs):
            return False
        return stage.get("checksums") == self._checksums(outputs)

    def run(
        self,
        name: str,
        status: str,
        config: object,
        outputs: list[str],
        fn: Callable[[], T],
        *,
        resume: bool = False,
    ) -> T | None:
        config_hash = stable_hash(config)
        if resume and self.cached(name, config_hash, outputs):
            return None
        prior = self.data.setdefault("stages", {}).get(name, {})
        record = {
            "status": "running",
            "started_at": utcnow(),
            "ended_at": None,
            "config_hash": config_hash,
            "data_snapshot": self.data.get("data_snapshot_id"),
            "expected_outputs": outputs,
            "actual_outputs": [],
            "checksums": {},
            "retry_count": prior.get("retry_count", 0) + (1 if prior else 0),
            "error": None,
        }
        self.data["stages"][name] = record
        self.set_status(status)
        try:
            result = fn()
            missing = [p for p in outputs if not (self.root / p).is_file()]
            if missing:
                raise FileNotFoundError(f"Stage {name} did not create {missing}")
            record.update(
                status="completed",
                ended_at=utcnow(),
                actual_outputs=outputs,
                checksums=self._checksums(outputs),
            )
            self.save()
            return result
        except Exception:
            record.update(status="failed", ended_at=utcnow(), error=traceback.format_exc())
            self.set_status("FAILED_RECOVERABLE")
            raise
