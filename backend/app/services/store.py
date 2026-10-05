from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.run import RunRecord
from app.services.paths import RUNS_DIR


def ensure_runs_dir() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_dir(run_id: str) -> Path:
    ensure_runs_dir()
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_path(run_id: str) -> Path:
    return run_dir(run_id) / "record.json"


def save_record(record: RunRecord) -> None:
    path = record_path(record.id)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def load_record(run_id: str) -> RunRecord:
    path = RUNS_DIR / run_id / "record.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    return RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))


def delete_record(run_id: str) -> None:
    path = RUNS_DIR / run_id
    if not path.exists():
        raise FileNotFoundError(run_id)
    shutil.rmtree(path, ignore_errors=True)


def list_records() -> list[RunRecord]:
    ensure_runs_dir()
    records: list[RunRecord] = []
    for path in RUNS_DIR.glob("*/record.json"):
        try:
            records.append(RunRecord.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    records.sort(key=_sort_stamp, reverse=True)
    return records


def _sort_stamp(record: RunRecord) -> datetime:
    """Never let a stray naive timestamp break sorting."""
    stamp = record.created_at
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
