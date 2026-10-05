from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Any

from app.schemas.run import RunRecord
from app.services import hisim_runner
from app.services.store import load_record, run_dir, save_record


def queue_thermal(run_ids: list[str]) -> dict[str, Any]:
    """Mark completed runs as queued for a thermal-only pass and start the worker."""
    queued: list[str] = []
    skipped: list[dict[str, str]] = []
    for run_id in dict.fromkeys(run_ids):
        try:
            record = load_record(run_id)
        except FileNotFoundError:
            skipped.append({"id": run_id, "reason": "run not found"})
            continue
        if record.status != "completed":
            skipped.append({"id": run_id, "reason": "run is not completed"})
            continue
        if record.thermal_status in ("queued", "running"):
            skipped.append({"id": run_id, "reason": f"thermal already {record.thermal_status}"})
            continue
        record.thermal_status = "queued"
        record.thermal_error = None
        record.owner_pid = os.getpid()
        record.updated_at = hisim_runner._now()
        save_record(record)
        queued.append(run_id)
    if queued:
        _start_worker(queued)
    return {"queued": queued, "skipped": skipped}


def _start_worker(run_ids: list[str]) -> None:
    threading.Thread(target=_process, args=(run_ids,), daemon=True).start()


def _process(run_ids: list[str]) -> None:
    for run_id in run_ids:
        _run_pass(run_id)


def merge_pass_result(record: RunRecord, payload: dict[str, Any]) -> None:
    """Fold a thermal-only payload into the run's gui_result.json and record."""
    result_path = run_dir(record.id) / "gui_result.json"
    base = hisim_runner._read_result(result_path) or {}
    base.setdefault("metrics", {}).update(payload.get("metrics") or {})
    base["thermal"] = payload.get("thermal")
    base["thermal_error"] = payload.get("thermal_error")
    result_path.write_text(json.dumps(base, indent=2, default=str), encoding="utf-8")
    hisim_runner.apply_thermal_payload(record, payload)
    if not payload.get("thermal") and payload.get("thermal_error"):
        record.thermal_status = "failed"
    record.updated_at = hisim_runner._now()
    save_record(record)


def _run_pass(run_id: str) -> None:
    """Thermal-only pass in a private engine copy (Official's per-run workspace): no shared state is touched."""
    record = load_record(run_id)
    target = run_dir(run_id)
    pass_result = target / "thermal_result.json"
    try:
        record.thermal_status = "running"
        save_record(record)
        config = record.config.model_copy(update={"thermal_sim": True})
        hisim_runner._validate_mapping_files(config)
        engine_dir, _ = hisim_runner._prepare_run_workspace(target, config)
        (target / "matplotlib").mkdir(exist_ok=True)
        env = {
            **os.environ,
            "MPLCONFIGDIR": str(target / "matplotlib"),
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "HISIM_GUI_RESULT": str(pass_result),
            "HISIM_GUI_THERMAL_ONLY": "1",
            "HISIM_GUI_THERMAL_DIR": str(target / "thermal"),
            "HISIM_THERMAL_OVERRIDES": json.dumps({"output": {"plots": False}}),
        }
        completed = subprocess.run(
            [sys.executable, "gui_run.py"], cwd=engine_dir, env=env, capture_output=True,
            text=True, timeout=hisim_runner._RUN_TIMEOUT_S, check=False,
        )
        (target / "thermal_stdout.log").write_text(completed.stdout, encoding="utf-8")
        (target / "thermal_stderr.log").write_text(completed.stderr, encoding="utf-8")
        payload = hisim_runner._read_result(pass_result) or {}
        if completed.returncode != 0 or not payload.get("thermal"):
            errors = payload.get("errors") or []
            message = errors[-1].removeprefix("Error: ") if errors else hisim_runner._friendly_error(
                completed.stderr or "thermal pass failed", completed.stdout)
            payload = {"thermal": None, "thermal_error": message}
        merge_pass_result(load_record(run_id), payload)
    except subprocess.TimeoutExpired:
        merge_pass_result(load_record(run_id), {"thermal": None, "thermal_error": "thermal pass timed out"})
    except Exception as exc:
        merge_pass_result(load_record(run_id), {"thermal": None, "thermal_error": str(exc)})


def load_thermal_maps(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id) / "thermal" / "maps.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    return json.loads(path.read_text(encoding="utf-8"))
