"""Pre-computed demo bundle (HISIM_GUI_nice/demo/), built by backend/scripts/build_demo.py.

Loading it copies the bundled runs into runs/ so the dashboard, Architecture 3D and Runs & Compare show a
complete 3.5D example instantly, without running the simulator (safe for presentations).
"""
from __future__ import annotations

import json
import shutil
from typing import Any

from app.schemas.run import RunRecord
from app.services.paths import ROOT_DIR, RUNS_DIR

DEMO_DIR = ROOT_DIR / "demo"


def load_demo() -> dict[str, Any]:
    manifest_path = DEMO_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "The demo bundle is missing. Build it with `python backend/scripts/build_demo.py` while the GUI server runs."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for run_id in manifest.get("runs", []):
        source = DEMO_DIR / "runs" / run_id
        target = RUNS_DIR / run_id
        if not source.exists():
            continue
        shutil.copytree(source, target, dirs_exist_ok=True)
        record_path = target / "record.json"
        record = RunRecord.model_validate(json.loads(record_path.read_text(encoding="utf-8")))
        record.stdout_path = str(target / "stdout.log")
        record.stderr_path = str(target / "stderr.log")
        record_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return manifest
