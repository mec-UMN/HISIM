"""Build the pre-computed demo bundle (HISIM_GUI_nice/demo/) from a running HISIM GUI server.

The Demo button loads this bundle instantly instead of running the simulator, so a presentation never
depends on a live simulation. Re-run this script whenever the engine or the thermal model changes:

    python backend/scripts/build_demo.py                # uses http://localhost:8001
    python backend/scripts/build_demo.py --api http://localhost:8011/api
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"
DEMO_DIR = ROOT / "demo"

# 3.5D showcase: 4 stacks side by side (2.5D), each 2 tiers (3D); identical chiplets so the package looks regular.
SHOWCASE = {"aimodel": "resnet50", "default_files_generic": False, "type_default_files": "3_5D_Mesh",
            "stack_count": 4, "chip_count": 2, "thermal_sim": True}
# The v1.1 thermal model does not support 3.5D yet, so the thermal showcase is the same workload on 2.5D.
THERMAL = {**SHOWCASE, "type_default_files": "2_5D_Mesh", "chip_count": 1, "thermal_sim": True}
SWEEPS = [
    ("n_sa", "Compute · Systolic arrays per tile", [1, 2, 4, 8]),
    ("stack_count", "Integration · Stacks (chiplets in package)", [2, 4, 6, 8]),
    ("n_2_5d_channels_per_chiplet_edge", "Interconnect · 2.5D channels per chiplet edge", [1, 2, 4, 8]),
]


def call(api, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(api + path, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def wait(api, ids, timeout_s=1800):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        records = [call(api, f"/runs/{i}") for i in ids]
        if all(r["status"] in ("completed", "failed") for r in records):
            return records
        time.sleep(5)
    raise TimeoutError("demo runs did not finish in time")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8001/api")
    api = parser.parse_args().api

    showcase = call(api, "/runs", {"name": "Demo · 3.5D resnet50 (4 stacks × 2 tiers)", "config": SHOWCASE})
    thermal = call(api, "/runs", {"name": "Demo · 2.5D resnet50 with thermal maps", "config": THERMAL})
    sweep_ids, groups = [], []
    for knob, label, values in SWEEPS:
        base = {**SHOWCASE, "thermal_sim": False}
        created = call(api, "/sweeps", {"config": base, "knob": knob, "knob_label": label, "values": values})
        sweep_ids += [r["id"] for r in created]
        groups.append(created[0]["group"])

    ids = [showcase["id"], thermal["id"], *sweep_ids]
    records = wait(api, ids)
    for r in records:
        print(f"{r['status']:9s} thermal={r['thermal_status']:6s} {r['name']}")
    failed = [r["name"] for r in records if r["status"] != "completed"]
    if failed:
        print("Some demo runs failed; the bundle was NOT written:", *failed, sep="\n  ")
        sys.exit(1)

    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    (DEMO_DIR / "runs").mkdir(parents=True)
    for run_id in ids:
        shutil.copytree(RUNS_DIR / run_id, DEMO_DIR / "runs" / run_id,
                        ignore=shutil.ignore_patterns("matplotlib", "artifacts", "config_used.py", "engine",
                                                     "input_files", "*.npy", "*.npz"))
    manifest = {"built_at": datetime.now(timezone.utc).isoformat(), "showcase": showcase["id"],
                "thermal": thermal["id"], "sweep_groups": groups, "runs": ids}
    (DEMO_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Demo bundle written to {DEMO_DIR} ({len(ids)} runs)")


if __name__ == "__main__":
    main()
