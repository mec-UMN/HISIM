from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.run import HisimConfig, RunCreate, RunRecord, SweepCreate
from app.services.paths import HISIM_DIR
from app.services.mapping_files import (
    KIND_PREFIXES,
    generated_file_path,
    mapping_files_overview,
    preferred_mapping_file_path,
)
from app.services.store import load_record, run_dir, save_record


_RUN_TIMEOUT_S = int(os.environ.get("HISIM_RUN_TIMEOUT", "1800"))

# Knobs that a sweep may vary. Maps the GUI knob id -> HisimConfig field.
SWEEPABLE_KNOBS = {
    "aimodel": "aimodel",
    "type_default_files": "type_default_files",
    "stack_count": "stack_count",
    "chip_count": "chip_count",
    "n_bank": "n_bank",
    "n_2d_links_per_tile": "n_2d_links_per_tile",
    "n_3d_links_per_tile": "n_3d_links_per_tile",
    "n_2_5d_channels_per_chiplet_edge": "n_2_5d_channels_per_chiplet_edge",
    "sa_size_x": "sa_size_x",
    "sa_size_y": "sa_size_y",
    "n_sa": "n_sa",
    "precision_bits": "precision_bits",
    "clock_hz": "clock_hz",
    "n_word": "n_word",
    "n_bit": "n_bit",
    "col_mux": "col_mux",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
#  config.py rewriting
# --------------------------------------------------------------------------- #
def _config_lines(config: HisimConfig, extra_values: dict[str, Any] | None = None) -> list[str]:
    values = config.to_config_values()
    if extra_values:
        values.update(extra_values)
    return [f"{key}={value!r}\n" for key, value in values.items()]


def _write_config(
    config: HisimConfig,
    original: str,
    config_path: Path,
    extra_values: dict[str, Any] | None = None,
) -> None:
    lines = _config_lines(config, extra_values)
    overrides = {line.split("=", 1)[0] for line in lines}
    kept_lines = []
    for line in original.splitlines(keepends=True):
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key not in overrides:
            kept_lines.append(line)
    config_path.write_text(
        "".join(kept_lines) + "\n# ---- written by the HISIM GUI ----\n" + "".join(lines),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
#  run execution
# --------------------------------------------------------------------------- #
def _missing_mapping_files(aimodel: str) -> list[str]:
    return [
        preferred_mapping_file_path(aimodel, kind).name
        for kind in KIND_PREFIXES
        if not preferred_mapping_file_path(aimodel, kind).exists()
    ]


def _validate_mapping_files(config: HisimConfig) -> None:
    if config.create_default_files:
        return
    missing = _missing_mapping_files(config.aimodel)
    if missing:
        raise ValueError(
            f"'Regenerate mapping files' is off for '{config.aimodel}', but "
            f"{', '.join(missing)} do not exist on disk. Turn 'Regenerate mapping files' "
            "on, or upload the missing input CSVs for this model first."
        )


def _validate_model(model: str) -> None:
    known = set(available_models())
    if known and model not in known:
        raise ValueError(f"Unknown AI model: {model}")


def create_run(payload: RunCreate) -> RunRecord:
    _validate_model(payload.config.aimodel)
    _validate_mapping_files(payload.config)
    record = _new_record(payload.config, payload.name)
    save_record(record)
    threading.Thread(target=_execute_run, args=(record,), daemon=True).start()
    return record


_MAX_SWEEP_RUNS = 96


def _display_sweep_value(value: Any) -> str:
    """Keep sweep names readable without exposing Python's e-notation."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric) or numeric == 0 or (1e-4 <= abs(numeric) < 1e6):
        return str(value)
    exponent = math.floor(math.log10(abs(numeric)))
    mantissa = numeric / (10 ** exponent)
    if abs(mantissa) >= 10:
        mantissa /= 10
        exponent += 1
    return f"{mantissa:.2f} × 10^{exponent}"


def create_sweep(payload: SweepCreate) -> list[RunRecord]:
    field = SWEEPABLE_KNOBS.get(payload.knob)
    if field is None:
        raise ValueError(f"Knob '{payload.knob}' cannot be swept")

    # A normal sweep uses the workload selected in Design Knobs unless the
    # caller explicitly turns on model comparison. This preserves a safe
    # single-model default for older API clients while allowing intentional
    # value × model studies from one launch.
    if payload.knob == "aimodel":
        pairs = [(str(value), value) for value in payload.values]
    else:
        selected_models = [payload.config.aimodel]
        if payload.compare_models:
            selected_models.extend(payload.models)
        selected_models = list(dict.fromkeys(selected_models))
        pairs = [(model, value) for model in selected_models for value in payload.values]
    models = list(dict.fromkeys(model for model, _ in pairs))

    known = set(available_models())
    unknown = [name for name in models if known and name not in known]
    if unknown:
        raise ValueError(f"Unknown AI model(s): {', '.join(unknown)}")

    total = len(pairs)
    if total > _MAX_SWEEP_RUNS:
        raise ValueError(
            f"{total} runs requested ({len(models)} model(s) × {len(payload.values)} values); "
            f"the limit is {_MAX_SWEEP_RUNS}."
        )

    errors: list[str] = []
    for model, value in pairs:
        _validate_model(model)
        candidate = payload.config.model_copy(deep=True)
        candidate.aimodel = model
        try:
            setattr(candidate, field, value)
        except (TypeError, ValueError) as exc:
            errors.append(f"Invalid value {value!r} for {payload.knob}: {exc}")
        if payload.knob == "type_default_files" and value not in {
            "2D_Mesh", "3D_Mesh", "2_5D_Mesh", "3_5D_Mesh",
            "2_5D_Mesh_Scaled", "3_5D_Mesh_Scaled",
        }:
            errors.append(f"Unknown topology template: {value!r}")
    if errors:
        raise ValueError(" | ".join(dict.fromkeys(errors)))

    group_id = uuid.uuid4().hex[:8]
    knob_label = payload.knob_label or payload.knob
    records: list[RunRecord] = []

    for model, value in pairs:
        config = payload.config.model_copy(deep=True)
        config.aimodel = model
        setattr(config, field, value)
        record = _new_record(config, f"{model} · {payload.knob}={_display_sweep_value(value)}")
        record.group = group_id
        record.group_label = f"{payload.knob}={value}"
        record.sweep_knob = payload.knob
        record.sweep_knob_label = knob_label
        record.sweep_value = str(value)
        record.sweep_model = config.aimodel
        save_record(record)
        records.append(record)

    ordered = records

    threading.Thread(target=_execute_sweep, args=(ordered,), daemon=True).start()
    return records


def _new_record(config: HisimConfig, name: str | None) -> RunRecord:
    now = _now()
    return RunRecord(
        id=uuid.uuid4().hex[:12],
        name=name or f"{config.aimodel} · {config.type_default_files}",
        status="queued",
        created_at=now,
        updated_at=now,
        config=config,
    )


def _execute_sweep(records: list[RunRecord]) -> None:
    for record in records:
        try:
            _execute_run(record)
        except Exception as exc:  # pragma: no cover - final safety net
            # A child failure must be terminal so the sweep worker can always
            # continue with the next point. _execute_run normally catches its
            # own failures; this covers errors before that handler can run.
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = _now()
            save_record(record)


def _copy_artifacts(target_dir: Path, aimodel: str, engine_dir: Path) -> list[str]:
    artifacts_dir = target_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    copied: list[str] = []
    candidates = [
        engine_dir / "Results",
        engine_dir / "output_summary.txt",
        engine_dir / "Module_1_Compute" / "HISIM_2_0_Files" / "HW_configs",
        engine_dir / "Module_2_Network" / "HISIM_2_0_Files" / "Network_configs",
        engine_dir / f"Chip_Map_{aimodel}.csv",
        engine_dir / f"Sys_Map_{aimodel}.csv",
        engine_dir / f"Layer_Mapping_{aimodel}.csv",
    ]
    def relevant(path: Path) -> bool:
        """Keep figures, reports and the spec files for this run's model only."""
        name = path.name.lower()
        if path.suffix.lower() in {".png", ".jpg", ".svg", ".txt", ".log"}:
            return True
        return aimodel.lower() in name or "lut" in name or "debug" in name

    for source in candidates:
        if not source.exists():
            continue
        destination = artifacts_dir / source.name
        try:
            if source.is_dir():
                shutil.copytree(
                    source, destination, dirs_exist_ok=True,
                    ignore=lambda directory, names: [
                        n for n in names
                        if not (Path(directory, n).is_dir() or relevant(Path(directory, n)))
                    ],
                )
            else:
                shutil.copy2(source, destination)
        except Exception:
            continue
        copied.append(str(destination.relative_to(artifacts_dir)))
    return copied


def _stage_input_files(aimodel: str, input_base: Path) -> None:
    """Copy the exact six selected CSVs into an isolated simulator workspace."""
    target_dir = input_base / aimodel
    target_dir.mkdir(parents=True, exist_ok=True)
    for kind, prefix in KIND_PREFIXES.items():
        source = preferred_mapping_file_path(aimodel, kind)
        if source.exists():
            shutil.copy2(source, target_dir / f"{prefix}_{aimodel}.csv")


def _set_workspace_network_clock(workspace: Path, clock_hz: float) -> None:
    """Keep the staged NoC/NoP JSON in lock-step with the GUI clock knob."""
    path = workspace / "Module_2_Network" / "HISIM_2_0_Files" / "Network.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["f_noc"] = float(clock_hz)
    payload["f_nop"] = float(clock_hz)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def _prepare_run_workspace(target: Path, config: HisimConfig) -> tuple[Path, Path]:
    """Stage a private engine copy so runs never rewrite shared HISIM state."""
    workspace = target / "engine"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(
        HISIM_DIR,
        workspace,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "Results", "*.png", "*.pkl"),
    )
    input_base = target / "input_files"
    if not config.create_default_files:
        _stage_input_files(config.aimodel, input_base)
    config_path = workspace / "config.py"
    _write_config(config, config_path.read_text(encoding="utf-8"), config_path, {"USER_FILES_DIR": str(input_base)})
    _set_workspace_network_clock(workspace, config.clock_hz)
    return workspace, input_base


def _publish_generated_input_files(aimodel: str, input_base: Path) -> None:
    """Expose generated CSVs for download without replacing user uploads."""
    source_dir = input_base / aimodel
    for kind, prefix in KIND_PREFIXES.items():
        source = source_dir / f"{prefix}_{aimodel}.csv"
        if source.exists():
            destination = generated_file_path(aimodel, kind)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


_NUM = r"([-+0-9.eE]+)"

_STDOUT_PATTERNS = {
    # stage timings
    "ai_mapping_sim_time": r"AI mapping sim time is:\s*" + _NUM,
    "compute_sim_time": r"Computing model sim time is:\s*" + _NUM,
    "network_sim_time": r"Network model sim time is:\s*" + _NUM,
    "cost_sim_time": r"Cost model sim time is:\s*" + _NUM,
    "total_sim_time": r"Total sim time is:\s*" + _NUM,
    # compute / memory
    "compute_area_mm2": r"Compute Area of the Chip is:\s*" + _NUM,
    "memory_area_mm2": r"Memory Area of the Chip is:\s*" + _NUM,
    "compute_latency_s": r"Compute Latency of the Chip is:\s*" + _NUM,
    "memory_latency_s": r"Memory Latency of the Chip is:\s*" + _NUM,
    "ddr_latency_s": r"DDR Latency of the Chip is:\s*" + _NUM,
    "compute_energy_j": r"Compute Energy of the Chip is:\s*" + _NUM,
    "memory_energy_j": r"Memory Energy of the Chip is:\s*" + _NUM,
    "ddr_energy_j": r"DDR Energy of the Chip is:\s*" + _NUM,
    # network
    "noc_area_mm2": r"Total NoC Area \(in mm\^2\)\s*" + _NUM,
    "nop_interface_area_mm2": r"Total NoP Interface Area \(in mm\^2\)\s*" + _NUM,
    "nop_router_area_mm2": r"Total NoP Router Area \(in mm\^2\)\s*" + _NUM,
    "noc_2d_latency_s": r"Total 2D Network NoC latency \(in s\)\s*" + _NUM,
    "noc_2d_energy_j": r"Total 2D Network NoC energy \(in J\)\s*" + _NUM,
    "nop_2_5d_latency_s": r"Total 2\.5D Network NoP latency \(in s\)\s*" + _NUM,
    "nop_2_5d_energy_j": r"Total 2\.5D Network NoP energy \(in J\)\s*" + _NUM,
    "noc_3d_latency_s": r"Total 3D Network NoC latency \(in s\)\s*" + _NUM,
    "noc_3d_energy_j": r"Total 3D Network NoC energy \(in J\)\s*" + _NUM,
    # combined + cost
    "total_area_mm2": r"Total Chip Area \(in mm\^2\)\s*" + _NUM,
    "total_latency_s": r"Total End-to-End Latency \(in s\)\s*" + _NUM,
    "total_energy_j": r"Total Chip Energy \(in J\)\s*" + _NUM,
    "manufacturing_volume": r"Total Recurring Cost for\s+" + _NUM + r"\s+units",
    "recurring_cost_usd": r"Total Recurring Cost for\s+[-+0-9.eE]+\s+units \(\$\):\s*" + _NUM,
    "nre_cost_usd": r"Total NRE Cost \(\$\):\s*" + _NUM,
    "cost_per_part_usd": r"Total Cost per Part \(\$\):\s*" + _NUM,
}


def _parse_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, pattern in _STDOUT_PATTERNS.items():
        match = re.search(pattern, stdout)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                continue
    return metrics


def _derive_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Add convenience metrics the dashboard likes to show."""
    derived = dict(metrics)
    latency = metrics.get("total_latency_s")
    energy = metrics.get("total_energy_j")
    area = metrics.get("total_area_mm2")
    if latency and latency > 0:
        derived["throughput_inferences_per_s"] = 1.0 / latency
        if energy is not None:
            derived["average_power_w"] = energy / latency
    if area and area > 0 and latency and latency > 0:
        derived["performance_per_area"] = (1.0 / latency) / area
    if energy and energy > 0:
        derived["inferences_per_joule"] = 1.0 / energy
    compute_l = metrics.get("compute_latency_s")
    if latency and latency > 0 and compute_l is not None:
        derived["compute_latency_fraction"] = compute_l / latency
    return derived


def _execute_run(record: RunRecord) -> None:
    target = run_dir(record.id)
    stdout_path = target / "stdout.log"
    stderr_path = target / "stderr.log"
    result_path = target / "gui_result.json"
    started = _now()

    try:
        record.status = "running"
        record.updated_at = _now()
        record.stdout_path = str(stdout_path)
        record.stderr_path = str(stderr_path)
        save_record(record)

        # Check each sweep point independently. A missing custom input set
        # marks only this point failed; _execute_sweep then advances to the
        # next model/value pair instead of rejecting the entire batch.
        _validate_mapping_files(record.config)
        engine_dir, input_base = _prepare_run_workspace(target, record.config)
        (target / "config_used.py").write_text(
            (engine_dir / "config.py").read_text(encoding="utf-8"), encoding="utf-8"
        )

        mpl_config_dir = target / "matplotlib"
        mpl_config_dir.mkdir(exist_ok=True)
        env = {
            **os.environ,
            "MPLCONFIGDIR": str(mpl_config_dir),
            "MPLBACKEND": "Agg",
            "HISIM_GUI_RESULT": str(result_path),
            "PYTHONUNBUFFERED": "1",
        }
        driver = "gui_run.py" if (engine_dir / "gui_run.py").exists() else "HISIM.py"
        completed = subprocess.run(
            [sys.executable, driver],
            cwd=engine_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=_RUN_TIMEOUT_S,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        record.return_code = completed.returncode
        metrics = _parse_metrics(completed.stdout)
        payload = _read_result(result_path)
        if payload:
            _annotate_result_inputs(payload, record.config)
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            metrics.update(
                {k: float(v) for k, v in (payload.get("metrics") or {}).items() if _is_number(v)}
            )
            record.counts = {
                k: int(v) for k, v in (payload.get("counts") or {}).items() if _is_number(v)
            }
            record.warnings = list(payload.get("warnings") or [])[:60]
            record.errors = list(payload.get("errors") or [])[:60]
        record.metrics = _derive_metrics(metrics)
        record.artifacts = _copy_artifacts(target, record.config.aimodel, engine_dir)
        if record.config.create_default_files:
            _publish_generated_input_files(record.config.aimodel, input_base)
        record.status = "completed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            record.error = _friendly_error(completed.stderr, completed.stdout)
    except subprocess.TimeoutExpired:
        record.status = "failed"
        record.error = f"HISIM run exceeded the {_RUN_TIMEOUT_S}s timeout."
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
    finally:
        record.duration_s = (_now() - started).total_seconds()
        record.updated_at = _now()
        save_record(record)


_ERROR_HINTS = [
    (
        "pygraphviz",
        "HISIM's DEBUG figure-generation mode needs pygraphviz (and the system graphviz "
        "package) to draw the layer graph. The GUI always runs with DEBUG off, so this "
        "usually means config.py was edited outside the GUI — install pygraphviz with "
        "`pip install pygraphviz`, or remove DEBUG=True from config.py.",
    ),
    (
        "Inconsistent mesh sizes in stack",
        "All chiplets inside one stack must use the same NoC mesh size. "
        "Reduce the tier count or use a generic topology template.",
    ),
    (
        "No module named",
        "A HISIM python dependency is missing. Install the packages listed in "
        "HISIM-SystolicArray/requirements.txt into the interpreter running the API.",
    ),
    (
        "Chip_Map_",
        "One or more HISIM input CSVs are missing for this AI model, and "
        "'Regenerate mapping files' is turned off, so HISIM has nothing to load. Turn "
        "'Regenerate mapping files' ON to auto-generate them, or upload the missing "
        "map/spec CSVs for this model first.",
    ),
]


def _friendly_error(stderr: str, stdout: str = "") -> str:
    """Surface the simulator/validator failure, not routine plotting noise."""
    raw = stdout if stdout.strip() else stderr
    if stderr.strip() and stdout.strip():
        raw = f"{stdout}\n\n[stderr]\n{stderr}"
    for needle, hint in _ERROR_HINTS:
        if needle in raw:
            return f"{hint}\n\n---\n{raw[-2500:]}"
    return raw[-4000:]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _annotate_result_inputs(payload: dict[str, Any], config: HisimConfig) -> dict[str, Any]:
    """Persist which input-file mode and files were used by this run."""
    overview = mapping_files_overview(config.aimodel)
    payload["input_file_overview"] = overview
    payload["input_mode_requested"] = "uploaded" if not config.create_default_files else "generated"
    payload["input_mode_used"] = overview.get("source", "unknown")
    return payload


def _diagnostics_from_log(text: str) -> tuple[list[str], list[str]]:
    warnings: dict[str, int] = {}
    errors: dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        bucket = warnings if stripped.startswith("Warning:") else errors if stripped.startswith("Error:") else None
        if bucket is not None:
            bucket[stripped] = bucket.get(stripped, 0) + 1

    def collapse(bucket: dict[str, int]) -> list[str]:
        return [message if count == 1 else f"{message}   [x{count}]" for message, count in list(bucket.items())[:60]]

    return collapse(warnings), collapse(errors)


def _fallback_result(run_id: str) -> dict[str, Any] | None:
    """Rebuild a partial result for runs recorded before gui_run.py existed."""
    log_path = run_dir(run_id) / "stdout.log"
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = _parse_metrics(text)
    if not metrics:
        return None
    warnings, errors = _diagnostics_from_log(text)

    config: dict[str, Any] = {}
    aimodel = None
    try:
        from app.services.store import load_record

        record = load_record(run_id)
        config = record.config.model_dump()
        aimodel = record.config.aimodel
    except Exception:
        pass

    return {
        "status": "completed",
        "partial": True,
        "partial_reason": (
            "This run was executed before the instrumented driver was added, so only the "
            "printed PPTAC summary could be recovered. Re-run the configuration to unlock "
            "the tile inventory, package topology and cost hierarchy."
        ),
        "aimodel": aimodel,
        "config": config,
        "stage_times": {
            "ai_mapping": metrics.get("ai_mapping_sim_time", 0.0),
            "compute": metrics.get("compute_sim_time", 0.0),
            "network": metrics.get("network_sim_time", 0.0),
            "cost": metrics.get("cost_sim_time", 0.0),
            "total": metrics.get("total_sim_time", 0.0),
        },
        "metrics": _derive_metrics(metrics),
        "topology": {"tiles": [], "chiplets": [], "stacks": [], "noc_edges": [], "nop_edges": []},
        "ai_model": {"layers": [], "edges": []},
        "cost": {
            "total_cost_usd": (metrics.get("recurring_cost_usd", 0.0) + metrics.get("nre_cost_usd", 0.0)) or None,
            "cost_per_die_usd": None,
            "breakdown": {},
        },
        "counts": {},
        "warnings": warnings,
        "errors": errors,
    }


def run_result(run_id: str) -> dict[str, Any]:
    payload = _read_result(run_dir(run_id) / "gui_result.json")
    if payload is None:
        payload = _fallback_result(run_id)
    if payload is None:
        raise FileNotFoundError(run_id)
    try:
        record = load_record(run_id)
        payload.setdefault("config", record.config.model_dump())
        if "input_file_overview" not in payload:
            _annotate_result_inputs(payload, record.config)
    except Exception:
        # Older or externally-created result folders may not have a record;
        # the result itself remains useful without the optional input summary.
        pass
    return payload


def backfill_record(record: RunRecord) -> RunRecord:
    """Populate PPTAC metrics for older records that only stored stage timings."""
    if record.status != "completed" or len(record.metrics) > 6:
        return record
    log_path = run_dir(record.id) / "stdout.log"
    if not log_path.exists():
        return record
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = _parse_metrics(text)
    if not metrics:
        return record
    warnings, errors = _diagnostics_from_log(text)
    record.metrics = _derive_metrics(metrics)
    if not record.warnings:
        record.warnings = warnings
    if not record.errors:
        record.errors = errors
    try:
        save_record(record)
    except Exception:
        pass
    return record


def run_logs(run_id: str, stream: str = "stdout", tail: int | None = None) -> str:
    name = "stderr.log" if stream == "stderr" else "stdout.log"
    path = run_dir(run_id) / name
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if tail:
        lines = text.splitlines()
        text = "\n".join(lines[-tail:])
    return text


def run_artifact_tree(run_id: str) -> list[dict[str, Any]]:
    base = run_dir(run_id) / "artifacts"
    if not base.exists():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(base)
        entries.append(
            {
                "path": str(relative).replace(os.sep, "/"),
                "name": path.name,
                "suffix": path.suffix.lower().lstrip("."),
                "size_bytes": path.stat().st_size,
                "kind": "image" if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"} else "text",
            }
        )
    return entries


def artifact_path(run_id: str, relative: str) -> Path:
    base = (run_dir(run_id) / "artifacts").resolve()
    target = (base / relative).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise FileNotFoundError(relative) from None
    if not target.is_file():
        raise FileNotFoundError(relative)
    return target


# --------------------------------------------------------------------------- #
#  metadata for the GUI
# --------------------------------------------------------------------------- #
def available_models() -> list[str]:
    base = HISIM_DIR / "Module_0_AI_Map" / "HISIM_2_0_AI_layer_information"
    if not base.exists():
        return []
    return sorted([path.name for path in base.iterdir() if path.is_dir()])


def model_info() -> list[dict[str, Any]]:
    """Model list with a cheap layer-count so the UI can show workload size."""
    base = HISIM_DIR / "Module_0_AI_Map" / "HISIM_2_0_AI_layer_information"
    infos: list[dict[str, Any]] = []
    for name in available_models():
        folder = base / name
        layers = None
        for candidate in sorted(folder.glob("*.csv")):
            try:
                with candidate.open("r", encoding="utf-8", errors="replace") as handle:
                    layers = max(sum(1 for _ in handle) - 1, 0)
                break
            except Exception:
                continue
        infos.append({"name": name, "layers": layers, "files": len(list(folder.glob("*")))})
    return infos


def default_config_from_file() -> dict[str, Any]:
    config_path = HISIM_DIR / "config.py"
    values: dict[str, Any] = {}
    if not config_path.exists():
        return values
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"main_dir", "os"}:
            continue
        try:
            values[key] = ast.literal_eval(value.split("#", 1)[0].strip())
        except Exception:
            continue
    return values
