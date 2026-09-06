from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import re
from typing import Any

from app.services.paths import GENERATED_FILES_DIR, HISIM_DIR, UPLOADED_FILES_DIR

# The six CSVs consumed by HISIM. The client sends only one of these fixed
# kinds; it never supplies a path or filename.
KIND_PREFIXES = {
    "chip_map": "Chip_Map",
    "sys_map": "Sys_Map",
    "layer_map": "Layer_Mapping",
    "sa_spec": "SA_Spec",
    "mem_spec": "Mem_Spec",
    "network_spec": "Network_Spec",
}

# Header-only files are intentionally offered before the first simulation. They
# make the expected schema discoverable without manufacturing a partial design
# that could be mistaken for a runnable HISIM configuration.
TEMPLATE_HEADERS = {
    "chip_map": "Chiplet ID,Tile ID,HW Type,NoC Position,AI Layer,NodeName",
    "sys_map": "Chiplet ID,Stack ID,Tier ID,NoP Position",
    "layer_map": (
        "Layer ID,Type,Function,in1_dim1,in1_dim2,in1_dim3,in1_dim4,"
        "in2_dim1,in2_dim2,in2_dim3,in2_dim4,Parallel,A,B,C"
    ),
    "sa_spec": "Chiplet ID,Tile ID,HW Type,SA_size_x,SA_size_y,n_SA,prec,Clock Frequency (Hz)",
    "mem_spec": "Chiplet ID,Tile ID,HW Type,Nbank,NW,NB,CM,Clock Frequency (Hz)",
    "network_spec": (
        "Stack ID,N_2D_Links_per_tile,N_3D_Links_per_tile,"
        "N_2.5D_channels_per_chiplet_edge"
    ),
}

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _validate_kind(kind: str) -> str:
    prefix = KIND_PREFIXES.get(kind)
    if prefix is None:
        raise ValueError(f"Unknown input file kind '{kind}' (expected one of {sorted(KIND_PREFIXES)})")
    return prefix


def _validate_model(aimodel: str) -> str:
    model = str(aimodel)
    if not _MODEL_RE.fullmatch(model):
        raise ValueError("Invalid model name")
    return model

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def mapping_file_path(aimodel: str, kind: str) -> Path:
    prefix = _validate_kind(kind)
    model = _validate_model(aimodel)
    if kind in {"sa_spec", "mem_spec"}:
        directory = HISIM_DIR / "Module_1_Compute" / "HISIM_2_0_Files" / "HW_configs"
    elif kind == "network_spec":
        directory = HISIM_DIR / "Module_2_Network" / "HISIM_2_0_Files" / "Network_configs"
    else:
        directory = HISIM_DIR
    return directory / f"{prefix}_{model}.csv"


def uploaded_file_path(aimodel: str, kind: str) -> Path:
    prefix = _validate_kind(kind)
    model = _validate_model(aimodel)
    return UPLOADED_FILES_DIR / model / f"{prefix}_{model}.csv"


def generated_file_path(aimodel: str, kind: str) -> Path:
    """Path for CSVs produced by a generated-model run, never user uploads."""
    prefix = _validate_kind(kind)
    model = _validate_model(aimodel)
    return GENERATED_FILES_DIR / model / f"{prefix}_{model}.csv"


def preferred_mapping_file_path(aimodel: str, kind: str) -> Path:
    uploaded = uploaded_file_path(aimodel, kind)
    if uploaded.exists():
        return uploaded
    generated = generated_file_path(aimodel, kind)
    return generated if generated.exists() else mapping_file_path(aimodel, kind)


def input_file_template(aimodel: str, kind: str) -> tuple[str, str]:
    """Return a safe filename and schema-only CSV for a missing input file."""
    prefix = _validate_kind(kind)
    model = _validate_model(aimodel)
    return f"{prefix}_{model}_template.csv", f"{TEMPLATE_HEADERS[kind]}\n"


def mapping_files_status(aimodel: str) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for kind in KIND_PREFIXES:
        uploaded = uploaded_file_path(aimodel, kind)
        generated = generated_file_path(aimodel, kind)
        legacy = mapping_file_path(aimodel, kind)
        if uploaded.exists():
            path, source = uploaded, "uploaded"
        elif generated.exists():
            path, source = generated, "generated"
        else:
            path, source = legacy, "generated"
        if path.exists():
            stat = path.stat()
            status[kind] = {
                "exists": True,
                "source": source,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            }
        else:
            status[kind] = {
                "exists": False,
                "source": "template",
                "size_bytes": None,
                "modified_at": None,
                "template_available": True,
            }
    return status


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [row for row in csv.DictReader(handle) if row]
    except (OSError, UnicodeError, csv.Error):
        return []


def _unique_text(rows: list[dict[str, str]], key: str) -> list[str]:
    values = {str(row.get(key, "")).strip() for row in rows}
    return sorted(value for value in values if value)


def _numeric_values(rows: list[dict[str, str]], key: str) -> list[int | float]:
    values: set[int | float] = set()
    for row in rows:
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        values.add(int(value) if value.is_integer() else value)
    return sorted(values)


def _sa_sizes(rows: list[dict[str, str]]) -> list[str]:
    sizes: set[str] = set()
    for row in rows:
        try:
            x = float(str(row.get("SA_size_x", "")).strip())
            y = float(str(row.get("SA_size_y", "")).strip())
        except (TypeError, ValueError):
            continue
        sizes.add(f"{x:g}×{y:g}")
    return sorted(sizes)


def mapping_files_overview(aimodel: str) -> dict[str, Any]:
    """Summarize the selected model's actual input CSVs for the UI."""
    model = _validate_model(aimodel)
    selected: dict[str, Path] = {}
    sources: list[str] = []
    for kind in KIND_PREFIXES:
        uploaded = uploaded_file_path(model, kind)
        generated = generated_file_path(model, kind)
        legacy = mapping_file_path(model, kind)
        path = uploaded if uploaded.exists() else generated if generated.exists() else legacy
        selected[kind] = path
        if path.exists():
            sources.append("uploaded" if path == uploaded else "generated")

    if not sources:
        source = "template"
    elif len(set(sources)) == 1:
        source = sources[0]
    else:
        source = "mixed"

    chip_rows = _csv_rows(selected["chip_map"])
    sys_rows = _csv_rows(selected["sys_map"])
    layer_rows = _csv_rows(selected["layer_map"])
    sa_rows = _csv_rows(selected["sa_spec"])
    mem_rows = _csv_rows(selected["mem_spec"])
    network_rows = _csv_rows(selected["network_spec"])

    hardware_types = Counter(
        str(row.get("HW Type", "")).strip()
        for row in chip_rows
        if str(row.get("HW Type", "")).strip()
    )
    sa_sizes = _sa_sizes(sa_rows)
    banks = _numeric_values(mem_rows, "Nbank")

    return {
        "model": model,
        "source": source,
        "files_available": len(sources),
        "files_total": len(KIND_PREFIXES),
        "chiplets": len(_unique_text(chip_rows, "Chiplet ID")),
        "tiles": len(chip_rows),
        "hardware_types": dict(sorted(hardware_types.items())),
        "stacks": len(_unique_text(sys_rows, "Stack ID")),
        "tiers": len(_unique_text(sys_rows, "Tier ID")),
        "ai_layers": len(_unique_text(layer_rows, "Layer ID")),
        "sa": {
            "tiles": len(sa_rows),
            "sizes": sa_sizes,
            "precisions": _numeric_values(sa_rows, "prec"),
            "arrays_per_tile": _numeric_values(sa_rows, "n_SA"),
        },
        "memory": {
            "tiles": len(mem_rows),
            "total_banks": sum(banks),
            "bank_counts": banks,
        },
        "network": {
            "stacks": len(network_rows),
            "noc_2d_links": _numeric_values(network_rows, "N_2D_Links_per_tile"),
            "links_3d": _numeric_values(network_rows, "N_3D_Links_per_tile"),
            "channels_2_5d": _numeric_values(network_rows, "N_2.5D_channels_per_chiplet_edge"),
        },
    }


def write_mapping_file(aimodel: str, kind: str, content: bytes) -> None:
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    if not content.strip():
        raise ValueError("Uploaded file is empty")
    path = uploaded_file_path(aimodel, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
