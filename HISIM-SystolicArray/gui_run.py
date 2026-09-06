"""Instrumented HISIM driver used by the HISIM GUI backend.

Mirrors HISIM.py, but in addition to the normal terminal output it captures a
machine-readable summary of the whole run into ``gui_result.json``:

    * stage-by-stage simulation timings
    * the printed PPTAC summary (compute / memory / DDR / NoC / NoP / 3D / cost)
    * per-tile records from the chip graph (area, latency, energy, router, links)
    * chiplet / stack / tier topology so the GUI can draw the 2.5D-3.5D package
    * the full hierarchical cost + yield breakdown
    * every Warning/Error line emitted by the analytical models

Run exactly like HISIM.py::

    python gui_run.py                       # writes ./gui_result.json
    HISIM_GUI_RESULT=/tmp/out.json python gui_run.py

The file is intentionally standalone so that ``python HISIM.py`` keeps working
untouched for command-line users.
"""

from __future__ import annotations

import io
import json
import math
import os
import pickle
import re
import sys
import time

import config

from Module_0_AI_Map.util_chip.HISIM_2_0_Files.HW_Map import load_ai_chip, load_ai_network
from Module_1_Compute.HISIM_2_0_Files.Compute import compute_main_fn
from Module_2_Network.HISIM_2_0_Files.Network import network_main_fn
from Module_3_Cost.Cost import cost_main_fn
from Module_5_ONNX.parser_filter import parse


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
class _Tee(io.TextIOBase):
    """Write to the real stdout (so the GUI can stream logs) and to a buffer."""

    def __init__(self, stream):
        self._stream = stream
        self.buffer_text = io.StringIO()

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self.buffer_text.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()


def _jsonable(value, _depth=0):
    """Best-effort conversion of arbitrary python/numpy/pandas values to JSON."""
    if _depth > 6:
        return str(value)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    # numpy scalars / arrays
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "shape", None) == ():
        try:
            return _jsonable(item(), _depth + 1)
        except Exception:
            return str(value)
    tolist = getattr(value, "tolist", None)
    if tolist is not None and not isinstance(value, (list, tuple, dict, set)):
        try:
            return _jsonable(tolist(), _depth + 1)
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, _depth + 1) for v in value]
    return str(value)


def _df_records(df, limit=20000):
    try:
        import pandas as pd  # noqa: F401

        records = df.head(limit).to_dict(orient="records")
        return [_jsonable(row) for row in records]
    except Exception:
        return []


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


# --------------------------------------------------------------------------- #
#  stdout summary parsing
# --------------------------------------------------------------------------- #
_NUM = r"([-+0-9.eE]+(?:[eE][-+]?\d+)?)"

SUMMARY_PATTERNS = {
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
    # combined
    "total_area_mm2": r"Total Chip Area \(in mm\^2\)\s*" + _NUM,
    "total_latency_s": r"Total End-to-End Latency \(in s\)\s*" + _NUM,
    "total_energy_j": r"Total Chip Energy \(in J\)\s*" + _NUM,
    # cost
    "manufacturing_volume": r"Total Recurring Cost for\s+" + _NUM + r"\s+units",
    "recurring_cost_usd": r"Total Recurring Cost for\s+[-+0-9.eE]+\s+units \(\$\):\s*" + _NUM,
    "nre_cost_usd": r"Total NRE Cost \(\$\):\s*" + _NUM,
    "cost_per_part_usd": r"Total Cost per Part \(\$\):\s*" + _NUM,
    # timings
    "ai_mapping_sim_time": r"AI mapping sim time is:\s*" + _NUM,
    "compute_sim_time": r"Computing model sim time is:\s*" + _NUM,
    "network_sim_time": r"Network model sim time is:\s*" + _NUM,
    "cost_sim_time": r"Cost model sim time is:\s*" + _NUM,
    "total_sim_time": r"Total sim time is:\s*" + _NUM,
}


def parse_summary(text):
    metrics = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            metrics[key] = float(match.group(1))
        except ValueError:
            continue
    return metrics


def collect_diagnostics(text):
    warnings, errors = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Warning:"):
            warnings.append(stripped)
        elif stripped.startswith("Error:"):
            errors.append(stripped)
    return warnings, errors


def _dedupe(messages, limit=60):
    counted, order = {}, []
    for message in messages:
        if message not in counted:
            counted[message] = 0
            order.append(message)
        counted[message] += 1
    out = []
    for message in order[:limit]:
        count = counted[message]
        out.append(message if count == 1 else f"{message}   [x{count}]")
    return out


# --------------------------------------------------------------------------- #
#  file housekeeping (identical semantics to HISIM.py)
# --------------------------------------------------------------------------- #
def del_files_folder(folder_path, exts):
    if not os.path.exists(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                if exts == "all" or any(file_path.endswith(ext) for ext in exts):
                    os.remove(file_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to delete {file_path}. Reason: {exc}")


def housekeeping(main_dir, aimodel, create_dfiles):
    results_dir = os.path.join(main_dir, "Results")
    del_files_folder(results_dir, exts="all")
    del_files_folder(os.path.join(results_dir, "Chip_Map"), exts=[".png"])
    del_files_folder(os.path.join(results_dir, "Network_Map"), exts=[".png"])
    del_files_folder(
        os.path.join(main_dir, "Module_2_Network", "HISIM_2_0_Files", "Network_configs"),
        exts=[".png", ".txt"],
    )
    del_files_folder(
        os.path.join(main_dir, "Module_1_Compute", "HISIM_2_0_Files", "HW_configs"),
        exts=[".png", ".txt"],
    )
    del_files_folder(
        os.path.join(main_dir, "Module_0_AI_Map", "HISIM_2_0_AI_layer_information", f"{aimodel}"),
        exts=[".png", ".txt"],
    )
    if create_dfiles:
        for filename in os.listdir(main_dir):
            if filename.startswith(("Chip_Map_", "Sys_Map_", "Layer_Mapping_")):
                try:
                    os.remove(os.path.join(main_dir, filename))
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"Failed to delete {filename}. Reason: {exc}")


# --------------------------------------------------------------------------- #
#  topology extraction
# --------------------------------------------------------------------------- #
def classify_hw(hw_type):
    """Collapse per-layer HW Type strings ('L4 Wg Mem') into a small set of classes."""
    text = str(hw_type or "")
    tokens = set(re.split(r"[^A-Za-z0-9.]+", text.upper()))
    if "DDR" in tokens:
        return "DDR"
    if "EMPTY" in tokens or not text.strip():
        return "Empty"
    if "SA" in tokens:
        return "SA"
    if "CPU" in tokens:
        return "CPU"
    if "WG" in tokens or "W" in tokens:
        return "Weight mem"
    if "OUT" in tokens or "O" in tokens:
        return "Output mem"
    if "IN" in tokens or "INPUT" in tokens or "I" in tokens:
        return "Input mem"
    if "MEM" in tokens:
        return "Memory"
    return "Other"


def _split_pos(raw, length):
    try:
        parts = [int(p) for p in str(raw).split(",")]
    except Exception:
        parts = []
    while len(parts) < length:
        parts.append(0)
    return parts[:length]


def build_topology(G_chip, G_sys, G_stack, mesh_size, mesh_size_nop):
    # The mapper represents unused coordinates as synthetic ``Empty`` nodes
    # (for example ``1_x-1_y-1``).  They are useful internally for graph
    # construction, but are not physical tiles and must never leak into the
    # user-facing topology, counts, or links.
    visible_tile_nodes = {
        node_id
        for node_id, attrs in G_chip.nodes(data=True)
        if classify_hw(attrs.get("HW Type")) != "Empty"
    }
    visible_chiplet_ids = {
        str(G_chip.nodes[node_id].get("Chiplet ID"))
        for node_id in visible_tile_nodes
        if G_chip.nodes[node_id].get("Chiplet ID") is not None
    }
    visible_stack_ids = {
        str(G_chip.nodes[node_id].get("Stack ID"))
        for node_id in visible_tile_nodes
        if G_chip.nodes[node_id].get("Stack ID") is not None
    }

    tiles = []
    for node_id, attrs in G_chip.nodes(data=True):
        if node_id not in visible_tile_nodes:
            continue
        noc_x, noc_y = _split_pos(attrs.get("NoC Position", "0,0"), 2)
        nop_x, nop_y, nop_z = _split_pos(attrs.get("NoP Position", "0,0,0"), 3)
        tiles.append(
            {
                "id": node_id,
                "chiplet": attrs.get("Chiplet ID"),
                "tile": attrs.get("Tile ID"),
                "hw_type": attrs.get("HW Type"),
                "hw_class": classify_hw(attrs.get("HW Type")),
                "stack": attrs.get("Stack ID"),
                "ai_layer": _jsonable(attrs.get("AI Layer")),
                "node_name": _jsonable(attrs.get("NodeName")),
                "noc": [noc_x, noc_y],
                "nop": [nop_x, nop_y, nop_z],
                "area_mm2": _jsonable(attrs.get("Tile_Area")),
                "latency_s": _jsonable(attrs.get("Tile_Latency")),
                "energy_j": _jsonable(attrs.get("Tile_Energy")),
                "router_area_mm2": _jsonable(attrs.get("router Area")),
                "router_latency_s": _jsonable(attrs.get("router Latency")),
                "router_energy_j": _jsonable(attrs.get("router Energy")),
                "router_power_w": _jsonable(attrs.get("router Power")),
                "link3d_latency_s": _jsonable(attrs.get("3D link Latency")),
                "link3d_energy_j": _jsonable(attrs.get("3D link Energy")),
                "link3d_power_w": _jsonable(attrs.get("3D link Power")),
                "ddr_latency_s": _jsonable(attrs.get("DDR_Latency")),
                "ddr_energy_j": _jsonable(attrs.get("DDR_Energy")),
            }
        )

    chiplets = []
    for node_id, attrs in G_sys.nodes(data=True):
        if visible_chiplet_ids and str(node_id) not in visible_chiplet_ids:
            continue
        nop_x, nop_y, nop_z = _split_pos(attrs.get("NoP Position", "0,0,0"), 3)
        chiplets.append(
            {
                "id": node_id,
                "stack": attrs.get("Stack ID"),
                "tier": attrs.get("Tier ID"),
                "nop": [nop_x, nop_y, nop_z],
                "nop_router_area_mm2": _jsonable(attrs.get("NoP Router Area")),
                "mesh_size": _jsonable(mesh_size.get(node_id) if isinstance(mesh_size, dict) else None),
            }
        )

    stacks = {}
    for node_id, attrs in G_stack.nodes(data=True):
        stack_id = str(node_id).rsplit("_", 1)[0]
        if visible_stack_ids and stack_id not in visible_stack_ids:
            continue
        entry = stacks.setdefault(
            stack_id,
            {
                "id": stack_id,
                "nop": _split_pos(attrs.get("NoP Position", "0,0,0"), 3),
                "link_area_mm2": 0.0,
                "edges": [],
            },
        )
        area = attrs.get("2.5d link Area")
        if isinstance(area, (int, float)):
            entry["link_area_mm2"] += float(area)
        entry["edges"].append(str(attrs.get("2.5d link position")))

    noc_edges = []
    for source, dest, attrs in G_chip.edges(data=True):
        if source not in visible_tile_nodes or dest not in visible_tile_nodes:
            continue
        noc_edges.append({"source": source, "target": dest, **_jsonable(dict(attrs))})

    nop_edges = []
    for source, dest, attrs in G_sys.edges(data=True):
        if visible_chiplet_ids and (str(source) not in visible_chiplet_ids or str(dest) not in visible_chiplet_ids):
            continue
        nop_edges.append({"source": source, "target": dest, **_jsonable(dict(attrs))})

    return {
        "tiles": tiles,
        "chiplets": chiplets,
        "stacks": list(stacks.values()),
        "noc_edges": noc_edges[:5000],
        "nop_edges": nop_edges[:5000],
        "mesh_size": _jsonable(mesh_size),
        "mesh_size_nop": _jsonable(mesh_size_nop),
    }


def build_ai_model(G_ai_model):
    layers = []
    for node_id, attrs in G_ai_model.nodes(data=True):
        layers.append({"id": str(node_id), **_jsonable(dict(attrs))})
    edges = [{"source": str(s), "target": str(t)} for s, t in G_ai_model.edges()]
    return {"layers": layers, "edges": edges}


def read_input_specs(main_dir, aimodel):
    """Return the generated/user-supplied input spec tables as JSON records."""
    import pandas as pd

    spec_files = {
        "chip_map": os.path.join(main_dir, f"Chip_Map_{aimodel}.csv"),
        "sys_map": os.path.join(main_dir, f"Sys_Map_{aimodel}.csv"),
        "layer_mapping": os.path.join(main_dir, f"Layer_Mapping_{aimodel}.csv"),
        "sa_spec": os.path.join(
            main_dir, "Module_1_Compute", "HISIM_2_0_Files", "HW_configs", f"SA_Spec_{aimodel}.csv"
        ),
        "mem_spec": os.path.join(
            main_dir, "Module_1_Compute", "HISIM_2_0_Files", "HW_configs", f"Mem_Spec_{aimodel}.csv"
        ),
        "network_spec": os.path.join(
            main_dir,
            "Module_2_Network",
            "HISIM_2_0_Files",
            "Network_configs",
            f"Network_Spec_{aimodel}.csv",
        ),
    }
    specs = {}
    for key, path in spec_files.items():
        if not os.path.exists(path):
            continue
        specs[key] = _safe(lambda p=path: _df_records(pd.read_csv(p)), [])
    return specs


def snapshot_config():
    keys = [
        "DEBUG",
        "CREATE_DEFAULT_FILES",
        "DEFAULT_FILES_GENERIC",
        "TYPE_DEFAULT_FILES",
        "SET_SUFF_BANKS",
        "stack_count",
        "chip_count",
        "tile_count_dict",
        "aimodel",
        "parse_mlir_output",
        "def_Nbank",
        "def_NW",
        "def_NB",
        "def_CM",
        "def_clk_hz",
        "def_SA_size_x",
        "def_SA_size_y",
        "def_n_SA",
        "def_prec",
        "def_n_2d_links_per_tile",
        "def_n_3d_links_per_tile",
        "def_n_2_5d_channels_per_chiplet_edge",
    ]
    return {key: _jsonable(getattr(config, key)) for key in keys if hasattr(config, key)}


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    aimodel = config.aimodel
    main_dir = config.main_dir
    result_path = os.environ.get("HISIM_GUI_RESULT", os.path.join(main_dir, "gui_result.json"))

    tee = _Tee(sys.stdout)
    real_stdout = sys.stdout
    sys.stdout = tee

    result = {
        "status": "failed",
        "config": snapshot_config(),
        "stage_times": {},
        "metrics": {},
        "warnings": [],
        "errors": [],
    }

    try:
        housekeeping(main_dir, aimodel, getattr(config, "CREATE_DEFAULT_FILES", False))

        if getattr(config, "parse_mlir_output", False):
            parse(aimodel)

        t0 = time.time()
        G_ai_model = load_ai_network(aimodel)

        # Keep the GUI driver aligned with HISIM.py: uploaded map/spec CSVs
        # are staged into the legacy consumer paths before they are loaded.
        if getattr(config, "USE_USER_FILES", True):
            from validate_maps import export_user_files, import_user_files

            user_files_dir = getattr(config, "USER_FILES_DIR", None)
            if getattr(config, "CREATE_DEFAULT_FILES", False):
                export_user_files(aimodel, main_dir, user_files_dir)
            else:
                import_user_files(aimodel, main_dir, user_files_dir)

        if getattr(config, "VALIDATE_MAPS", True):
            from validate_maps import validate_or_exit

            validate_or_exit(
                aimodel,
                main_dir,
                strict=getattr(config, "VALIDATE_STRICT", False),
                verbose=getattr(config, "VALIDATE_VERBOSE", False),
            )

        (
            G_sys,
            G_chip,
            G_stack,
            noc_tile_dict,
            nop_chip_dict,
            tier_ids,
            stack_ids,
            tile_ids,
            mesh_size,
            mesh_size_nop,
        ) = load_ai_chip(
            os.path.join(main_dir, f"Chip_Map_{aimodel}.csv"),
            os.path.join(main_dir, f"Sys_Map_{aimodel}.csv"),
        )
        t_map = time.time()
        print("AI mapping sim time is:", (t_map - t0), "s")

        G_chip, tile_map, mem_req, ip_list = compute_main_fn(G_ai_model, G_chip, tile_ids)
        t_compute = time.time()
        print("Computing model sim time is:", (t_compute - t_map), "s")

        G_chip, G_sys, G_stack, nw_df, stack_area, n_signal_IO = network_main_fn(
            G_ai_model,
            G_chip,
            G_sys,
            G_stack,
            noc_tile_dict,
            nop_chip_dict,
            tile_map,
            mem_req,
            mesh_size,
            mesh_size_nop,
            stack_ids,
        )
        t_network = time.time()
        print("Network model sim time is:", (t_network - t_compute), "s")

        outs = (G_chip, G_sys, nw_df, stack_area, stack_ids, mesh_size, ip_list, n_signal_IO)
        with open(os.path.join(main_dir, "input_params_to_cost.pkl"), "wb") as handle:
            pickle.dump(outs, handle)

        total_cost, cost_breakdown, cost_per_die = cost_main_fn(
            G_chip, G_sys, nw_df, stack_area, stack_ids, mesh_size, ip_list, n_signal_IO
        )
        t_cost = time.time()
        print("Cost model sim time is:", (t_cost - t_network), "s")
        print("Total sim time is:", (t_cost - t0), "s")

        captured = tee.buffer_text.getvalue()
        warnings, errors = collect_diagnostics(captured)

        topology = _safe(
            lambda: build_topology(G_chip, G_sys, G_stack, mesh_size, mesh_size_nop), {}
        )
        result.update(
            {
                "status": "completed",
                "aimodel": aimodel,
                "stage_times": {
                    "ai_mapping": t_map - t0,
                    "compute": t_compute - t_map,
                    "network": t_network - t_compute,
                    "cost": t_cost - t_network,
                    "total": t_cost - t0,
                },
                "metrics": parse_summary(captured),
                "topology": topology,
                "ai_model": _safe(lambda: build_ai_model(G_ai_model), {}),
                "input_specs": _safe(lambda: read_input_specs(main_dir, aimodel), {}),
                "cost": {
                    "total_cost_usd": _jsonable(total_cost),
                    "cost_per_die_usd": _jsonable(cost_per_die),
                    "breakdown": _jsonable(cost_breakdown),
                },
                "network_table": _safe(lambda: _df_records(nw_df), []),
                "stack_area_mm2": _jsonable(stack_area),
                "n_signal_io": _jsonable(n_signal_IO),
                "warnings": _dedupe(warnings),
                "errors": _dedupe(errors),
                "counts": {
                    "tiles": len(topology.get("tiles", [])),
                    "chiplets": len(topology.get("chiplets", [])),
                    "stacks": len(topology.get("stacks", [])),
                    "tiers": len({chiplet.get("tier") for chiplet in topology.get("chiplets", [])}),
                    "ai_layers": G_ai_model.number_of_nodes(),
                },
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced to the GUI
        import traceback

        captured = tee.buffer_text.getvalue()
        warnings, errors = collect_diagnostics(captured)
        result["warnings"] = _dedupe(warnings)
        result["errors"] = _dedupe(errors + [f"Error: {exc}"])
        result["traceback"] = traceback.format_exc()
        result["metrics"] = parse_summary(captured)
        sys.stdout = real_stdout
        print(result["traceback"], file=sys.stderr)
        _write(result_path, result)
        raise
    finally:
        sys.stdout = real_stdout

    _write(result_path, result)


def _write(path, payload):
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to write GUI result file {path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
