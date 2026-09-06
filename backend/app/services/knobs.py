"""Declarative description of every HISIM knob exposed by the GUI.

The web UI renders its control panel straight from this payload, so adding a new
knob here (plus the matching field in ``HisimConfig``) is enough to make it
appear in the browser.
"""

from __future__ import annotations

from typing import Any

from app.services.hisim_runner import SWEEPABLE_KNOBS, available_models


TOPOLOGY_OPTIONS = [
    {
        "value": "2D_Mesh",
        "label": "2D Mesh",
        "hint": "Single monolithic die, one tile per layer compute/memory type.",
        "dimension": "2D",
    },
    {
        "value": "3D_Mesh",
        "label": "3D Mesh",
        "hint": "Single stack, multiple vertically bonded tiers connected by 3D links.",
        "dimension": "3D",
    },
    {
        "value": "2_5D_Mesh",
        "label": "2.5D Mesh",
        "hint": "3-chiplet package: DDR memory chiplet, SA+Mem chiplet, CPU+Mem chiplet.",
        "dimension": "2.5D",
    },
    {
        "value": "3_5D_Mesh",
        "label": "3.5D Mesh",
        "hint": "2.5D package where each chiplet is a 2-tier 3D stack.",
        "dimension": "3.5D",
    },
    {
        "value": "2_5D_Mesh_Scaled",
        "label": "2.5D Mesh (scaled)",
        "hint": "One component (CPU / Mem / SA) per chiplet, one chiplet per layer type.",
        "dimension": "2.5D",
    },
    {
        "value": "3_5D_Mesh_Scaled",
        "label": "3.5D Mesh (scaled)",
        "hint": "One stack per AI layer, 3 tiers: compute, output memory, weight memory.",
        "dimension": "3.5D",
    },
]


def knob_schema() -> dict[str, Any]:
    models = available_models()
    return {
        "sweepable": sorted(SWEEPABLE_KNOBS.keys()),
        "groups": [
            {
                "id": "workload",
                "label": "Workload",
                "icon": "brain",
                "description": "AI model whose layer graph is mapped onto the hardware.",
                "fields": [
                    {
                        "id": "aimodel",
                        "label": "AI model",
                        "type": "select",
                        "config_name": "aimodel",
                        "options": [{"value": m, "label": m} for m in models],
                        "help": "Layer graph loaded from Module_0_AI_Map/HISIM_2_0_AI_layer_information.",
                    },
                ],
            },
            {
                "id": "integration",
                "label": "Integration & package",
                "icon": "layers",
                "description": "How the design is partitioned across dies, stacks and tiers.",
                "fields": [
                    {
                        "id": "type_default_files",
                        "label": "Topology template",
                        "type": "select",
                        "config_name": "TYPE_DEFAULT_FILES",
                        "options": TOPOLOGY_OPTIONS,
                        "help": "Template used when HISIM auto-generates Chip_Map / Sys_Map / Layer_Mapping.",
                    },
                    {
                        "id": "default_files_generic",
                        "label": "Generic auto-mapping",
                        "type": "bool",
                        "config_name": "DEFAULT_FILES_GENERIC",
                        "help": "ON: tile count derived from the AI model. OFF: use the explicit stack/chip/tile counts below.",
                    },
                    {
                        "id": "create_default_files",
                        "label": "Regenerate mapping files",
                        "type": "bool",
                        "config_name": "CREATE_DEFAULT_FILES",
                        "help": "OFF re-uses the Chip_Map / Sys_Map CSVs already on disk (hand-edited heterogeneous designs).",
                    },
                    {
                        "id": "set_suff_banks",
                        "label": "Size banks for full on-package memory",
                        "type": "bool",
                        "config_name": "SET_SUFF_BANKS",
                        "help": "Auto-scales SRAM banks so weights/activations fit on package, minimising DDR traffic.",
                    },
                    {
                        "id": "stack_count",
                        "label": "Stacks (chiplets in package)",
                        "type": "int",
                        "config_name": "stack_count",
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "presets": [1, 2, 3, 4, 8, 16],
                        "help": ">1 gives a 2.5D / 3.5D package. =1 is monolithic 2D / 3D. Ignored while generic auto-mapping is ON.",
                        "disabled_when": {"field": "default_files_generic", "equals": True},
                    },
                    {
                        "id": "chip_count",
                        "label": "Tiers per stack",
                        "type": "int",
                        "config_name": "chip_count",
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "presets": [1, 2, 3, 4],
                        "help": ">1 adds vertically bonded dies connected by 3D links. Ignored while generic auto-mapping is ON.",
                        "disabled_when": {"field": "default_files_generic", "equals": True},
                    },
                    {
                        "id": "tile_count_dict",
                        "label": "Tiles per chiplet",
                        "type": "dict_int",
                        "config_name": "tile_count_dict",
                        "keys": [
                            {"key": "SA", "label": "Systolic array"},
                            {"key": "CPU", "label": "CPU (non-linear)"},
                            {"key": "Mem_I", "label": "Input mem"},
                            {"key": "Mem_W", "label": "Weight mem"},
                            {"key": "Mem_O", "label": "Output mem"},
                        ],
                        "min": 1,
                        "max": 64,
                        "help": "Only used when generic auto-mapping is OFF.",
                        "disabled_when": {"field": "default_files_generic", "equals": True},
                    },
                ],
            },
            {
                "id": "compute",
                "label": "Compute · systolic array",
                "icon": "cpu",
                "description": "Per-tile systolic array geometry driving Module_1_Compute.",
                "fields": [
                    {
                        "id": "sa_size_x",
                        "label": "SA rows (X)",
                        "type": "int",
                        "config_name": "def_SA_size_x",
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "presets": [8, 16, 32, 64, 128, 256],
                    },
                    {
                        "id": "sa_size_y",
                        "label": "SA columns (Y)",
                        "type": "int",
                        "config_name": "def_SA_size_y",
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "presets": [8, 16, 32, 64, 128, 256],
                    },
                    {
                        "id": "n_sa",
                        "label": "Arrays per tile",
                        "type": "int",
                        "config_name": "def_n_SA",
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "presets": [1, 2, 4, 8, 16],
                    },
                    {
                        "id": "precision_bits",
                        "label": "Datapath precision",
                        "type": "select_int",
                        "config_name": "def_prec",
                        "options": [
                            {"value": 4, "label": "INT4"},
                            {"value": 8, "label": "INT8"},
                            {"value": 16, "label": "FP16 / INT16"},
                            {"value": 32, "label": "FP32"},
                        ],
                    },
                    {
                        "id": "clock_hz",
                        "label": "Clock frequency",
                        "type": "select_float",
                        "config_name": "def_clk_hz",
                        "unit": "Hz",
                        "options": [
                            {"value": 2.5e8, "label": "250 MHz"},
                            {"value": 5e8, "label": "500 MHz"},
                            {"value": 1e9, "label": "1.0 GHz"},
                            {"value": 1.5e9, "label": "1.5 GHz"},
                            {"value": 2e9, "label": "2.0 GHz"},
                        ],
                    },
                ],
            },
            {
                "id": "memory",
                "label": "Memory & DDR",
                "icon": "database",
                "description": "SRAM tile organisation used by Module_1_Compute/Mem.py.",
                "fields": [
                    {
                        "id": "n_bank",
                        "label": "Banks per memory tile",
                        "type": "int",
                        "config_name": "def_Nbank",
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "presets": [1, 5, 9, 13, 17],
                        "help": "Ignored when 'size banks for full on-package memory' is ON.",
                    },
                    {
                        "id": "n_word",
                        "label": "Rows per bank (NW)",
                        "type": "int",
                        "config_name": "def_NW",
                        "min": 32,
                        "max": 16384,
                        "step": 32,
                        "presets": [128, 256, 512, 1024],
                        "help": "Rows per bank; the engine snaps this to the nearest calibrated Mem_LUT row.",
                    },
                    {
                        "id": "n_bit",
                        "label": "Columns per bank (NB)",
                        "type": "int",
                        "config_name": "def_NB",
                        "min": 32,
                        "max": 8192,
                        "step": 32,
                        "presets": [64, 160, 240, 320],
                        "help": "Columns per bank; these presets are calibrated Mem_LUT values.",
                    },
                    {
                        "id": "col_mux",
                        "label": "Column mux (CM)",
                        "type": "select_int",
                        "config_name": "def_CM",
                        "options": [{"value": 4, "label": "4 (calibrated)"}],
                        "help": "The bundled memory LUT is calibrated for CM=4.",
                    },
                ],
            },
            {
                "id": "network",
                "label": "Interconnect · NoC / NoP / 3D",
                "icon": "network",
                "description": "Link budgets consumed by Module_2_Network.",
                "fields": [
                    {
                        "id": "n_2d_links_per_tile",
                        "label": "2D NoC links per tile",
                        "type": "int",
                        "config_name": "def_n_2d_links_per_tile",
                        "min": 4,
                        "max": 1024,
                        "step": 4,
                        "presets": [20, 40, 60, 80, 160],
                        "help": "Wire count between neighbouring tiles; drives NoC bandwidth and router area.",
                    },
                    {
                        "id": "n_3d_links_per_tile",
                        "label": "3D links per tile",
                        "type": "int",
                        "config_name": "def_n_3d_links_per_tile",
                        "min": 4,
                        "max": 1024,
                        "step": 4,
                        "presets": [20, 40, 80, 160],
                        "help": "Vertical (TSV/hybrid-bond) links; only active when tiers > 1.",
                    },
                    {
                        "id": "n_2_5d_channels_per_chiplet_edge",
                        "label": "2.5D channels per chiplet edge",
                        "type": "int",
                        "config_name": "def_n_2_5d_channels_per_chiplet_edge",
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "presets": [1, 2, 4, 8],
                        "help": "AIB die-to-die channels on each chiplet edge; only active when stacks > 1.",
                    },
                ],
            },
        ],
    }
