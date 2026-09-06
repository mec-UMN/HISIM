from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.paths import POSTPROCESSING_DIR


PLOT_SOURCES: dict[str, dict[str, Any]] = {
    "chiplet-cost": {
        "title": "Cost per part vs. chiplet count",
        "unit": "Cost per part ($)",
        "axis": "Number of chiplets (stacks)",
        "sweep": "chiplet",
        "log": True,
        "path": POSTPROCESSING_DIR / "chiplet_dse" / "output_summary_cost.csv",
    },
    "chiplet-area": {
        "title": "Total chip area vs. chiplet count",
        "unit": "Chip area (mm²)",
        "axis": "Number of chiplets (stacks)",
        "sweep": "chiplet",
        "log": True,
        "path": POSTPROCESSING_DIR / "chiplet_dse" / "output_summary_chip_area.csv",
    },
    "chiplet-sim-time": {
        "title": "Simulator runtime vs. chiplet count",
        "unit": "Simulation time (s)",
        "axis": "Number of chiplets (stacks)",
        "sweep": "chiplet",
        "log": False,
        "path": POSTPROCESSING_DIR / "chiplet_dse" / "output_summary_sim_time.csv",
    },
    "memory-latency": {
        "title": "On-chip memory latency vs. banks per tile",
        "unit": "Memory latency (s)",
        "axis": "Banks per memory tile",
        "sweep": "memory",
        "log": True,
        "path": POSTPROCESSING_DIR / "mem_dse" / "output_summary_mem_latency.csv",
    },
    "ddr-latency": {
        "title": "DDR latency vs. banks per tile",
        "unit": "DDR latency (s)",
        "axis": "Banks per memory tile",
        "sweep": "memory",
        "log": True,
        "path": POSTPROCESSING_DIR / "mem_dse" / "output_summary_ddr_latency.csv",
    },
    "network-latency": {
        "title": "Network latency vs. 2D links per tile",
        "unit": "Network latency (s)",
        "axis": "2D NoC links per tile",
        "sweep": "network",
        "log": True,
        "path": POSTPROCESSING_DIR / "network_dse" / "output_summary_network_latency.csv",
    },
    "chiplet-yield": {
        "title": "Die yield vs. chiplet count",
        "unit": "Die yield",
        "axis": "Number of chiplets (stacks)",
        "sweep": "chiplet",
        "log": False,
        "derived_from": "chiplet-area",
    },
}

# Illustrative negative-binomial yield model (paper Eq. 30), applied to the
# swept Total Chip Area series so the DSE tab can show the same "small dies
# yield better" story as Fig. 15(b) without a dedicated yield sweep dataset.
YIELD_DEFECT_DENSITY_PER_CM2 = 0.12
YIELD_CLUSTERING_ALPHA = 2.0
YIELD_CRITICAL_AREA_RATIO = 1.0


def _negative_binomial_yield(area_mm2: float | None) -> float | None:
    if area_mm2 is None or area_mm2 <= 0:
        return None
    area_cm2 = area_mm2 / 100.0
    defects = YIELD_DEFECT_DENSITY_PER_CM2 * area_cm2 * YIELD_CRITICAL_AREA_RATIO
    return (1 + defects / YIELD_CLUSTERING_ALPHA) ** (-YIELD_CLUSTERING_ALPHA)


def _source_ready(source: dict[str, Any]) -> bool:
    derived_from = source.get("derived_from")
    if derived_from:
        base = PLOT_SOURCES.get(derived_from)
        return bool(base and base.get("path") and base["path"].exists())
    path = source.get("path")
    return bool(path and path.exists())


def available_plots() -> list[dict[str, Any]]:
    plots = []
    for key, source in PLOT_SOURCES.items():
        if _source_ready(source):
            plots.append(
                {
                    "id": key,
                    "title": source["title"],
                    "unit": source["unit"],
                    "axis": source["axis"],
                    "sweep": source["sweep"],
                    "log": source["log"],
                }
            )
    return plots


# Column headers carry more than one number (e.g. "Stack Count: 16 Chip Count: 1"),
# so both the sort order and the short display label must key off the same
# specific swept parameter — never just "the last number in the string", which
# picks up an unrelated, constant field and produces a scrambled axis order.
_SWEPT_VALUE_PATTERNS = (r"Nbank:\s*(\d+)", r"2DLinkspertile:\s*(\d+)", r"Stack Count:\s*(\d+)")


def _swept_value(label: str) -> str | None:
    for pattern in _SWEPT_VALUE_PATTERNS:
        match = re.search(pattern, label)
        if match:
            return match.group(1)
    return None


def _sort_key(label: str) -> tuple[float, str]:
    swept = _swept_value(label)
    if swept is not None:
        return (float(swept), label)
    numbers = re.findall(r"\d+\.?\d*", label)
    return (float(numbers[-1]) if numbers else 0.0, label)


def _short_label(label: str) -> str:
    """Turn 'Type: User-Defined Stack Count: 16 Chip Count: 1 Nbank: 5' into '5'."""
    return _swept_value(label) or label


def _clean(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def plot_payload(plot_id: str) -> dict[str, Any]:
    if plot_id not in PLOT_SOURCES:
        raise KeyError(plot_id)
    source = PLOT_SOURCES[plot_id]

    derived_from = source.get("derived_from")
    if derived_from:
        base = plot_payload(derived_from)
        # Yield is a per-die property, but the source series is *total* package
        # area summed across every chiplet. Divide by the swept chiplet count
        # (the category itself, e.g. "16") to recover an average per-die area
        # before applying the yield model — otherwise splitting into more,
        # smaller dies would look like it *hurts* yield instead of helping it.
        chiplet_counts = [_clean(category) or 1.0 for category in base["categories"]]
        series = [
            {
                "model": entry["model"],
                "values": [
                    _negative_binomial_yield(area / count if area is not None else None)
                    for area, count in zip(entry["values"], chiplet_counts)
                ],
            }
            for entry in base["series"]
        ]
        return {
            "id": plot_id,
            "title": source["title"],
            "unit": source["unit"],
            "axis": source["axis"],
            "sweep": source["sweep"],
            "log": source["log"],
            "source": f"derived:{derived_from}",
            "categories": base["categories"],
            "raw_categories": base["raw_categories"],
            "series": series,
        }

    path: Path = source["path"]
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)
    if "Model" not in df.columns:
        raise ValueError(f"{path} does not contain a Model column")

    value_columns = sorted([col for col in df.columns if col != "Model"], key=_sort_key)
    categories = [_short_label(col) for col in value_columns]

    series = []
    for _, row in df.iterrows():
        series.append(
            {
                "model": str(row["Model"]).lower(),
                "values": [_clean(row[col]) for col in value_columns],
            }
        )

    return {
        "id": plot_id,
        "title": source["title"],
        "unit": source["unit"],
        "axis": source["axis"],
        "sweep": source["sweep"],
        "log": source["log"],
        "source": str(path),
        "categories": categories,
        "raw_categories": value_columns,
        "series": series,
    }
