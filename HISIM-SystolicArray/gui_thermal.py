"""Convert a hisim_thermal result directory into compact JSON for the HISIM GUI."""
from __future__ import annotations

import json
import math
import os

import numpy as np

# The solve time and memory grow with the number of unknowns: ~150k unknowns
# take about 10 s, ~14M (a 4x4 qwen design) did not finish in 30 minutes.
MAX_UNKNOWNS = 400_000       # solve at the production resolution up to here, coarsen above it
GIVE_UP_UNKNOWNS = 1_500_000  # still larger than this after coarsening: skip thermal


def export_maps(thermal_dir, t_min=None, t_avg=None, t_max=None):
    T = np.load(os.path.join(thermal_dir, "T_full.npy"))
    with open(os.path.join(thermal_dir, "layers.json"), encoding="utf-8") as handle:
        layers = json.load(handle)
    edges = np.load(os.path.join(thermal_dir, "grid.npz"))

    ks_by_die = {}
    for sub in layers["sublayers"]:
        if sub.get("role") == "active" and sub.get("die") is not None:
            ks_by_die.setdefault(int(sub["die"]), []).append(int(sub["k"]))
    if not ks_by_die:
        raise ValueError("thermal result has no active layers")

    dies, grids = [], {}
    for die in sorted(ks_by_die):
        plane = T[ks_by_die[die]].mean(axis=0)
        dies.append({"die": die, "peak_k": float(plane.max()), "min_k": float(plane.min()),
                     "avg_k": float(plane.mean())})
        grids[str(die)] = {"grid": np.round(plane, 3).tolist()}

    maps = {
        "x_edges_mm": (np.asarray(edges["x_edges"], float) * 1e3).tolist(),
        "y_edges_mm": (np.asarray(edges["y_edges"], float) * 1e3).tolist(),
        "dies": grids,
    }
    with open(os.path.join(thermal_dir, "maps.json"), "w", encoding="utf-8") as handle:
        json.dump(maps, handle)

    ambient = None
    meta_path = os.path.join(thermal_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            ambient = ((json.load(handle).get("config") or {}).get("bc") or {}).get("T_amb")

    return {
        "ambient_k": float(ambient) if ambient is not None else None,
        "peak_k": float(t_max) if t_max is not None else max(d["peak_k"] for d in dies),
        "min_k": float(t_min) if t_min is not None else min(d["min_k"] for d in dies),
        "avg_k": float(t_avg) if t_avg is not None else float(np.mean([d["avg_k"] for d in dies])),
        "dies": dies,
    }


def coarsen_factor(unknowns, limit=MAX_UNKNOWNS):
    """Lateral edge multiplier that brings ``unknowns`` down to about ``limit`` (cells scale ~1/f^2)."""
    return max(1.0, math.sqrt(unknowns / limit))


def plan_resolution(G_chip, G_stack, G_sys, mesh_size, preset=None, rule="rowshelf", limit=MAX_UNKNOWNS):
    """Size the thermal grid without solving: returns the grid size, the coarsening needed to fit ``limit``
    and the reference edge (mm) to pass as ``mesh_size_resolution`` (None = production resolution)."""
    from hisim_thermal import get_config, mesh
    from hisim_thermal.adapters.sa import from_sa
    from hisim_thermal.stack import build_stack

    cfg = get_config(preset or "v1.1", {})
    ti = from_sa(G_chip, G_stack, G_sys, cfg, mesh_size=mesh_size, rule=rule)

    def count(edge=None):
        ti.ref_edge_mm = float(edge) if edge is not None else base_edge
        g = mesh.build(ti, build_stack(ti, cfg), cfg)
        return int(g.nk) * int(g.ny) * int(g.nx)

    base_edge = float(ti.ref_edge_mm)
    return fit_resolution(count, base_edge, limit)


def fit_resolution(count, base_edge, limit=MAX_UNKNOWNS):
    """Coarsen the reference edge until ``count(edge)`` unknowns fit ``limit``. Stops (keeping the production
    resolution) when coarsening barely shrinks the grid: every tile/strip boundary needs a cell, so huge
    designs are bounded below by their geometry, not by the target edge."""
    unknowns = final = count(base_edge)
    factor = 1.0
    for _ in range(6):
        if final <= limit:
            break
        trial = factor * max(1.1, coarsen_factor(final, limit))
        shrunk = count(base_edge * trial)
        if shrunk > 0.9 * final:          # no real progress: the geometry sets the floor
            return {"unknowns": unknowns, "final_unknowns": unknowns, "coarsen_factor": 1.0, "resolution_mm": None}
        factor, final = trial, shrunk
    return {"unknowns": unknowns, "final_unknowns": final, "coarsen_factor": factor,
            "resolution_mm": base_edge * factor if factor > 1.0 else None}
