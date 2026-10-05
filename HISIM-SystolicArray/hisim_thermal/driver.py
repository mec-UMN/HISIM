"""run(ti, cfg, out_dir): neutral input -> temperature field and result files (plan §3.4)."""
import hashlib
import json
import time

import numpy as np

from . import assemble, layout, materials, mesh, output, power, solve
from .config import VERSION, provenance
from .stack import build_stack, summary, total_thickness_um


class Result:
    pass


def prepare(ti, cfg):
    """Geometry, materials and power (no solve).  Useful for tests and exporters."""
    ti.check()
    r = Result()
    r.ti, r.cfg = ti, cfg
    r.stack = build_stack(ti, cfg)
    r.grid = mesh.build(ti, r.stack, cfg)
    r.maps = layout.build_maps(ti, r.grid)
    r.K, r.M = materials.assign(r.grid, r.stack, r.maps, ti, cfg)
    pw = power.assign(ti, r.grid, r.stack, cfg)
    r.P, r.trace, r.ledger = pw.P, pw.trace, pw.ledger
    return r


def run(ti, cfg, out_dir=None, write=True, solver_force=None, P_override=None):
    t0 = time.time()
    r = prepare(ti, cfg)
    if P_override is not None:
        if P_override.shape != r.P.shape:
            raise ValueError('P_override shape mismatch')
        r.P = np.array(P_override, dtype=float)
        # the ledgers and the trace describe the neutral-input power, not this map (final audit driver F1)
        r.trace = []
        r.ledger = {'ok': None, 'P_override': True, 'P_total_W': float(r.P.sum()),
                    'note': 'power map given directly (P_override): source / region ledgers and trace not applicable'}
        if not np.all(np.isfinite(r.P)) or np.any(r.P < 0):
            raise ValueError('P_override must be finite and non-negative')
    t1 = time.time()
    r.op = assemble.build_operator(r.grid, r.K, cfg['bc'])
    t2 = time.time()
    theta, r.solver_info = solve.run(r.op.G, r.P.ravel(), cfg['solver'], force=solver_force)
    t3 = time.time()
    r.theta = theta.reshape(r.P.shape)
    r.energy = assemble.energy_balance(r.op, r.theta, float(r.P.sum()))
    if not r.energy['ok']:
        raise RuntimeError('energy balance failed: {}'.format(r.energy))
    T_amb = cfg['bc']['T_amb']
    r.T = r.theta + T_amb
    r.peaks = output.compute_peaks(r.T, r.grid, r.stack, r.maps, ti)
    r.tiles = output.tile_labels(r.T, r.grid, r.stack, ti)
    W, H = ti.bbox_mm
    A_top = W * H * 1e-6
    h_top = cfg['bc']['h_top']
    r.meta = {
        'hisim_thermal_version': VERSION, 'preset': cfg['preset'], 'config': cfg, 'provenance': provenance(cfg),
        'input': {'arch': ti.arch, 'n_tier': ti.n_tier, 'chiplet_num': ti.chiplet_num, 'mesh_edge': ti.mesh_edge,
                  'bbox_mm': list(ti.bbox_mm), 'ref_edge_mm': ti.ref_edge_mm, 'source': ti.source,
                  'n_tiles': len(ti.tiles), 'n_links': len(ti.links), 'layout_hash': ti.layout_hash(),
                  'budget_W': ti.budget(), 'adapter_meta': ti.meta},
        'stack': summary(r.stack), 'stack_total_um': total_thickness_um(r.stack), 'grid': r.grid.describe(),
        'A_top_m2': A_top, 'R_sa_eq_K_per_W': (1.0 / (h_top * A_top)) if h_top > 0 else None,
        'solver': r.solver_info, 'energy_balance': r.energy, 'power_ledger': r.ledger,
        'peaks_K': {c: (v['T_K'] if v.get('valid') else None) for c, v in r.peaks.items()},
        'certified': {'tile_level': 'covered only by a tools/run_gci.py certificate_*.json with the same production_grid_id, '
                                    'layout_hash and physics_id; read its verdicts (verdict_d11 = plan §5.4 as written, '
                                    'verdict_d11_noise_exempt, verdict_relaxed) and README §4 / FINAL_REPORT §3',
                      'global_router_peaks': 'not certified (D11)'},
        'validation_state': {'input_profile': ti.meta.get('input_profile', 'upstream_raw' if ti.source != 'synthetic' else 'explicit_W'),
                             'upstream_physical_validation': 'unverified', 'numerical_validation': 'pending',
                             'physical_calibration': 'not_performed'},
        'timing_s': {'prepare': t1 - t0, 'assemble': t2 - t1, 'solve': t3 - t2},
        'notes': list(ti.meta.get('notes', [])),
        'production_grid_id': grid_id(r.grid, cfg),
        'physics_id': physics_id(cfg),
    }
    if r.peaks['device'].get('valid') and abs(r.peaks['device']['T_K'] - r.peaks['domain']['T_K']) > 0:
        r.meta['peak_warning'] = 'domain peak ({:.6f} K, {}) differs from the device peak ({:.6f} K) used for PPA'.format(
            r.peaks['domain']['T_K'], r.peaks['domain']['layer'], r.peaks['device']['T_K'])
    r.out_dir = out_dir
    if write and out_dir:
        output.write_all(out_dir, r)
        r.meta['timing_s']['write'] = time.time() - t3
    return r


def physics_id(cfg):
    """Hash of every setting that changes the physical problem (preset, materials, stack, bonding, gap fill, BCs,
    power semantics, alpha); mesh, solver and output settings excluded.  A GCI certificate covers a run only if
    production_grid_id, layout_hash and physics_id all match (final audit driver F3)."""
    keep = {k: v for k, v in cfg.items() if k not in ('mesh', 'solver', 'output', 'check_conservation')}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def grid_id(grid, cfg):
    """Hash of the grid (edges, sublayers) and mesh settings: identifies the production grid (plan §5.4)."""
    h = hashlib.sha1()
    for a in (grid.x_edges, grid.y_edges, grid.z_edges):
        h.update(np.round(a, 15).tobytes())
    h.update(repr(sorted(cfg['mesh'].items())).encode())
    return h.hexdigest()[:16]
