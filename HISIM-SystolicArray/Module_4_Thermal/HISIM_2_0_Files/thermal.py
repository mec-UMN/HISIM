"""HISIM 2.0 (systolic array) thermal module: G_chip / G_stack / G_sys -> hisim_thermal (neutral input -> FV solver).

The original file was an empty stub (interface comments only; it also opened a pickle and called pdb at import).
The interface of the stub is kept:
    thermal_analysis(G_chip, G_stack, Device_map, Thermal_info, T_ambient, mesh_size_resolution) -> (T_min, T_avg, T_max)
Device_map / Thermal_info (stack sub-layers and materials) come from hisim_thermal's presets instead of CSV files;
T_ambient overrides the preset's T_amb; mesh_size_resolution (mm, optional) sets the reference cell edge.
Extra keyword arguments: G_sys (required: chiplet tiers, NoP positions, areas), mesh_size, preset, overrides,
out_dir, rule ('rowshelf' | 'grid' floorplan interpretation, see hisim_thermal/adapters/sa.py).
Temperatures in K; T_min / T_avg / T_max are taken over the active-layer cells of all dies.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SA_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))


def _find_pkg():
    for c in (os.environ.get('HISIM_THERMAL_ROOT'), _SA_ROOT, os.path.dirname(_SA_ROOT)):
        if c and os.path.isfile(os.path.join(c, 'hisim_thermal', '__init__.py')):
            return c
    return None


_PKG = _find_pkg()
if _PKG and _PKG not in sys.path:
    sys.path.insert(0, _PKG)


def thermal_analysis(G_chip, G_stack, Device_map=None, Thermal_info=None, T_ambient=None, mesh_size_resolution=None,
                     G_sys=None, mesh_size=None, preset=None, overrides=None, out_dir=None, rule='rowshelf'):
    if _PKG is None:
        raise ImportError('hisim_thermal not found: set HISIM_THERMAL_ROOT (looked next to {})'.format(_SA_ROOT))
    if G_sys is None:
        raise ValueError('thermal_analysis needs G_sys (chiplet tiers, NoP positions, chiplet areas)')
    if Device_map is not None or Thermal_info is not None:
        raise NotImplementedError('Device_map / Thermal_info CSVs are not used: the stack and materials come from the '
                                  'hisim_thermal preset (pass overrides instead)')
    import json
    import numpy as np
    from hisim_thermal import get_config
    from hisim_thermal.adapters.sa import from_sa
    from hisim_thermal.driver import run
    import copy
    over = copy.deepcopy(overrides or {})          # never modify the caller's dicts (delta review r_sa F1)
    if os.environ.get('HISIM_THERMAL_OVERRIDES'):            # deep merge (final review F7)
        def merge(dst, src):
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    merge(dst[k], v)
                else:
                    dst[k] = v
        merge(over, json.loads(os.environ['HISIM_THERMAL_OVERRIDES']))
    if T_ambient is not None:
        over.setdefault('bc', {})['T_amb'] = float(T_ambient)
    cfg = get_config(preset or os.environ.get('HISIM_THERMAL_PRESET') or 'v1.1', over)
    ti = from_sa(G_chip, G_stack, G_sys, cfg, mesh_size=mesh_size, rule=rule)
    if mesh_size_resolution:
        ti.ref_edge_mm = float(mesh_size_resolution)
    out = out_dir or os.path.join(_SA_ROOT, 'Results', 'thermal')
    r = run(ti, cfg, out)
    # active layers, die cells only (no package gap / empty), volume-weighted mean (review v0.2 fixes F7)
    vol = r.grid.cell_volume()
    vals, wts = [], []
    for p, L in enumerate(r.stack):
        if L['role'] == 'active':
            k0, k1 = r.grid.krange(p)
            die = r.maps.footprint[L['die']]
            vals.append(r.T[k0:k1][:, die].ravel())
            wts.append(vol[k0:k1][:, die].ravel())
    vals, wts = np.concatenate(vals), np.concatenate(wts)
    t_min, t_max, t_avg = float(vals.min()), float(vals.max()), float((vals * wts).sum() / wts.sum())
    print('hisim_thermal [SA, {}]: power {:.6g} W, energy balance rel {:.2e}, peaks: {}'.format(
        ti.arch, r.meta['input']['budget_W']['total'], r.energy['rel_err'],
        ', '.join('{} {:.3f} K'.format(c, v['T_K']) for c, v in r.peaks.items() if v.get('valid'))))
    print('hisim_thermal [SA]: die area / upstream area = {} (floorplan rule {}); results in {}'.format(
        {k: round(v, 3) for k, v in ti.meta['die_area_vs_upstream'].items() if v}, rule, out))
    if cfg['output'].get('plots', True):
        from hisim_thermal import plots
        plots.make_all(out)
    if cfg['output'].get('viewer', False):
        from hisim_thermal import viewer
        viewer.write_viewer(out)
    return t_min, t_avg, t_max
