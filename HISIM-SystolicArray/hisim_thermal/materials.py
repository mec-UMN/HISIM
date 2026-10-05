"""Cell conductivities (plan §4.1-4.3, §4.7).

Die layers (role beol / active / si / bond of die d) take their material from the region map of die d:
  gap / empty            -> cfg['gap_fill'][role]                      (2.5D, D12)
  tsv  (3D stacks only)  -> tsv_beol / tsv_active / tsv_si / tsv bond   (M3D TSV landing zone)
  aib_tx / aib_rx, si    -> cfg['aib_si']                               (default 'si', as original HISIM)
  everything else        -> beol / si_active / si / bond
Package layers: spreader cu; tim1 (footprint of die 0, else gap_fill); c4 (footprint of the bottom die, else gap_fill);
emib (bridge -> emib_si, else subs_org); subs subs_org; v1compat heatsink / tim2 / heatspread / top beol uniform.
"""
import numpy as np

from .config import bond_props, k_of
from .layout import REGION_CODES

DIE_MAT = {'beol': 'beol', 'active': 'si_active', 'si': 'si'}
TSV_MAT = {'beol': 'tsv_beol', 'active': 'tsv_active', 'si': 'tsv_si'}
UNIFORM = {'spreader': 'cu', 'subs': 'subs_org', 'heatsink': 'heatsink', 'tim2': 'tim2', 'heatspread': 'heatspread'}


def has_3d(arch, n_tier=2):
    """TSV regions take TSV materials only in real 3D stacks (more than one die level)."""
    return arch in ('M3D', 'M3_5D') and n_tier > 1


def material_table(cfg):
    names = sorted(cfg['materials'])
    return {m: i for i, m in enumerate(names)}


def die_material(role, region_name, cfg, three_d):
    _, bond_mat, tsv_bond_mat = bond_props(cfg)
    if region_name in ('gap', 'empty'):
        return cfg['gap_fill'][role]
    if region_name == 'tsv' and three_d:
        return tsv_bond_mat if role == 'bond' else TSV_MAT[role]
    if role == 'si' and region_name in ('aib_tx', 'aib_rx'):
        return cfg['aib_si']
    return bond_mat if role == 'bond' else DIE_MAT[role]


def assign(grid, stack, maps, ti, cfg):
    """K[k, iy, ix] (W/mK) and M[k, iy, ix] (material index into material_table(cfg))."""
    mt = material_table(cfg)
    nk, ny, nx = grid.nk, grid.ny, grid.nx
    K = np.zeros((nk, ny, nx))
    M = np.full((nk, ny, nx), -1, np.int16)
    three_d = has_3d(ti.arch, ti.n_tier)
    n_die = ti.n_tier
    for p, L in enumerate(stack):
        k0, k1 = grid.krange(p)
        role = L['role']
        mat2d = np.empty((ny, nx), dtype=object)
        if L['die'] is not None:
            reg = maps.region[L['die']]
            for code in np.unique(reg):
                name = [k for k, v in REGION_CODES.items() if v == code][0]
                mat2d[reg == code] = die_material(role, name, cfg, three_d)
        elif role == 'tim1':
            mat2d[:] = cfg['gap_fill']['tim1']
            mat2d[maps.footprint[0]] = 'tim1'
        elif role == 'c4':
            mat2d[:] = cfg['gap_fill']['c4']
            mat2d[maps.footprint[n_die - 1]] = 'c4'
        elif role == 'emib':
            mat2d[:] = 'subs_org'
            mat2d[maps.emib >= 0] = 'emib_si'
        elif role == 'beol':        # v1compat: BEOL of the top die, no TSV region
            mat2d[:] = 'beol'
        elif role in UNIFORM:
            mat2d[:] = UNIFORM[role]
        else:
            raise ValueError('no material rule for layer {} (role {})'.format(L['name'], role))
        kk = np.zeros((ny, nx))
        mm = np.zeros((ny, nx), np.int16)
        for name in set(mat2d.ravel().tolist()):
            sel = mat2d == name
            kk[sel] = k_of(cfg, name)
            mm[sel] = mt[name]
        K[k0:k1] = kk[None]
        M[k0:k1] = mm[None]
    if np.any(K <= 0) or not np.all(np.isfinite(K)):
        raise ValueError('non-positive or non-finite conductivity in K')
    return K, M
