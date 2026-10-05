"""Result files (plan §9.1) and derived quantities: peaks, tile labels, surface power density."""
import csv
import json
import os

import numpy as np

from .layout import REGION_CODES, REGION_NAMES, TILE_KINDS
from .materials import has_3d, material_table

MAT_CODE_BASE = 32
PEAK_CLASSES = ('domain', 'device', 'active', 'imc', 'tile', 'router', 'tsv', 'aib', 'emib')


def sub_roles(grid, stack):
    roles = [stack[p]['role'] for p in grid.sub_phys]
    dies = [stack[p]['die'] for p in grid.sub_phys]
    return roles, dies


def class_masks(grid, stack, maps, ti):
    """Boolean masks [nk, ny, nx] of every peak class."""
    roles, dies = sub_roles(grid, stack)
    shape = (grid.nk, grid.ny, grid.nx)
    m = {c: np.zeros(shape, bool) for c in PEAK_CLASSES}
    m['domain'][:] = True
    reg_imc = REGION_CODES['imc']
    tile_codes = [REGION_CODES[k] for k in TILE_KINDS]
    aib_codes = [REGION_CODES['aib_tx'], REGION_CODES['aib_rx']]
    three_d = has_3d(ti.arch, ti.n_tier)
    for k, (role, d) in enumerate(zip(roles, dies)):
        if d is not None:
            m['device'][k] = True
            reg = maps.region[d]
            if three_d:
                m['tsv'][k] = np.isin(reg, [REGION_CODES['tsv'], REGION_CODES['l3d']])
            if role == 'active':
                m['active'][k] = True
                m['imc'][k] = reg == reg_imc
                m['tile'][k] = np.isin(reg, tile_codes)
                m['router'][k] = reg == REGION_CODES['router']
                m['aib'][k] = np.isin(reg, aib_codes)
        if role == 'emib':
            m['emib'][k] = maps.emib >= 0
    return m


def compute_peaks(T, grid, stack, maps, ti):
    roles, dies = sub_roles(grid, stack)
    xc = 0.5 * (grid.x_edges[:-1] + grid.x_edges[1:]) * 1e3
    yc = 0.5 * (grid.y_edges[:-1] + grid.y_edges[1:]) * 1e3
    zc = 0.5 * (grid.z_edges[:-1] + grid.z_edges[1:]) * 1e6
    out = {}
    for c, mask in class_masks(grid, stack, maps, ti).items():
        if not mask.any():
            out[c] = {'valid': False}
            continue
        Tm = np.where(mask, T, -np.inf)
        flat = int(np.argmax(Tm))
        k, iy, ix = np.unravel_index(flat, T.shape)
        d = dies[k]
        rec = {'valid': True, 'T_K': float(T[k, iy, ix]), 'k': int(k), 'iy': int(iy), 'ix': int(ix),
               'x_mm': float(xc[ix]), 'y_mm': float(yc[iy]), 'z_um': float(zc[k]),
               'layer': stack[grid.sub_phys[k]]['name'], 'tier': d, 'chiplet': None, 'tile': None}
        if d is not None:
            rec['chiplet'] = int(maps.chiplet[d, iy, ix])
            rec['tile'] = ti.tiles[int(maps.owner[d, iy, ix])].id
        elif roles[k] == 'emib' and maps.emib[iy, ix] >= 0:
            rec['tile'] = ti.tiles[int(maps.emib[iy, ix])].id
        out[c] = rec
    return out


def tile_labels(T, grid, stack, ti):
    """Per tile (power-carrying kinds): volume-weighted mean and max temperature over the tile's cells in all
    sublayers of the active layer of its die (plan §9.1)."""
    from .stack import find
    vol_z = grid.dz
    rows = []
    for t in ti.tiles:
        if t.kind not in TILE_KINDS + ('router',):
            continue
        p = find(stack, 'active', t.tier)
        k0, k1 = grid.krange(p)
        sy, sx = grid.tile_slices(t)
        blk = T[k0:k1, sy, sx]
        w = vol_z[k0:k1, None, None] * grid.dy[None, sy, None] * grid.dx[None, None, sx]
        rows.append({'id': t.id, 'chiplet': t.chiplet, 'tier': t.tier, 'kind': t.kind,
                     'x0_mm': t.x0, 'y0_mm': t.y0, 'w_mm': t.w, 'h_mm': t.h,
                     'T_mean_K': float((blk * w).sum() / w.sum()), 'T_max_K': float(blk.max()),
                     'P_W': t.p_dyn_W + t.p_leak_W, 'P_dyn_W': t.p_dyn_W, 'P_leak_W': t.p_leak_W,
                     'slot': t.slot, 'mapped': bool(t.src.get('mapped', t.p_dyn_W > 0)),
                     'pos': t.src.get('pos')})
    return rows


def qsurf(P, grid, stack):
    """W/cm^2 per physical active layer: sum over its sublayers / (dx dy)."""
    area_cm2 = (grid.dy[:, None] * grid.dx[None, :]) * 1e4
    out = {}
    for p, L in enumerate(stack):
        if L['role'] == 'active':
            k0, k1 = grid.krange(p)
            out[L['name']] = P[k0:k1].sum(axis=0) / area_cm2
    return out


def label_array(grid, stack, maps, M, cfg):
    """N_full: region code in die layers / emib bridge, material code (MAT_CODE_BASE + index) elsewhere."""
    N = (MAT_CODE_BASE + M).astype(np.int16)
    for p, L in enumerate(stack):
        k0, k1 = grid.krange(p)
        if L['die'] is not None:
            N[k0:k1] = maps.region[L['die']][None]
        elif L['role'] == 'emib':
            br = maps.emib >= 0
            N[k0:k1][:, br] = REGION_CODES['emib']
    names = {int(c): n for c, n in REGION_NAMES.items()}
    for m, i in material_table(cfg).items():
        names[MAT_CODE_BASE + i] = m
    return N, names


def layers_json(grid, stack):
    subs = []
    for k in range(grid.nk):
        L = stack[grid.sub_phys[k]]
        subs.append({'k': k, 'name': '{}.s{}'.format(L['name'], k - grid.krange(grid.sub_phys[k])[0]),
                     'phys': L['name'], 'role': L['role'], 'die': L['die'], 'tier': L['die'],
                     'z_top_m': float(grid.z_edges[k]), 'z_bot_m': float(grid.z_edges[k + 1]), 'dz_m': float(grid.dz[k])})
    phys = [{'name': L['name'], 'role': L['role'], 'die': L['die'], 'tier': L['die'],
             'k_range': list(grid.krange(p)), 'thickness_m': L['thickness_m']} for p, L in enumerate(stack)]
    return {'axis_order': 'k,iy,ix', 'z_origin': 'top surface, z grows downward', 'sublayers': subs, 'phys_layers': phys}


MANAGED = ('T_full.npy', 'P_full.npy', 'K_full.npy', 'N_full.npy', 'qvol_full.npy', 'grid.npz', 'layers.json',
           'peaks.json', 'tiles.json', 'trace.csv', 'meta.json', 'input.json', 'regions.json', 'T_tile.npy',
           'P_tile.npy', 'tile_labels.npz', 'viewer.html')


def _ours(d):
    try:
        with open(os.path.join(d, 'meta.json')) as f:
            return 'hisim_thermal_version' in json.load(f)
    except (OSError, ValueError):
        return False


def clean_dir(out_dir):
    """Remove stale results of an earlier run (review v0.2 e2e F1): only the files this package writes, its figures,
    and side-layout sub-directories that carry a hisim_thermal meta.json."""
    import shutil
    if not os.path.isdir(out_dir) or not _ours(out_dir):
        return
    for f in os.listdir(out_dir):
        p = os.path.join(out_dir, f)
        if os.path.isfile(p) and (f in MANAGED or (f.startswith('qsurf_') and f.endswith('.npy'))):
            os.remove(p)
        elif os.path.isdir(p) and f == 'figures':
            shutil.rmtree(p)
        elif os.path.isdir(p) and f.startswith('layout_') and _ours(p):
            shutil.rmtree(p)


def write_all(out_dir, res):
    clean_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    g = res.grid
    np.save(os.path.join(out_dir, 'T_full.npy'), res.T)
    np.save(os.path.join(out_dir, 'P_full.npy'), res.P)
    np.save(os.path.join(out_dir, 'K_full.npy'), res.K)
    np.save(os.path.join(out_dir, 'qvol_full.npy'), res.P / g.cell_volume())
    N, names = label_array(g, res.stack, res.maps, res.M, res.cfg)
    np.save(os.path.join(out_dir, 'N_full.npy'), N)
    with open(os.path.join(out_dir, 'regions.json'), 'w') as f:
        json.dump({str(k): v for k, v in sorted(names.items())}, f, indent=1)
    np.savez(os.path.join(out_dir, 'grid.npz'), x_edges=g.x_edges, y_edges=g.y_edges, z_edges=g.z_edges)
    for name, q in qsurf(res.P, g, res.stack).items():
        np.save(os.path.join(out_dir, 'qsurf_{}.npy'.format(name)), q)
    with open(os.path.join(out_dir, 'layers.json'), 'w') as f:
        json.dump(layers_json(g, res.stack), f, indent=1)
    with open(os.path.join(out_dir, 'peaks.json'), 'w') as f:
        json.dump(res.peaks, f, indent=1)
    with open(os.path.join(out_dir, 'tiles.json'), 'w') as f:
        json.dump(res.tiles, f, indent=1)
    _write_tile_arrays(out_dir, res)
    with open(os.path.join(out_dir, 'trace.csv'), 'w', newline='') as f:
        keys = ['item_id', 'category', 'source', 'chiplet', 'tier', 'slot', 'tile_id', 'k0', 'k1', 'iy0', 'iy1',
                'ix0', 'ix1', 'n_rect', 'power_W']
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in res.trace:
            w.writerow(r)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(res.meta, f, indent=1, default=str)
    res.ti.to_json(os.path.join(out_dir, 'input.json'))


def _write_tile_arrays(out_dir, res):
    """IMC: N x N tile label maps per tier (M2D/M3D) or per chiplet (H2_5D)."""
    N = res.ti.mesh_edge
    if not N:
        return
    imc = [r for r in res.tiles if r['kind'] == 'imc' and r.get('pos') is not None]
    if not imc:
        return
    planes = sorted({(r['chiplet'], r['tier']) for r in imc})
    pidx = {pl: i for i, pl in enumerate(planes)}
    shape = (len(planes), N, N)
    Tm, Tx, Pw = np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan)
    slot = np.full(shape, -1, int)
    mapped = np.zeros(shape, bool)
    for r in imc:
        i, j = r['pos']
        q = pidx[(r['chiplet'], r['tier'])]
        Tm[q, i, j], Tx[q, i, j], Pw[q, i, j] = r['T_mean_K'], r['T_max_K'], r['P_W']
        slot[q, i, j], mapped[q, i, j] = r['slot'], r['mapped']
    np.save(os.path.join(out_dir, 'T_tile.npy'), Tm)
    np.save(os.path.join(out_dir, 'P_tile.npy'), Pw)
    np.savez(os.path.join(out_dir, 'tile_labels.npz'), T_mean=Tm, T_max=Tx, P_W=Pw, slot=slot, mapped=mapped,
             planes=np.array(planes))
