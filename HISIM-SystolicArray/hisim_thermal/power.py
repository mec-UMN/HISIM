"""Power distribution onto the grid (plan §6).

Every power item has a support (a set of cells) and is spread with uniform volumetric density over it:
  tile (p_dyn + p_leak)   -> the tile rectangle in all sublayers of the active layer of its die
  link p_src / p_dst      -> union of the src / dst tiles' rectangles, active layer of each tile's die
  link p_mid, mid_path    -> union of the mid tiles' rectangles in the listed die roles of the src die
                              (TSV via mode: share of each segment = its thickness, plan §6.4)
  link p_mid, emib layer  -> the mid tiles (bridge) in the emib layer
Ledgers (plan §6.7):
  source ledger    - every item is injected exactly once (per-item check);
  placement ledger - injected totals per category equal the INDEPENDENT upstream budget computed by the adapter
                     from the raw inputs (ti.meta['source_ledger']); without it, the neutral-input budget.
"""
import numpy as np

from .stack import find

CATS = ('tile_dyn', 'leak', 'router', 'tsv', 'aib', 'emib')


class PowerResult:
    def __init__(self, P, trace, ledger):
        self.P, self.trace, self.ledger = P, trace, ledger


def _support_volume(grid, cells):
    V = 0.0
    for k0, k1, sy, sx, _ in cells:
        V += float(grid.dz[k0:k1].sum() * grid.dy[sy].sum() * grid.dx[sx].sum())
    return V


def _inject(P, grid, cells, p):
    """Spread p (W) over cells with uniform volumetric density; returns [(cell entry, injected W), ...]."""
    if p == 0.0:
        return [(c, 0.0) for c in cells]
    keys = [(c[0], c[1], c[2].start, c[2].stop, c[3].start, c[3].stop) for c in cells]
    if len(set(keys)) != len(keys):                       # final audit power F1: an entry counted twice breaks the
        raise ValueError('duplicate support entries')     # uniform volumetric density (and the via shares)
    V = _support_volume(grid, cells)
    if not V > 0:
        raise ValueError('empty support for positive power')
    q = p / V
    out = []
    for c in cells:
        k0, k1, sy, sx, _ = c
        blk = q * grid.dz[k0:k1, None, None] * grid.dy[None, sy, None] * grid.dx[None, None, sx]
        P[k0:k1, sy, sx] += blk
        out.append((c, float(blk.sum())))
    return out


def _tile_cells(grid, stack, t, roles=('active',), die=None):
    sy, sx = grid.tile_slices(t)
    d = t.tier if die is None else die
    out = []
    for role in roles:
        try:
            p = find(stack, role, d)
        except KeyError:
            raise ValueError('tile {}: the stack has no {} layer for die {}'.format(t.id, role, d))
        k0, k1 = grid.krange(p)
        out.append((k0, k1, sy, sx, t.id))
    return out


def _emib_cells(grid, stack, t):
    sy, sx = grid.tile_slices(t)
    try:
        p = find(stack, 'emib', None)
    except KeyError:
        raise ValueError('tile {}: the stack has no emib layer'.format(t.id))
    k0, k1 = grid.krange(p)
    return [(k0, k1, sy, sx, t.id)]


def _close(a, b):
    return abs(a - b) <= 1e-12 + 1e-9 * max(abs(a), abs(b))


def assign(ti, grid, stack, cfg):
    P = np.zeros((grid.nk, grid.ny, grid.nx))
    trace = []
    worst = 0.0
    item_in = dict.fromkeys(CATS, 0.0)
    injected = dict.fromkeys(CATS, 0.0)
    tidx = ti.tile_index()

    def rec(item, cat, p_in, parts, info):
        nonlocal worst
        p_out = sum(w for _, w in parts)
        item_in[cat] += p_in
        injected[cat] += p_out
        worst = max(worst, abs(p_out - p_in) / p_in if p_in else abs(p_out))
        for (k0, k1, sy, sx, tid), w in parts:
            trace.append(dict(info, item_id=item, category=cat, tile_id=tid, k0=k0, k1=k1, iy0=sy.start, iy1=sy.stop,
                              ix0=sx.start, ix1=sx.stop, power_W=w))

    for t in ti.tiles:
        if t.kind in ('gap', 'empty', 'emib', 'tsv', 'l3d', 'aib_tx', 'aib_rx', 'filler'):
            continue
        cells = _tile_cells(grid, stack, t)
        info = dict(source=str(t.src.get('layer', t.src.get('node', ''))), chiplet=t.chiplet, tier=t.tier, slot=t.slot)
        if t.kind == 'router':
            p = t.p_dyn_W + t.p_leak_W
            rec(t.id, 'router', p, _inject(P, grid, cells, p), info)
        else:
            rec(t.id + ':dyn', 'tile_dyn', t.p_dyn_W, _inject(P, grid, cells, t.p_dyn_W), info)
            if t.p_leak_W:
                rec(t.id + ':leak', 'leak', t.p_leak_W, _inject(P, grid, cells, t.p_leak_W), info)

    for l in ti.links:
        cat_sd = 'tsv' if l.kind == 'tsv' else 'aib'
        cat_mid = 'tsv' if l.kind == 'tsv' else 'emib'
        base = dict(source='{}:{}'.format(l.kind, l.src_row), chiplet=l.src[0], tier=l.src[1], slot=-1)
        for part, p, key in (('src', l.p_src_W, 'src_tiles'), ('dst', l.p_dst_W, 'dst_tiles')):
            cells = []
            for tid in l.region.get(key, []):
                cells += _tile_cells(grid, stack, ti.tiles[tidx[tid]])
            if cells or p:
                rec('{}:{}'.format(l.id, part), cat_sd, p, _inject(P, grid, cells, p), base)
        if l.p_mid_W or l.region.get('mid_tiles'):
            cells = []
            for tid in l.region.get('mid_tiles', []):
                t = ti.tiles[tidx[tid]]
                if l.region.get('mid_layer') == 'emib':
                    cells += _emib_cells(grid, stack, t)
                else:
                    cells += _tile_cells(grid, stack, t, roles=tuple(l.region['mid_path']), die=l.src[1])
            rec('{}:mid'.format(l.id), cat_mid, l.p_mid_W, _inject(P, grid, cells, l.p_mid_W), base)

    # placement ledger by region (review v0.2 F15): P summed over the region labels of the grid versus the trace rows
    # grouped by the kind of their target tile -- two independent code paths
    kind_of = {t.id: t.kind for t in ti.tiles}
    by_kind = {}
    for row in trace:
        kd = kind_of.get(row['tile_id'], '?')
        by_kind[kd] = by_kind.get(kd, 0.0) + row['power_W']
    from .layout import build_maps, REGION_NAMES
    maps = build_maps(ti, grid)
    by_region = {}
    for pidx, L in enumerate(stack):
        k0, k1 = grid.krange(pidx)
        blk = P[k0:k1].sum(axis=0)
        if L['die'] is not None:
            reg = maps.region[L['die']]
            for code in np.unique(reg):
                nm = REGION_NAMES[int(code)]
                by_region[nm] = by_region.get(nm, 0.0) + float(blk[reg == code].sum())
        elif L['role'] == 'emib':
            by_region['emib'] = by_region.get('emib', 0.0) + float(blk[maps.emib >= 0].sum())
            by_region['subs_under_emib_layer'] = by_region.get('subs_under_emib_layer', 0.0) + float(blk[maps.emib < 0].sum())
        else:
            by_region['package:' + L['role']] = by_region.get('package:' + L['role'], 0.0) + float(blk.sum())
    region_ok = all(_close(by_region.get(k, 0.0), v) for k, v in by_kind.items())
    region_ok &= all(abs(v) <= 1e-15 or k in by_kind for k, v in by_region.items())
    src = ti.meta.get('source_ledger')
    expected = {c: float(src[c]) for c in CATS} if src else {c: ti.budget()[c] for c in CATS}
    ledger = {'expected_from': 'adapter source_ledger (raw upstream inputs)' if src else 'neutral-input budget',
              'expected_W': expected, 'items_W': item_in, 'injected_W': injected,
              'grid_total_W': float(P.sum()), 'item_max_rel_err': worst}
    ok = worst <= 1e-9
    for c in CATS:
        ok &= _close(item_in[c], expected[c]) and _close(injected[c], expected[c])
    ok &= _close(ledger['grid_total_W'], sum(expected.values()))
    ledger['by_region_W'] = by_region
    ledger['by_target_kind_W'] = by_kind
    ledger['region_ok'] = bool(region_ok)
    ok &= region_ok
    if src and src.get('aib_rows') is not None:      # per AIB row: Tx strip / Rx strip / EMIB (final review plan_b4 F1)
        got = {}
        for row in trace:
            if row['category'] in ('aib', 'emib') and str(row.get('source', '')).startswith('aib:'):
                j = int(row['source'].split(':')[1])
                part = row['item_id'].rsplit(':', 1)[1]
                got[(j, part)] = got.get((j, part), 0.0) + row['power_W']
        rows_out, rows_ok = [], True
        for r in src['aib_rows']:
            g = {k: got.pop((r['row'], part), 0.0) for k, part in (('tx_W', 'src'), ('rx_W', 'dst'), ('emib_W', 'mid'))}
            good = all(_close(g[k], r[k]) for k in g)
            rows_ok &= good
            rows_out.append({'row': r['row'], 'expected': {k: r[k] for k in g}, 'injected': g, 'ok': bool(good)})
        rows_ok &= all(abs(v) <= 1e-15 for (j, _), v in got.items() if j >= 0)    # no power on unknown rows
        ledger['aib_rows'] = rows_out
        ledger['aib_rows_ok'] = bool(rows_ok)
        ok &= rows_ok
    ledger['ok'] = bool(ok)
    if cfg.get('check_conservation', True) and not ok:
        raise ValueError('power conservation failed: {}'.format(ledger))
    if np.any(P < 0) or not np.all(np.isfinite(P)):
        raise ValueError('negative or non-finite cell power')
    return PowerResult(P, trace, ledger)
