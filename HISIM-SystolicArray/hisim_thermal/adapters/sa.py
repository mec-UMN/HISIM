"""HISIM-SystolicArray (HISIM 2.0) adapter: G_chip / G_stack / G_sys -> ThermalInput.

Upstream facts used (see docs/SA_vs_IMC.md):
  G_sys  : chiplet 'C<k>': Stack ID 'S<k>', Tier ID 'TR<z>' (z = 0 is the BOTTOM die, on the interposer),
           NoP Position 'x,y,z' (stack position on the NoP grid; y grows to the north), Chiplet Actual Area (mm^2).
           Chiplet ID 'Empty' = empty NoP site.
  G_chip : tile '<chip>_T<n>' (or '<chip>_x<i>_y<j>' for Empty padding nodes): HW Type (last word SA / CPU / Mem /
           Tile (= 'Mem Tile', incl. DDR) / Empty), NoC Position 'x,y' (x east, y north; -1 / m = boundary NoP
           interface sites), Tile_Area (mm^2), Tile_Power (W, average over the tile's own busy time), Tile_Latency (s),
           Tile_Energy (J), router Area (mm^2), router Power (W), 3D link Area (mm^2), 3D link Power (W).
  G_stack: '<stack>_N|S|E|W' NoP interface of the TR0 die on that edge: 2.5d link Area (mm^2), 2.5d link Power (W).

Geometry interpretation (an ASSUMPTION, recorded in meta; upstream gives areas and grid indices only):
  * every tile uses HISIM's own beachhead template (Network.py:278-316): core a x a (a = sqrt(Tile_Area)),
    3D-link strips a x b and b x a, router b x b (b = sqrt(router Area));
  * the NoC mesh of a stack is a grid of slots: column x width = max over tiers / rows of (a + b), row y height =
    max over tiers / columns of (a + b); tiles sit at the top-left of their slot, the rest of the slot is filler Si;
    all tiers of a stack share the grid (3D links line up, equal die footprints, like 'Chiplet Actual Area');
  * the TR0 die carries a 2.5D PHY strip on every edge with 2.5d link Area > 0: full edge length, depth = area/edge;
  * stacks sit on the NoP grid (north up), centred in cells sized by the largest die of each NoP row / column,
    separated by gap_mm; package area outside the dies is 'gap'; facing PHY strips of NoP-adjacent stacks are joined
    by an EMIB bridge (geometry only; upstream gives no wire power);
  * boundary NoP-interface routers (Empty nodes outside the mesh) put their power into the PHY strip of their edge.
"""
import math

import numpy as np

from ..neutral_input import Link, ThermalInput, Tile

KIND = {'SA': 'sa', 'CPU': 'cpu', 'Mem': 'mem', 'Tile': 'mem'}


def _nodes(g):
    if g is None:
        return {}
    if hasattr(g, 'nodes') and callable(getattr(g, 'nodes')):
        return {k: dict(v) for k, v in g.nodes(data=True)}
    return {k: dict(v) for k, v in g.items()}


def _xy(s):
    v = [int(float(t)) for t in str(s).split(',')]
    return v


def _num(v, name, node):
    try:
        f = float(v or 0.0)
    except (TypeError, ValueError):
        raise ValueError('{}: {} is not a number ({!r})'.format(node, name, v))
    if not math.isfinite(f) or f < 0:
        raise ValueError('{}: {} = {} must be finite and >= 0'.format(node, name, f))
    return f


def _infer_mesh(tiles):
    """NoC mesh size m when mesh_size is not given (review v1.0 understanding F1).  HISIM surrounds the m x m mesh
    with a ring of Empty NoP nodes at -1 and m; a coordinate M = max(x, y) whose nodes are all Empty is that ring."""
    xy = [(t['x'], t['y'], t['last']) for t in tiles]
    if not any(x >= 0 and y >= 0 for x, y, _ in xy):
        return 1
    M = max(max(x, y) for x, y, _ in xy)
    if any(x == -1 or y == -1 for x, y, _ in xy) and all(l == 'Empty' for x, y, l in xy if x == M or y == M):
        return M
    return M + 1


def from_sa(G_chip, G_stack, G_sys, cfg, mesh_size=None, gap_mm=0.1, rule='rowshelf', floorplan=None):
    pw = cfg['power']
    if pw['tsv_mode'] != 'driver' or pw['emib_wire_fraction'] != 0.0 or pw['include_leakage']:
        raise NotImplementedError('SA path: tsv_mode via / emib_wire_fraction / include_leakage have no upstream data '
                                  'in HISIM 2.0 (review v0.2 SA F4)')
    if floorplan is not None:
        raise NotImplementedError('user floorplan coordinates: planned (see docs/SA_vs_IMC.md); not implemented yet')
    chip_nodes, stack_nodes, sys_nodes = _nodes(G_chip), _nodes(G_stack), _nodes(G_sys)
    mode = cfg['power']['mode']
    # ---------------------------------------------------------------- chiplets, stacks, tiers
    chips = {}
    for cid, a in sys_nodes.items():
        if a.get('Chiplet ID') == 'Empty':
            continue
        nop = _xy(a['NoP Position'])
        chips[cid] = {'stack': a['Stack ID'], 'z': int(str(a['Tier ID'])[2:]), 'nop': (nop[0], nop[1]),
                      'actual': float(a.get('Chiplet Actual Area', 0.0) or 0.0)}
    stacks = {}
    for cid, c in chips.items():
        stacks.setdefault(c['stack'], []).append(cid)
    st_ids = sorted(stacks, key=lambda s: (chips[stacks[s][0]]['nop'][1], chips[stacks[s][0]]['nop'][0]))
    tiers = {len(stacks[s]) for s in st_ids}
    if len(tiers) != 1:
        raise NotImplementedError('stacks with different tier counts ({}) are not supported'.format(
            {s: len(stacks[s]) for s in st_ids}))
    n_tier = tiers.pop()
    for s in st_ids:
        zs = sorted(chips[c]['z'] for c in stacks[s])
        if zs != list(range(n_tier)):
            raise ValueError('stack {} tiers {} are not 0..{}'.format(s, zs, n_tier - 1))
        if len({chips[c]['nop'] for c in stacks[s]}) != 1:
            raise ValueError('stack {} chiplets have different NoP (x, y)'.format(s))
    n_stack = len(st_ids)
    arch = ('M2D' if n_tier == 1 else 'M3D') if n_stack == 1 else ('H2_5D' if n_tier == 1 else 'M3_5D')
    # ---------------------------------------------------------------- tiles per chiplet
    per_chip = {c: [] for c in chips}
    for nid, a in chip_nodes.items():
        cid = nid.split('_')[0]
        if cid not in chips:
            raise ValueError('tile {} belongs to unknown chiplet {}'.format(nid, cid))
        ht = str(a.get('HW Type', '')).split()
        last = ht[-1] if ht else ''
        x, y = _xy(a['NoC Position'])[:2]
        per_chip[cid].append({'id': nid, 'last': last, 'x': x, 'y': y,
                              'A': _num(a.get('Tile_Area'), 'Tile_Area', nid),
                              'P': _num(a.get('Tile_Power'), 'Tile_Power', nid),
                              'E': _num(a.get('Tile_Energy'), 'Tile_Energy', nid),
                              'L': _num(a.get('Tile_Latency'), 'Tile_Latency', nid),
                              'Ar': _num(a.get('router Area'), 'router Area', nid),
                              'Pr': _num(a.get('router Power'), 'router Power', nid),
                              'A3': _num(a.get('3D link Area'), '3D link Area', nid),
                              'P3': _num(a.get('3D link Power'), '3D link Power', nid),
                              'type': ' '.join(ht), 'layers': a.get('AI Layer')})
    # duty (plan §6.1 analogue): concurrent = busy-average Tile_Power as given
    busy = [t for c in per_chip for t in per_chip[c] if t['last'] not in ('Empty', '') and t['L'] > 0]
    L_sum = sum(t['L'] for t in busy) or 1.0
    L_max = max([t['L'] for t in busy] or [1.0])

    def p_eff(t):
        if mode == 'concurrent':
            return t['P']
        if mode == 'pipeline':
            return t['P'] * (t['L'] / L_max)
        return t['E'] / L_sum            # time_avg: energy over the serial end-to-end compute time

    # ---------------------------------------------------------------- slot grid per stack
    geo, inferred = {}, []
    edge_b = [math.sqrt(t['Ar']) for c in per_chip for t in per_chip[c] if t['Ar'] > 0]
    b_typ = float(np.median(edge_b)) if edge_b else 0.01
    for s in st_ids:
        cs = stacks[s]
        if not (mesh_size and all(c in mesh_size for c in cs)):
            inferred.append('stack {}: mesh_size not given, NoC size inferred from the node coordinates'.format(s))
        m = max(int(mesh_size[c]) if mesh_size and c in mesh_size else _infer_mesh(per_chip[c]) for c in cs)
        e = np.zeros((m, m))                       # slot edge per NoC position: max over the tiers of a + b
        for c in cs:
            for t in per_chip[c]:
                if 0 <= t['x'] < m and 0 <= t['y'] < m:
                    e[t['x'], t['y']] = max(e[t['x'], t['y']], math.sqrt(t['A']) + math.sqrt(t['Ar']))
        e[e <= 0] = b_typ
        slots = {}
        if rule == 'grid':                          # strict NoC grid: column / row maxima
            colw, rowh = e.max(axis=1), e.max(axis=0)
            cx = np.concatenate([[0.0], np.cumsum(colw)])
            ry = np.concatenate([[0.0], np.cumsum(rowh[::-1])])
            for xq in range(m):
                for yq in range(m):
                    slots[(xq, yq)] = (cx[xq], ry[m - 1 - yq], colw[xq], rowh[yq])
            Wm, Hm = float(colw.sum()), float(rowh.sum())
        elif rule == 'rowshelf':                    # NoC rows as shelves (north on top), square slots left to right
            Wm, yy = 0.0, 0.0
            for yq in range(m - 1, -1, -1):
                hrow, xx = float(e[:, yq].max()), 0.0
                for xq in range(m):
                    slots[(xq, yq)] = (xx, yy, float(e[xq, yq]), hrow)
                    xx += float(e[xq, yq])
                Wm = max(Wm, xx)
                yy += hrow
            Hm = yy
        else:
            raise ValueError("SA floorplan rule must be 'rowshelf' or 'grid'")
        tr0 = [c for c in cs if chips[c]['z'] == 0][0]
        d = {}
        for e in 'NSEW':
            A_e = float(stack_nodes.get('{}_{}'.format(s, e), {}).get('2.5d link Area', 0.0) or 0.0)
            d[e] = A_e / (Wm if e in 'NS' else Hm) if A_e > 0 else 0.0
        geo[s] = {'m': m, 'slots': slots, 'Wm': Wm, 'Hm': Hm, 'd': d, 'tr0': tr0,
                  'Wd': Wm + d['W'] + d['E'], 'Hd': Hm + d['N'] + d['S']}
    for s in st_ids:                 # review v0.2 SA F5: only Empty boundary NoP sites may lie outside the mesh
        m = geo[s]['m']
        for c in stacks[s]:
            for t in per_chip[c]:
                if 0 <= t['x'] < m and 0 <= t['y'] < m:
                    continue
                ring = t['x'] in (-1, m) or t['y'] in (-1, m)
                if t['last'] != 'Empty' or not ring or not (-1 <= t['x'] <= m and -1 <= t['y'] <= m):
                    raise ValueError('{}: node at NoC {} lies outside the {}x{} mesh and is not an Empty boundary '
                                     'NoP site (check mesh_size)'.format(t['id'], (t['x'], t['y']), m, m))
    # ---------------------------------------------------------------- package (NoP grid, north up)
    xs = sorted({chips[stacks[s][0]]['nop'][0] for s in st_ids})
    ys = sorted({chips[stacks[s][0]]['nop'][1] for s in st_ids}, reverse=True)
    cw = {x: max(geo[s]['Wd'] for s in st_ids if chips[stacks[s][0]]['nop'][0] == x) for x in xs}
    rh = {y: max(geo[s]['Hd'] for s in st_ids if chips[stacks[s][0]]['nop'][1] == y) for y in ys}
    gx, gy, acc = {}, {}, 0.0
    for x in xs:
        gx[x] = acc
        acc += cw[x] + gap_mm
    W = acc - gap_mm
    acc = 0.0
    for y in ys:
        gy[y] = acc
        acc += rh[y] + gap_mm
    H = acc - gap_mm
    for s in st_ids:
        x, y = chips[stacks[s][0]]['nop']
        geo[s]['X0'] = gx[x] + 0.5 * (cw[x] - geo[s]['Wd'])
        geo[s]['Y0'] = gy[y] + 0.5 * (rh[y] - geo[s]['Hd'])
    # ---------------------------------------------------------------- tiles
    tiles, links, notes = [], [], list(inferred)
    l3d_area = [0.0, 0.0]          # sum of upstream 3D link areas, sum of template strip areas
    nop_power = {}
    die_rects = {}
    for ci, s in enumerate(st_ids):
        g = geo[s]
        X0, Y0 = g['X0'], g['Y0']
        die_rects[ci] = (X0, Y0, g['Wd'], g['Hd'])
        mx0, my0 = X0 + g['d']['W'], Y0 + g['d']['N']
        for c in stacks[s]:
            z = chips[c]['z']
            die = n_tier - 1 - z
            pref = '{}.{}'.format(s, c)
            occ = {}
            for t in per_chip[c]:
                if 0 <= t['x'] < g['m'] and 0 <= t['y'] < g['m']:
                    if (t['x'], t['y']) in occ:
                        raise ValueError('{}: two tiles at NoC position {}'.format(c, (t['x'], t['y'])))
                    occ[(t['x'], t['y'])] = t
            for xq in range(g['m']):
                for yq in range(g['m']):
                    r = g['m'] - 1 - yq
                    ox, oy, sw, sh = g['slots'][(xq, yq)]
                    sx0, sy0 = mx0 + ox, my0 + oy
                    t = occ.get((xq, yq))
                    a = math.sqrt(t['A']) if t else 0.0
                    b = math.sqrt(t['Ar']) if t else 0.0
                    tag = '{}.x{}y{}'.format(pref, xq, yq)
                    if t and t['last'] != 'Empty' and a > 0:
                        kind = KIND.get(t['last'], 'logic')
                        tiles.append(Tile(tag + '.core', ci, die, kind, sx0, sy0, a, a, p_dyn_W=p_eff(t),
                                          src={'node': t['id'], 'hw_type': t['type'], 'layers': t['layers'],
                                               'busy_W': t['P'], 'latency_s': t['L'], 'energy_J': t['E'],
                                               'pos': [r, xq], 'noc': [xq, yq], 'sa_tier': z, 'mapped': True}))
                        if b > 0:
                            # 3D-link landing strips of the beachhead template; plain die Si (review v0.2 SA F3):
                            # the upstream 3D link area is ~1e-4 of the strip area, so TSV material is not used
                            tiles.append(Tile(tag + '.l3dR', ci, die, 'l3d', sx0 + a, sy0, b, a))
                            tiles.append(Tile(tag + '.l3dB', ci, die, 'l3d', sx0, sy0 + a, a, b))
                            l3d_area[0] += t['A3']
                            l3d_area[1] += 2 * a * b
                    elif t and t['P'] > 0:
                        raise ValueError('{}: positive Tile_Power on a tile with zero area'.format(t['id']))
                    if t and b > 0:
                        tiles.append(Tile(tag + '.r', ci, die, 'router', sx0 + a, sy0 + a, b, b, p_dyn_W=t['Pr'],
                                          src={'node': t['id'], 'noc': [xq, yq], 'sa_tier': z}))
                    elif t and t['Pr'] > 0:
                        raise ValueError('{}: router power without router area'.format(t['id']))
                    e = a + b
                    if e == 0:
                        tiles.append(Tile(tag + '.f', ci, die, 'filler', sx0, sy0, sw, sh))
                    else:
                        if sw - e > 1e-12:
                            tiles.append(Tile(tag + '.fR', ci, die, 'filler', sx0 + e, sy0, sw - e, sh))
                        if sh - e > 1e-12:
                            tiles.append(Tile(tag + '.fB', ci, die, 'filler', sx0, sy0 + e, e, sh - e))
                    if t and t['P3'] > 0:
                        if t['last'] != 'Empty' and a > 0 and b > 0:
                            src_t, note = [tag + '.l3dR', tag + '.l3dB'], 'SA 3D link power, injected in its own strips'
                        elif b > 0:     # Empty slot that passes vertical traffic (review v0.2 SA F2): at its router
                            src_t, note = [tag + '.r'], 'SA 3D link power of an Empty slot, injected at its router'
                        else:
                            raise ValueError('{}: 3D link power but neither strips nor a router'.format(t['id']))
                        links.append(Link('l3d.{}'.format(t['id']), 'tsv', (ci, die), (ci, die), p_src_W=t['P3'],
                                          region={'src_tiles': src_t}, note=note))
            if rule == 'rowshelf':
                for yq in range(g['m']):
                    ox, oy, sw, sh = g['slots'][(g['m'] - 1, yq)]
                    xe = ox + sw
                    if g['Wm'] - xe > 1e-12:
                        tiles.append(Tile('{}.rowend{}'.format(pref, yq), ci, die, 'filler', mx0 + xe, my0 + oy,
                                          g['Wm'] - xe, sh))
            # PHY strips (TR0 only; other tiers: filler) and die corners
            d = g['d']
            strips = {'N': (mx0, Y0, g['Wm'], d['N']), 'S': (mx0, my0 + g['Hm'], g['Wm'], d['S']),
                      'W': (X0, my0, d['W'], g['Hm']), 'E': (mx0 + g['Wm'], my0, d['E'], g['Hm'])}
            for e, (x0, y0, w, h) in strips.items():
                if w > 1e-12 and h > 1e-12:
                    k = 'aib_tx' if z == 0 else 'filler'
                    tiles.append(Tile('{}.phy{}'.format(pref, e), ci, die, k, x0, y0, w, h,
                                      src={'edge': e, 'stack': s}))
            for nm, (x0, y0, w, h) in {'NW': (X0, Y0, d['W'], d['N']), 'NE': (mx0 + g['Wm'], Y0, d['E'], d['N']),
                                       'SW': (X0, my0 + g['Hm'], d['W'], d['S']),
                                       'SE': (mx0 + g['Wm'], my0 + g['Hm'], d['E'], d['S'])}.items():
                if w > 1e-12 and h > 1e-12:
                    tiles.append(Tile('{}.corner{}'.format(pref, nm), ci, die, 'filler', x0, y0, w, h))
            # 2.5D link power and boundary NoP routers -> PHY strip of the edge
            if z == 0:
                bnd = {e: 0.0 for e in 'NSEW'}
                for t in per_chip[c]:
                    if 0 <= t['x'] < g['m'] and 0 <= t['y'] < g['m']:
                        continue
                    e = 'W' if t['x'] < 0 else 'E' if t['x'] >= g['m'] else 'S' if t['y'] < 0 else 'N'
                    bnd[e] += t['Pr']
                for e in 'NSEW':
                    node = stack_nodes.get('{}_{}'.format(s, e), {})
                    p25 = float(node.get('2.5d link Power', 0.0) or 0.0)
                    tid = '{}.phy{}'.format(pref, e)
                    has = d[e] > 1e-12
                    for nm, p in (('link', p25), ('noc_boundary_router', bnd[e])):
                        if p > 0:
                            if not has:
                                raise ValueError('{} edge {}: {} power {} W but no PHY strip'.format(s, e, nm, p))
                            links.append(Link('aib.{}.{}.{}'.format(s, e, nm), 'aib', (ci, die), (ci, die), p_src_W=p,
                                              region={'src_tiles': [tid]}, note='SA 2.5D {} power'.format(nm)))
            # NoP router power of the TR0 chiplet (G_sys 'NoP Router Energy' / 'NoP Router Latency', Network.py:480-559;
            # review v0.2 SA F1): spread over the PHY strips of that die
            if z == 0:
                sn = sys_nodes[c]
                E_nop = float(sn.get('NoP Router Energy', 0.0) or 0.0)
                L_nop = float(sn.get('NoP Router Latency', 0.0) or 0.0)
                p_nop = E_nop / L_nop if (E_nop > 0 and L_nop > 0) else 0.0
                if E_nop > 0 and L_nop <= 0:
                    raise ValueError('{}: NoP Router Energy without latency'.format(c))
                if p_nop > 0:
                    phy = ['{}.phy{}'.format(pref, e) for e in 'NSEW' if g['d'][e] > 1e-12]
                    if not phy:
                        raise ValueError('{}: NoP router power {} W but no PHY strip'.format(c, p_nop))
                    links.append(Link('aib.{}.nop_router'.format(s), 'aib', (ci, die), (ci, die), p_src_W=p_nop,
                                      region={'src_tiles': phy}, note='SA NoP router power E/L (G_sys)'))
                    nop_power[s] = p_nop
    for k, a_ in sys_nodes.items():
        if a_.get('Chiplet ID') == 'Empty' and float(a_.get('NoP Router Energy', 0.0) or 0.0) > 0:
            raise NotImplementedError('empty NoP site {} carries NoP router energy (no die to put it on)'.format(k))
    # EMIB bridges between facing PHY strips of NoP-adjacent stacks (geometry only)
    pos = {chips[stacks[s][0]]['nop']: ci for ci, s in enumerate(st_ids)}
    bridges = []
    for (x, y), ci in pos.items():
        for (dx, dy, ea, eb) in ((1, 0, 'E', 'W'), (0, -1, 'S', 'N')):
            cj = pos.get((x + dx, y + dy))
            if cj is None:
                continue
            ga, gb = geo[st_ids[ci]], geo[st_ids[cj]]
            if ga['d'][ea] <= 0 or gb['d'][eb] <= 0:
                continue
            if ea == 'E':
                x0 = ga['X0'] + ga['Wd'] - ga['d']['E']
                x1 = gb['X0'] + gb['d']['W']
                y0 = max(ga['Y0'] + ga['d']['N'], gb['Y0'] + gb['d']['N'])
                y1 = min(ga['Y0'] + ga['d']['N'] + ga['Hm'], gb['Y0'] + gb['d']['N'] + gb['Hm'])
                rect = (x0, y0, x1 - x0, y1 - y0)
            else:
                y0 = ga['Y0'] + ga['Hd'] - ga['d']['S']
                y1 = gb['Y0'] + gb['d']['N']
                x0 = max(ga['X0'] + ga['d']['W'], gb['X0'] + gb['d']['W'])
                x1 = min(ga['X0'] + ga['d']['W'] + ga['Wm'], gb['X0'] + gb['d']['W'] + gb['Wm'])
                rect = (x0, y0, x1 - x0, y1 - y0)
            if rect[2] > 1e-9 and rect[3] > 1e-9:
                bid = 'emib.{}-{}'.format(st_ids[ci], st_ids[cj])
                tiles.append(Tile(bid, -1, 0, 'emib', rect[0], rect[1], rect[2], rect[3], layer='emib'))
                bridges.append({'id': bid, 'stacks': [st_ids[ci], st_ids[cj]], 'rect_mm': list(rect)})
            else:
                notes.append('no facing overlap for a bridge between {} and {}'.format(st_ids[ci], st_ids[cj]))
    # package gaps: coarse cells (die edges + bbox) outside every die, per tier
    if n_stack > 1:
        ex = sorted({0.0, W} | {v for (x0, y0, w, h) in die_rects.values() for v in (x0, x0 + w)})
        ey = sorted({0.0, H} | {v for (x0, y0, w, h) in die_rects.values() for v in (y0, y0 + h)})
        for i in range(len(ex) - 1):
            for j in range(len(ey) - 1):
                cxm, cym = 0.5 * (ex[i] + ex[i + 1]), 0.5 * (ey[j] + ey[j + 1])
                inside = any(x0 < cxm < x0 + w and y0 < cym < y0 + h for (x0, y0, w, h) in die_rects.values())
                if not inside and ex[i + 1] - ex[i] > 1e-12 and ey[j + 1] - ey[j] > 1e-12:
                    for die in range(n_tier):
                        tiles.append(Tile('gap.d{}.{}_{}'.format(die, i, j), -1, die, 'gap', ex[i], ey[j],
                                          ex[i + 1] - ex[i], ey[j + 1] - ey[j]))
    # independent source ledger straight from the graphs (plan §6.7)
    led = {'tile_dyn': 0.0, 'leak': 0.0, 'router': 0.0, 'tsv': 0.0, 'aib': 0.0, 'emib': 0.0}
    for s in st_ids:
        m = geo[s]['m']
        for c in stacks[s]:
            for t in per_chip[c]:
                inside = 0 <= t['x'] < m and 0 <= t['y'] < m
                if t['last'] != 'Empty':
                    led['tile_dyn'] += p_eff(t)                 # every mapped tile, wherever the slot test says
                if inside:
                    led['router'] += t['Pr']
                    led['tsv'] += t['P3']
                    if t['last'] == 'Empty' and (t['P'] > 0 or t['E'] > 0):
                        raise ValueError('{}: Empty node with tile power'.format(t['id']))
                else:
                    if t['P3'] > 0:
                        raise ValueError('{}: 3D link power on a boundary NoP site'.format(t['id']))
                    if t['Pr'] > 0 and chips[c]['z'] != 0:
                        raise NotImplementedError('{}: NoP boundary router power on an upper tier'.format(t['id']))
                    led['aib'] += t['Pr']
        for e in 'NSEW':
            led['aib'] += float(stack_nodes.get('{}_{}'.format(s, e), {}).get('2.5d link Power', 0.0) or 0.0)
        tr0 = [c for c in stacks[s] if chips[c]['z'] == 0][0]
        En = float(sys_nodes[tr0].get('NoP Router Energy', 0.0) or 0.0)
        Ln = float(sys_nodes[tr0].get('NoP Router Latency', 0.0) or 0.0)
        led['aib'] += En / Ln if (En > 0 and Ln > 0) else 0.0
    cores = [t.w for t in tiles if t.kind in ('sa', 'cpu', 'mem', 'logic')]
    ref = float(np.median(cores)) if cores else max(W, H)
    actual = {s: chips[geo[s]['tr0']]['actual'] for s in st_ids}
    meta = {'adapter': 'sa', 'power_mode': mode, 'gap_mm': gap_mm, 'n_stack': n_stack, 'stacks': st_ids,
            'tier_order': 'SA TR0 (bottom, on the interposer) -> die index n_tier-1; TR(n-1) -> die 0 (top)',
            'die_mm': {s: [geo[s]['Wd'], geo[s]['Hd']] for s in st_ids},
            'die_area_vs_upstream': {s: (geo[s]['Wd'] * geo[s]['Hd'] / actual[s] if actual[s] > 0 else None)
                                     for s in st_ids},
            'phy_depth_mm': {s: geo[s]['d'] for s in st_ids}, 'bridges': bridges, 'notes': notes,
            'geometry': 'ASSUMED: rule={} (slots per NoC position shared by the tiers of a stack, HISIM beachhead '
                        'template, NoP grid placement); see adapter docstring'.format(rule), 'floorplan_rule': rule,
            'input_profile': 'upstream_raw', 'source_ledger': led,
            'l3d': {'upstream_3d_link_area_mm2': l3d_area[0], 'template_strip_area_mm2': l3d_area[1],
                    'note': '3D-link strips modelled as die Si: TSV area fraction {:.2e}'.format(
                        l3d_area[0] / l3d_area[1] if l3d_area[1] else 0.0)},
            'nop_router_power_W': nop_power,
            'power_semantics': 'Tile_Power = average over the tile busy time (Compute.py:401-412), no leakage; '
                               'router / 3D / 2.5D link powers as given by the network model (not corrected)'}
    cb = {ci: die_rects[ci] for ci in die_rects}
    ti = ThermalInput(arch, n_tier, n_stack, 0, (W, H), ref, tiles, links, source='sa', chiplet_bbox=cb, meta=meta)
    ti.meta['layout_hash'] = ti.layout_hash()
    ti.check()
    return ti
