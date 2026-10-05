"""Floorplan builders that produce neutral-input tiles (plan §5.1-5.2, §7.3).

IMC tile template (original basicblock / snakewalk): per slot at tile row i, column j
    imc (a x a) | tsv (b wide)
    tsv (b high)| router (b x b)
with tile pitch a + b.  Slot order inside a tier / chiplet (original snakewalk and snake_walk_new, start top-left,
first direction down): column j = s // N; row i = s % N for even j, N - 1 - s % N for odd j.

2.5D layouts (original create_global_structure):
  chain : C0 | Tx0 gap0 Rx0 | C1 | Tx1 gap1 Rx1 | C2 ...          (any N)
  L     : C0 (top-left) -> C1 (top-right) -> C2 (bottom-right); empty bottom-left block       (N = 3)
  ring  : C0 TL -> C1 TR -> C2 BR -> C3 BL -> C0 (4th bridge carries no power); empty centre  (N = 4)
"""
from .neutral_input import Tile


def snake_pos(s, N):
    j = s // N
    r = s % N
    i = r if j % 2 == 0 else N - 1 - r
    return i, j


def imc_block(prefix, chiplet, tier, N, a, b, x_off, y_off, slot_recs, router_W, kind_tsv='tsv'):
    """Tiles of one IMC tier / chiplet.  slot_recs[s] = dict(p_dyn_W, p_leak_W, src) for slot s (len N*N)."""
    if len(slot_recs) != N * N:
        raise ValueError('need {} slot records, got {}'.format(N * N, len(slot_recs)))
    tiles = []
    for s in range(N * N):
        i, j = snake_pos(s, N)
        x0, y0 = x_off + j * (a + b), y_off + i * (a + b)
        tag = '{}.s{:03d}'.format(prefix, s)
        rec = slot_recs[s]
        src = dict(rec.get('src', {}))
        src['pos'] = [i, j]
        tiles.append(Tile(tag + '.imc', chiplet, tier, 'imc', x0, y0, a, a, slot=s,
                          p_dyn_W=float(rec['p_dyn_W']), p_leak_W=float(rec.get('p_leak_W', 0.0)), src=src))
        tiles.append(Tile(tag + '.tsvR', chiplet, tier, kind_tsv, x0 + a, y0, b, a, slot=s, src={'pos': [i, j]}))
        tiles.append(Tile(tag + '.tsvB', chiplet, tier, kind_tsv, x0, y0 + a, a, b, slot=s, src={'pos': [i, j]}))
        tiles.append(Tile(tag + '.r', chiplet, tier, 'router', x0 + a, y0 + a, b, b, slot=s,
                          p_dyn_W=float(router_W), src={'pos': [i, j]}))
    return tiles


def _bridge_x(j, x, E, y0, wtx, wgap, wrx, tx_chip, rx_chip, rx_first=False):
    """Vertical strips (span y0..y0+E) starting at x, ordered Tx | gap | Rx (or Rx | gap | Tx)."""
    order = [('rx', wrx), ('gap', wgap), ('tx', wtx)] if rx_first else [('tx', wtx), ('gap', wgap), ('rx', wrx)]
    tiles, xx = [], x
    for part, w in order:
        if w > 0:
            if part == 'gap':
                tiles.append(Tile('b{}.gap'.format(j), -1, 0, 'gap', xx, y0, w, E))
            else:
                tiles.append(Tile('b{}.{}'.format(j, part), tx_chip if part == 'tx' else rx_chip, 0,
                                  'aib_tx' if part == 'tx' else 'aib_rx', xx, y0, w, E))
        xx += w
    if wtx + wgap + wrx > 0:
        tiles.append(Tile('b{}.emib'.format(j), -1, 0, 'emib', x, y0, wtx + wgap + wrx, E, layer='emib'))
    return tiles


def _bridge_y(j, y, E, x0, wtx, wgap, wrx, tx_chip, rx_chip, rx_first=False):
    """Horizontal strips (span x0..x0+E) starting at y, ordered Tx | gap | Rx downward (or reversed)."""
    order = [('rx', wrx), ('gap', wgap), ('tx', wtx)] if rx_first else [('tx', wtx), ('gap', wgap), ('rx', wrx)]
    tiles, yy = [], y
    for part, w in order:
        if w > 0:
            if part == 'gap':
                tiles.append(Tile('b{}.gap'.format(j), -1, 0, 'gap', x0, yy, E, w))
            else:
                tiles.append(Tile('b{}.{}'.format(j, part), tx_chip if part == 'tx' else rx_chip, 0,
                                  'aib_tx' if part == 'tx' else 'aib_rx', x0, yy, E, w))
        yy += w
    if wtx + wgap + wrx > 0:
        tiles.append(Tile('b{}.emib'.format(j), -1, 0, 'emib', x0, y, E, wtx + wgap + wrx, layer='emib'))
    return tiles


def layout_2p5d(kind, n_chip, E, widths):
    """Chiplet origins and bridge tiles.  widths[j] = (w_tx, w_gap, w_rx) in mm for bridge j (j = 0..n_bridge-1).
    Returns (origins {chiplet: (x0, y0)}, bridge_tiles, empty_tiles, bbox (W, H), bridges [(j, tx_chip, rx_chip)])."""
    if kind == 'chain':
        origins, tiles, x = {}, [], 0.0
        bridges = []
        for c in range(n_chip):
            origins[c] = (x, 0.0)
            x += E
            if c < n_chip - 1:
                wtx, wg, wrx = widths[c]
                tiles += _bridge_x(c, x, E, 0.0, wtx, wg, wrx, c, c + 1)
                bridges.append((c, c, c + 1))
                x += wtx + wg + wrx
        return origins, tiles, [], (x, E), bridges
    if kind == 'L':
        if n_chip != 3:
            raise ValueError('L layout needs 3 chiplets')
        (t0, g0, r0), (t1, g1, r1) = widths[0], widths[1]
        w0, w1 = t0 + g0 + r0, t1 + g1 + r1
        origins = {0: (0.0, 0.0), 1: (E + w0, 0.0), 2: (E + w0, E + w1)}
        tiles = _bridge_x(0, E, E, 0.0, t0, g0, r0, 0, 1) + _bridge_y(1, E, E, E + w0, t1, g1, r1, 1, 2)
        empty = [Tile('empty.bl', -1, 0, 'empty', 0.0, E, E + w0, E + w1)]
        return origins, tiles, empty, (2 * E + w0, 2 * E + w1), [(0, 0, 1), (1, 1, 2)]
    if kind == 'ring':
        if n_chip != 4:
            raise ValueError('ring layout needs 4 chiplets')
        (t0, g0, r0), (t1, g1, r1), (t2, g2, r2), (t3, g3, r3) = widths
        wx = max(t0 + g0 + r0, t2 + g2 + r2)      # x gap shared by bridge 0 (top) and bridge 2 (bottom)
        wy = max(t1 + g1 + r1, t3 + g3 + r3)      # y gap shared by bridge 1 (right) and bridge 3 (left)
        origins = {0: (0.0, 0.0), 1: (E + wx, 0.0), 2: (E + wx, E + wy), 3: (0.0, E + wy)}
        # every strip touches its own chiplet; the bridge spans the whole gap between its two strips
        tiles = (_bridge_x(0, E, E, 0.0, t0, wx - t0 - r0, r0, 0, 1)                      # top: C0 (Tx) -> C1 (Rx)
                 + _bridge_y(1, E, E, E + wx, t1, wy - t1 - r1, r1, 1, 2)                 # right: C1 (Tx) -> C2
                 + _bridge_x(2, E, E, E + wy, t2, wx - t2 - r2, r2, 2, 3, rx_first=True)  # bottom: C3 (Rx) <- C2
                 + _bridge_y(3, E, E, 0.0, t3, wy - t3 - r3, r3, 3, 0, rx_first=True))   # left: C0 (Rx) <- C3
        empty = [Tile('empty.c', -1, 0, 'empty', E, E, wx, wy)]
        return origins, tiles, empty, (2 * E + wx, 2 * E + wy), [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 0)]
    raise ValueError('unknown 2.5D layout ' + repr(kind))
