"""HISIM-IMC adapter: thermal_model() arguments + the two Debug CSVs -> ThermalInput (plan §6, §7).

CSV columns are ZERO-BASED indices (as in the code: computing_data[i][k]):
  layer_inform.csv     [1] tiles of the layer, [7] cumulative slots (placement 1-4 includes skipped slots),
                       [9] tier, [15] stack
  layer_performance.csv[2] layer latency L (s), [4] leakage per tile (W), [5] layer-active dynamic power per tile (mW)
Upstream values are used as they are (D15): router / TSV (tier_2d / tier_3d, mW), AIB rows (pJ/ns = mW).
"""
import hashlib
import math

import numpy as np

from ..floorplan import imc_block, layout_2p5d
from ..neutral_input import Link, ThermalInput


def read_csv(path):
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    if a.ndim != 2 or a.shape[0] < 1:
        raise ValueError('empty CSV ' + path)
    return a


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def duty_factors(perf, mode):
    L = perf[:, 2].astype(float)
    if mode == 'concurrent':
        return np.ones(len(perf))
    if mode == 'pipeline':
        return L / L.max()
    if mode == 'time_avg':
        return L / L.sum()
    raise ValueError('unknown power mode ' + repr(mode))


def _orig_order_1to4(inform, NN):
    """Literal original util.py:507-519 on layer indices (None = padding)."""
    out = []
    n = len(inform)
    for i in range(n - 1):
        out += [i] * int(inform[i][1])
        gap = (inform[i][9] + 1) * NN - inform[i][7]
        if gap != 0 and gap < inform[i + 1][1]:
            out += [None] * int(gap)
    out += [n - 1] * int(inform[-1][1])
    gap = (inform[-1][9] + 1) * NN - inform[-1][7]
    if gap != 0:
        out += [None] * int(gap)
    return out


def _grouped(inform, key_col, n_groups, NN, what):
    out = []
    for g in range(n_groups):
        rows = [i for i in range(len(inform)) if int(inform[i][key_col]) == g]
        cnt = sum(int(inform[i][1]) for i in rows)
        if cnt > NN:
            raise ValueError('{} {} holds {} tiles > {} slots'.format(what, g, cnt, NN))
        for i in rows:
            out += [i] * int(inform[i][1])
        out += [None] * (NN - cnt)
    bad = [int(v) for v in inform[:, key_col] if not (0 <= int(v) < n_groups)]
    if bad:
        raise ValueError('{} index out of range in layer_inform.csv: {}'.format(what, sorted(set(bad))))
    return out


def slot_records(inform, perf, n_groups, N, placement, cfg, key_col, notes):
    """Records in slot order (len n_groups * N^2): (p_eff_W, mapped, layer, stack, tier_logical, p_busy_W, duty)."""
    NN = N * N
    order = _grouped(inform, key_col, n_groups, NN, 'tier' if key_col == 9 else 'stack')
    if key_col == 9 and placement in (1, 2, 3, 4):
        lit = _orig_order_1to4(inform, NN)
        if lit != order[:len(lit)] or any(v is not None for v in order[len(lit):]):
            raise ValueError('placement {}: original padding formula and tier grouping disagree '
                             '(layer_inform.csv col 7/9 not consistent)'.format(placement))
        if len(lit) != len(order):
            notes.append('original padding gives {} slots, expected {}: the original would fail its length assert; '
                         'grouping pads the empty tiers'.format(len(lit), len(order)))
    duty = duty_factors(perf, cfg['power']['mode'])
    recs = []
    for l in order:
        if l is None:
            recs.append({'p': 0.0, 'mapped': False, 'layer': None, 'stack': None, 'tier_l': None, 'p_busy': 0.0, 'duty': 0.0})
        else:
            pb = float(perf[l][5]) / 1000.0
            recs.append({'p': pb * float(duty[l]), 'mapped': True, 'layer': int(l), 'stack': int(inform[l][15]),
                         'tier_l': int(inform[l][9]), 'p_busy': pb, 'duty': float(duty[l])})
    if key_col == 9:
        if placement == 2:
            recs = recs[::-1]
        elif placement in (3, 4):
            recs = sorted(recs, key=lambda r: -r['p'])
            a, b = recs[0::2], recs[1::2]
            if placement == 3:
                b = sorted(b, key=lambda r: r['p'])
            recs = a + b
    return recs


def _leak(rec, perf, cfg):
    pw = cfg['power']
    if not pw['include_leakage']:
        return 0.0
    mode = pw['leakage_slots']
    if mode == 'mapped':
        return float(perf[rec['layer']][4]) if rec['mapped'] else 0.0
    if mode == 'all':
        return float(perf[rec['layer'] if rec['mapped'] else 0][4])
    if mode == 'p_positive':                 # v1: leak_tile of row 0 on every slot with power > 0
        return float(perf[0][4]) if rec['p'] > 0 else 0.0
    raise ValueError(mode)


def _slot_dicts(recs, perf, cfg, placement):
    out = []
    for s, r in enumerate(recs):
        out.append({'p_dyn_W': r['p'], 'p_leak_W': _leak(r, perf, cfg),
                    'src': {'layer': r['layer'], 'stack': r['stack'], 'tier_logical': r['tier_l'],
                            'mapped': r['mapped'], 'p_busy_W': r['p_busy'], 'duty': r['duty'],
                            'placement': placement}})
    return out


UPSTREAM_DIRS = ('Module_Compute', 'Module_Network', 'Module_AI_Map')
D15_NOTES = ['NoC (router / TSV) power taken as provided: the ~103x unit issue of the network model is NOT corrected (D15)',
             'tile area (area_single_tile) taken as provided: upstream area over-estimate NOT corrected (D15)',
             'tile power = layer-active (busy) power per tile: concurrent = all layers busy at once (stress case, D3)']


def upstream_md5(root):
    """md5 of every .py file of the upstream models (no git metadata in the copy, plan §1.5)."""
    import os
    out = {}
    if not root:
        return out
    for d in UPSTREAM_DIRS:
        base = os.path.join(root, d)
        for dp, _, fs in os.walk(base):
            for f in sorted(fs):
                if f.endswith('.py'):
                    pth = os.path.join(dp, f)
                    out[os.path.relpath(pth, root)] = md5(pth)
    return out


def _check_layers(tiles, inform, perf, duty, notes):
    """Per AI layer: the tiles that carry it hold exactly n_l slots and p_l * n_l of power."""
    cnt, pw = {}, {}
    for t in tiles:
        l = t.src.get('layer') if t.kind == 'imc' else None
        if l is not None:
            cnt[l] = cnt.get(l, 0) + 1
            pw[l] = pw.get(l, 0.0) + t.p_dyn_W
    for l in range(len(inform)):
        n_l = int(inform[l][1])
        p_l = float(perf[l][5]) / 1000.0 * float(duty[l])
        if cnt.get(l, 0) != n_l or abs(pw.get(l, 0.0) - p_l * n_l) > 1e-12 + 1e-9 * p_l * n_l:
            raise ValueError('AI layer {}: {} slots / {} W placed, expected {} / {} W'.format(
                l, cnt.get(l, 0), pw.get(l, 0.0), n_l, p_l * n_l))


def _source_ledger(inform, perf, duty, cfg, n_slots, t2d_W, tsv_W, aib_rows):
    """Category totals computed straight from the raw upstream inputs (independent of the slot / tile code)."""
    n = inform[:, 1].astype(float)
    p = perf[:, 5].astype(float) / 1000.0 * duty
    pw = cfg['power']
    leak = 0.0
    if pw['include_leakage']:
        if pw['leakage_slots'] == 'mapped':
            leak = float((n * perf[:, 4]).sum())
        elif pw['leakage_slots'] == 'all':
            leak = float((n * perf[:, 4]).sum() + (n_slots - n.sum()) * perf[0, 4])
        else:
            leak = float(n[p > 0].sum() * perf[0, 4])
    f = pw['emib_wire_fraction']
    rows = [{'row': j, 'tx_circuit_W': r['tx_circuit_W'], 'rx_circuit_W': r['rx_circuit_W'], 'p_wire_W': r['p_wire_W'],
             'tx_W': r['tx_circuit_W'] + (1 - f) * r['p_wire_W'], 'rx_W': r['rx_circuit_W'], 'emib_W': f * r['p_wire_W']}
            for j, r in enumerate(aib_rows)]                 # per AIB row: Tx / Rx / P_wire each placed once (plan 6.7)
    return {'tile_dyn': float((n * p).sum()), 'leak': leak, 'router': float(sum(t2d_W)), 'tsv': float(sum(tsv_W)),
            'aib': float(sum(r['tx_W'] + r['rx_W'] for r in rows)), 'emib': float(sum(r['emib_W'] for r in rows)),
            'aib_rows': rows}


def from_imc(chip_architect, chiplet_num_list, N_tile, placement_method, tier_2d_hop_list_power_1,
             tier_3d_hop_list_power_1, area_single_tile, single_router_area, mesh_edge, layer_aib_list,
             inform_csv, perf_csv, cfg, layout='chain', upstream_root=None):
    """ThermalInput for M2D / M3D (layout ignored) or H2_5D (layout 'chain' | 'L' | 'ring')."""
    if placement_method not in (1, 2, 3, 4, 5):
        raise ValueError('placement_method must be 1..5, got {!r}'.format(placement_method))
    N = int(mesh_edge)
    if N < 1 or N * N != int(N_tile):
        raise ValueError('N_tile = {} is not mesh_edge^2 = {}^2'.format(N_tile, mesh_edge))
    if not (area_single_tile > 0 and single_router_area > 0):
        raise ValueError('tile / router area must be > 0')
    inform, perf = read_csv(inform_csv), read_csv(perf_csv)
    if inform.shape[1] < 16 or perf.shape[1] < 6:
        raise ValueError('unexpected CSV widths {} / {}'.format(inform.shape, perf.shape))
    if len(inform) != len(perf):
        raise ValueError('layer_inform.csv and layer_performance.csv have different row counts')
    if not (np.all(np.isfinite(inform)) and np.all(np.isfinite(perf))):
        raise ValueError('non-finite value in the Debug CSVs')
    a = math.sqrt(area_single_tile)
    b = math.sqrt(single_router_area)
    notes = list(D15_NOTES)
    duty = duty_factors(perf, cfg['power']['mode'])
    meta = {'adapter': 'imc', 'chip_architect': chip_architect, 'placement': int(placement_method),
            'power_mode': cfg['power']['mode'], 'a_mm': a, 'b_mm': b, 'N': N,
            'area_single_tile_mm2': float(area_single_tile), 'single_router_area_mm2': float(single_router_area),
            'raw': {'chiplet_num_list': list(chiplet_num_list), 'tier_2d_mW': tier_2d_hop_list_power_1,
                    'tier_3d_mW': tier_3d_hop_list_power_1, 'layer_aib_list': layer_aib_list},
            'csv_md5': {'layer_inform': md5(inform_csv), 'layer_performance': md5(perf_csv)},
            'upstream_md5': upstream_md5(upstream_root), 'input_profile': 'upstream_raw', 'notes': notes}
    if chip_architect in ('M2D', 'M3D'):
        n_tier = int(chiplet_num_list[0])
        if chip_architect == 'M2D' and n_tier != 1:
            raise ValueError('M2D with {} tiers'.format(n_tier))
        t2d = [float(v) for v in tier_2d_hop_list_power_1[0]]
        t3d = [float(v) for v in tier_3d_hop_list_power_1[0]]
        if len(t2d) != n_tier:
            raise ValueError('tier_2d list has {} entries for {} tiers'.format(len(t2d), n_tier))
        if not all(math.isfinite(v) and v >= 0 for v in t2d + t3d):
            raise ValueError('router / TSV power must be finite and >= 0')
        recs = slot_records(inform, perf, n_tier, N, int(placement_method), cfg, 9, notes)
        slots = _slot_dicts(recs, perf, cfg, int(placement_method))
        router_tier = t2d[::-1] if placement_method == 2 else list(t2d)
        if not cfg['power'].get('router_fix', True):
            raise NotImplementedError('router_fix = False (original M05 behaviour) exists only in the legacy preset')
        tiles = []
        for t in range(n_tier):
            tiles += imc_block('t{}'.format(t), 0, t, N, a, b, 0.0, 0.0, slots[t * N * N:(t + 1) * N * N],
                               router_tier[t] / 1000.0 / (N * N))
        links, tsv_W = [], []
        if chip_architect == 'M3D' and n_tier > 1:
            if len(t3d) != n_tier:
                raise ValueError('tier_3d list has {} entries, expected {} (last one is a copy)'.format(len(t3d), n_tier))
            p_if = list(enumerate(t3d[:n_tier - 1]))          # (upstream interface index, mW)
            meta['tier_3d_dropped_last_mW'] = t3d[-1]
            if placement_method == 2:
                p_if = p_if[::-1]                             # physical interface i <- upstream N-2-i
            mode = cfg['power']['tsv_mode']
            for i, (i_up, pmw) in enumerate(p_if):
                p = pmw / 1000.0
                tsv_W.append(p)
                src = [x.id for x in tiles if x.kind == 'tsv' and x.tier == i]
                dst = [x.id for x in tiles if x.kind == 'tsv' and x.tier == i + 1]
                if mode == 'driver':
                    links.append(Link('tsv.{}-{}'.format(i, i + 1), 'tsv', (0, i), (0, i + 1), p_src_W=p / 2,
                                      p_dst_W=p / 2, src_row=i_up, region={'src_tiles': src, 'dst_tiles': dst},
                                      note='driver mode: half to each joined tier; upstream interface {}'.format(i_up)))
                else:
                    links.append(Link('tsv.{}-{}'.format(i, i + 1), 'tsv', (0, i), (0, i + 1), p_mid_W=p, src_row=i_up,
                                      region={'mid_tiles': src, 'mid_path': list(cfg['power']['tsv_via_path'])},
                                      note='via mode: TSV path of the upper tier; upstream interface {}'.format(i_up)))
        _check_layers(tiles, inform, perf, duty, notes)
        meta['source_ledger'] = _source_ledger(inform, perf, duty, cfg, n_tier * N * N,
                                               [v / 1000.0 for v in t2d], tsv_W, [])
        W = N * (a + b)
        meta['layout_recomputed'] = bool(placement_method in (3, 4) and cfg['power']['mode'] != 'concurrent')
        ti = ThermalInput(chip_architect, n_tier, 1, N, (W, W), a, tiles, links, source='imc', meta=meta)
    elif chip_architect == 'H2_5D':
        ti = _from_imc_2p5d(chiplet_num_list, N, placement_method, tier_2d_hop_list_power_1, a, b, layer_aib_list,
                            inform, perf, cfg, layout, meta, notes, duty)
    else:
        raise NotImplementedError('chip_architect {} is not supported by hisim_thermal'.format(chip_architect))
    ti.meta['layout_hash'] = ti.layout_hash()
    ti.check()
    return ti


def _aib_rows(layer_aib_list, n_chip, cfg):
    rows = [[float(v) for v in r] for r in (layer_aib_list or [])]
    if len(rows) != max(n_chip - 1, 0):
        raise ValueError('layer_aib_list has {} rows for {} chiplets (expected {})'.format(len(rows), n_chip, n_chip - 1))
    f = cfg['power']['emib_wire_fraction']
    out = []
    for r in rows:
        if len(r) < 11:
            raise ValueError('layer_aib_list row has {} columns, expected 11'.format(len(r)))
        if not all(math.isfinite(v) and v >= 0 for v in r) or r[9] <= 0 or r[10] <= 0:
            raise ValueError('layer_aib_list row has a non-finite / negative value or a zero latency: {}'.format(r))
        tx_c = r[6] / r[9] / 1000.0          # E_Tx / L_Tx  (pJ/ns = mW) -> W
        rx_c = r[7] / r[10] / 1000.0
        pw = r[8] / 1000.0                   # P_wire (mW) -> W
        out.append({'A_tx': r[3], 'A_rx': r[4], 'A_wire': r[5], 'p_tx': tx_c + (1.0 - f) * pw, 'p_rx': rx_c,
                    'p_emib': f * pw, 'tx_circuit_W': tx_c, 'rx_circuit_W': rx_c, 'p_wire_W': pw})
    return out


def _from_imc_2p5d(chiplet_num_list, N, placement, t2d1, a, b, layer_aib_list, inform, perf, cfg, layout, meta, notes,
                   duty):
    n_chip = int(sum(chiplet_num_list))
    if layout not in ('chain', 'L', 'ring'):
        raise ValueError('unknown 2.5D layout {!r} (chain | L | ring)'.format(layout))
    if (layout == 'L' and n_chip != 3) or (layout == 'ring' and n_chip != 4):
        raise ValueError('layout {} needs {} chiplets, got {}'.format(layout, 3 if layout == 'L' else 4, n_chip))
    t2d = [float(v) for v in sum([list(x) for x in t2d1], [])]
    if len(t2d) != n_chip:
        raise ValueError('router list has {} entries for {} chiplets'.format(len(t2d), n_chip))
    stacks = [int(v) for v in inform[:, 15]]
    seq = [stacks[0]] + [s for i, s in enumerate(stacks[1:], 1) if s != stacks[i - 1]]
    if any(seq[i + 1] != seq[i] + 1 for i in range(len(seq) - 1)) or seq[0] != 0:
        raise ValueError('stack sequence {} is not 0, 1, 2, ... (AIB row j must be the transfer j -> j+1)'.format(seq))
    recs = slot_records(inform, perf, n_chip, N, int(placement), cfg, 15, notes)
    slots = _slot_dicts(recs, perf, cfg, int(placement))
    E = N * (a + b)
    rows = _aib_rows(layer_aib_list, n_chip, cfg)
    if n_chip > 1 and len(seq) - 1 != len(rows):
        raise ValueError('{} stack transitions but {} AIB rows'.format(len(seq) - 1, len(rows)))
    nb = {'chain': n_chip - 1, 'L': 2, 'ring': 4}[layout]
    mode = cfg['power']['aib_area_mode']
    if rows:
        mean_w = (sum(r['A_tx'] for r in rows) / len(rows) / E, sum(r['A_wire'] for r in rows) / len(rows) / E,
                  sum(r['A_rx'] for r in rows) / len(rows) / E)
        if mode == 'mean':
            A_aib = sum(r['A_tx'] + r['A_rx'] for r in rows) / (2 * len(rows))
            widths = [(A_aib / E, mean_w[1], A_aib / E)] * nb
        else:
            widths = [(r['A_tx'] / E, r['A_wire'] / E, r['A_rx'] / E) for r in rows]
            widths += [mean_w] * (nb - len(widths))          # ring closing bridge: no AIB row -> mean geometry
    else:
        widths = []
    origins, btiles, empty, bbox, bridges = layout_2p5d(layout, n_chip, E, widths)
    tiles = []
    for c in range(n_chip):
        x0, y0 = origins[c]
        tiles += imc_block('c{}'.format(c), c, 0, N, a, b, x0, y0, slots[c * N * N:(c + 1) * N * N],
                           t2d[c] / 1000.0 / (N * N))
    tiles += btiles + empty
    byid = {t.id: t for t in tiles}
    links = []
    for (j, txc, rxc) in bridges:
        r = rows[j] if j < len(rows) else None          # ring: 4th bridge has no AIB row -> no power
        reg = {'src_tiles': ['b{}.tx'.format(j)], 'dst_tiles': ['b{}.rx'.format(j)], 'mid_tiles': ['b{}.emib'.format(j)],
               'mid_layer': 'emib'}
        for k in ('src_tiles', 'dst_tiles', 'mid_tiles'):
            reg[k] = [x for x in reg[k] if x in byid]
        p = (r['p_tx'], r['p_rx'], r['p_emib']) if r else (0.0, 0.0, 0.0)
        links.append(Link('aib.{}'.format(j), 'aib', (txc, 0), (rxc, 0), p_src_W=p[0], p_dst_W=p[1], p_mid_W=p[2],
                          src_row=j if r else -1, region=reg, note='' if r else 'ring closing bridge: no AIB row, no power'))

    def area(tid):
        t = byid.get(tid)
        return t.w * t.h if t else 0.0
    bmeta = []
    for (j, txc, rxc) in bridges:
        inp = ({'A_tx': rows[j]['A_tx'], 'A_rx': rows[j]['A_rx'], 'A_wire': rows[j]['A_wire']} if j < len(rows) else None)
        mod = {'A_tx': area('b{}.tx'.format(j)), 'A_rx': area('b{}.rx'.format(j)),
               'A_bridge': area('b{}.emib'.format(j)), 'A_gap': area('b{}.gap'.format(j))}
        w = widths[j] if j < len(widths) else None          # requested strip widths (final review F3 / F4)
        src = ('aib_area_mode=mean' if mode == 'mean' else 'aib row {}'.format(j) if j < len(rows)
               else 'mean of the AIB rows (ring closing bridge, no AIB row)')
        # widths of the modelled strips (strip area / chiplet edge E): in the ring, bridges facing each other share
        # the wider gap, so the modelled gap can exceed the requested one (delta review r_power F1)
        bmeta.append({'j': j, 'tx_chiplet': txc, 'rx_chiplet': rxc, 'input_area_mm2': inp, 'modeled_area_mm2': mod,
                      'strip_width_mm': {'tx': mod['A_tx'] / E, 'gap': mod['A_gap'] / E, 'rx': mod['A_rx'] / E},
                      'requested_width_mm': {'tx': w[0], 'gap': w[1], 'rx': w[2]} if w else None,
                      'width_source': src})
    notes.append('2.5D: tile power injected into the imc region only (original: imc + 2 tsv cells by area), '
                 'intentional change (plan §6.1)')
    _check_layers(tiles, inform, perf, duty, notes)
    meta.update({'layout_2p5d': layout, 'E_mm': E, 'aib_area_mode': mode, 'bridges': bmeta, 'aib_rows': rows,
                 'layout_recomputed': False,
                 'area_vs_ppa': {'package_bbox_mm2': bbox[0] * bbox[1], 'chiplet_blocks_mm2': n_chip * E * E}})
    meta['source_ledger'] = _source_ledger(inform, perf, duty, cfg, n_chip * N * N, [v / 1000.0 for v in t2d], [], rows)
    cb = {c: (origins[c][0], origins[c][1], E, E) for c in range(n_chip)}
    return ThermalInput('H2_5D', 1, n_chip, N, bbox, a, tiles, links, source='imc', chiplet_bbox=cb, meta=meta)
