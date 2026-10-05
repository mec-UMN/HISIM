"""Publication figures for hisim_thermal result directories.

Every field map is drawn with ``pcolormesh`` on the true (strongly non-uniform)
cell edges, so tile and strip dimensions are physically faithful.  Plan views
use the solver convention: x to the right, y DOWN, origin at the top-left of
the package bounding box.  Vertical sections use a *stretched* z axis (every
physical layer gets a minimum band height) with the true thicknesses printed
next to the axis.

Public API
----------
load_result(dir)                 -> dict of arrays + json
make_all(dir, out_dir=None)      -> list of written files
plot_parity / plot_diff_map / plot_gci / plot_tornado  (validation helpers)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
import functools  # noqa: E402

DPI = 200
CMAP_T = 'inferno'
CMAP_Q = 'viridis'
MAIN_KINDS = ('imc', 'sa', 'cpu', 'mem', 'logic', 'empty')
LINK_STYLE = {   # kind -> (edge colour, line style, label)
    'aib_tx': ('#4fc3f7', '-', 'AIB Tx'),
    'aib_rx': ('#81d4fa', '--', 'AIB Rx'),
    'gap': ('#bdbdbd', ':', 'gap (mold)'),
    'emib': ('#76ff03', '-', 'EMIB bridge'),
}
ROLE_COLOR = {   # material colour for the stack diagram (by layer role)
    'spreader': '#c77c3a', 'tim1': '#9e9e9e', 'beol': '#7fb3d5', 'active': '#e74c3c',
    'si': '#5d6d7e', 'bond': '#f5cba7', 'c4': '#d4ac0d', 'emib': '#58d68d',
    'subs': '#27ae60',
}
STYLE = {
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.direction': 'out',
    'ytick.direction': 'out', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.titleweight': 'bold',
}


def _styled(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        with plt.rc_context(STYLE):
            return fn(*a, **k)
    return wrap


def _cbar(fig, ax, im, label, size='3.5%', pad=0.08):
    cax = make_axes_locatable(ax).append_axes('right', size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=6.5)
    if hasattr(cb.formatter, 'set_useOffset'):
        cb.formatter.set_useOffset(False)
        cb.update_ticks()
    return cb


# ----------------------------------------------------------------------------- loading
def _json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_result(d):
    """Load a result directory into a dict (lengths: x/y in mm, z in um)."""
    g = np.load(os.path.join(d, 'grid.npz'))
    r = {'dir': d,
         'xe': np.asarray(g['x_edges'], float) * 1e3,
         'ye': np.asarray(g['y_edges'], float) * 1e3,
         'ze': np.asarray(g['z_edges'], float) * 1e6}
    for key, fn in (('T', 'T_full.npy'), ('P', 'P_full.npy'), ('K', 'K_full.npy'), ('N', 'N_full.npy')):
        p = os.path.join(d, fn)
        r[key] = np.load(p) if os.path.exists(p) else None
    r['regions'] = {int(k): v for k, v in (_json(os.path.join(d, 'regions.json'), {}) or {}).items()}
    r['layers'] = _json(os.path.join(d, 'layers.json'))
    r['meta'] = _json(os.path.join(d, 'meta.json'), {})
    r['peaks'] = _json(os.path.join(d, 'peaks.json'), {})
    r['tiles'] = _json(os.path.join(d, 'tiles.json'), [])
    r['input'] = _json(os.path.join(d, 'input.json'), {})
    r['phys'] = r['layers']['phys_layers']
    r['qsurf'] = {}
    for p in r['phys']:
        fp = os.path.join(d, 'qsurf_%s.npy' % p['name'])
        if os.path.exists(fp):
            r['qsurf'][p['name']] = np.load(fp)
    inp = r['meta'].get('input', {}) or {}
    r['arch'] = inp.get('arch', r['input'].get('arch', '?'))
    r['source'] = inp.get('source', r['input'].get('source', '?'))
    r['bc'] = (r['meta'].get('config', {}) or {}).get('bc', {})
    return r


def layer_mean(r, p, field='T'):
    """Thickness-weighted mean of a [nk,ny,nx] field over one physical layer."""
    k0, k1 = p['k_range']
    dz = np.diff(r['ze'])[k0:k1]
    return np.tensordot(dz / dz.sum(), r[field][k0:k1], axes=(0, 0))


def _title(r):
    m = r['meta'].get('input', {}) or {}
    s = '%s · %s' % (r['arch'], str(r['source']).upper())
    if m.get('n_tier', 1) > 1:
        s += ' · %d tiers' % m['n_tier']
    if m.get('chiplet_num', 1) > 1:
        s += ' · %d chiplets' % m['chiplet_num']
    return s


def _save(fig, out_dir, stem, pdf=False):
    files = []
    p = os.path.join(out_dir, stem + '.png')
    fig.savefig(p, dpi=DPI)
    files.append(p)
    if pdf:
        p = os.path.join(out_dir, stem + '.pdf')
        fig.savefig(p)
        files.append(p)
    plt.close(fig)
    return files


def _fmt_um(t_um):
    if t_um >= 999.5:
        return '%.3g mm' % (t_um / 1000)
    return '%.3g µm' % t_um


# ----------------------------------------------------------------------------- geometry
def _in_tiles(r):
    return r['input'].get('tiles', []) or []


def chiplet_boxes(r):
    """{chiplet: (x0, y0, w, h)} in mm."""
    cb = r['input'].get('chiplet_bbox') or {}
    if cb:
        return {int(k): tuple(v) for k, v in cb.items()}
    out = {}
    for t in _in_tiles(r):
        c = t.get('chiplet', -1)
        if c is None or c < 0:
            continue
        x0, y0, x1, y1 = t['x0'], t['y0'], t['x0'] + t['w'], t['y0'] + t['h']
        if c in out:
            a = out[c]
            out[c] = (min(a[0], x0), min(a[1], y0), max(a[2], x1), max(a[3], y1))
        else:
            out[c] = (x0, y0, x1, y1)
    return {c: (a[0], a[1], a[2] - a[0], a[3] - a[1]) for c, a in out.items()}


def layer_tiles(r, p):
    """Neutral-input tiles that live in physical layer ``p``."""
    tl = _in_tiles(r)
    if p['role'] == 'emib':
        return [t for t in tl if t.get('layer') == 'emib']
    if p.get('die') is None:
        return []
    return [t for t in tl if t.get('layer', 'device') == 'device' and t.get('tier') == p['die']]


def _short_id(tid):
    parts = tid.split('.')
    return ('.'.join(parts[1:-1]) + '\n' + parts[-1]) if len(parts) > 2 else tid


def draw_outlines(ax, r, p=None, tiles=True, ids=None, color='white', link_tiles=None):
    """Tile outlines (thin), chiplet outlines (bold) and AIB / gap / EMIB outlines."""
    tl = layer_tiles(r, p) if (p is not None and tiles) else []
    main = [t for t in tl if t['kind'] in MAIN_KINDS]
    if main:
        pc = PatchCollection([Rectangle((t['x0'], t['y0']), t['w'], t['h']) for t in main],
                             facecolor='none', edgecolor=color, linewidth=0.3, alpha=0.55)
        ax.add_collection(pc)
    if ids is None:
        ids = 0 < len(main) < 40
    if ids:
        bw = r['xe'][-1] - r['xe'][0]
        for t in main:
            if min(t['w'], t['h']) < 0.06 * bw:
                continue
            ax.text(t['x0'] + t['w'] / 2, t['y0'] + t['h'] / 2, _short_id(t['id']), ha='center',
                    va='center', fontsize=4.5, color=color, alpha=0.85, clip_on=True)
    for c, (x0, y0, w, h) in chiplet_boxes(r).items():
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor='none', edgecolor='#00e5ff',
                               linewidth=0.9, alpha=0.9))
    if link_tiles is None:
        if p is None:
            link_tiles = [t for t in _in_tiles(r) if t['kind'] in LINK_STYLE]
        elif p['role'] == 'emib' or p.get('die') is None:
            link_tiles = [t for t in _in_tiles(r) if t['kind'] in ('emib', 'gap')]
        else:
            link_tiles = [t for t in tl if t['kind'] in LINK_STYLE]
    seen = set()
    for t in link_tiles:
        ec, ls, lab = LINK_STYLE[t['kind']]
        ax.add_patch(Rectangle((t['x0'], t['y0']), t['w'], t['h'], facecolor='none', edgecolor=ec,
                               linestyle=ls, linewidth=0.8,
                               label=None if lab in seen else lab))
        seen.add(lab)
    return seen


def setup_plan(ax, r):
    ax.set_xlim(r['xe'][0], r['xe'][-1])
    ax.set_ylim(r['ye'][-1], r['ye'][0])        # y grows downward (origin top-left)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)  ↓')


def mesh(ax, r, A, **kw):
    kw.setdefault('shading', 'flat')
    kw.setdefault('rasterized', True)
    return ax.pcolormesh(r['xe'], r['ye'], A, **kw)


def _peak_groups(r, phys_name=None):
    """Peaks from peaks.json grouped by cell; optionally only those in one layer."""
    groups = {}
    for cls, v in (r['peaks'] or {}).items():
        if not v or not v.get('valid'):
            continue
        if phys_name is not None and v.get('layer') != phys_name:
            continue
        key = (v['k'], v['iy'], v['ix'])
        groups.setdefault(key, {'v': v, 'cls': []})['cls'].append(cls)
    return list(groups.values())


def mark_peaks(ax, r, phys_name, A=None, fs=6, legend=True, text=True):
    """Star + label for every peaks.json class in this layer; open circle for the map max."""
    first = True
    for g in _peak_groups(r, phys_name):
        v = g['v']
        lab = 'global max' if 'domain' in g['cls'] else ' / '.join(c + ' peak' for c in g['cls'])
        ax.plot(v['x_mm'], v['y_mm'], marker='*', ms=7, mfc='#00e676', mec='k', mew=0.5, zorder=6, ls='none',
                label='peak cell (peaks.json)' if (first and legend) else None)
        first = False
        if text:
            _label(ax, v['x_mm'], v['y_mm'], '%s\n%.2f K' % (lab, v['T_K']), fs)
    if A is not None:
        iy, ix = np.unravel_index(np.nanargmax(A), A.shape)
        xc = 0.5 * (r['xe'][ix] + r['xe'][ix + 1])
        yc = 0.5 * (r['ye'][iy] + r['ye'][iy + 1])
        ax.plot(xc, yc, marker='o', ms=6, mfc='none', mec='#00e676', mew=1.0, zorder=5, ls='none',
                label='max of this map' if legend else None)
        if text and not _peak_groups(r, phys_name):
            _label(ax, xc, yc, 'layer max\n%.2f K' % A[iy, ix], fs)


def _label(ax, x, y, s, fs, extra=0):
    placed = ax.__dict__.setdefault('_hs_placed', [])
    px, py = ax.transData.transform((x, y))
    extra += 16 * sum(1 for (qx, qy) in placed if abs(qx - px) < 90 and abs(qy - py) < 40)
    placed.append((px, py))
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()
    fx = (x - x0) / (x1 - x0)
    fy = (y - y0) / (y1 - y0)
    dx = -10 if fx > 0.6 else 10
    dy = -(10 + extra) if fy > 0.6 else (10 + extra)   # axis inverted: +offset in points goes up
    ax.annotate(s, (x, y), xytext=(dx, -dy), textcoords='offset points', fontsize=fs,
                ha='right' if dx < 0 else 'left', va='center', color='k', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85),
                arrowprops=dict(arrowstyle='-', lw=0.5, color='white'))


# ----------------------------------------------------------------------------- stretched z
def _band_weights(t, min_frac):
    """Band fractions: proportional to t, but no band below min_frac (sum = 1)."""
    n = len(t)
    min_frac = min(min_frac, 0.8 / n)
    small = np.zeros(n, bool)
    for _ in range(n + 1):
        rest = 1.0 - min_frac * small.sum()
        w = np.where(small, min_frac, rest * t / max(t[~small].sum(), 1e-30))
        new = small | (w < min_frac)
        if (new == small).all():
            break
        small = new
    return w


def stretched_z(r, min_frac=0.06):
    """Map sublayer z edges to a display coordinate in [0, 1] (0 = top surface).

    Each physical layer gets a band proportional to its true thickness but at
    least ``min_frac`` of the total (bands are then renormalised); sublayers
    are spaced linearly inside their band.  Returns (zdisp_edges, bands) with
    bands = [(name, d_top, d_bot, true_thickness_um)].
    """
    ze = r['ze']
    phys = r['phys']
    tot = ze[-1] - ze[0]
    w = _band_weights(np.array([ze[p['k_range'][1]] - ze[p['k_range'][0]] for p in phys]) / tot, min_frac)
    zd = np.zeros_like(ze)
    bands, top = [], 0.0
    for p, wi in zip(phys, w):
        k0, k1 = p['k_range']
        frac = (ze[k0:k1 + 1] - ze[k0]) / (ze[k1] - ze[k0])
        zd[k0:k1 + 1] = top + wi * frac
        bands.append((p['name'], top, top + wi, ze[k1] - ze[k0]))
        top += wi
    return zd, bands


def draw_section(ax, r, A2, zd, bands, cmap=CMAP_T, norm=None, labels=True, true_t=True, lw=0.35):
    """pcolormesh of an x–z slice on true x and stretched z, with layer labels."""
    im = ax.pcolormesh(r['xe'], zd, A2, cmap=cmap, norm=norm, shading='flat', rasterized=True)
    ax.set_ylim(1, 0)
    ax.set_xlim(r['xe'][0], r['xe'][-1])
    for _, a, b, _t in bands[1:]:
        ax.axhline(a, color='white', lw=lw, alpha=0.7)
    centers = [0.5 * (a + b) for _, a, b, _ in bands]
    ax.set_yticks(centers)
    ax.set_yticklabels([n for n, *_ in bands] if labels else [], fontsize=5.5)
    ax.tick_params(axis='y', length=0)
    if true_t:
        ax2 = ax.twinx()
        ax2.set_ylim(1, 0)
        ax2.set_yticks(centers)
        ax2.set_yticklabels([_fmt_um(t) for *_, t in bands], fontsize=5.5, color='#444')
        ax2.tick_params(axis='y', length=0)
        for s in ax2.spines.values():
            s.set_visible(False)
    ax.set_xlabel('x (mm)')
    return im


# ----------------------------------------------------------------------------- plan views
def _plan_fig(r, ncols=1):
    bw, bh = r['xe'][-1] - r['xe'][0], r['ye'][-1] - r['ye'][0]
    asp = bh / bw
    w = 4.6 if asp > 0.7 else 6.4
    return plt.figure(figsize=(w, max(2.2, (w - 1.0) * asp + 0.9)))


def _legend_links(ax, seen):
    h, l = ax.get_legend_handles_labels()
    if seen and h:
        fig = ax.figure
        h_pt = ax.get_position().height * fig.get_figheight() * 72
        ax.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, -30.0 / max(h_pt, 1)), ncol=min(len(h), 5), frameon=False,
                  fontsize=6, handlelength=1.6, columnspacing=1.2)


def _zoom_box(r, p):
    """(x0, x1, y0, y1) zoom window on the chiplet-to-chiplet bridge, or None."""
    win = _link_window(r)
    if win is None or (p.get('die') is None and p['role'] != 'emib'):
        return None
    pk = r['peaks'] or {}
    yc = None
    for c in ('aib', 'emib'):
        if pk.get(c, {}).get('valid'):
            yc = pk[c]['y_mm']
            break
    if yc is None:
        yc = 0.5 * (r['ye'][0] + r['ye'][-1])
    hh = 1.6 * (win[1] - win[0])
    y0 = min(max(yc - hh, r['ye'][0]), r['ye'][-1] - 2 * hh)
    return win[0], win[1], y0, y0 + 2 * hh


def _plan_axes(r, p):
    zb = _zoom_box(r, p)
    if zb is None:
        fig = _plan_fig(r)
        return fig, fig.add_subplot(111), None, None
    bw, bh = r['xe'][-1] - r['xe'][0], r['ye'][-1] - r['ye'][0]
    W = 7.4
    fig = plt.figure(figsize=(W, max(2.6, (W - 2.4) * bh / bw + 1.0)))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.0], wspace=0.28)
    return fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), zb


def _draw_zoom(r, p, ax, axz, zb, A, **kw):
    mesh(axz, r, A, **kw)
    draw_outlines(axz, r, p, ids=False)
    axz.set_xlim(zb[0], zb[1])
    axz.set_ylim(zb[3], zb[2])
    axz.set_aspect('equal')
    axz.set_xlabel('x (mm)')
    axz.set_title('zoom: bridge region\n(true scale)', fontsize=7, fontweight='normal')
    axz.tick_params(labelsize=6)
    ax.add_patch(Rectangle((zb[0], zb[2]), zb[1] - zb[0], zb[3] - zb[2], fc='none', ec='#ff4081', lw=0.9,
                           zorder=8))
    # name the lateral columns (region codes of this layer) along the zoom's upper part
    yc = zb[2] + 0.12 * (zb[3] - zb[2])
    iy = int(np.clip(np.searchsorted(r['ye'], yc) - 1, 0, len(r['ye']) - 2))
    xc = 0.5 * (r['xe'][:-1] + r['xe'][1:])
    sel = np.where((xc > zb[0]) & (xc < zb[1]))[0]
    if len(sel):
        codes = r['N'][p['k_range'][0], iy, sel]
        s = 0
        for i in range(1, len(sel) + 1):
            if i == len(sel) or codes[i] != codes[s]:
                x = 0.5 * (r['xe'][sel[s]] + r['xe'][sel[i - 1] + 1])
                axz.text(x, yc, r['regions'].get(int(codes[s]), '?'), rotation=90, ha='center', va='top',
                         fontsize=4.8, color='white', zorder=9,
                         bbox=dict(boxstyle='round,pad=0.1', fc='k', ec='none', alpha=0.45))
                s = i


@_styled
def plot_plan(r, p, out_dir, pdf=False):
    A = layer_mean(r, p, 'T')
    fig, ax, axz, zb = _plan_axes(r, p)
    norm = mcolors.Normalize(A.min(), A.max())
    im = mesh(ax, r, A, cmap=CMAP_T, norm=norm)
    setup_plan(ax, r)
    seen = draw_outlines(ax, r, p)
    mark_peaks(ax, r, p['name'], A)
    if axz is not None:
        _draw_zoom(r, p, ax, axz, zb, A, cmap=CMAP_T, norm=norm)
        mark_peaks(axz, r, p['name'], None, legend=False, text=False)
        _cbar(fig, axz, im, 'T (K)', size='12%', pad=0.06)
    else:
        _cbar(fig, ax, im, 'T (K)')
    t_um = r['ze'][p['k_range'][1]] - r['ze'][p['k_range'][0]]
    nsub = p['k_range'][1] - p['k_range'][0]
    ax.set_title('%s — layer %s (%s, %d sublayer%s)\n%s; min %.2f K, max %.2f K'
                 % (_title(r), p['name'], _fmt_um(t_um), nsub, '' if nsub == 1 else 's',
                    'T' if nsub == 1 else 'thickness-weighted mean T', A.min(), A.max()),
                 fontsize=7.5, fontweight='normal')
    _legend_links(ax, True)
    return _save(fig, out_dir, 'plan_%s' % p['name'], pdf)


@_styled
def plot_qsurf(r, name, q, out_dir, pdf=False):
    p = next(pp for pp in r['phys'] if pp['name'] == name)
    fig, ax, axz, zb = _plan_axes(r, p)
    pos = q[q > 0]
    vmax = float(q.max()) if q.size else 1.0
    area = np.diff(r['ye'])[:, None] * np.diff(r['xe'])[None, :]
    hot_frac = float(area[q > 0.1 * vmax].sum() / area.sum()) if vmax > 0 else 1.0
    use_log = vmax > 0 and hot_frac < 0.05 and pos.size and pos.min() < 1e-2 * vmax
    if use_log:   # power concentrated in a few small blocks: log scale keeps the rest readable
        norm = mcolors.LogNorm(max(pos.min(), vmax * 1e-4), vmax)
        q = np.where(q > 0, q, np.nan)
    else:
        norm = mcolors.Normalize(0, vmax if vmax > 0 else 1)
    im = mesh(ax, r, q, cmap=CMAP_Q, norm=norm)
    if use_log:
        ax.set_facecolor('#d9d9d9')
    setup_plan(ax, r)
    seen = draw_outlines(ax, r, p, color='white')
    if axz is not None:
        _draw_zoom(r, p, ax, axz, zb, q, cmap=CMAP_Q, norm=norm)
        _cbar(fig, axz, im, 'power density (W/cm²)' + (', log' if use_log else ''), size='12%', pad=0.06)
    else:
        _cbar(fig, ax, im, 'power density (W/cm²)' + (', log scale; grey = 0' if use_log else ''))
    q0 = np.nan_to_num(q)
    ptot = float((q0 * area * 1e-2).sum())   # mm² -> cm²
    qmean = float((q0 * area).sum() / area[q0 > 0].sum()) if (q0 > 0).any() else 0.0
    ax.set_title('%s — %s power density\ntotal %.4g W; max %.4g W/cm²; mean over powered area %.4g W/cm²'
                 % (_title(r), name, ptot, vmax, qmean),
                 fontsize=7.5, fontweight='normal')
    _legend_links(ax, seen)
    return _save(fig, out_dir, 'qsurf_%s' % name, pdf)


# ----------------------------------------------------------------------------- overview
def _actives(r):
    return [p for p in r['phys'] if p['role'] == 'active']


def _global_peak(r):
    pk = r['peaks'] or {}
    for c in ('domain', 'device', 'active'):
        if pk.get(c, {}).get('valid'):
            return pk[c]
    k, iy, ix = np.unravel_index(np.argmax(r['T']), r['T'].shape)
    return {'k': int(k), 'iy': int(iy), 'ix': int(ix), 'T_K': float(r['T'][k, iy, ix]),
            'x_mm': 0.5 * (r['xe'][ix] + r['xe'][ix + 1]), 'y_mm': 0.5 * (r['ye'][iy] + r['ye'][iy + 1])}


@_styled
def plot_overview(r, out_dir, pdf=True):
    acts = _actives(r)
    maps = [layer_mean(r, p, 'T') for p in acts]
    gp = _global_peak(r)
    iy = gp['iy']
    sec = r['T'][:, iy, :]
    vmin = min(min(m.min() for m in maps), sec.min())
    vmax = max(max(m.max() for m in maps), sec.max())
    norm = mcolors.Normalize(vmin, vmax)
    n = len(acts)
    bw, bh = r['xe'][-1] - r['xe'][0], r['ye'][-1] - r['ye'][0]
    W = 7.2
    pw = (W - 0.9) / max(n, 1)
    ph = min(pw * bh / bw, 3.4)
    sh = max(2.4, 0.15 * len(r['phys']))
    fig = plt.figure(figsize=(W, ph + sh + 0.9))
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.04], height_ratios=[ph, sh],
                          hspace=0.45, wspace=0.25)
    im = None
    for i, (p, A) in enumerate(zip(acts, maps)):
        ax = fig.add_subplot(gs[0, i])
        im = mesh(ax, r, A, cmap=CMAP_T, norm=norm)
        setup_plan(ax, r)
        draw_outlines(ax, r, p, ids=False)
        ax.axhline(gp['y_mm'], color='#00e5ff', lw=0.6, ls='--')
        iyx = np.unravel_index(np.argmax(A), A.shape)
        ax.plot(0.5 * (r['xe'][iyx[1]] + r['xe'][iyx[1] + 1]), 0.5 * (r['ye'][iyx[0]] + r['ye'][iyx[0] + 1]),
                marker='*' if p['name'] == gp.get('layer', acts[0]['name']) else 'o',
                ms=6, mfc='#00e676' if p['name'] == gp.get('layer') else 'none', mec='#00e676', mew=0.8)
        tier = '' if p.get('die') is None else ' (die %d%s)' % (p['die'], ', top' if p['die'] == 0 and n > 1
                                                             else (', bottom' if p['die'] == n - 1 and n > 1 else ''))
        ax.set_title('%s%s\nmax %.2f K' % (p['name'], tier, A.max()), fontsize=7)
        if i:
            ax.set_ylabel('')
    if n == 1:
        _cbar(fig, ax, im, 'T (K)')
    else:
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, n]))
        cb.set_label('T (K)')
        cb.formatter.set_useOffset(False)
        cb.update_ticks()
    ax = fig.add_subplot(gs[1, :n])
    zd, bands = stretched_z(r)
    draw_section(ax, r, sec, zd, bands, norm=norm)
    ax.plot(gp['x_mm'], 0.5 * (zd[gp['k']] + zd[gp['k'] + 1]), marker='*', ms=8, mfc='#00e676',
            mec='k', mew=0.5)
    ax.set_title('x–z section at y = %.2f mm (row iy = %d, through the global hotspot %.2f K in %s)'
                 % (gp['y_mm'], iy, gp['T_K'], gp.get('layer', '?')), fontsize=7)
    ax.text(0.0, -0.75 / sh, 'Vertical axis stretched: every physical layer is drawn with a band of at least %.1f %% of '
            'the height (true stack %s); true layer thicknesses are given on the right. '
            'Horizontal axis is to scale.' % (100 * min(b - a for _, a, b, _ in bands),
                                              _fmt_um(r['ze'][-1] - r['ze'][0])),
            transform=ax.transAxes, fontsize=6, color='#333', va='top')
    bc = r['bc'] or {}
    fig.suptitle('%s — temperature overview  (T_amb %.1f K, h_top %.3g, h_bottom %.3g W/m²K)'
                 % (_title(r), bc.get('T_amb', float('nan')), bc.get('h_top', float('nan')),
                    bc.get('h_bottom', float('nan'))), fontsize=8.5, fontweight='bold', y=1.0)
    return _save(fig, out_dir, 'overview', pdf)


# ----------------------------------------------------------------------------- stack
def layer_k_summary(r, p):
    """(k_main, main_name, [(name, k), ...]) for one physical layer (area weighted)."""
    k0 = p['k_range'][0]
    N, K = r['N'][k0], r['K'][k0]
    area = np.diff(r['ye'])[:, None] * np.diff(r['xe'])[None, :]
    stats = []
    for c in np.unique(N):
        m = N == c
        stats.append((float(area[m].sum()), r['regions'].get(int(c), str(c)), float(np.median(K[m]))))
    stats.sort(reverse=True)
    kmain = stats[0][2]
    extra, seen = [], {round(kmain, 3)}
    for a, nm, kv in stats[1:]:
        if round(kv, 3) not in seen:
            extra.append((nm, kv))
            seen.add(round(kv, 3))
    return kmain, stats[0][1], extra


def _section_row(r):
    pk = r['peaks'] or {}
    for c in (('aib', 'emib') if _link_window(r) else ()):
        if pk.get(c, {}).get('valid'):
            return pk[c]['iy'], c
    return _global_peak(r)['iy'], 'global hotspot'


def _link_window(r):
    lt = [t for t in _in_tiles(r) if t['kind'] in LINK_STYLE]
    if not any(t['kind'] in ('gap', 'emib') for t in lt):
        return None
    t0 = lt[0]
    grp = [t for t in lt if abs(t['y0'] - t0['y0']) < 1e-9 and abs(t['x0'] - t0['x0']) < 2.0]
    x0 = min(t['x0'] for t in grp)
    x1 = max(t['x0'] + t['w'] for t in grp)
    pad = max(1.0 * (x1 - x0), 0.15)
    return x0 - pad, x1 + pad


@_styled
def plot_stack(r, out_dir, pdf=True):
    zd, bands = stretched_z(r)
    win = _link_window(r)
    ncol = 3 if win else 2
    H = max(3.6, 0.2 * len(bands) + 1.2)
    fig = plt.figure(figsize=(7.6 if ncol == 2 else 9.4, H))
    wr = [1.15, 2.2, 0.04] if ncol == 2 else [1.15, 2.0, 1.0, 0.04]
    gs = fig.add_gridspec(1, ncol + 1, width_ratios=wr, wspace=0.35 if ncol == 2 else 0.5)
    ax = fig.add_subplot(gs[0, 0])
    for p, (nm, a, b, t) in zip(r['phys'], bands):
        ax.add_patch(Rectangle((0, a), 0.36, b - a, fc=ROLE_COLOR.get(p['role'], '#ccc'), ec='white', lw=0.6))
        km, mn, extra = layer_k_summary(r, p)
        ax.text(0.04, 0.5 * (a + b), nm, va='center', fontsize=6, color='k' if p['role'] not in
                ('si', 'active', 'spreader') else 'white', fontweight='bold')
        ex = ', '.join('%s %.3g' % (e, v) for e, v in extra[:3]) + (' …' if len(extra) > 3 else '')
        ax.text(0.39, 0.5 * (a + b), '%s · k %.3g%s' % (_fmt_um(t), km, ('\n[%s]' % ex) if ex else ''),
                va='center', fontsize=5 if ex else 5.5, clip_on=False, linespacing=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis('off')
    ax.set_title('stack (top → bottom), thickness · k (W/mK)\n[other in-layer materials / regions]',
                 fontsize=7, loc='left', fontweight='normal')
    iy, why = _section_row(r)
    sec = r['K'][:, iy, :]
    norm = mcolors.LogNorm(max(sec.min(), 1e-3), sec.max())
    ax1 = fig.add_subplot(gs[0, 1])
    im = draw_section(ax1, r, sec, zd, bands, cmap='cividis', norm=norm, labels=False, true_t=False)
    ax1.set_title('k on the x–z section at y = %.2f mm (%s row)' % (0.5 * (r['ye'][iy] + r['ye'][iy + 1]), why),
                  fontsize=7)
    if win:
        ax2 = fig.add_subplot(gs[0, 2])
        draw_section(ax2, r, sec, zd, bands, cmap='cividis', norm=norm, labels=False, true_t=False)
        ax2.set_xlim(*win)
        ax2.set_title('zoom: chiplet edge / AIB / gap / EMIB', fontsize=7)
        ax1.axvspan(*win, fc='none', ec='#ff4081', lw=0.8)
        _label_columns(r, ax2, iy, win, zd)
    cax = fig.add_subplot(gs[0, ncol])
    fig.colorbar(im, cax=cax).set_label('k (W/mK), log scale')
    fig.suptitle('%s — layer stack and materials (true stack %s; z stretched: each layer ≥ %.1f %% of the height, '
                 'x to scale)' % (_title(r), _fmt_um(r['ze'][-1] - r['ze'][0]),
                                  100 * min(b - a for _, a, b, _ in bands)), fontsize=8.5, y=1.02)
    return _save(fig, out_dir, 'stack', pdf)


def _label_columns(r, ax, iy, win, zd):
    """Name the lateral columns (regions) of the device layer and the EMIB layer inside a zoom."""
    xc = 0.5 * (r['xe'][:-1] + r['xe'][1:])
    sel = np.where((xc > win[0]) & (xc < win[1]))[0]
    for p in r['phys']:
        if p['role'] not in ('active', 'emib'):
            continue
        k = p['k_range'][0]
        codes = r['N'][k, iy, sel]
        runs, s = [], 0
        for i in range(1, len(sel) + 1):
            if i == len(sel) or codes[i] != codes[s]:
                runs.append((sel[s], sel[i - 1], int(codes[s])))
                s = i
        ymid = 0.5 * (zd[k] + zd[p['k_range'][1]])
        for a, b, c in runs:
            x = 0.5 * (r['xe'][a] + r['xe'][b + 1])
            ax.text(x, ymid, r['regions'].get(c, str(c)), rotation=90, ha='center', va='center', fontsize=5,
                    color='white', bbox=dict(boxstyle='round,pad=0.1', fc='k', ec='none', alpha=0.45))


# ----------------------------------------------------------------------------- tile labels
def _text_color(cmap, norm, v):
    rgb = plt.get_cmap(cmap)(norm(v))[:3]
    return 'k' if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 0.5 else 'white'


def _decimals(span):
    return 1 if span >= 2 else (2 if span >= 0.2 else 3)


def _tile_panel(ax, r, sub, key, cmap, norm, dec):
    bw, bh = r['xe'][-1] - r['xe'][0], r['ye'][-1] - r['ye'][0]
    ax.add_patch(Rectangle((r['xe'][0], r['ye'][0]), bw, bh, fc='#eeeeee', ec='none'))
    pc = PatchCollection([Rectangle((t['x0_mm'], t['y0_mm']), t['w_mm'], t['h_mm']) for t in sub],
                         cmap=cmap, norm=norm, edgecolor='white', linewidth=0.3)
    pc.set_array(np.array([t[key] for t in sub]))
    ax.add_collection(pc)
    for t in sub:
        if not t.get('mapped', True):
            ax.add_patch(Rectangle((t['x0_mm'], t['y0_mm']), t['w_mm'], t['h_mm'], fc='none',
                                   ec='#9e9e9e', hatch='////', lw=0.0))
    for c, (x0, y0, w, h) in chiplet_boxes(r).items():
        ax.add_patch(Rectangle((x0, y0), w, h, fc='none', ec='#0097a7', lw=0.8))
    setup_plan(ax, r)
    lim = 0.045 * max(bw, bh)
    big = [t for t in sub if min(t['w_mm'], t['h_mm']) > lim]
    small = [t for t in sub if min(t['w_mm'], t['h_mm']) <= lim]
    fs = 4.6 if len(big) <= 100 else 0
    for t in big if fs else []:
        ax.text(t['x0_mm'] + t['w_mm'] / 2, t['y0_mm'] + t['h_mm'] / 2, '%.*f' % (dec, t[key]),
                ha='center', va='center', fontsize=fs, color=_text_color(cmap, norm, t[key]))
    if small:   # tiles too small to see at package scale: coloured marker (+ callout if few)
        xs = [t['x0_mm'] + t['w_mm'] / 2 for t in small]
        ys = [t['y0_mm'] + t['h_mm'] / 2 for t in small]
        ax.scatter(xs, ys, c=[t[key] for t in small], cmap=cmap, norm=norm, marker='s', s=14,
                   edgecolors='k', linewidths=0.4, zorder=5)
        if len(small) <= 16:
            x0, x1 = r['xe'][0], r['xe'][-1]
            for t, x, y in zip(small, xs, ys):
                fx = (x - x0) / (x1 - x0)
                fy = (y - r['ye'][0]) / bh
                ha = 'left' if fx < 0.2 else ('right' if fx > 0.8 else 'center')
                ax.annotate('%s %.*f' % (t['kind'], dec, t[key]), (x, y), xytext=(0, 5 if fy > 0.5 else -5),
                            textcoords='offset points', ha=ha, va='bottom' if fy > 0.5 else 'top',
                            fontsize=4.4, zorder=6,
                            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.8))


@_styled
def plot_tiles(r, out_dir, pdf=True):
    """Per-tier tile-label maps (T_mean, T_max) as rectangles at true positions."""
    tl = [t for t in r['tiles'] if t['kind'] in MAIN_KINDS and t.get('T_mean_K') is not None]
    if not tl:
        return []
    tiers = sorted({t['tier'] for t in tl})
    n = len(tiers)
    bw, bh = r['xe'][-1] - r['xe'][0], r['ye'][-1] - r['ye'][0]
    metrics = (('T_mean_K', 'tile mean T'), ('T_max_K', 'tile max T'))
    if n == 1:      # one die: the two metrics side by side
        W = 7.2
        ph = min((W - 1.4) / 2 * bh / bw, 3.4)
        fig = plt.figure(figsize=(W, ph + 1.1))
        gs = fig.add_gridspec(1, 2, wspace=0.42)
        cells = [[(gs[0, 0], None)], [(gs[0, 1], None)]]
    else:
        W = 7.2
        ph = min((W - 0.8) / n * bh / bw, 3.2)
        fig = plt.figure(figsize=(W, 2 * ph + 1.4))
        gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.04], hspace=0.38, wspace=0.22)
        cells = [[(gs[row, j], gs[row, n]) for j in range(n)] for row in range(2)]
    cmap = CMAP_T
    for mi, (key, lab) in enumerate(metrics):
        vals = np.array([t[key] for t in tl])
        lo, hi = vals.min(), vals.max()
        norm = mcolors.Normalize(lo, hi if hi > lo else lo + 1e-6)
        dec = _decimals(hi - lo)
        for j, tier in enumerate(tiers):
            ax = fig.add_subplot(cells[mi][j][0])
            _tile_panel(ax, r, [t for t in tl if t['tier'] == tier], key, cmap, norm, dec)
            if j or (n == 1 and mi):
                ax.set_ylabel('')
            if n > 1 and mi == 0:
                ax.set_xlabel('')
            ax.set_title('%s — die %d%s' % (lab, tier, ' (all chiplets)' if n == 1 and len(chiplet_boxes(r)) > 1
                                              else ''), fontsize=7)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        if cells[mi][0][1] is None:
            _cbar(fig, ax, sm, '%s (K)' % lab)
        else:
            cb = fig.colorbar(sm, cax=fig.add_subplot(cells[mi][0][1]))
            cb.set_label('%s (K)' % lab)
            cb.formatter.set_useOffset(False)
            cb.update_ticks()
    nm = sum(1 for t in tl if not t.get('mapped', True))
    fig.suptitle('%s — per-tile temperature labels (%d compute/memory tiles%s; router/TSV strips not shown)'
                 % (_title(r), len(tl), ', %d unmapped hatched' % nm if nm else ''), fontsize=8.5, y=1.0)
    return _save(fig, out_dir, 'tiles', pdf)


# ----------------------------------------------------------------------------- validation helpers
def _out(out):
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    stem, ext = os.path.splitext(out)
    return stem, (ext or '.png')


def _save_to(fig, out, pdf=True):
    stem, ext = _out(out)
    files = [stem + ext]
    fig.savefig(files[0], dpi=DPI)
    if pdf and ext != '.pdf':
        fig.savefig(stem + '.pdf')
        files.append(stem + '.pdf')
    plt.close(fig)
    return files


@_styled
def plot_parity(ours, ref, labels=None, out='parity.png', xlabel='reference T (K)', ylabel='hisim_thermal T (K)',
                title='Parity', ref_value=None, pdf=True):
    """Scatter of ours vs ref with y = x, ±1 % bands of the rise (if ref_value=T_amb given) and stats.

    ``labels``: optional per-point names (annotated when ≤ 30 points) or a single series name.
    """
    o, f = np.asarray(ours, float).ravel(), np.asarray(ref, float).ravel()
    e = o - f
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lo, hi = min(o.min(), f.min()), max(o.max(), f.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad
    xx = np.array([lo, hi])
    ax.plot(xx, xx, color='#555', lw=0.8, label='y = x')
    if ref_value is not None:
        for s in (1, -1):
            ax.plot(xx, ref_value + (xx - ref_value) * (1 + s * 0.01), color='#999', lw=0.6, ls='--',
                    label='±1 % of rise' if s == 1 else None)
    series = labels if isinstance(labels, str) else None
    ax.scatter(f, o, s=14, c='#1f77b4', edgecolors='white', linewidths=0.4, zorder=3, label=series)
    if labels is not None and not isinstance(labels, str) and len(labels) <= 30:
        for x, y, s in zip(f, o, labels):
            ax.annotate(str(s), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.ticklabel_format(useOffset=False)
    ss = ((f - f.mean()) ** 2).sum()
    r2 = 1 - (e ** 2).sum() / ss if ss > 0 else float('nan')
    txt = 'n = %d\nRMSE = %.3g\nmax |Δ| = %.3g\nmean Δ = %+.3g\nR² = %.5f' % (
        len(o), np.sqrt((e ** 2).mean()), np.abs(e).max(), e.mean(), r2)
    if ref_value is not None:
        rise = np.abs(f - ref_value)
        ok = rise > 0
        if ok.any():
            txt += '\nmax |Δ|/rise = %.2f %%' % (100 * np.max(np.abs(e[ok]) / rise[ok]))
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va='top', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc'))
    ax.legend(loc='lower right', frameon=False)
    ax.grid(lw=0.3, alpha=0.4)
    return _save_to(fig, out, pdf)


@_styled
def plot_diff_map(diff2d, x_edges, y_edges, out='diff.png', title='ours − reference', label='ΔT (K)',
                  edges_unit='auto', outlines=None, pdf=True):
    """Diverging map of a 2-D difference on true edges (y down).  ``outlines``: list of (x0, y0, w, h) mm."""
    xe, ye = np.asarray(x_edges, float), np.asarray(y_edges, float)
    if edges_unit == 'm' or (edges_unit == 'auto' and max(abs(xe).max(), abs(ye).max()) < 1.0):
        xe, ye = xe * 1e3, ye * 1e3
    D = np.asarray(diff2d, float)
    vm = np.nanmax(np.abs(D)) or 1e-12
    asp = (ye[-1] - ye[0]) / (xe[-1] - xe[0])
    fig, ax = plt.subplots(figsize=(4.6, max(2.2, 3.6 * asp + 0.8)))
    im = ax.pcolormesh(xe, ye, D, cmap='RdBu_r', norm=mcolors.TwoSlopeNorm(0, -vm, vm), shading='flat',
                       rasterized=True)
    for (x0, y0, w, h) in outlines or []:
        ax.add_patch(Rectangle((x0, y0), w, h, fc='none', ec='k', lw=0.3, alpha=0.6))
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[-1], ye[0])
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)  ↓')
    i = np.unravel_index(np.nanargmax(np.abs(D)), D.shape)
    ax.set_title('%s\nmax |Δ| = %.3g at (%.2f, %.2f) mm; RMS %.3g'
                 % (title, abs(D[i]), 0.5 * (xe[i[1]] + xe[i[1] + 1]), 0.5 * (ye[i[0]] + ye[i[0] + 1]),
                    np.sqrt(np.nanmean(D ** 2))), fontsize=7.5)
    _cbar(fig, ax, im, label)
    return _save_to(fig, out, pdf)


@_styled
def plot_gci(h, phi, phi_ext, out='gci.png', ylabel='peak T (K)', title='Grid convergence', p_obs=None, pdf=True):
    """phi vs representative cell size h with the Richardson extrapolate, and relative error log-log."""
    h, phi = np.asarray(h, float), np.asarray(phi, float)
    o = np.argsort(h)
    h, phi = h[o], phi[o]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw={'wspace': 0.42})
    a1.plot(h, phi, 'o-', color='#1f77b4', ms=4, lw=1, label='solutions')
    a1.plot([0], [phi_ext], marker='*', ms=9, color='#d62728', ls='none', label='Richardson extrapolate')
    a1.axhline(phi_ext, color='#d62728', lw=0.6, ls='--')
    a1.set_xlim(left=0)
    a1.set_xlabel('representative cell size h')
    a1.set_ylabel(ylabel)
    a1.ticklabel_format(useOffset=False)
    a1.legend(frameon=False)
    a1.set_title(title)
    err = np.abs(phi - phi_ext) / max(abs(phi_ext), 1e-30)
    m = err > 0
    a2.loglog(h[m], 100 * err[m], 's-', color='#2ca02c', ms=4, lw=1)
    if m.sum() >= 2:
        hh = np.array([h[m].min(), h[m].max()])
        e0 = 100 * err[m][0]
        for pp, ls in ((1, ':'), (2, '--')):
            a2.loglog(hh, e0 * (hh / hh[0]) ** pp, color='#888', lw=0.7, ls=ls, label='slope %d' % pp)
    hm = h[m]
    for x, y in zip(hm, 100 * err[m]):
        right = x >= hm.max()
        a2.annotate('%.3g %%' % y, (x, y), xytext=(-4 if right else 4, -8 if right else 3),
                    textcoords='offset points', fontsize=5.5, ha='right' if right else 'left')
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for axis in (a2.xaxis, a2.yaxis):
        f = ScalarFormatter()
        f.set_scientific(False)
        axis.set_major_formatter(f)
        axis.set_minor_formatter(NullFormatter())
    a2.set_xticks(hm)
    a2.set_xlabel('h')
    a2.set_ylabel('|φ − φ_ext| / |φ_ext| (%)')
    a2.set_title('relative error' + ('' if p_obs is None else ' (p_obs = %.2f)' % p_obs))
    a2.grid(which='both', lw=0.3, alpha=0.4)
    if m.sum() >= 2:
        a2.legend(frameon=False)
    return _save_to(fig, out, pdf)


@_styled
def plot_tornado(d, out='tornado.png', base=None, xlabel='Δ peak T (K)', title='Sensitivity', pdf=True):
    """Tornado chart.  d: {name: (low, high)}; values are deltas, or absolute values if ``base`` given."""
    items = []
    for k, v in d.items():
        lo, hi = (float(v[0]), float(v[1]))
        if base is not None:
            lo, hi = lo - base, hi - base
        items.append((k, lo, hi))
    items.sort(key=lambda t: max(abs(t[1]), abs(t[2])))
    fig, ax = plt.subplots(figsize=(5.0, 0.32 * len(items) + 1.0))
    y = np.arange(len(items))
    # low and high side by side (final audit plots_b F1: overlaid bars hid a same-sign low bar)
    ax.barh(y - 0.18, [t[1] for t in items], color='#4c72b0', height=0.34, label='low setting')
    ax.barh(y + 0.18, [t[2] for t in items], color='#dd8452', height=0.34, label='high setting')
    for yi, (_, lo, hi) in zip(y, items):
        for v, dy in ((lo, -0.18), (hi, 0.18)):
            ax.text(v, yi + dy, ' %+.3g ' % v, va='center', ha='left' if v >= 0 else 'right', fontsize=5.5)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in items])
    span = max([max(abs(t[1]), abs(t[2])) for t in items] + [1e-12])
    ax.set_xlim(-1.3 * span, 1.3 * span)
    ax.set_xlabel(xlabel if base is None else '%s (base %.4g)' % (xlabel, base))
    ax.set_title(title)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.32 if len(items) < 4 else -0.14),
              ncol=2)
    ax.grid(axis='x', lw=0.3, alpha=0.4)
    return _save_to(fig, out, pdf)


# ----------------------------------------------------------------------------- driver
def make_all(d, out_dir=None, pdf_planes=True):
    """Write all standard figures of one result directory; return the list of files.

    PNG (200 dpi) for everything; PDF for the key figures (overview, stack, tiles,
    and the plan / power maps of the active layers).
    """
    r = load_result(d)
    out_dir = out_dir or os.path.join(d, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for p in r['phys']:
        files += plot_plan(r, p, out_dir, pdf=pdf_planes and p['role'] == 'active')
    for name, q in r['qsurf'].items():
        files += plot_qsurf(r, name, q, out_dir, pdf=pdf_planes)
    files += plot_overview(r, out_dir)
    files += plot_stack(r, out_dir)
    files += plot_tiles(r, out_dir)
    return files


if __name__ == '__main__':   # python -m hisim_thermal.plots <result_dir> [out_dir]
    import sys
    for f in make_all(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None):
        print(f)
