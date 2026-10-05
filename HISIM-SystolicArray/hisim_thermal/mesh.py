"""Tensor-product finite-volume grid (plan §5).

Lateral: the union of all tile / chiplet / package rectangle edges (mm) gives the base intervals; interval i is
cut into n_i = max(1, round_half_up(w_i / target)) * 2**lat_level equal cells, target = ref_edge_mm / cells_per_tile.
For the IMC template (a = sqrt(A_tile), b = sqrt(A_router)): a-intervals -> r cells, b-intervals -> 1 cell.

Vertical: graded sublayers (v1 rule, plan §5.3) for the package layers, one sublayer per die layer; every
sublayer is then cut into 2**vert_level equal parts (GCI, plan §5.4).
"""
import bisect
import math

import numpy as np


def graded_sublayers(total_m, dz0_m, growth, max_sub, fine_at_end):
    """v1 util.py:267-284 verbatim in behaviour.  Returns sublayer thicknesses in stack (top->bottom) order.
    fine_at_end=True: thinnest sublayer at the bottom (layer above the devices)."""
    if not (total_m > 0 and dz0_m > 0 and growth >= 1.0 and max_sub >= 1):
        raise ValueError('graded_sublayers: bad arguments')
    if total_m <= 1.5 * dz0_m or max_sub == 1:
        return [total_m]
    dz, d = [], dz0_m
    while sum(dz) + d < total_m and len(dz) < max_sub - 1:
        dz.append(d)
        d *= growth
    dz.append(total_m - sum(dz))
    if len(dz) >= 2 and dz[-1] < dz[-2]:
        dz[-2] += dz[-1]
        dz.pop()
    if abs(sum(dz) - total_m) > 1e-12 * total_m:
        raise ValueError('graded_sublayers: thickness not conserved')
    return dz[::-1] if fine_at_end else dz


def merge_edges(vals, tol):
    vals = sorted(float(v) for v in vals)
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] > tol * max(1.0, abs(v)):
            out.append(v)
    return out


class Grid:
    """Cell (k, iy, ix).  k: sublayer from the top; iy: row (y downward); ix: column (x rightward).
    x_edges/y_edges/z_edges in metres; z = 0 at the top surface, growing downward."""

    def __init__(self, xb_mm, yb_mm, nxi, nyi, stack, sub_dz, sub_phys, tol):
        self.xb_mm, self.yb_mm, self.tol = list(xb_mm), list(yb_mm), tol   # merged base edges
        self.nxi, self.nyi = list(nxi), list(nyi)                          # cells per base interval
        xe = [xb_mm[0]]
        for i, n in enumerate(nxi):
            a, b = xb_mm[i], xb_mm[i + 1]
            xe += [a + (b - a) * (j + 1) / n for j in range(n)]
            xe[-1] = b
        ye = [yb_mm[0]]
        for i, n in enumerate(nyi):
            a, b = yb_mm[i], yb_mm[i + 1]
            ye += [a + (b - a) * (j + 1) / n for j in range(n)]
            ye[-1] = b
        self.x_edges = np.array(xe) * 1e-3
        self.y_edges = np.array(ye) * 1e-3
        self.dx = np.diff(self.x_edges)
        self.dy = np.diff(self.y_edges)
        self.dz = np.array(sub_dz, dtype=float)
        self.z_edges = np.concatenate([[0.0], np.cumsum(self.dz)])
        self.sub_phys = np.array(sub_phys, dtype=int)
        self.stack = stack
        self.nx, self.ny, self.nk = len(self.dx), len(self.dy), len(self.dz)
        self.n = self.nx * self.ny * self.nk
        self._xoff = np.concatenate([[0], np.cumsum(nxi)]).astype(int)   # first cell of each base interval
        self._yoff = np.concatenate([[0], np.cumsum(nyi)]).astype(int)
        self.phys_krange = []
        for p in range(len(stack)):
            ks = np.nonzero(self.sub_phys == p)[0]
            if len(ks) == 0:
                raise ValueError('physical layer {} has no sublayer'.format(stack[p]['name']))
            if not np.all(np.diff(ks) == 1):
                raise ValueError('physical layer {} sublayers not contiguous'.format(stack[p]['name']))
            self.phys_krange.append((int(ks[0]), int(ks[-1]) + 1))
        if np.any(self.dx <= 0) or np.any(self.dy <= 0) or np.any(self.dz <= 0):
            raise ValueError('non-positive cell size')

    def _edge_index(self, edges, v):
        tol = self.tol * max(1.0, abs(v))
        i = bisect.bisect_left(edges, v - tol)
        if i >= len(edges) or abs(edges[i] - v) > 2 * tol:
            raise ValueError('coordinate {:.12g} mm is not a grid edge'.format(v))
        return i

    def rect_slices(self, x0, y0, w, h):
        """(slice over iy, slice over ix) of the cells covering a rectangle given in mm (must be grid-aligned)."""
        i0 = self._edge_index(self.xb_mm, x0)
        i1 = self._edge_index(self.xb_mm, x0 + w)
        j0 = self._edge_index(self.yb_mm, y0)
        j1 = self._edge_index(self.yb_mm, y0 + h)
        return slice(int(self._yoff[j0]), int(self._yoff[j1])), slice(int(self._xoff[i0]), int(self._xoff[i1]))

    def tile_slices(self, t):
        return self.rect_slices(t.x0, t.y0, t.w, t.h)

    def krange(self, p):
        return self.phys_krange[p]

    def cell_volume(self):
        return self.dz[:, None, None] * self.dy[None, :, None] * self.dx[None, None, :]

    def describe(self):
        return {'nx': self.nx, 'ny': self.ny, 'nk': self.nk, 'n': self.n,
                'lateral_base_intervals': [len(self.nxi), len(self.nyi)],
                'dx_min_um': float(self.dx.min() * 1e6), 'dx_max_um': float(self.dx.max() * 1e6),
                'dz_min_um': float(self.dz.min() * 1e6), 'dz_max_um': float(self.dz.max() * 1e6)}


def _round_half_up(v):
    return int(math.floor(v + 0.5))


def build(ti, stack, cfg):
    me = cfg['mesh']
    tol = me['edge_tol_mm']
    W, H = ti.bbox_mm
    xs = [0.0, W]
    ys = [0.0, H]
    for t in ti.tiles:
        xs += [t.x0, t.x0 + t.w]
        ys += [t.y0, t.y0 + t.h]
    for (bx, by, bw, bh) in ti.chiplet_bbox.values():
        xs += [bx, bx + bw]
        ys += [by, by + bh]
    xb = merge_edges(xs, tol)
    yb = merge_edges(ys, tol)
    if abs(xb[0]) > tol or abs(yb[0]) > tol or abs(xb[-1] - W) > tol * max(1, W) or abs(yb[-1] - H) > tol * max(1, H):
        raise ValueError('tiles extend beyond the package bounding box')
    target = ti.ref_edge_mm / me['cells_per_tile']
    mult = 2 ** me['lat_level']
    narrow = 0.5 * ti.ref_edge_mm          # router / TSV / AIB strips and gaps: one cell at production level (§5.1)

    def ncell(w):
        return (1 if w < narrow else max(1, _round_half_up(w / target))) * mult
    nxi = [ncell(xb[i + 1] - xb[i]) for i in range(len(xb) - 1)]
    nyi = [ncell(yb[i + 1] - yb[i]) for i in range(len(yb) - 1)]
    vmult = 2 ** me['vert_level']
    sub_dz, sub_phys = [], []
    for p, L in enumerate(stack):
        t = L['thickness_m']
        if L['grade'] is None:
            base = [t]
        else:
            base = graded_sublayers(t, me['dz0_um'] * 1e-6, me['growth'], me['max_sub'], L['grade'] == 'fine_bottom')
        parts = []
        for d in base:
            parts += [d / vmult] * vmult
        if abs(sum(parts) - t) > 1e-12 * t:
            raise ValueError('sublayers of {} do not add up to its thickness'.format(L['name']))
        sub_dz += parts
        sub_phys += [p] * len(parts)
    g = Grid(xb, yb, nxi, nyi, stack, sub_dz, sub_phys, tol)
    tot = sum(L['thickness_m'] for L in stack)
    if abs(g.z_edges[-1] - tot) > 1e-12 * tot:
        raise ValueError('grid thickness {} != stack thickness {}'.format(g.z_edges[-1], tot))
    # independent check against the closed stack formula of plan §4.1 / §4.2 (final review plan_b3a F5): catches a
    # layer missing from or added to the stack list, which the sum above cannot see
    from .config import bond_props
    from .stack import is_2p5d
    st, n = cfg['stack'], ti.n_tier
    n_bond = (n - 1 if st['bond_between_dies'] else 0) + (1 if st['bond_below_last'] else 0)
    bottom = st['bottom_2p5d'] if is_2p5d(ti.arch) else st['bottom']
    exp_um = (sum(t for _, t, _ in st['top']) + n * sum(t for _, t, _ in st['per_die'])
              + n_bond * bond_props(cfg)[0] * 1e6 + sum(t for _, t, _ in bottom))
    if abs(g.z_edges[-1] * 1e6 - exp_um) > 1e-9 * exp_um:
        raise ValueError('grid thickness {} um != stack formula {} um'.format(g.z_edges[-1] * 1e6, exp_um))
    return g
