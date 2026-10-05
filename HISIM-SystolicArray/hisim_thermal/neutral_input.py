"""Neutral thermal input (plan §3.2): the only thing the physics/numerics of hisim_thermal ever see.

Units: lengths in mm, powers in W.  Plan view origin at the top-left corner of the package bounding box,
x to the right (column ix), y downward (row iy).  Tier 0 is the top die.

Tile kinds
  power-carrying : 'imc', 'sa', 'cpu', 'mem', 'logic', 'router'
  geometry only  : 'tsv' (3D-link / TSV region), 'aib_tx', 'aib_rx' (2.5D interface PHY strips), 'gap'
                   (inter-chiplet gap), 'empty' (unused package area), 'filler' (unused die area), 'emib' (bridge,
                   layer='emib')
Geometry-only tiles carry no power; their power comes only from Links (no double counting, rule 2).
"""
import copy
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

SCHEMA = 'hisim_thermal.input/1'
POWER_KINDS = ('imc', 'sa', 'cpu', 'mem', 'logic', 'router')
GEOM_KINDS = ('tsv', 'l3d', 'aib_tx', 'aib_rx', 'gap', 'empty', 'filler', 'emib')
ALL_KINDS = POWER_KINDS + GEOM_KINDS
LAYERS = ('device', 'emib')
LINK_KINDS = ('tsv', 'aib')


@dataclass
class Tile:
    id: str
    chiplet: int            # chiplet index; -1 for package-level regions (gap / empty / emib)
    tier: int               # die level, 0 = top; for layer 'emib' use 0
    kind: str
    x0: float               # mm
    y0: float               # mm
    w: float                # mm
    h: float                # mm
    layer: str = 'device'   # 'device' (die layers of that tier) | 'emib' (bridge layer at the substrate top)
    slot: int = -1
    p_dyn_W: float = 0.0    # effective dynamic power (duty already applied, already in W)
    p_leak_W: float = 0.0
    src: dict = field(default_factory=dict)


@dataclass
class Link:
    id: str
    kind: str               # 'tsv' | 'aib'
    src: tuple              # (chiplet, tier)
    dst: tuple              # (chiplet, tier)
    p_src_W: float = 0.0    # injected over region['src_tiles'] (active layer of their die)
    p_dst_W: float = 0.0    # injected over region['dst_tiles'] (active layer of their die)
    p_mid_W: float = 0.0    # injected over region['mid_tiles'] in region['mid_path'] roles of the src die, or the emib layer
    src_row: int = -1       # tier_3d interface index / layer_aib_list row
    region: dict = field(default_factory=dict)
    note: str = ''


@dataclass
class ThermalInput:
    arch: str               # 'M2D' | 'M3D' | 'H2_5D' | 'M3_5D'
    n_tier: int             # number of die levels in z
    chiplet_num: int
    mesh_edge: int          # IMC tile mesh edge N (0 if not applicable)
    bbox_mm: tuple          # (W, H) package bounding box
    ref_edge_mm: float      # reference tile edge for the production mesh target (a for IMC)
    tiles: list
    links: list
    source: str = 'imc'     # 'imc' | 'sa' | 'synthetic'
    chiplet_bbox: dict = field(default_factory=dict)   # chiplet -> (x0, y0, w, h)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def tile_index(self):
        return {t.id: i for i, t in enumerate(self.tiles)}

    def layout_hash(self):
        h = hashlib.sha1()
        for t in sorted(self.tiles, key=lambda t: t.id):
            h.update(repr((t.id, t.chiplet, t.tier, t.kind, t.layer, round(t.x0, 12), round(t.y0, 12),
                           round(t.w, 12), round(t.h, 12), t.slot, t.src.get('layer'), t.src.get('node'),
                           t.src.get('mapped'))).encode())
        for l in sorted(self.links, key=lambda l: l.id):
            h.update(repr((l.id, l.kind, tuple(l.src), tuple(l.dst), json.dumps(l.region, sort_keys=True))).encode())
        return h.hexdigest()

    def with_powers(self, tile_powers=None, link_powers=None, scale=None):
        """Same geometry and layout, new power magnitudes (frozen layout, plan §6.1 / §11.2).
        tile_powers: {tile_id: (p_dyn_W, p_leak_W)}; link_powers: {link_id: (p_src, p_dst, p_mid)};
        scale: multiply every power by this factor (applied after the explicit values)."""
        ti = copy.deepcopy(self)
        idx = ti.tile_index()
        for tid, (pd, pl) in (tile_powers or {}).items():
            if tid not in idx:
                raise ValueError('with_powers: unknown tile ' + tid)
            ti.tiles[idx[tid]].p_dyn_W, ti.tiles[idx[tid]].p_leak_W = float(pd), float(pl)
        lidx = {l.id: i for i, l in enumerate(ti.links)}
        for lid, (ps, pd, pm) in (link_powers or {}).items():
            if lid not in lidx:
                raise ValueError('with_powers: unknown link ' + lid)
            l = ti.links[lidx[lid]]
            l.p_src_W, l.p_dst_W, l.p_mid_W = float(ps), float(pd), float(pm)
        if scale is not None:
            for t in ti.tiles:
                t.p_dyn_W *= scale
                t.p_leak_W *= scale
            for l in ti.links:
                l.p_src_W *= scale
                l.p_dst_W *= scale
                l.p_mid_W *= scale
        ti.meta = dict(ti.meta)
        if ti.meta.pop('source_ledger', None) is not None:     # the upstream ledger no longer applies (review v0.2 F5)
            ti.meta['power_override'] = True
        ti.meta['layout_recomputed'] = False
        ti.meta['layout_hash'] = self.layout_hash()
        ti.check()
        return ti

    def zero_powers(self):
        return self.with_powers({t.id: (0.0, 0.0) for t in self.tiles}, {l.id: (0.0, 0.0, 0.0) for l in self.links})

    def budget(self):
        """Input power totals per category (W)."""
        b = {'tile_dyn': 0.0, 'leak': 0.0, 'router': 0.0, 'tsv': 0.0, 'aib': 0.0, 'emib': 0.0}
        for t in self.tiles:
            if t.kind == 'router':
                b['router'] += t.p_dyn_W + t.p_leak_W
            else:
                b['tile_dyn'] += t.p_dyn_W
                b['leak'] += t.p_leak_W
        for l in self.links:
            if l.kind == 'tsv':
                b['tsv'] += l.p_src_W + l.p_dst_W + l.p_mid_W
            else:
                b['aib'] += l.p_src_W + l.p_dst_W
                b['emib'] += l.p_mid_W
        b['total'] = sum(b.values())
        return b

    # ------------------------------------------------------------------ JSON
    def to_json(self, path):
        d = {'schema': SCHEMA}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == 'tiles':
                v = [dataclasses.asdict(t) for t in v]
            elif f.name == 'links':
                v = [dataclasses.asdict(l) for l in v]
            elif f.name == 'chiplet_bbox':
                v = {str(k): list(b) for k, b in v.items()}
            d[f.name] = v
        with open(path, 'w') as f:
            json.dump(d, f, indent=1, sort_keys=False)

    @staticmethod
    def from_json(path):
        with open(path) as f:
            d = json.load(f)
        if d.get('schema') != SCHEMA:
            raise ValueError('not a {} file: {}'.format(SCHEMA, path))
        d.pop('schema')
        d['tiles'] = [Tile(**t) for t in d['tiles']]
        links = []
        for l in d['links']:
            l['src'], l['dst'] = tuple(l['src']), tuple(l['dst'])
            links.append(Link(**l))
        d['links'] = links
        d['bbox_mm'] = tuple(d['bbox_mm'])
        d['chiplet_bbox'] = {int(k): tuple(v) for k, v in d.get('chiplet_bbox', {}).items()}
        ti = ThermalInput(**d)
        ti.check()
        return ti

    # ------------------------------------------------------------------ validation
    def check(self, tol=1e-9):
        """Explicit checks of the neutral-input rules (plan §3.2); raises ValueError / NotImplementedError."""
        if self.arch not in ('M2D', 'M3D', 'H2_5D', 'M3_5D'):
            raise ValueError('unknown arch ' + repr(self.arch))
        if self.arch == 'M3_5D':
            raise NotImplementedError('M3_5D (3D stacks on a 2.5D package) is not implemented / validated in v1.1')
        if not isinstance(self.n_tier, int) or self.n_tier < 1:
            raise ValueError('n_tier must be an integer >= 1')
        W, H = self.bbox_mm
        for name, v in (('bbox W', W), ('bbox H', H), ('ref_edge_mm', self.ref_edge_mm)):
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError('{} must be finite and > 0, got {!r}'.format(name, v))
        ids = set()
        for t in self.tiles:
            if t.id in ids:
                raise ValueError('duplicate tile id ' + t.id)
            ids.add(t.id)
            if t.kind not in ALL_KINDS:
                raise ValueError('tile {}: unknown kind {!r}'.format(t.id, t.kind))
            if t.layer not in LAYERS:
                raise ValueError('tile {}: unknown layer {!r}'.format(t.id, t.layer))
            if (t.layer == 'emib') != (t.kind == 'emib'):
                raise ValueError('tile {}: kind emib <-> layer emib'.format(t.id))
            for nm in ('x0', 'y0', 'w', 'h', 'p_dyn_W', 'p_leak_W'):
                v = getattr(t, nm)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValueError('tile {}: {} must be finite, got {!r}'.format(t.id, nm, v))
            if t.w <= 0 or t.h <= 0:
                raise ValueError('tile {}: zero/negative size {} x {}'.format(t.id, t.w, t.h))
            if t.p_dyn_W < 0 or t.p_leak_W < 0:
                raise ValueError('tile {}: negative power'.format(t.id))
            if t.kind in GEOM_KINDS and (t.p_dyn_W != 0 or t.p_leak_W != 0):
                raise ValueError('tile {} of kind {} must carry no power (power comes from links)'.format(t.id, t.kind))
            if not (0 <= t.tier < self.n_tier):
                raise ValueError('tile {}: tier {} outside 0..{}'.format(t.id, t.tier, self.n_tier - 1))
            if t.x0 < -tol or t.y0 < -tol or t.x0 + t.w > W + tol * max(1, W) or t.y0 + t.h > H + tol * max(1, H):
                raise ValueError('tile {} lies outside the package bounding box'.format(t.id))
            if t.kind in ('gap', 'empty', 'emib') and t.chiplet != -1:
                raise ValueError('tile {}: package-level kind {} must have chiplet = -1'.format(t.id, t.kind))
            if t.kind not in ('gap', 'empty', 'emib') and not (0 <= t.chiplet < self.chiplet_num):
                raise ValueError('tile {}: chiplet {} outside 0..{}'.format(t.id, t.chiplet, self.chiplet_num - 1))
        tidx = self.tile_index()
        lids = set()
        for l in self.links:
            if l.id in lids:
                raise ValueError('duplicate link id ' + l.id)
            lids.add(l.id)
            if l.kind not in LINK_KINDS:
                raise ValueError('link {}: unknown kind {!r}'.format(l.id, l.kind))
            for nm in ('p_src_W', 'p_dst_W', 'p_mid_W'):
                v = getattr(l, nm)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                    raise ValueError('link {}: {} must be finite and >= 0, got {!r}'.format(l.id, nm, v))
            for key, p in (('src_tiles', l.p_src_W), ('dst_tiles', l.p_dst_W), ('mid_tiles', l.p_mid_W)):
                lst = l.region.get(key, [])
                for tid in lst:
                    if tid not in tidx:
                        raise ValueError('link {}: unknown tile {} in {}'.format(l.id, tid, key))
                if len(set(lst)) != len(lst):                       # final audit power F1: a support counted twice
                    raise ValueError('link {}: a tile is listed twice in {}'.format(l.id, key))
                if p > 0 and not lst:
                    raise ValueError('link {}: positive power but empty {}'.format(l.id, key))
            if l.p_mid_W > 0 and not (l.region.get('mid_path') or l.region.get('mid_layer') == 'emib'):
                raise ValueError('link {}: p_mid needs region mid_path or mid_layer = emib'.format(l.id))
            if l.region.get('mid_layer') == 'emib' and self.arch not in ('H2_5D',):
                raise ValueError('link {}: an emib-layer mid part needs a 2.5D stack (arch {})'.format(l.id, self.arch))
            for end, key in ((l.src, 'src_tiles'), (l.dst, 'dst_tiles')):
                if len(end) != 2 or not (0 <= int(end[1]) < self.n_tier) or not (0 <= int(end[0]) < self.chiplet_num):
                    raise ValueError('link {}: end {} has no valid (chiplet, tier)'.format(l.id, end))
                for tid in l.region.get(key, []):
                    t = self.tiles[tidx[tid]]
                    if t.layer != 'device' or t.tier != int(end[1]):
                        raise ValueError('link {}: {} tile {} is not on tier {}'.format(l.id, key, tid, end[1]))
                    if t.chiplet != int(end[0]):                    # final audit ni2_floorplan F2
                        raise ValueError('link {}: {} tile {} belongs to chiplet {}, not to the link end chiplet {}'
                                         .format(l.id, key, tid, t.chiplet, end[0]))
            for tid in l.region.get('mid_tiles', []):
                t = self.tiles[tidx[tid]]
                if l.region.get('mid_layer') == 'emib':
                    if t.layer != 'emib':
                        raise ValueError('link {}: mid tile {} is not an emib-layer tile'.format(l.id, tid))
                elif t.layer != 'device' or t.tier != int(l.src[1]):
                    raise ValueError('link {}: mid tile {} is not on the src tier {}'.format(l.id, tid, l.src[1]))
            if 'bond' in (l.region.get('mid_path') or []) and int(l.src[1]) >= self.n_tier - 1:
                raise ValueError('link {}: mid_path uses the bond below tier {}, which is the bottom die'.format(
                    l.id, l.src[1]))
        self._check_partition(tol)
        return True

    def _check_partition(self, tol):
        """Per tier: device-layer tiles must partition the package bbox exactly (no overlap, no hole); each
        chiplet's tiles must lie inside its bbox when given.  Emib-layer tiles must not overlap each other."""
        W, H = self.bbox_mm
        groups = {}
        for t in self.tiles:
            groups.setdefault((t.layer, t.tier if t.layer == 'device' else 0), []).append(t)
        for (layer, tier), ts in groups.items():
            xs = _merge_edges([0.0, W] + [v for t in ts for v in (t.x0, t.x0 + t.w)], tol)
            ys = _merge_edges([0.0, H] + [v for t in ts for v in (t.y0, t.y0 + t.h)], tol)
            cover = [[0] * (len(xs) - 1) for _ in range(len(ys) - 1)]
            for t in ts:
                i0, i1 = _span(xs, t.x0, t.x0 + t.w, tol)
                j0, j1 = _span(ys, t.y0, t.y0 + t.h, tol)
                for j in range(j0, j1):
                    row = cover[j]
                    for i in range(i0, i1):
                        row[i] += 1
                        if row[i] > 1:
                            raise ValueError('tiles overlap in layer {} tier {} near tile {}'.format(layer, tier, t.id))
            if layer == 'device':
                for j, row in enumerate(cover):
                    for i, c in enumerate(row):
                        if c == 0:
                            raise ValueError('tier {}: package area not covered at x={:.6g} y={:.6g} mm '
                                             '(use explicit gap/empty tiles)'.format(tier, xs[i], ys[j]))
        # rule 1: inside a chiplet's bbox, every device-layer tile must belong to that chiplet (review v0.1 F11)
        for c, (bx, by, bw, bh) in self.chiplet_bbox.items():
            for t in self.tiles:
                if t.layer != 'device' or t.chiplet == c:
                    continue
                ox = min(t.x0 + t.w, bx + bw) - max(t.x0, bx)
                oy = min(t.y0 + t.h, by + bh) - max(t.y0, by)
                if ox > tol * max(1.0, bw) and oy > tol * max(1.0, bh):
                    raise ValueError('tile {} (chiplet {}) intrudes into the bbox of chiplet {}'.format(t.id, t.chiplet, c))
        for t in self.tiles:
            # AIB PHY strips belong to their chiplet die but sit at its edge, outside the compute block bbox
            if t.chiplet >= 0 and t.chiplet in self.chiplet_bbox and t.kind not in ('aib_tx', 'aib_rx'):
                bx, by, bw, bh = self.chiplet_bbox[t.chiplet]
                if t.x0 < bx - tol or t.y0 < by - tol or t.x0 + t.w > bx + bw + tol or t.y0 + t.h > by + bh + tol:
                    raise ValueError('tile {} lies outside chiplet {} bbox'.format(t.id, t.chiplet))


def _merge_edges(vals, tol):
    vals = sorted(float(v) for v in vals)
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] > tol * max(1.0, abs(v)):
            out.append(v)
    return out


def _span(edges, a, b, tol):
    import bisect
    def find(v):
        i = bisect.bisect_left(edges, v - tol * max(1.0, abs(v)))
        if i >= len(edges) or abs(edges[i] - v) > tol * max(1.0, abs(v)) * 2:
            raise ValueError('edge {:.12g} not on the merged grid'.format(v))
        return i
    return find(a), find(b)
