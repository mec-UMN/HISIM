"""Plan-view region maps on the grid (plan §5.1-5.2, §7.3).

For every die level t: region code, owning tile index and chiplet of each (iy, ix) cell of the device layers;
for the emib layer: bridge tile index (-1 = plain substrate).  The device planes must be covered exactly once.
"""
import numpy as np

REGION_CODES = {'none': 0, 'imc': 1, 'router': 2, 'tsv': 3, 'aib_tx': 4, 'aib_rx': 5, 'gap': 6, 'empty': 7,
                'emib': 8, 'sa': 9, 'cpu': 10, 'mem': 11, 'logic': 12, 'filler': 13, 'l3d': 14}
REGION_NAMES = {v: k for k, v in REGION_CODES.items()}
TILE_KINDS = ('imc', 'sa', 'cpu', 'mem', 'logic')      # compute / memory tiles (tile-level labels)


class Maps:
    def __init__(self, region, owner, chiplet, emib):
        self.region = region          # int16 [n_tier, ny, nx]
        self.owner = owner            # int32 [n_tier, ny, nx] tile index
        self.chiplet = chiplet        # int16 [n_tier, ny, nx]; -1 package-level (gap / empty)
        self.emib = emib              # int32 [ny, nx] emib tile index or -1
        self.footprint = chiplet >= 0  # bool [n_tier, ny, nx]: inside some chiplet


def build_maps(ti, grid):
    nt, ny, nx = ti.n_tier, grid.ny, grid.nx
    region = np.zeros((nt, ny, nx), np.int16)
    owner = np.full((nt, ny, nx), -1, np.int32)
    chip = np.full((nt, ny, nx), -1, np.int16)
    emib = np.full((ny, nx), -1, np.int32)
    for i, t in enumerate(ti.tiles):
        sy, sx = grid.tile_slices(t)
        if t.layer == 'emib':
            if np.any(emib[sy, sx] >= 0):
                raise ValueError('emib tiles overlap at ' + t.id)
            emib[sy, sx] = i
            continue
        blk = owner[t.tier, sy, sx]
        if np.any(blk >= 0):
            raise ValueError('device tiles overlap at {} (tier {})'.format(t.id, t.tier))
        owner[t.tier, sy, sx] = i
        region[t.tier, sy, sx] = REGION_CODES[t.kind]
        chip[t.tier, sy, sx] = t.chiplet
    if np.any(owner < 0):
        t, iy, ix = [int(v[0]) for v in np.nonzero(owner < 0)]
        raise ValueError('device plane of tier {} not covered at cell (iy={}, ix={})'.format(t, iy, ix))
    return Maps(region, owner, chip, emib)
