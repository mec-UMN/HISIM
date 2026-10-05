"""Physical layer stack (plan §4).  Top to bottom, active face up (D4).

v1.1 (M2D/M3D):   spreader | tim1 | [beol, active, si] x die d (+ bond between d and d+1) | c4 | subs
v1.1 (2.5D):      spreader | tim1 | beol | active | si | c4 | emib (substrate top) | subs (950 um)
v1compat:         heatsink | tim2 | heatspread | tim1 | beol(top die) | [active, si, bond] x tier | c4 | subs
"""
from .config import bond_props

def is_2p5d(arch):
    return arch in ('H2_5D', 'M3_5D')


def build_stack(ti, cfg):
    """List of physical layers (dicts): name, role, die (None for package layers), thickness_m, grade, part."""
    st = cfg['stack']
    n = ti.n_tier
    t_bond_m, _, _ = bond_props(cfg)
    layers = []

    def add(name, role, t_m, grade, die, part):
        layers.append({'name': name, 'role': role, 'die': die, 'thickness_m': float(t_m), 'grade': grade, 'part': part})

    for name, t_um, grade in st['top']:
        add(name, name, t_um * 1e-6, grade, None, 'top')
    for d in range(n):
        for name, t_um, grade in st['per_die']:
            add('{}.d{}'.format(name, d), name, t_um * 1e-6, grade, d, 'die')
        if (d < n - 1 and st['bond_between_dies']) or (d == n - 1 and st['bond_below_last']):
            add('bond.d{}'.format(d), 'bond', t_bond_m, None, d, 'die')
    if is_2p5d(ti.arch):
        bottom = st.get('bottom_2p5d')
        if not bottom:
            raise NotImplementedError('preset {} has no 2.5D stack (v1 defined only M2D / M3D; use preset v1.1)'
                                      .format(cfg['preset']))
    else:
        bottom = st['bottom']
    for name, t_um, grade in bottom:
        add(name, name, t_um * 1e-6, grade, None, 'bottom')
    names = [L['name'] for L in layers]
    if len(set(names)) != len(names):
        raise ValueError('duplicate layer names in stack: {}'.format(names))
    return layers


def total_thickness_um(layers):
    return sum(L['thickness_m'] for L in layers) * 1e6


def find(layers, role, die=None):
    """Index of the physical layer with this role (and die)."""
    for i, L in enumerate(layers):
        if L['role'] == role and L['die'] == die:
            return i
    raise KeyError('no layer role={} die={}'.format(role, die))


def summary(layers):
    return [{'name': L['name'], 'role': L['role'], 'die': L['die'], 'thickness_um': L['thickness_m'] * 1e6,
             'grade': L['grade']} for L in layers]
