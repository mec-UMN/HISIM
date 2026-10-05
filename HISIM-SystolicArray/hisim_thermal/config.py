"""Presets and configuration for hisim_thermal.

Every number used by the solver lives here, together with its unit and source (plan §4).
get_config(preset, overrides) returns a validated deep copy; unknown keys raise ValueError.

Presets
  'v1.1'          default physical model (plan §4.1-4.6)
  'v1compat'      numerically equivalent to the v1 reference implementation (plan §4.7)
  'legacy'        handled by Module_Thermal/legacy (original code + patches L1-L7); not run by this package
  'legacy_exact'  handled by Module_Thermal/legacy_exact (original code + L1, L2 only)
"""
import copy
import math
import os

VERSION = '1.1.0'

def _find_up(*parts):
    """First existing directory <ancestor>/<parts> above this package (works from the workspace and from snapshots)."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    while True:
        cand = os.path.join(d, *parts)
        if os.path.isdir(cand):
            return cand
        if os.path.dirname(d) == d:
            return os.path.normpath(os.path.join(here, '..', '..', *parts))
        d = os.path.dirname(d)


_PYAMG_DEFAULT = _find_up('3DICE_baseline', 'compare', 'pylib')     # bundled pyamg (no package installs)

# ----------------------------------------------------------------------------------------------
# Materials, W/(m K).  Each entry: value, source.
# ----------------------------------------------------------------------------------------------
MATERIALS_V11 = {
    'cu':              (400.0, 'Cu lid/spreader; CRC/NIST 401, HotSpot/MFIT lids 398-400'),
    'si':              (148.0, 'bulk Si at 300 K; CRC 148-149, McConnell & Goodson 2005'),
    'si_active':       (120.0, 'engineering choice: 2 um doped active film slightly below bulk (HISIM used 110); insensitive'),
    'tim1':            (4.0,   'die-to-lid TIM: HotSpot default 20 um / k 4; Vendra 2018 20 um / 4'),
    'beol':            (2.25,  '3D-ICE stack.stk BEOL material'),
    'bond_ubump':      (4.0,   'micro-bump + underfill layer, middle value (Tavakkoli paste 5, eff. 7.9 with bumps, underfill ~0.5); scanned 2-8'),
    'bond_hybrid':     (1.4,   'hybrid-bond oxide (3D-ICE SIO2 1.38); engineering value'),
    'c4':              (8.0,   'C4 bumps + underfill, ~20% bump area; parallel bound with paste ~15, with underfill ~12; MFIT kz 2.35'),
    'subs_org':        (1.0,   'organic package substrate through-plane 0.3-1 (MFIT 0.378)'),
    'underfill':       (0.5,   'Henkel E1216M 0.42; typical 0.4-0.6'),
    'mold':            (0.8,   'epoxy mold compound < 1 (Shibuya & Nguyen, ECTC 2017); typical 0.7-1'),
    'air':             (0.026, 'air (only if gap_fill selects it)'),
    'tsv_active':      (148.0, 'Si keep-out zone around TSVs (HISIM used 142.8)'),
    'tsv_si':          (200.0, '~20% Cu TSV array: 0.2*400 + 0.8*148 = 198 (HISIM 200); vertical parallel mix used isotropically'),
    'tsv_bond_ubump':  (7.9,   'Tavakkoli IJHMT 2016 Table 3 micro-bump TIM effective k (as used by HISIM in TSV cells)'),
    'tsv_bond_hybrid': (81.0,  '20% Cu pads in SiO2: 0.2*400 + 0.8*1.4; engineering value'),
    'tsv_beol':        (2.25,  'numerical choice = beol (TSV region through-plane taken low); options 7.9 or 12 (3D-ICE TUNGSTEN_BEOL)'),
    'emib_si':         (148.0, 'passive Si bridge (Intel EMIB; bridge < 75 um, Mahajan ECTC 2016)'),
}
MATERIALS_V1 = {   # v1 reference implementation THERMAL_MAT
    'heatsink':   (398.0, 'v1 THERMAL_MAT heatsink'),
    'tim2':       (3.0,   'v1 THERMAL_MAT tim2'),
    'heatspread': (398.0, 'v1 THERMAL_MAT heatspread'),
    'tim1':       (4.0,   'v1 THERMAL_MAT tim1'),
    'beol':       (2.0,   'v1 THERMAL_MAT beol'),
    'si_active':  (110.0, 'v1 THERMAL_MAT si_active'),
    'si':         (142.8, 'v1 THERMAL_MAT si_bulk'),
    'bond':       (1.93,  'v1 THERMAL_MAT bond (hybrid bond + BEOL of the die below, in series)'),
    'tsv_active': (142.8, 'v1 THERMAL_MAT tsv_active'),
    'tsv_si':     (200.0, 'v1 THERMAL_MAT tsv_bulk'),
    'tsv_bond':   (7.9,   'v1 THERMAL_MAT tsv_bond'),
    'tsv_beol':   (2.0,   'v1: the BEOL layer has no TSV region (same k as beol)'),
    'c4':         (5.0,   'v1 THERMAL_MAT c4'),
    'subs_org':   (1.0,   'v1 THERMAL_MAT subs'),
    'cu':         (398.0, 'v1 THERMAL_MAT cu'),
    'underfill':  (0.5,   'not used by v1'),
    'mold':       (0.8,   'not used by v1'),
    'air':        (0.026, 'v1 THERMAL_MAT air (unused)'),
    'emib_si':    (148.0, 'not used by v1'),
}

# ----------------------------------------------------------------------------------------------
# Stack templates.  Thickness in um.  grade: None | 'fine_bottom' (fine side toward devices below) |
# 'fine_top' (fine side toward devices above).  'per_die' expands for every die d = 0..n_die-1 (top first).
# ----------------------------------------------------------------------------------------------
STACK_V11 = {
    'top':     [('spreader', 2000.0, 'fine_bottom'), ('tim1', 20.0, 'fine_bottom')],
    'per_die': [('beol', 10.0, None), ('active', 2.0, None), ('si', 100.0, None)],
    'bond_between_dies': True,      # one bond layer between die d and d+1 (Q1 (a))
    'bond_below_last': False,
    'bottom':  [('c4', 100.0, 'fine_top'), ('subs', 1000.0, 'fine_top')],
    'bottom_2p5d': [('c4', 100.0, 'fine_top'), ('emib', 50.0, 'fine_top'), ('subs', 950.0, 'fine_top')],
}
STACK_V1 = {
    'top':     [('heatsink', 5000.0, 'fine_bottom'), ('tim2', 50.0, 'fine_bottom'), ('heatspread', 2000.0, 'fine_bottom'),
                ('tim1', 30.0, 'fine_bottom'), ('beol', 10.0, 'fine_bottom')],
    'per_die': [('active', 2.0, None), ('si', 100.0, None)],
    'bond_between_dies': True,
    'bond_below_last': True,        # v1: every tier carries an 11 um bond sublayer, the bottom tier included
    'bottom':  [('c4', 70.0, 'fine_top'), ('subs', 1000.0, 'fine_top')],
    'bottom_2p5d': None,
}
BONDING = {   # D6
    'ubump':  {'t_bond_um': 20.0, 'k_bond': 'bond_ubump',  'k_tsv_bond': 'tsv_bond_ubump',
               'source': 'Tavakkoli / HISIM micro-bump layer'},
    'hybrid': {'t_bond_um': 1.0,  'k_bond': 'bond_hybrid', 'k_tsv_bond': 'tsv_bond_hybrid',
               'source': 'hybrid bonding, engineering estimate'},
    'v1':     {'t_bond_um': 11.0, 'k_bond': 'bond',        'k_tsv_bond': 'tsv_bond',
               'source': 'v1 THERMAL_Z device[2] = 0.011 mm'},
}


def _base_v11():
    return {
        'preset': 'v1.1',
        'materials': {k: v[0] for k, v in MATERIALS_V11.items()},
        'stack': copy.deepcopy(STACK_V11),
        'bonding': 'ubump',
        # 2.5D gaps / empty regions (D12): per physical role, material of the non-chiplet columns
        'gap_fill': {'tim1': 'mold', 'beol': 'mold', 'active': 'mold', 'si': 'mold', 'bond': 'mold', 'c4': 'underfill'},
        'aib_si': 'si',                      # 'si' (original HISIM: AIB region = body) | 'tsv_si' (through-Si links)
        'bc': {'h_top': 1.0e4, 'h_bottom': 10.0, 'T_amb': 298.0},
        'mesh': {'cells_per_tile': 1, 'dz0_um': 30.0, 'growth': 1.5, 'max_sub': 12,
                 'lat_level': 0, 'vert_level': 0, 'edge_tol_mm': 1e-9},
        'power': {'mode': 'concurrent', 'include_leakage': False, 'leakage_slots': 'mapped',
                  'tsv_mode': 'driver', 'tsv_via_path': ['active', 'si', 'bond'],
                  'emib_wire_fraction': 0.0, 'aib_area_mode': 'per_row', 'router_fix': True},
        'layout_2p5d': {'extra_layouts': True},     # N = 3 -> also L, N = 4 -> also ring (side directories)
        'orientation': 'face_up',
        'alpha': 1.0,
        'solver': {'direct_max_n': 60000, 'rtol': 1e-12, 'resid_tol': 1e-10, 'maxiter': 2000,
                   'berr_tol': 1e-12, 'berr_stop': 1e-13, 'berr_check_every': 10, 'amg_seed': 0,
                   'amg_strength': ['symmetric', {'theta': 0.25}],
                   'pyamg_path': os.environ.get('HISIM_THERMAL_PYAMG_PATH', _PYAMG_DEFAULT)},
        'output': {'write_arrays': True, 'plots': True, 'legacy_plots': False, 'viewer': False},
        'check_conservation': True,
    }


def _base_v1compat():
    c = _base_v11()
    c['preset'] = 'v1compat'
    c['materials'] = {k: v[0] for k, v in MATERIALS_V1.items()}
    c['stack'] = copy.deepcopy(STACK_V1)
    c['bonding'] = 'v1'
    c['bc'] = {'h_top': 1.0e4, 'h_bottom': 0.0, 'T_amb': 298.0}
    c['mesh'].update({'growth': 1.6})
    c['power'].update({'include_leakage': True, 'leakage_slots': 'p_positive', 'tsv_via_path': ['active', 'si', 'bond']})
    c['aib_si'] = 'si'
    c['layout_2p5d'] = {'extra_layouts': False}
    return c


PRESETS = {'v1.1': _base_v11, 'v1compat': _base_v1compat}
LEGACY_PRESETS = ('legacy', 'legacy_exact')

# provenance of scalar settings (dotted key -> (unit, source)); materials and stack carry their own
SOURCES = {
    'bc.h_top':    ('W/(m^2 K)', 'lumped TIM2 + sink + convection over the lid top; R_sa,eq = 1/(h_top*A_top) recorded in meta (plan §4.5)'),
    'bc.h_bottom': ('W/(m^2 K)', 'Tavakkoli ASME JHT 138 (2016) nominal h_b = 10 (dissertation UCR 2017 §2.3); Incropera natural convection 2-25'),
    'bc.T_amb':    ('K', 'ambient 298 K (option 318.15)'),
    'mesh.dz0_um': ('um', 'first sublayer next to the devices (v1 rule, plan §5.3)'),
    'mesh.growth': ('-', 'geometric growth of sublayers away from the devices'),
    'mesh.max_sub': ('-', 'max sublayers per graded layer'),
    'mesh.cells_per_tile': ('-', 'production lateral refinement r of the a-intervals (plan §5.1); certified by GCI (§5.4)'),
    'power.emib_wire_fraction': ('-', 'scenario value only: resistive split ~0.057, distributed RC 0.04-0.11 (plan §6.5); default 0'),
}

_ALLOWED = {
    'preset', 'materials', 'stack', 'bonding', 'gap_fill', 'aib_si', 'bc', 'mesh', 'power', 'layout_2p5d',
    'orientation', 'alpha', 'solver', 'output', 'check_conservation',
}
_SUB_ALLOWED = {
    'bc': {'h_top', 'h_bottom', 'T_amb'},
    'mesh': {'cells_per_tile', 'dz0_um', 'growth', 'max_sub', 'lat_level', 'vert_level', 'edge_tol_mm'},
    'power': {'mode', 'include_leakage', 'leakage_slots', 'tsv_mode', 'tsv_via_path', 'emib_wire_fraction',
              'aib_area_mode', 'router_fix'},
    'solver': {'direct_max_n', 'rtol', 'resid_tol', 'maxiter', 'pyamg_path', 'berr_tol', 'berr_stop',
               'berr_check_every', 'amg_seed', 'amg_strength'},
    'output': {'write_arrays', 'plots', 'legacy_plots', 'viewer'},
    'layout_2p5d': {'extra_layouts'},
    'gap_fill': {'tim1', 'beol', 'active', 'si', 'bond', 'c4'},
}


def _deep_update(dst, src, path=''):
    for k, v in src.items():
        key = path + k
        if path == '' and k not in _ALLOWED:
            raise ValueError('unknown config key: ' + key)
        top = key.split('.')[0]
        if path and top in _SUB_ALLOWED and k not in _SUB_ALLOWED[top]:
            raise ValueError('unknown config key: ' + key)
        if isinstance(v, dict) and isinstance(dst.get(k), dict) and k not in ('materials',):
            _deep_update(dst[k], v, key + '.')
        elif k == 'materials' and isinstance(v, dict):
            for m, kv in v.items():
                if m not in dst['materials']:
                    raise ValueError('unknown material: ' + m)
                dst['materials'][m] = kv
        else:
            dst[k] = copy.deepcopy(v)


def _finite(x, name, lo=None, hi=None, lo_open=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('{} must be a finite number, got {!r}'.format(name, x))
    if lo is not None and (x < lo or (lo_open and x <= lo)):
        raise ValueError('{} = {} out of range (lower bound {}{})'.format(name, x, lo, ' exclusive' if lo_open else ''))
    if hi is not None and x > hi:
        raise ValueError('{} = {} out of range (upper bound {})'.format(name, x, hi))


def validate(cfg):
    """Explicit checks (no assert): finiteness and legal ranges of every setting."""
    for m, k in cfg['materials'].items():
        _finite(k, 'materials.' + m, 0.0, lo_open=True)
    _finite(cfg['alpha'], 'alpha', 0.0, lo_open=True)
    bc = cfg['bc']
    _finite(bc['h_top'], 'bc.h_top', 0.0)
    _finite(bc['h_bottom'], 'bc.h_bottom', 0.0)
    _finite(bc['T_amb'], 'bc.T_amb', 0.0, lo_open=True)
    if bc['h_top'] == 0.0 and bc['h_bottom'] == 0.0:
        raise ValueError('h_top and h_bottom are both 0: the conduction matrix would be singular')
    me = cfg['mesh']
    for k in ('cells_per_tile', 'max_sub'):
        if type(me[k]) is not int or me[k] < 1:
            raise ValueError('mesh.{} must be an integer >= 1, got {!r}'.format(k, me[k]))
    for k in ('lat_level', 'vert_level'):
        if not isinstance(me[k], int) or isinstance(me[k], bool) or me[k] < 0:
            raise ValueError('mesh.{} must be an integer >= 0, got {!r}'.format(k, me[k]))
    _finite(me['dz0_um'], 'mesh.dz0_um', 0.0, lo_open=True)
    _finite(me['growth'], 'mesh.growth', 1.0)
    _finite(me['edge_tol_mm'], 'mesh.edge_tol_mm', 0.0, lo_open=True)
    pw = cfg['power']
    if pw['mode'] not in ('concurrent', 'pipeline', 'time_avg'):
        raise ValueError('power.mode must be concurrent | pipeline | time_avg, got {!r}'.format(pw['mode']))
    if pw['tsv_mode'] not in ('driver', 'via'):
        raise ValueError('power.tsv_mode must be driver | via, got {!r}'.format(pw['tsv_mode']))
    if pw['leakage_slots'] not in ('mapped', 'all', 'p_positive'):
        raise ValueError('power.leakage_slots must be mapped | all | p_positive (v1: slots with power > 0)')
    if pw['aib_area_mode'] not in ('per_row', 'mean'):
        raise ValueError('power.aib_area_mode must be per_row | mean')
    _finite(pw['emib_wire_fraction'], 'power.emib_wire_fraction', 0.0, 1.0)
    if not pw['tsv_via_path'] or any(r not in ('beol', 'active', 'si', 'bond') for r in pw['tsv_via_path']):
        raise ValueError('power.tsv_via_path must be a non-empty subset of beol/active/si/bond')
    if len(set(pw['tsv_via_path'])) != len(pw['tsv_via_path']):       # final audit power F1
        raise ValueError('power.tsv_via_path lists a layer role twice: {}'.format(pw['tsv_via_path']))
    if cfg['bonding'] not in BONDING:
        raise ValueError('bonding must be one of ' + ', '.join(BONDING))
    for role, mat in cfg['gap_fill'].items():
        if mat not in cfg['materials']:
            raise ValueError('gap_fill.{} = {!r} is not a material'.format(role, mat))
    if cfg['aib_si'] not in ('si', 'tsv_si'):
        raise ValueError("aib_si must be 'si' or 'tsv_si'")
    if cfg['orientation'] != 'face_up':
        raise NotImplementedError("orientation {!r} is not implemented (only 'face_up')".format(cfg['orientation']))
    st = cfg['stack']
    for path, v in (('power.include_leakage', pw['include_leakage']), ('power.router_fix', pw['router_fix']),
                    ('check_conservation', cfg['check_conservation']),
                    ('layout_2p5d.extra_layouts', cfg['layout_2p5d']['extra_layouts']),
                    ('stack.bond_between_dies', st['bond_between_dies']), ('stack.bond_below_last', st['bond_below_last']))\
            + tuple(('output.' + k, v) for k, v in cfg['output'].items()):
        if not isinstance(v, bool):
            raise ValueError('{} must be true / false, got {!r}'.format(path, v))
    so = cfg['solver']
    _finite(so['rtol'], 'solver.rtol', 0.0, lo_open=True)
    _finite(so['berr_tol'], 'solver.berr_tol', 0.0, lo_open=True)
    _finite(so['berr_stop'], 'solver.berr_stop', 0.0, lo_open=True)
    _finite(so['resid_tol'], 'solver.resid_tol', 0.0, lo_open=True)
    for k in ('direct_max_n', 'maxiter', 'berr_check_every'):
        if type(so[k]) is not int or so[k] < 1:
            raise ValueError('solver.{} must be an integer >= 1'.format(k))
    st = cfg['stack']
    unknown = set(st) - {'top', 'per_die', 'bond_between_dies', 'bond_below_last', 'bottom', 'bottom_2p5d'}
    if unknown:
        raise ValueError('unknown stack keys: {}'.format(sorted(unknown)))
    b = BONDING[cfg['bonding']]
    for key in ('k_bond', 'k_tsv_bond'):
        if b[key] not in cfg['materials']:
            raise ValueError('bonding preset {} needs material {!r}, missing in this preset'.format(cfg['bonding'], b[key]))
    parts = ['top', 'per_die', 'bottom'] + (['bottom_2p5d'] if st.get('bottom_2p5d') else [])
    for part in parts:
        for name, t, grade in st[part]:
            _finite(t, 'stack.{}.{}'.format(part, name), 0.0, lo_open=True)
            if grade not in (None, 'fine_bottom', 'fine_top'):
                raise ValueError('bad grade {!r} for layer {}'.format(grade, name))
    return cfg


def get_config(preset='v1.1', overrides=None):
    """Deep copy of a preset with overrides applied and validated."""
    if preset in LEGACY_PRESETS:
        raise ValueError("preset {!r} is run by Module_Thermal/{}, not by hisim_thermal".format(preset, preset))
    if preset not in PRESETS:
        raise ValueError('unknown preset {!r}; choose one of {}'.format(preset, ', '.join(list(PRESETS) + list(LEGACY_PRESETS))))
    cfg = PRESETS[preset]()
    if overrides:
        if 'preset' in overrides and overrides['preset'] != preset:      # final audit cfg2_ni1 F1
            raise ValueError('overrides may not change the preset ({!r} -> {!r}); pass the preset argument instead'
                             .format(preset, overrides['preset']))
        _deep_update(cfg, overrides)
    return validate(cfg)


def k_of(cfg, material):
    """Conductivity of a material name, divided by alpha (plan §4.3: alpha != 1 only for experiments)."""
    if material not in cfg['materials']:
        raise ValueError('unknown material ' + repr(material))
    return cfg['materials'][material] / cfg['alpha']


def bond_props(cfg):
    """(t_bond_m, bond material, TSV-region bond material) of the selected bonding preset."""
    b = BONDING[cfg['bonding']]
    return b['t_bond_um'] * 1e-6, b['k_bond'], b['k_tsv_bond']


THICKNESS_SOURCES = {
    'spreader': 'chip-sized Cu lid 2 mm (plan §4.1; HotSpot / MFIT lids)',
    'tim1': 'die-to-lid TIM 20 um (HotSpot default, Vendra 2018)',
    'beol': 'BEOL 10 um per die (Q1 (a); 3D-ICE stack.stk has a BEOL per die; Bloomfield 2026: 4.3 um)',
    'active': 'active / FEOL 2 um (HISIM device sublayer; Bloomfield 2026: 1.5 um)',
    'si': 'thinned die Si 100 um (HISIM / Tavakkoli; scanned 50 / 100)',
    'bond': 'die-to-die bond: micro-bump 20 um / hybrid 1 um (D6)',
    'c4': 'C4 + underfill 100 um (Tavakkoli Table 2: SnAgCu d = 100 um, pitch 200 um)',
    'emib': 'EMIB bridge layer 50 um at the substrate top (Mahajan ECTC 2016: bridge < 75 um)',
    'subs': 'organic substrate 1 mm (950 um under a 50 um EMIB layer in 2.5D)',
    'heatsink': 'v1 THERMAL_Z heatsink', 'tim2': 'v1 THERMAL_Z tim2', 'heatspread': 'v1 THERMAL_Z heatspread',
}


def provenance(cfg):
    """Unit / source strings for meta.json; values that differ from the preset default are flagged."""
    mats = MATERIALS_V11 if cfg['preset'] == 'v1.1' else MATERIALS_V1
    out_m = {}
    for m, k in cfg['materials'].items():
        dflt = mats.get(m)
        if dflt is None:
            src = 'not in the preset table'
        elif k != dflt[0]:
            src = 'USER OVERRIDE (preset default {} W/mK: {})'.format(dflt[0], dflt[1])
        else:
            src = dflt[1]
        out_m[m] = {'k_W_mK': k, 'source': src}
    base = PRESETS[cfg['preset']]()['stack']
    thick = {}
    for part in ('top', 'per_die', 'bottom', 'bottom_2p5d'):
        for name, t, grade in (cfg['stack'].get(part) or []):
            dflt = {n: tt for n, tt, _ in (base.get(part) or [])}.get(name)
            src = THICKNESS_SOURCES.get(name, '')
            if dflt is not None and dflt != t:
                src = 'USER OVERRIDE (preset default {} um) '.format(dflt) + src
            thick['{}.{}'.format(part, name)] = {'thickness_um': t, 'grade': grade, 'source': src}
    return {'materials': out_m, 'thickness': thick,
            'settings': {k: {'unit': u, 'source': s} for k, (u, s) in SOURCES.items()},
            'bonding': BONDING[cfg['bonding']]}
