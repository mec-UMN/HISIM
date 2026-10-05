"""Grid-convergence certification of tile-level labels (plan §5.4, review R03/R04).

Family: G0 = production grid (mesh settings of cfg), G1 / G2 = every lateral interval AND every sublayer split
2x / 4x (lat_level = vert_level = 1, 2 on top of the production cells_per_tile).  Diagnostic families refine only
laterally or only vertically.  The neutral input is built once (frozen layout); only the mesh changes.

Per tile (fixed tile id) and per quantity phi in {mean rise, max rise} of its active-layer cells:
  e_f = phi_m - phi_f,  e_c = phi_c - phi_m,  R = e_f / e_c
  0 < R < 1: p = ln(e_c / e_f) / ln 2,  phi_ext = phi_f + (phi_f - phi_m) / (2^p - 1)
  E_prod   = 1.25 |phi_c - phi_ext| / Theta_max    (error of the PRODUCTION grid; Theta_max = largest tile max rise)
  GCI_fine = 1.25 |e_f| / (Theta_max (2^p - 1))
special states: 'oscillatory' (R <= 0), 'divergent' (R >= 1), 'roundoff' (|e_f|, |e_c| < 1e-9 Theta_max),
'p_out_of_range' (p outside [0.5, 4]).  Non-monotone tiles get a range-based estimate (Roache, Fs = 3).
Two verdicts are reported (final review plan_b3b1 F1-F4):
  'd11'     - plan §5.4 / D11 as written: every tile, for BOTH labels (mean, max), E_prod < 1% AND state monotone or
              roundoff, AND the linear-solve error bound < 1e-3 min(|e_f|, |e_c|) (or roundoff); only the joint family
              is an acceptance family, lateral / vertical are diagnostic;
  'relaxed' - tile-MEAN label only, every tile E_prod < 1% (non-monotone tiles by the range estimate).
Linear-solve error per level (plan §5.4): every level gets one step of mixed-precision iterative refinement (residual
in extended precision, correction by the same solver), then an a-posteriori estimate that is a bound up to the
accuracy of its own solve: G is a nonsingular M-matrix, so G^-1 >= 0 and the solve error obeys
|theta - theta*| <= G^-1 (|res| + gamma) componentwise (res formed in extended precision, gamma = its rounding
allowance); one extra solve per level gives this bound, reduced per tile like the labels.
"""
import copy
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FS = 1.25          # three-grid monotone estimate (Roache)
FS_RANGE = 3.0     # non-monotone / range-based estimate: Roache's conservative factor (final review F3)


def _level_cfg(cfg, lat, vert):
    c = copy.deepcopy(cfg)
    c['mesh']['lat_level'] = cfg['mesh']['lat_level'] + lat
    c['mesh']['vert_level'] = cfg['mesh']['vert_level'] + vert
    c['output'] = dict(c['output'], plots=False)
    c['solver'] = dict(c['solver'], keep_amg=True)    # refinement and bound solves reuse the level's AMG hierarchy
    return c


def _resid_ld(G, th, b):
    """Residual b - G th and |G| |th| in extended precision (np.longdouble)."""
    L = np.longdouble
    prod = G.data.astype(L) * th[G.indices].astype(L)
    nz = np.diff(G.indptr) > 0                    # reduceat over the non-empty rows only (delta review r_gci F1)
    Gth = np.zeros(G.shape[0], L)
    aGth = np.zeros(G.shape[0], L)
    if G.nnz:
        Gth[nz] = np.add.reduceat(prod, G.indptr[:-1][nz])
        aGth[nz] = np.add.reduceat(np.abs(prod), G.indptr[:-1][nz])
    return b.astype(L) - Gth, aGth


def _refine(r, steps=1):
    """Mixed-precision iterative refinement of a driver result (residual in np.longdouble, correction by the same
    solver): the linear-solve error becomes negligible next to the grid differences (plan §5.4).  Labels and peaks
    are recomputed from the refined solution."""
    from . import solve
    from .output import compute_peaks, tile_labels
    G = r.op.G.tocsr()
    inner = dict(r.cfg['solver'], rtol=1e-8, berr_tol=1e-6, berr_stop=1e-9)   # a low-accuracy correction suffices
    for _ in range(steps):
        res, _ = _resid_ld(G, r.theta.ravel(), r.P.ravel())
        d, _ = solve.run(G, res.astype(float), inner)
        r.theta = (r.theta.ravel().astype(np.longdouble) + d).astype(float).reshape(r.P.shape)
    r.T = r.theta + r.cfg['bc']['T_amb']
    r.peaks = compute_peaks(r.T, r.grid, r.stack, r.maps, r.ti)
    r.tiles = tile_labels(r.T, r.grid, r.stack, r.ti)
    return r


def _solve_error_bound(r):
    """A-posteriori estimate of the linear-solve error of a driver result, per cell: e = G^-1 (|res| + gamma).  With an
    exact solve of this system it is a componentwise upper bound (G^-1 >= 0); e itself comes from the iterative solver
    (normwise backward error 1e-12) plus a 0.1 % margin, so for very small components it is an estimate, not a proof
    (delta review r_gci F2).  The residual is formed in extended precision (np.longdouble): its rounding allowance
    gamma is ~1e-19 relative, and e is ~10x the error instead of ~300x with a double residual (checked against direct
    solves)."""
    from . import solve
    G = r.op.G.tocsr()
    b = r.P.ravel()
    res, aGth = _resid_ld(G, r.theta.ravel(), b)
    gam = 16 * np.finfo(np.longdouble).eps * (aGth + np.abs(b).astype(np.longdouble))
    rhs = (np.abs(res) + gam).astype(float) * (1.0 + 1e-12)
    e, info = solve.run(G, rhs, r.cfg['solver'])
    return np.maximum(e, 0.0).reshape(r.P.shape) * 1.001, info      # 0.1 % margin for the bound solve itself


def _solve_level(args):
    ti, cfg, lat, vert = args[:4]
    with_bound = args[4] if len(args) > 4 else True
    from .driver import run
    from .output import tile_labels
    r = run(ti, _level_cfg(cfg, lat, vert), None, write=False)
    if with_bound:
        _refine(r)                                   # one refinement step (residual in extended precision)
    T_amb = cfg['bc']['T_amb']
    tiles = {t['id']: (t['T_mean_K'] - T_amb, t['T_max_K'] - T_amb) for t in r.tiles if t['kind'] != 'router'}
    peaks = {c: (v['T_K'] - T_amb if v.get('valid') else None) for c, v in r.peaks.items()}
    hot = max(tiles.items(), key=lambda kv: kv[1][1])[0] if tiles else None
    out = {'lat': lat, 'vert': vert, 'n': r.grid.n, 'grid_id': r.meta['production_grid_id'], 'tiles': tiles,
           'peaks_rise_K': peaks, 'hottest_tile': hot, 'solver': r.solver_info, 'energy_rel_err': r.energy['rel_err']}
    if with_bound:
        e, info = _solve_error_bound(r)
        out['solve_err_tiles'] = {t['id']: (t['T_mean_K'], t['T_max_K']) for t in tile_labels(e, r.grid, r.stack, ti)
                                  if t['kind'] != 'router'}
        out['solve_err_max_K'] = float(e.max())
        out['solve_err_solver'] = {k: info.get(k) for k in ('method', 'iterations', 'berr', 'time_s')}
    out['refined'] = bool(with_bound)
    from .solve import _ML_CACHE
    _ML_CACHE.clear()                                # release the level's AMG hierarchy (delta review r_solve F4)
    return out


def _three(pc, pm, pf, theta_max):
    """Three-grid estimate (Roache 2009 / Celik 2008).  Non-monotone cases get a range-based estimate
    E = Fs * max(|e_f|, |e_c|) / Theta_max.  An apparent order above 2 (the formal order of the scheme) is replaced by 2
    for the error estimate, which enlarges it; a low order is used as it is (also conservative, review plan_b3b1 F1).
    The raw p is reported and p outside [0.5, 4] is flagged 'p_out_of_range'."""
    ef, ec = pm - pf, pc - pm
    tol = 1e-9 * theta_max
    if abs(ef) < tol and abs(ec) < tol:
        return {'state': 'roundoff', 'E_prod': FS * abs(pc - pf) / theta_max, 'p': None, 'phi_ext': pf, 'GCI_fine': 0.0}
    R = ef / ec if ec != 0 else float('inf')
    if R <= 0 or R >= 1:
        rng = max(abs(ef), abs(ec))
        return {'state': 'oscillatory' if R <= 0 else 'divergent', 'R': R, 'p': None, 'phi_ext': None,
                'E_prod': FS_RANGE * rng / theta_max, 'GCI_fine': FS_RANGE * abs(ef) / theta_max, 'range_rel': rng / theta_max}
    p_raw = math.log(ec / ef) / math.log(2.0)
    p = min(p_raw, 2.0)
    ext = pf + (pf - pm) / (2.0 ** p - 1.0)
    st = 'monotone' if 0.5 <= p_raw <= 4.0 else 'p_out_of_range'
    return {'state': st, 'R': R, 'p': p_raw, 'p_used': p, 'phi_ext': ext, 'E_prod': FS * abs(pc - ext) / theta_max,
            'GCI_fine': FS * abs(ef) / (theta_max * (2.0 ** p - 1.0))}


NOISE_FLOOR = 1e-4   # 'd11_noise_exempt': a label that changes by < 1e-4 Theta_max over the levels counts as converged


def d11_verdict(per, raw, theta_max, family_kind, have_bound, noise_floor=None):
    """plan §5.4 / D11 verdict from the per-tile three-grid records `per` and the raw level values `raw`
    ({tile: [[mean, max] per level]}).  noise_floor = None: the plan as written (roundoff = 1e-9 Theta_max);
    otherwise a label whose largest change between the levels is below noise_floor * Theta_max is treated like the
    roundoff state (non-monotone noise far below the 1 % target).  Also used by tools/run_gci.py --remake."""
    out = {'acceptance_family': family_kind == 'joint', 'solve_error_checked': have_bound,
           'noise_floor_rel': noise_floor, 'labels': {}}
    for q, idx in (('mean', 0), ('max', 1)):
        bad, exempt = [], 0
        for tid, rec in per.items():
            r_ = rec[q]
            vals = [lv[idx] for lv in raw[tid]]
            chg = max(abs(vals[1] - vals[2]), abs(vals[0] - vals[1])) / theta_max
            if noise_floor is not None and chg < noise_floor:
                exempt += 1
                continue
            why = []
            if not (r_['E_prod'] is not None and r_['E_prod'] < 0.01):
                why.append('E_prod')
            if r_['state'] not in ('monotone', 'roundoff'):
                why.append(r_['state'])
            if have_bound and not r_.get('solve_err_ok', False):
                why.append('solve_error')
            if why:
                bad.append({'tile': tid, 'why': why, 'E_prod': r_['E_prod'], 'max_change_rel': chg})
        out['labels'][q] = {'pass': (not bad) if have_bound else None, 'n_fail': len(bad), 'n_noise_exempt': exempt,
                            'fail': sorted(bad, key=lambda x: -(x['E_prod'] or 0))[:50]}
    # None = not evaluated (no solve-error estimate, run with --no_solve_error; delta review r_solve F2)
    out['pass'] = (bool(out['acceptance_family'] and all(v['pass'] for v in out['labels'].values()))
                   if have_bound else None)
    return out


def analyse(levels, family_kind='joint'):
    """levels: three _solve_level results, coarse to fine."""
    c, m, f = levels
    theta_max = max(v[1] for v in f['tiles'].values())
    have_bound = all('solve_err_tiles' in lv for lv in levels)
    per = {}
    for tid in f['tiles']:
        rec = {}
        for q, idx in (('mean', 0), ('max', 1)):
            rec[q] = _three(c['tiles'][tid][idx], m['tiles'][tid][idx], f['tiles'][tid][idx], theta_max)
            if have_bound:                          # plan §5.4: solve error << grid error
                sb = max(lv['solve_err_tiles'][tid][idx] for lv in levels)
                vals = [lv['tiles'][tid][idx] for lv in levels]
                emin = min(abs(vals[1] - vals[2]), abs(vals[0] - vals[1]))
                rec[q]['solve_err_bound_K'] = sb
                rec[q]['solve_err_ok'] = bool(rec[q]['state'] == 'roundoff' or sb < 1e-3 * emin)
        per[tid] = rec
    summ = {}
    for q in ('mean', 'max'):
        states = {}
        eps = []
        worst = (None, -1.0)
        for tid, rec in per.items():
            st = rec[q]['state']
            states[st] = states.get(st, 0) + 1
            if rec[q]['E_prod'] is not None:
                eps.append(rec[q]['E_prod'])
                if rec[q]['E_prod'] > worst[1]:
                    worst = (tid, rec[q]['E_prod'])
        ps = [rec[q]['p'] for rec in per.values() if rec[q]['p'] is not None]
        ok = all(rec[q]['E_prod'] is not None and rec[q]['E_prod'] < 0.01 for rec in per.values())
        summ[q] = {'states': states, 'E_prod_max': max(eps) if eps else None, 'E_prod_median': float(np.median(eps)) if eps else None,
                   'worst_tile': worst[0], 'p_median': float(np.median(ps)) if ps else None,
                   'p_min': min(ps) if ps else None, 'p_max': max(ps) if ps else None, 'certified': bool(ok)}
    raw = {tid: [list(lv['tiles'][tid]) for lv in levels] for tid in f['tiles']}
    d11 = d11_verdict(per, raw, theta_max, family_kind, have_bound)
    d11_noise = d11_verdict(per, raw, theta_max, family_kind, have_bound, noise_floor=NOISE_FLOOR)
    peaks = {}
    for cl in c['peaks_rise_K']:
        vals = [lv['peaks_rise_K'][cl] for lv in levels]
        if None not in vals:
            peaks[cl] = {'rise_K': vals, 'three_grid': _three(vals[0], vals[1], vals[2], theta_max)}
    return {'theta_max_K': theta_max, 'summary': summ, 'per_tile': per, 'peaks_report_only': peaks,
            'hottest_tile_per_level': [lv['hottest_tile'] for lv in levels],
            'certified': bool(summ['mean']['certified']),            # relaxed: tile-MEAN label, range estimate allowed
            'certified_max_label': bool(summ['max']['certified']),   # tile max = single-cell maximum (not smooth)
            'd11': d11,                                               # plan §5.4 / D11 as written
            'd11_noise_exempt': d11_noise,                            # D11, roundoff threshold raised to NOISE_FLOOR
            'solve_err_max_K': [lv.get('solve_err_max_K') for lv in levels]}


def family(ti, cfg, kind='joint', workers=3, solve_error_bound=True):
    """Solve the three levels of a refinement family in parallel processes; returns (levels, analysis)."""
    steps = {'joint': [(0, 0), (1, 1), (2, 2)], 'lateral': [(0, 0), (1, 0), (2, 0)],
             'vertical': [(0, 0), (0, 1), (0, 2)]}[kind]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        levels = list(ex.map(_solve_level, [(ti, cfg, a, b, solve_error_bound) for a, b in steps]))
    out = analyse(levels, kind)
    out['family'] = kind
    out['levels'] = [{k: v for k, v in lv.items() if k not in ('tiles', 'solve_err_tiles')} for lv in levels]
    out['tiles_raw_rise_K'] = {tid: [list(lv['tiles'][tid]) for lv in levels] for tid in levels[-1]['tiles']}
    return levels, out
