"""Linear solvers (plan §8.2): SuperLU for n <= direct_max_n, otherwise pyamg smoothed-aggregation + CG.

Acceptance (revised after review v0.1, F2/F4/F8): the relative residual ||P - G theta|| / ||P|| has a round-off floor
of about eps * ||G|| ||theta|| / ||P||, which exceeds 1e-10 on thin-layer stacks even for an exact direct solve.
The solution is therefore accepted on the normwise backward error
    berr = ||P - G theta||_inf / (||G||_inf ||theta||_inf + ||P||_inf)  <=  berr_tol
(the energy balance is checked separately by the driver).  CG stops when berr <= berr_stop (checked on the
iterate every few iterations) or when its own rtol is met; non-convergence raises.
AMG setup uses pyamg's random spectral-radius estimate: the global RNG is seeded for the setup and restored, so
results are bitwise reproducible.
"""
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

AMG_STRENGTH_DEFAULT = ('symmetric', {'theta': 0.25})


def amg_params(scfg):
    st = scfg.get('amg_strength') or AMG_STRENGTH_DEFAULT
    if isinstance(st, list):
        st = tuple(st)
    if isinstance(st, tuple) and len(st) == 2 and isinstance(st[1], list):
        st = (st[0], dict(st[1]))
    return dict(symmetry='symmetric', strength=st,
                presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                max_coarse=2000, coarse_solver='splu')


def _import_pyamg(path):
    try:
        import pyamg
        return pyamg, getattr(pyamg, '__file__', '')
    except ImportError:
        pass
    if path and path not in sys.path:
        sys.path.insert(0, path)
    try:
        import pyamg
        return pyamg, getattr(pyamg, '__file__', '')
    except ImportError:
        return None, None


class _Done(Exception):
    def __init__(self, x):
        self.x = x


def backward_error(G, x, b, Ginf=None):
    r = b - G @ x
    if Ginf is None:
        Ginf = float(abs(G).sum(axis=1).max())
    den = Ginf * float(np.max(np.abs(x))) + float(np.max(np.abs(b)))
    return float(np.max(np.abs(r))) / den if den > 0 else 0.0, r


_ML_CACHE = {}     # one AMG hierarchy kept per process when scfg['keep_amg'] (GCI: refinement + error-bound solves)


def _fingerprint(G):
    d = G.data
    return (G.shape, G.nnz, float(d.sum()), float(d[::max(1, d.size // 4096)].sum()), float(np.abs(d).max()))


def run(G, b, scfg, force=None):
    """Solve G x = b.  force: None | 'direct' | 'amg'.  Returns (x, info); raises on failure."""
    n = G.shape[0]
    t0 = time.time()
    bn = float(np.linalg.norm(b))
    info = {'n': int(n), 'nnz': int(G.nnz)}
    if bn == 0.0:
        info.update(method='zero_rhs', resid=0.0, berr=0.0, iterations=0, time_s=0.0)
        return np.zeros(n), info
    if not np.all(np.isfinite(b)):
        raise ValueError('non-finite right-hand side')
    method = force or ('direct' if n <= scfg['direct_max_n'] else 'amg')
    Ginf = float(abs(G).sum(axis=1).max())
    berr_tol = scfg.get('berr_tol', 1e-12)
    if method == 'amg':
        pyamg, where = _import_pyamg(scfg.get('pyamg_path'))
        if pyamg is None:
            why = ('AMG was requested' if force == 'amg' else
                   'n = {} is above direct_max_n = {}: refusing a direct factorisation of unknown fill-in'
                   .format(n, scfg['direct_max_n']))
            raise RuntimeError('pyamg not found (solver.pyamg_path = {!r}); {}'.format(scfg.get('pyamg_path'), why))
    if method == 'direct':
        x = spla.spsolve(G.tocsc(), b)
        info.update(method='spsolve', iterations=0)
    elif method == 'amg':
        prm = amg_params(scfg)
        key = (_fingerprint(G), repr(prm), scfg.get('amg_seed', 0)) if scfg.get('keep_amg') else None
        ml = _ML_CACHE.get('ml') if key is not None and _ML_CACHE.get('key') == key else None
        if ml is None:
            _ML_CACHE.clear()                         # drop an old hierarchy before building (delta review r_solve F4)
            state = np.random.get_state()
            np.random.seed(scfg.get('amg_seed', 0))
            try:
                ml = pyamg.smoothed_aggregation_solver(G, **prm)      # seeded: the same G gives the same hierarchy
            finally:
                np.random.set_state(state)
            if key is not None:
                _ML_CACHE.update(key=key, ml=ml)
        M = ml.aspreconditioner(cycle='V')
        it = [0]
        stop = scfg.get('berr_stop', 1e-13)
        every = scfg.get('berr_check_every', 10)

        def cb(xk):
            it[0] += 1
            if it[0] % every == 0:
                be, _ = backward_error(G, xk, b, Ginf)
                if be <= stop:
                    raise _Done(np.array(xk, copy=True))
        try:
            x, flag = spla.cg(G, b, rtol=scfg['rtol'], atol=0.0, maxiter=scfg['maxiter'], M=M, callback=cb)
        except _Done as d:
            x, flag = d.x, 0
        info.update(method='cg+sa_amg', iterations=it[0], cg_flag=int(flag), pyamg_version=pyamg.__version__,
                    pyamg_file=where, amg_levels=len(ml.levels), amg_seed=scfg.get('amg_seed', 0),
                    amg_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in prm.items()})
        if flag != 0:
            be, _ = backward_error(G, x, b, Ginf)
            if be > berr_tol:
                raise RuntimeError('CG did not converge (flag {}, {} iterations, berr {:.2e})'.format(flag, it[0], be))
    else:
        raise ValueError('unknown solver method ' + repr(method))
    if not np.all(np.isfinite(x)):
        raise RuntimeError('non-finite solution')
    berr, r = backward_error(G, x, b, Ginf)
    info.update(resid=float(np.linalg.norm(r)) / bn, berr=berr, time_s=time.time() - t0)
    if berr > berr_tol:
        raise RuntimeError('backward error {:.3e} above tolerance {:.1e} ({})'.format(berr, berr_tol, method))
    return x, info
