"""Finite-volume conduction operator (plan §8.1).

Two-point flux between neighbouring cells i, j across a face of area A:  G = A / (d_i/(2 k_i) + d_j/(2 k_j)).
Robin boundaries: top G_top = dx dy / (dz_0/(2 k_0) + 1/h_top), bottom likewise with h_bottom (if > 0); sides adiabatic.
Unknown: temperature rise theta = T - T_amb (same T_amb on both boundaries), so  G theta = P.
"""
import numpy as np
import scipy.sparse as sp


class Operator:
    def __init__(self, G, g_top, g_bot, diag):
        self.G, self.g_top, self.g_bot, self.diag = G, g_top, g_bot, diag


def build_operator(grid, K, bc):
    nk, ny, nx = grid.nk, grid.ny, grid.nx
    if K.shape != (nk, ny, nx):
        raise ValueError('K shape mismatch')
    h_top, h_bot = float(bc['h_top']), float(bc['h_bottom'])
    if h_top < 0 or h_bot < 0 or not (np.isfinite(h_top) and np.isfinite(h_bot)):
        raise ValueError('h must be finite and >= 0')
    if h_top == 0 and h_bot == 0:
        raise ValueError('both boundaries adiabatic: singular operator')
    if np.any(K <= 0) or not np.all(np.isfinite(K)):
        raise ValueError('conductivity must be finite and > 0')
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    n = nk * ny * nx
    idx = np.arange(n, dtype=np.int64).reshape(nk, ny, nx)
    rows, cols, vals = [], [], []
    diag = np.zeros((nk, ny, nx))

    def add_faces(Gf, a, b):
        rows.extend([a.ravel(), b.ravel()])
        cols.extend([b.ravel(), a.ravel()])
        vals.extend([-Gf.ravel(), -Gf.ravel()])

    if nx > 1:
        r = dx[None, None, :] / (2.0 * K)
        Gx = (dz[:, None, None] * dy[None, :, None]) / (r[:, :, :-1] + r[:, :, 1:])
        add_faces(Gx, idx[:, :, :-1], idx[:, :, 1:])
        diag[:, :, :-1] += Gx
        diag[:, :, 1:] += Gx
    if ny > 1:
        r = dy[None, :, None] / (2.0 * K)
        Gy = (dz[:, None, None] * dx[None, None, :]) / (r[:, :-1, :] + r[:, 1:, :])
        add_faces(Gy, idx[:, :-1, :], idx[:, 1:, :])
        diag[:, :-1, :] += Gy
        diag[:, 1:, :] += Gy
    if nk > 1:
        r = dz[:, None, None] / (2.0 * K)
        Gz = (dy[None, :, None] * dx[None, None, :]) / (r[:-1] + r[1:])
        add_faces(Gz, idx[:-1], idx[1:])
        diag[:-1] += Gz
        diag[1:] += Gz
    area = dy[:, None] * dx[None, :]
    g_top = np.zeros((ny, nx))
    g_bot = np.zeros((ny, nx))
    if h_top > 0:
        g_top = area / (dz[0] / (2.0 * K[0]) + 1.0 / h_top)
    if h_bot > 0:
        g_bot = area / (dz[-1] / (2.0 * K[-1]) + 1.0 / h_bot)
    diag[0] += g_top
    diag[-1] += g_bot
    rows.append(idx.ravel())
    cols.append(idx.ravel())
    vals.append(diag.ravel())
    G = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    G.sum_duplicates()
    op = Operator(G, g_top, g_bot, diag)
    check_operator(op, grid)
    return op


def check_operator(op, grid):
    """Symmetry, positive diagonal, row sums = boundary conductance (plan §8.1)."""
    G = op.G
    asym = abs(G - G.T)
    if asym.nnz and asym.max() > 0:
        raise ValueError('operator not symmetric')
    d = G.diagonal()
    if np.any(d <= 0):
        raise ValueError('non-positive diagonal')
    rs = np.asarray(G.sum(axis=1)).ravel()
    bvec = np.zeros(G.shape[0])
    nlay = grid.ny * grid.nx
    bvec[:nlay] += op.g_top.ravel()
    bvec[-nlay:] += op.g_bot.ravel()
    err = np.max(np.abs(rs - bvec)) / np.max(d)
    if err > 1e-12:
        raise ValueError('row sums differ from boundary conductances (rel {:.3e})'.format(err))
    return True


def energy_balance(op, theta, P_total):
    th = theta.reshape(-1)
    nlay = op.g_top.size
    q_top = float(np.dot(op.g_top.ravel(), th[:nlay]))
    q_bot = float(np.dot(op.g_bot.ravel(), th[-nlay:]))
    err = abs(q_top + q_bot - P_total)
    ok = err <= 1e-12 + 1e-6 * abs(P_total)
    return {'Q_top_W': q_top, 'Q_bottom_W': q_bot, 'P_W': float(P_total), 'abs_err_W': err,
            'rel_err': err / P_total if P_total else 0.0, 'ok': bool(ok)}
