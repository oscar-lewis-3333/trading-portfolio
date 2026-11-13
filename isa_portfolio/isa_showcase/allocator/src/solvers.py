



import numpy as np
import quadprog
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import linprog
from scipy.spatial.distance import squareform

EPS_T = 1e-10
EPS_W = 1e-14




def waterfill(p, caps, total):

    p = np.maximum(np.asarray(p, float), 0.0)
    caps = np.asarray(caps, float)
    if total <= 0 or p.sum() <= 0:
        return np.zeros_like(p)
    
    act = p > 0
    if caps[act].sum() <= total + 1e-15:
        return np.where(act, caps, 0.0)
    
    lo, hi = 0.0, total / p[act].min() + 1.0
    while np.minimum(caps, hi * p).sum() < total:
        hi *= 2

    for _ in range(200):
        mid = (lo + hi) / 2
        if np.minimum(caps, mid * p).sum() < total:
            lo = mid
        else:
            hi = mid
    w = np.minimum(caps, hi * p)
    return w * (total / w.sum())


def clip_scale(w, caps, total_max):

    w = np.clip(np.asarray(w, float), 0.0, caps)
    s = w.sum()
    if s > total_max + 1e-15 and s > 0:
        return w * (total_max / s), True
    
    return w, False




def _qp(G, a, C, b, meq):
    G = (G + G.T) / 2
    x, f, *_ = quadprog.solve_qp(G, a, C, b, meq)

    return x


def _feasible_rows(k, caps, total, eq, a_prev):

    n = 2 * k
    rows, rhs, meq = [], [], 0
    if eq:
        r = np.zeros(n); r[:k] = 1.0
        rows.append(r); rhs.append(total); meq = 1
    else:
        r = np.zeros(n); r[:k] = -1.0
        rows.append(r); rhs.append(-total)
    for i in range(k):
        r = np.zeros(n); r[i] = 1.0; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[i] = -1.0; rows.append(r); rhs.append(-caps[i])
        r = np.zeros(n); r[k + i] = 1.0; r[i] = -1.0; rows.append(r); rhs.append(-a_prev[i])
        r = np.zeros(n); r[k + i] = 1.0; r[i] = 1.0; rows.append(r); rhs.append(a_prev[i])
    return np.array(rows).T, np.array(rhs), meq


def mean_variance(mu, Sig, c, gamma, cost, a_prev, caps, total, eq):

    k = len(mu)
    if k == 0:
        return np.zeros(0)

    if eq and total <= 0:
        return np.zeros(k)
    
    G = np.zeros((2 * k, 2 * k))
    G[:k, :k] = gamma * Sig + EPS_W * np.eye(k)
    G[k:, k:] = EPS_T * np.eye(k)
    a = np.r_[mu - gamma * c, -cost]
    C, b, meq = _feasible_rows(k, caps, total, eq, a_prev)
    x = _qp(G, a, C, b, meq)
    return np.clip(x[:k], 0.0, caps)


def tangency(mu_M, var_M, mu, Sig, c, cost, a_prev, caps, total, eq):


    k = len(mu)
    if k == 0:
        return np.zeros(0)
    if eq and total <= 0:
        return np.zeros(k)
    n = 1 + 2 * k
    S = np.zeros((1 + k, 1 + k))
    S[0, 0], S[0, 1:], S[1:, 0], S[1:, 1:] = var_M, c, c, Sig
    G = np.zeros((n, n))
    G[:1 + k, :1 + k] = S + EPS_W * np.eye(1 + k)
    G[1 + k:, 1 + k:] = EPS_T * np.eye(k)
    rows, rhs, meq = [], [], 0
    if eq:
        r = np.zeros(n); r[1:1 + k] = 1.0; r[0] = -total
        rows.append(r); rhs.append(0.0); meq = 1
    r = np.zeros(n); r[0] = mu_M; r[1:1 + k] = mu; r[1 + k:] = -cost
    rows.append(r); rhs.append(1.0)
    if not eq:
        r = np.zeros(n); r[1:1 + k] = -1.0; r[0] = total
        rows.append(r); rhs.append(0.0)
    r = np.zeros(n); r[0] = 1.0; rows.append(r); rhs.append(0.0)
    for i in range(k):
        r = np.zeros(n); r[1 + i] = 1.0; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[1 + i] = -1.0; r[0] = caps[i]; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[1 + k + i] = 1.0; r[1 + i] = -1.0; r[0] = a_prev[i]; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[1 + k + i] = 1.0; r[1 + i] = 1.0; r[0] = -a_prev[i]; rows.append(r); rhs.append(0.0)
    C, b = np.array(rows).T, np.array(rhs)
    try:
        x = _qp(G, np.zeros(n), C, b, meq)
    except ValueError:

        return mean_variance(np.zeros(k), Sig, c, 1.0, cost, a_prev, caps, total, eq)
    v0 = x[0]
    if v0 <= 1e-14:
        return mean_variance(np.zeros(k), Sig, c, 1.0, cost, a_prev, caps, total, eq)
    return np.clip(x[1:1 + k] / v0, 0.0, caps)


def sharpe_of(w, mu_M, var_M, mu, Sig, c, cost=None, a_prev=None):
    num = mu_M + mu @ w - (0.0 if cost is None else cost @ np.abs(w - a_prev))
    den = np.sqrt(var_M + 2 * c @ w + w @ Sig @ w)
    return num / den


def max_diversification(sd, Sig):

    k = len(sd)
    if k == 1:
        return np.ones(1)
    
    G = Sig + 1e-14 * np.eye(k)
    C = np.column_stack([sd, np.eye(k)])
    b = np.r_[1.0, np.zeros(k)]
    y = np.clip(_qp(G, np.zeros(k), C, b, 1), 0.0, None)
    return y / y.sum()


def erc(Sig, budget=None, tol=1e-12, maxit=100):

    k = Sig.shape[0]
    if k == 1:
        return np.ones(1)
    
    b = np.full(k, 1.0 / k) if budget is None else np.asarray(budget, float)
    y = b / np.sqrt(np.diag(Sig))
    y = y / np.sqrt(y @ Sig @ y)
    f = lambda z: 0.5 * z @ Sig @ z - b @ np.log(z)
    for _ in range(maxit):
        g = Sig @ y - b / y
        H = Sig + np.diag(b / y ** 2)
        d = np.linalg.solve(H, g)
        if np.abs(g).max() < tol:
            break

        s = 1.0
        while np.any(y - s * d <= 0):
            s *= 0.5
        f0 = f(y)

        while f(y - s * d) > f0 - 1e-4 * s * (g @ d) and s > 1e-12:
            s *= 0.5
        y = y - s * d

    return y / y.sum()


def _ivp(Sig, idx):
    iv = 1 / np.diag(Sig)[idx]

    return iv / iv.sum()


def _cluster_var(Sig, idx):
    w = _ivp(Sig, idx)

    return float(w @ Sig[np.ix_(idx, idx)] @ w)


def _tree(R):
    k = R.shape[0]
    d = np.sqrt(np.clip((1 - R) / 2, 0.0, None))
    np.fill_diagonal(d, 0.0)

    return linkage(squareform(d, checks=False), method="single")


def hrp(Sig, R):

    k = Sig.shape[0]
    if k == 1:
        return np.ones(1)
    
    order = to_tree(_tree(R)).pre_order()
    w = np.ones(k)
    items = [order]
    while items:
        items = [i[j:k2] for i in items for j, k2 in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for a in range(0, len(items), 2):
            c0, c1 = items[a], items[a + 1]
            v0, v1 = _cluster_var(Sig, c0), _cluster_var(Sig, c1)
            al = 1 - v0 / (v0 + v1)
            w[c0] *= al
            w[c1] *= 1 - al

    return w / w.sum()


def herc(Sig, R):


    k = Sig.shape[0]
    if k == 1:
        return np.ones(1)
    
    w = np.ones(k)

    def walk(node, scale):
        if node.is_leaf():
            w[node.id] = scale
            return
        
        L, Rn = node.get_left().pre_order(), node.get_right().pre_order()
        vL, vR = _cluster_var(Sig, L), _cluster_var(Sig, Rn)
        aL = (1 / vL) / (1 / vL + 1 / vR)
        walk(node.get_left(), scale * aL)
        walk(node.get_right(), scale * (1 - aL))

    walk(to_tree(_tree(R)), 1.0)
    return w / w.sum()




def min_cvar(core, X, alpha, cost, a_prev, caps, total, eq):

    T, k = X.shape
    if k == 0 or (eq and total <= 0):
        return np.zeros(k)

    nv = 2 * k + 1 + T
    cvec = np.r_[np.zeros(k), cost, 1.0, np.full(T, 1.0 / ((1 - alpha) * T))]
    A, bub = [], []

    Au = np.zeros((T, nv)); Au[:, :k] = -X; Au[:, 2 * k] = -1.0; Au[np.arange(T), 2 * k + 1 + np.arange(T)] = -1.0
    A.append(Au); bub.append(core)
    At = np.zeros((2 * k, nv))
    At[:k, :k] = np.eye(k); At[:k, k:2 * k] = -np.eye(k)
    At[k:, :k] = -np.eye(k); At[k:, k:2 * k] = -np.eye(k)
    A.append(At); bub.append(np.r_[a_prev, -a_prev])
    Aeq = beq = None
    if eq:
        Aeq = np.zeros((1, nv)); Aeq[0, :k] = 1.0; beq = [total]
    else:
        r = np.zeros((1, nv)); r[0, :k] = 1.0; A.append(r); bub.append([total])

    bounds = [(0, caps[i]) for i in range(k)] + [(0, None)] * k + [(None, None)] + [(0, None)] * T
    res = linprog(cvec, A_ub=np.vstack(A), b_ub=np.concatenate(bub), A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"CVaR LP failed: {res.message}")
    
    return np.clip(res.x[:k], 0.0, caps)


def min_cdar(core, X, alpha, cost, a_prev, caps, total, eq):


    T, k = X.shape
    if k == 0 or (eq and total <= 0):
        return np.zeros(k)
    
    cumc, cumX = np.cumsum(core), np.cumsum(X, axis=0)

    iz, ih, iu = 2 * k, 2 * k + 1, 2 * k + 1 + T
    nv = 2 * k + 1 + 2 * T
    cvec = np.zeros(nv); cvec[k:2 * k] = cost; cvec[iz] = 1.0; cvec[iu:] = 1.0 / ((1 - alpha) * T)
    rows, rhs = [], []

    A1 = np.zeros((T, nv)); A1[:, :k] = cumX; A1[np.arange(T), ih + np.arange(T)] = -1.0
    rows.append(A1); rhs.append(-cumc)

    A2 = np.zeros((T - 1, nv)); A2[np.arange(T - 1), ih + np.arange(T - 1)] = 1.0; A2[np.arange(T - 1), ih + 1 + np.arange(T - 1)] = -1.0
    rows.append(A2); rhs.append(np.zeros(T - 1))

    A3 = np.zeros((T, nv)); A3[np.arange(T), ih + np.arange(T)] = 1.0; A3[:, :k] = -cumX; A3[:, iz] = -1.0
    A3[np.arange(T), iu + np.arange(T)] = -1.0
    rows.append(A3); rhs.append(cumc)
    At = np.zeros((2 * k, nv))

    At[:k, :k] = np.eye(k); At[:k, k:2 * k] = -np.eye(k)
    At[k:, :k] = -np.eye(k); At[k:, k:2 * k] = -np.eye(k)
    rows.append(At); rhs.append(np.r_[a_prev, -a_prev])
    Aeq = beq = None

    if eq:
        Aeq = np.zeros((1, nv)); Aeq[0, :k] = 1.0; beq = [total]
    else:
        r = np.zeros((1, nv)); r[0, :k] = 1.0; rows.append(r); rhs.append([total])
    bounds = [(0, caps[i]) for i in range(k)] + [(0, None)] * k + [(None, None)] + [(0, None)] * T + [(0, None)] * T
    res = linprog(cvec, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"CDaR LP failed: {res.message}")
    
    return np.clip(res.x[:k], 0.0, caps)


def path_drawdowns(core_paths, X_paths, w):


    r = core_paths + X_paths @ w
    y = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r, axis=1)], axis=1)
    peak = np.maximum.accumulate(y, axis=1)
    dd = peak - y
    tr = dd.argmax(axis=1)
    pk = np.array([np.argmax(y[p, :tr[p] + 1]) for p in range(len(tr))])
    return dd[np.arange(len(tr)), tr], pk, tr


def cvar_dd_and_grad(core_paths, X_paths, w, alpha):


    D, pk, tr = path_drawdowns(core_paths, X_paths, w)
    P = len(D)
    m = int(np.ceil((1 - alpha) * P))
    top = np.argsort(-D)[:m]
    val = float(D[top].mean())
    cumX = np.concatenate([np.zeros((P, 1, X_paths.shape[2])), np.cumsum(X_paths, axis=1)], axis=1)
    g = (cumX[top, pk[top], :] - cumX[top, tr[top], :]).mean(axis=0)
    return val, g


def m19_solve(core_paths, X_paths, mean_vec, alpha, limit, cost, a_prev, caps, total, eq, tol=1e-7, maxit=100):


    k = len(mean_vec)
    if k == 0 or (eq and total <= 0):
        return np.zeros(k), {"status": "empty"}

    nv = 2 * k
    base_A, base_b = [], []
    At = np.zeros((2 * k, nv))
    At[:k, :k] = np.eye(k); At[:k, k:] = -np.eye(k)
    At[k:, :k] = -np.eye(k); At[k:, k:] = -np.eye(k)
    base_A.append(At); base_b.append(np.r_[a_prev, -a_prev])
    Aeq = beq = None
    if eq:
        Aeq = np.zeros((1, nv)); Aeq[0, :k] = 1.0; beq = [total]

    else:
        r = np.zeros((1, nv)); r[0, :k] = 1.0; base_A.append(r); base_b.append([total])

    bounds = [(0, caps[i]) for i in range(k)] + [(0, None)] * k
    if not eq:
        v0, _ = cvar_dd_and_grad(core_paths, X_paths, np.zeros(k), alpha)
        if v0 > limit + tol:
            return np.zeros(k), {"status": "infeasible_core", "cvar0": v0}
    cuts_A, cuts_b = [], []

    obj = np.r_[-mean_vec, cost]
    w = None
    for it in range(maxit):
        A = np.vstack(base_A + ([np.array(cuts_A)] if cuts_A else []))
        b = np.concatenate(base_b + ([np.array(cuts_b)] if cuts_b else []))
        res = linprog(obj, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if res.status != 0:
            break

        w = np.clip(res.x[:k], 0.0, caps)
        val, g = cvar_dd_and_grad(core_paths, X_paths, w, alpha)
        if val <= limit + tol:
            return w, {"status": "optimal", "iterations": it + 1, "cvar": val}

        row = np.zeros(nv); row[:k] = g
        cuts_A.append(row); cuts_b.append(limit - val + g @ w)

    return min_cvar_dd(core_paths, X_paths, alpha, caps, total, eq, a_prev, cost, maxit), {"status": "min_cvar_fallback"}


def min_cvar_dd(core_paths, X_paths, alpha, caps, total, eq, a_prev, cost, maxit=100, tol=1e-9):

    k = X_paths.shape[2]
    nv = 2 * k + 1
    At = np.zeros((2 * k, nv))
    At[:k, :k] = np.eye(k); At[:k, k:2 * k] = -np.eye(k)
    At[k:, :k] = -np.eye(k); At[k:, k:2 * k] = -np.eye(k)
    base_A, base_b = [At], [np.r_[a_prev, -a_prev]]
    Aeq = beq = None
    if eq:
        Aeq = np.zeros((1, nv)); Aeq[0, :k] = 1.0; beq = [total]
    else:
        r = np.zeros((1, nv)); r[0, :k] = 1.0; base_A.append(r); base_b.append([total])
    bounds = [(0, caps[i]) for i in range(k)] + [(0, None)] * k + [(0, None)]
    obj = np.r_[np.zeros(k), cost, 1.0]
    cuts_A, cuts_b, best, best_v = [], [], None, np.inf
    for it in range(maxit):
        A = np.vstack(base_A + ([np.array(cuts_A)] if cuts_A else []))
        b = np.concatenate(base_b + ([np.array(cuts_b)] if cuts_b else []))
        res = linprog(obj, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        w = np.clip(res.x[:k], 0.0, caps)
        val, g = cvar_dd_and_grad(core_paths, X_paths, w, alpha)
        tot = val + cost @ np.abs(w - a_prev)
        if tot < best_v:
            best, best_v = w, tot
        if val - res.x[-1] <= tol:
            break

        row = np.zeros(nv); row[:k] = g; row[-1] = -1.0
        cuts_A.append(row); cuts_b.append(g @ w - val)
    return best


def tangency_many(problems, cost, a_prev, caps, total, eq):


    k = len(cost)
    if k == 0:
        return [np.zeros(0) for _ in problems]
    
    if eq and total <= 0:
        return [np.zeros(k) for _ in problems]
    
    n = 1 + 2 * k
    rows, rhs, meq = [], [], 0
    if eq:
        r = np.zeros(n); r[1:1 + k] = 1.0; r[0] = -total
        rows.append(r); rhs.append(0.0); meq = 1
    imean = len(rows)
    rows.append(np.zeros(n)); rhs.append(1.0)
    if not eq:
        r = np.zeros(n); r[1:1 + k] = -1.0; r[0] = total
        rows.append(r); rhs.append(0.0)
    r = np.zeros(n); r[0] = 1.0; rows.append(r); rhs.append(0.0)
    for i in range(k):
        r = np.zeros(n); r[1 + i] = 1.0; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[1 + i] = -1.0; r[0] = caps[i]; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[1 + k + i] = 1.0; r[1 + i] = -1.0; r[0] = a_prev[i]; rows.append(r); rhs.append(0.0)
        r = np.zeros(n); r[1 + k + i] = 1.0; r[1 + i] = 1.0; r[0] = -a_prev[i]; rows.append(r); rhs.append(0.0)

    C, b = np.array(rows).T.copy(), np.array(rhs)
    out = []
    G = np.zeros((n, n))
    G[1 + k:, 1 + k:] = EPS_T * np.eye(k)
    for mu_M, var_M, mu, Sig, c in problems:
        C[0, imean] = mu_M
        C[1:1 + k, imean] = mu
        C[1 + k:, imean] = -cost
        G[0, 0] = var_M + EPS_W
        G[0, 1:1 + k] = c
        G[1:1 + k, 0] = c
        G[1:1 + k, 1:1 + k] = Sig + EPS_W * np.eye(k)
        try:
            x = quadprog.solve_qp(G, np.zeros(n), C, b, meq)[0]
            v0 = x[0]
            w = np.clip(x[1:1 + k] / v0, 0.0, caps) if v0 > 1e-14 else None
        except ValueError:
            w = None

        if w is None:
            w = mean_variance(np.zeros(k), Sig, c, 1.0, cost, a_prev, caps, total, eq)
            
        out.append(w)
    return out
