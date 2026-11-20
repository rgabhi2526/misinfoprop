"""Radar (Li, Dani, Hu, Liu, IJCAI 2017) residual-analysis anomaly scorer, and the
E05 planted-misfit benchmark re-run with Radar added as an extra baseline.

  min_{W,R} ||X - W^T X - R||_F^2 + a||W||_{2,1} + b||R||_{2,1} + g tr(R^T L R)
Updates (paper Alg. 1): W = (X X^T + a D_W)^-1 X (X-R)^T ; R = (I + b D_R + g L)^-1 (X - W^T X)
with D_W,ii = 1/(2||w_i||), D_R,ii = 1/(2||r_i||). Score = ||r_i||_2.
The n x n inverse is never formed: Woodbury on (a D_W + X X^T) keeps everything d x d, and
row norms of W are computed exactly via ||w_i||^2 = (MX)_i C (MX)_i^T, C=(X-R)^T(X-R).
R solves (I + b D_R + g L) R = X - W^T X with Jacobi-preconditioned CG on all columns (sparse LU/dense Cholesky are slow at n=4000).

    python radar.py            # self-check + full E05 grid (3 seeds) -> results/detection/synthetic_benchmark_radar.json
"""
import json, os, sys, time
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

EPS = 1e-8
HP = (10.0, 0.001, 100.0)   # (alpha, beta, gamma): paper defaults a=b=g=1 collapse R to ~1e-9 (n>>d), see JSON note


def _pcg(Ls, dd, B, X0, tol=1e-4, maxit=100):
    """Column-wise Jacobi PCG for (Ls + diag(dd)) X = B, all columns at once."""
    Dg = Ls.diagonal() + dd
    X = X0.copy()
    r = B - (Ls @ X + dd[:, None] * X)
    z = r / Dg[:, None]
    p = z.copy()
    rz = (r * z).sum(0)
    bn = np.linalg.norm(B, axis=0) + EPS
    for _ in range(maxit):
        Ap = Ls @ p + dd[:, None] * p
        al = rz / np.maximum((p * Ap).sum(0), EPS * EPS)
        X += al * p
        r -= al * Ap
        if (np.linalg.norm(r, axis=0) / bn).max() < tol:
            break
        z = r / Dg[:, None]
        rz2 = (r * z).sum(0)
        p = z + (rz2 / np.maximum(rz, EPS * EPS)) * p
        rz = rz2
    return X


def radar(X, A, alpha=1.0, beta=1.0, gamma=1.0, iters=20, tol=1e-4):
    n, d = X.shape
    A = sp.csr_matrix(A)
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    R = np.zeros_like(X)
    dr = np.ones(n) / 2
    dw = np.ones(n) / 2
    Ls = (gamma * L + sp.identity(n)).tocsr()
    for _ in range(iters):
        # W step (Woodbury): M = (a D_W + X X^T)^-1
        ainv = 1.0 / (alpha * dw)
        AX = ainv[:, None] * X
        S = np.linalg.inv(np.eye(d) + X.T @ AX)
        MX = AX - AX @ (S @ (X.T @ AX))
        Y = X - R
        WtX = Y @ (X.T @ MX)                       # W^T X
        C = Y.T @ Y
        wn = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", MX, C, MX), 0))
        dw = 1.0 / (2 * np.maximum(wn, EPS))
        # R step
        Rn = _pcg(Ls, beta * dr, X - WtX, R)
        rn = np.linalg.norm(Rn, axis=1)
        dr = 1.0 / (2 * np.maximum(rn, EPS))
        delta = np.linalg.norm(Rn - R) / (np.linalg.norm(Rn) + EPS)
        R = Rn
        if delta < tol:
            break
    return np.linalg.norm(R, axis=1)


def self_check():
    # two dense blocks with distinct attributes; one node has the other block's attribute
    rng = np.random.default_rng(0)
    n = 60
    blk = np.repeat([0, 1], n // 2)
    A = (rng.random((n, n)) < np.where(blk[:, None] == blk[None], 0.4, 0.01)).astype(float)
    A = np.triu(A, 1); A = A + A.T
    C = 3 * rng.normal(size=(2, 8))
    X = C[blk] + 0.1 * rng.normal(size=(n, 8))
    X[7] = C[1] + 0.1 * rng.normal(size=8)        # node 7 sits in block 0 but carries block-1 attributes
    s = radar(X, A, 1.0, 0.1, 10.0, iters=20)
    assert np.argmax(s) == 7, (np.argmax(s), s[7], s.max())


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    import benchmark, views
    self_check()
    print("self-check ok", flush=True)
    t0 = time.time()
    rows = []
    for nu in (0.0, 1.5):
        for mu in (0.1, 0.3, 0.5):
            for sg in (0.5, 1.0, 2.0):
                per = []
                for sd in (0, 1, 2):
                    ei, text, comm, truth = benchmark.planted(n=4000, mu=mu, sigma=sg, seed=sd, nuisance=nu)
                    b = views._bundle(ei, text, np.ones(len(comm), bool), comm)
                    s = radar(text, b["A"], *HP, iters=10)
                    a = benchmark.aucs({"radar": s}, truth, b["deg"])["radar"]
                    per.append(a)
                rows.append({"nuisance": nu, "mu": mu, "sigma": sg, "seeds": [0, 1, 2],
                             "auc_per_seed": [p[0] for p in per], "auc_within_per_seed": [p[1] for p in per],
                             "auc": [round(float(np.mean([p[0] for p in per])), 3),
                                     round(float(np.mean([p[1] for p in per])), 3)]})
                print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "detection", "synthetic_benchmark_radar.json")
    note = ("Radar a,b,g=%s, 10 iters. Paper-style defaults a=b=g=1 (and 0.01) collapse R to ~1e-9 on this "
            "n=4000,d=64/128 data (single cell nu1.5 mu0.5 sg2 seed0: AUC 0.494). HP picked on one cell "
            "(nu1.5 mu0.5 sg2 seed0) from a ~12-setting sweep using the planted labels, i.e. favourable to Radar." % (HP,))
    json.dump({"note": note, "hp": HP, "rows": rows, "runtime_s": round(time.time() - t0, 1)}, open(out, "w"), indent=1)
    print("runtime_s", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
