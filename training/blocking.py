"""Phase 4 of the make-it-work plan: does blocking high-disagreement nodes stop
spread better than structural targeting? Independent Cascade, source unknown.

Design (fixed before results):
  - Graph: the undirected social graph from views.py (R5: direction not required).
  - Spread: live-edge IC. Edge u->v transmits with p_uv:
      'wc'      weighted cascade, p_uv = 1/deg(v) (assumed model, R7)
      'ic:c'    uniform p = c * p_c, p_c = <k>/(<k^2>-<k>) the epidemic threshold
                (a sweep of c, because who wins flips with transmissibility —
                Hebert-Dufresne et al. 2013)
  - Source unknown: n_seeds seeds drawn uniformly from nodes that NO strategy
    blocks at the largest budget (so no strategy wins by deleting the source).
  - Common random numbers: every strategy and budget sees the same live-edge
    samples and seeds, so differences are not Monte Carlo noise.
  - Outcome per run: infected share (seeds excluded) and cross-community share
    (infected nodes outside every seed's T community).

    python blocking.py timme --sims 200
    python blocking.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

import views
from metrics import log


# --------------------------------------------------------------------------- #
# targeting strategies: each returns node ids ranked best-first
def rank(score):
    return np.argsort(-np.asarray(score), kind="stable")


def adaptive_degree(a, kmax):
    """Remove the current highest-degree node, update degrees, repeat."""
    a = a.tocsr()
    deg = np.asarray(a.sum(1)).ravel().astype(np.float64)
    out = []
    for _ in range(kmax):
        i = int(np.argmax(deg))
        out.append(i)
        nb = a.indices[a.indptr[i]:a.indptr[i + 1]]
        deg[nb] -= 1
        deg[i] = -np.inf
    return np.asarray(out)


def collective_influence(a):
    """CI at radius 1: (k_i - 1) * sum over neighbours of (k_j - 1)
    (Morone & Makse 2015; radius 2's frontier is most of TIMME)."""
    k = np.asarray(a.sum(1)).ravel()
    return (k - 1) * (a @ (k - 1))


def gateway(a, t, p_c=None):
    """Expected flow a node relays OUT of a community: sum over communities A of
    (edges into A) * (acceptance of A's content) * (edges outside A).
    p_c=None -> acceptance 1/k for every community: the same formula without
    content, so gateway vs gateway_struct isolates what the text adds."""
    import ambiguity
    tt = np.where(t < 0, int(t.max()) + 1, t)
    h = (a @ ambiguity.onehot(tt)).toarray()[:, : int(t.max()) + 1]
    deg = np.asarray(a.sum(1)).ravel()[:, None]
    acc = np.full(h.shape, 1.0 / h.shape[1]) if p_c is None else p_c
    return (h * acc * (deg - h)).sum(1)


def strategies(b, s, kmax, seed=0):
    """Ranked node lists. s = ambiguity.compute() output."""
    rng = np.random.default_rng(seed)
    pr = s["pagerank"] / s["pagerank"].mean()
    deg = b["deg"]
    import igraph as ig  # noqa: F401  (betweenness via the bundle's graph)
    piv = rng.choice(b["n"], min(b["n"], 500), replace=False).tolist()
    btw = np.asarray(b["g"].betweenness(sources=piv), dtype=np.float64)
    return {
        "random": rng.permutation(b["n"]),
        "degree": rank(deg),
        "adaptive_degree": adaptive_degree(b["A"], kmax),
        "pagerank": rank(pr),
        "betweenness": rank(btw),
        "collective_influence": rank(collective_influence(b["A"])),
        "participation": rank(s["participation"]),
        "participation_x_pr": rank(s["participation"] * pr),
        "consensus_entropy_x_pr": rank(s["consensus_entropy"] * pr),
        "ambiguity_T_x_pr": rank(s["ambiguity_T"] * pr),
        "content_outlier_x_pr": rank(s["content_outlier"] * pr),
        "disagreement": rank(s["disagreement"]),
        "disagreement_x_pr": rank(s["disagreement"] * pr),
        "gateway": rank(gateway(b["A"], s["t"], s["p_C"])),
        "gateway_struct": rank(gateway(b["A"], s["t"])),
        "misfit": rank(s["misfit"]),
        "misfit_x_pr": rank(s["misfit"] * pr),
    }


# --------------------------------------------------------------------------- #
# simulation
def edge_probs(a, model):
    a = a.tocoo()
    deg = np.asarray(a.tocsr().sum(1)).ravel()
    if model == "wc":
        return a.row, a.col, 1.0 / deg[a.col]
    c = float(model.split(":")[1])
    pc = deg.mean() / ((deg ** 2).mean() - deg.mean())
    return a.row, a.col, np.full(a.nnz, min(1.0, c * pc))


def reach(live, seeds, blocked):
    """Nodes reachable from seeds in the live-edge graph with blocked nodes removed."""
    n = live.shape[0]
    keep = np.ones(n, bool)
    keep[blocked] = False
    # super-source: one extra node linked to every seed
    src = np.full(len(seeds), n)
    m = sp.csr_matrix((np.ones(len(seeds)), (src, seeds)), shape=(n + 1, n + 1))
    lv = live.multiply(keep[:, None]).multiply(keep[None, :]).tocsr()
    lv = sp.bmat([[lv, None], [None, sp.csr_matrix((1, 1))]]).tocsr() + m
    order = breadth_first_order(lv, n, directed=True, return_predecessors=False)
    hit = np.zeros(n, bool)
    hit[order[order < n]] = True
    return hit


def simulate(b, ranked, comm, budgets, model, sims=200, n_seeds=10, seed=0, affinity=None,
             raw=False):
    """Returns {strategy: {budget: (mean infected share, mean cross-community share)}}
    plus the no-blocking baseline under key 'none'.

    affinity: optional (n, k) matrix, rows summing to 1 = how much each node agrees
    with each community's content. Content-aware spread: each cascade starts in
    one community A (seeds drawn inside A) and edge u->v transmits with
    p_uv * k * affinity[v, A] (mean multiplier ~1, so the overall rate is kept).
    Selective exposure / echo chambers, Del Vicario et al. PNAS 2016."""
    rng = np.random.default_rng(seed)
    r, c, p = edge_probs(b["A"], model)
    if affinity is not None:
        kc = affinity.shape[1]
    n = b["n"]
    kmax = max(budgets)
    blocked_any = np.unique(np.concatenate([v[:kmax] for v in ranked.values()]))
    pool = np.setdiff1d(np.arange(n), blocked_any)
    res = {name: {k: [] for k in budgets} for name in ranked}
    res["none"] = {0: []}
    for it in range(sims):
        if affinity is None:
            live_e = rng.random(len(p)) < p
            seeds = rng.choice(pool, n_seeds, replace=False)
        else:
            src = int(comm[rng.choice(pool)])          # source community, size-weighted
            src = min(src, kc - 1)
            live_e = rng.random(len(p)) < np.minimum(1.0, p * kc * affinity[c, src])
            inside = pool[comm[pool] == src]
            seeds = rng.choice(inside, min(n_seeds, len(inside)), replace=False)
        live = sp.csr_matrix((np.ones(live_e.sum()), (r[live_e], c[live_e])), shape=(n, n))
        seed_comms = np.unique(comm[seeds])

        def outcome(blocked):
            hit = reach(live, seeds, blocked)
            hit[seeds] = False
            cross = hit & ~np.isin(comm, seed_comms)
            return hit.sum() / (n - len(seeds)), cross.sum() / (n - len(seeds))

        res["none"][0].append(outcome(np.array([], dtype=np.int64)))
        for name, order in ranked.items():
            for k in budgets:
                res[name][k].append(outcome(order[:k]))
        if (it + 1) % 25 == 0:
            log(f"simulate[{model}]: {it + 1}/{sims}")
    if raw:   # per-run (infected, cross) arrays, for paired comparisons
        return {name: {k: np.asarray(v) for k, v in d.items()} for name, d in res.items()}
    # per cell: [mean infected, mean cross, SE infected, SE cross] over runs
    return {name: {k: np.r_[np.mean(v, axis=0), np.std(v, axis=0, ddof=1) / np.sqrt(len(v))].tolist()
                   for k, v in d.items()} for name, d in res.items()}


def table(res, budgets, col=0):
    """col 0 = total infected, col 1 = cross-community infected."""
    base = res["none"][0]
    lines = [f"{'strategy':24s} " + " ".join(f"k={k:<11d}" for k in budgets) +
             f"   (no blocking: infected {base[0]:.4f}, cross {base[1]:.4f}; cells = % of infected removed)"]
    lines[0] += " [" + ("total" if col == 0 else "cross-community") + "]"
    order = sorted([s for s in res if s != "none"], key=lambda s: res[s][budgets[-1]][col])
    for s in order:
        # +- = 1.96 SE of this strategy's mean, as % of the unblocked mean (CRN makes
        # paired differences tighter than this, so it is a conservative band)
        cells = [f"{100 * (1 - res[s][k][col] / max(base[col], 1e-12)):5.1f}%"
                 f"\u00b1{196 * res[s][k][col + 2] / max(base[col], 1e-12):4.1f}" for k in budgets]
        lines.append(f"{s:24s} " + " ".join(cells))
    return "\n".join(lines)


def selftest():
    """Barbell-like case: two dense groups joined through 3 bridge nodes. Blocking
    the bridges must cut cross-community spread more than blocking the same
    number of random nodes; no blocking must spread across."""
    rng = np.random.default_rng(0)
    n = 200
    gt = np.repeat([0, 1], n // 2)
    p = np.where(gt[:, None] == gt[None, :], 0.2, 0.0)
    up = np.triu(rng.random((n, n)) < p, 1)
    bridges = np.array([0, 1, 2])
    for x in bridges:
        up[x, rng.choice(np.arange(100, 200), 15, replace=False)] = True
    s_, d_ = np.nonzero(up | up.T)
    b = views._bundle(np.stack([s_, d_]), np.eye(n)[:, :8], np.ones(n, bool), gt)
    ranked = {"bridges": np.r_[bridges, np.arange(3, n)],
              "random": rng.permutation(np.arange(3, n))}
    res = simulate(b, ranked, gt, [3], "ic:3", sims=40, n_seeds=2)
    assert res["none"][0][1] > 0.05, res
    assert res["bridges"][3][1] < 0.5 * res["random"][3][1], res
    # content-aware: receivers in group 1 reject group-0 messages and vice versa
    # (affinity one-hot on own group) -> almost no cross-group spread even unblocked
    aff = np.eye(2)[gt] * 0.98 + 0.01
    res = simulate(b, ranked, gt, [3], "ic:3", sims=40, n_seeds=2, affinity=aff)
    assert res["none"][0][1] < 0.02, res
    print("blocking self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=["timme", *views.POKEC_SLICES])
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--models", default="wc,ic:0.5,ic:1,ic:2,ic:4")
    ap.add_argument("--budgets", default="20,50,100,200")
    ap.add_argument("--scores", default="ambiguity.npz", help="file in out/<ds>_views/")
    ap.add_argument("--affinity", default=None,
                    help="timme: 'glove' = content-aware spread, affinity from a "
                         "cross-fitted GloVe-only classifier (score file should be MiniLM-only)")
    ap.add_argument("--only", default=None, help="comma list of strategies to run")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.dataset:
        selftest()
    else:
        b = views.load_timme() if a.dataset == "timme" else views.load_pokec(a.dataset)
        d = os.path.join(views.OUT, f"{a.dataset}_views")
        s = dict(np.load(os.path.join(d, a.scores)))
        budgets = [int(x) for x in a.budgets.split(",")]
        ranked = strategies(b, s, max(budgets))
        if a.only:
            ranked = {k: v for k, v in ranked.items() if k in a.only.split(",")}
        comm = np.where(s["t"] < 0, s["t"].max() + 1, s["t"])
        aff = None
        if a.affinity == "glove":
            import ambiguity
            aff = ambiguity.content_probs(b["glove"], s["t"])
            log(f"affinity: GloVe-only classifier, acc {(aff.argmax(1) == s['t']).mean():.3f}")
        allres = {}
        for model in a.models.split(","):
            res = simulate(b, ranked, comm, budgets, model, a.sims, affinity=aff)
            allres[model] = res
            print(f"\n== {model} ==\n" + table(res, budgets) + "\n" + table(res, budgets, 1), flush=True)
        tag = os.path.splitext(a.scores)[0] + (f"_{a.affinity}" if a.affinity else "")
        json.dump(allres, open(os.path.join(d, f"blocking_{tag}.json"), "w"), indent=1)
