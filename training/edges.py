"""Edge-level intervention test: add friction to LINKS instead of removing accounts.

Motivation (regime sweep, 2026-10-04): mismatched people are receivers, not relays,
so node removal always favours hubs. Under agreement-driven spread, the links that
matter are those carrying community A's message into people receptive to A in
other communities -- receptivity that structure-only edge scores cannot see.

Edge scores (undirected link u-v, both directions blocked together):
  random            uniform
  degree_product    k_u * k_v
  pr_product        PR_u * PR_v
  bridge_struct     cross-community links, PR_u * PR_v                (structure only)
  bridge_content    cross-community links, max over directions of
                    PR_u * p_C(v)[t_u]: influential sender x receptive receiver
  bridge_struct_x_recv  PR_u * PR_v * max receptivity: bridge_struct re-weighted by text
bridge_content vs bridge_struct isolates what the text adds.

    python edges.py synthetic     # one planted setting (mu 0.3, beta 4, 5% hub mismatches)
    python edges.py timme         # content-aware spread, GloVe affinity, MiniLM scores
    python edges.py --selftest
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

import ambiguity
import blocking
import views
from metrics import log


def edge_scores(b, s, seed=0):
    rng = np.random.default_rng(seed)
    up = sp.triu(b["A"]).tocoo()
    u, v = up.row, up.col
    pr = s["pagerank"] / s["pagerank"].mean()
    t, pc = s["t"], s["p_C"]
    deg = b["deg"]
    cross = (t[u] != t[v]) & (t[u] >= 0) & (t[v] >= 0)
    tu, tv = np.where(t[u] < 0, 0, t[u]), np.where(t[v] < 0, 0, t[v])
    recv = np.maximum(pr[u] * pc[v, tu], pr[v] * pc[u, tv])
    # structural bridge score re-weighted by receptivity (both ends relay)
    recv2 = pr[u] * pr[v] * np.maximum(pc[v, tu], pc[u, tv])
    sc = {
        "random": rng.random(len(u)),
        "degree_product": deg[u] * deg[v],
        "pr_product": pr[u] * pr[v],
        "bridge_struct": np.where(cross, pr[u] * pr[v], -1.0),
        "bridge_content": np.where(cross, recv, -1.0),
        "bridge_struct_x_recv": np.where(cross, recv2, -1.0),
    }
    return u, v, {k: np.argsort(-x, kind="stable") for k, x in sc.items()}


def reach_edges(live, seeds, cut):
    """Reachable set from seeds after deleting the links in `cut` (sparse 0/1)."""
    n = live.shape[0]
    lv = (live - live.multiply(cut)).tocsr()
    lv.eliminate_zeros()
    src = np.full(len(seeds), n)
    m = sp.csr_matrix((np.ones(len(seeds)), (src, seeds)), shape=(n + 1, n + 1))
    lv = sp.bmat([[lv, None], [None, sp.csr_matrix((1, 1))]]).tocsr() + m
    order = breadth_first_order(lv, n, directed=True, return_predecessors=False)
    hit = np.zeros(n, bool)
    hit[order[order < n]] = True
    return hit


def simulate_edges(b, u, v, ranked, comm, budgets, model, affinity, sims=60, n_seeds=10, seed=0):
    """Content-aware IC (as blocking.simulate) with link cuts. Returns per-run
    cross-community infected shares: {strategy: {budget: array}} + 'none'."""
    rng = np.random.default_rng(seed)
    r, c, p = blocking.edge_probs(b["A"], model)
    n, kc = b["n"], affinity.shape[1]
    cuts = {}
    for name, order in ranked.items():
        for k in budgets:
            e = order[:k]
            cuts[name, k] = sp.csr_matrix((np.ones(2 * k), (np.r_[u[e], v[e]], np.r_[v[e], u[e]])),
                                          shape=(n, n))
    res = {name: {k: [] for k in budgets} for name in ranked}
    res["none"] = {0: []}
    for it in range(sims):
        src = min(int(comm[rng.integers(n)]), kc - 1)
        live_e = rng.random(len(p)) < np.minimum(1.0, p * kc * affinity[c, src])
        live = sp.csr_matrix((np.ones(live_e.sum()), (r[live_e], c[live_e])), shape=(n, n))
        inside = np.flatnonzero(comm == src)
        seeds = rng.choice(inside, min(n_seeds, len(inside)), replace=False)

        def cross(cut):
            hit = reach_edges(live, seeds, cut)
            hit[seeds] = False
            return (hit & (comm != src)).sum() / n

        res["none"][0].append(cross(sp.csr_matrix((n, n))))
        for name in ranked:
            for k in budgets:
                res[name][k].append(cross(cuts[name, k]))
        if (it + 1) % 20 == 0:
            log(f"simulate_edges[{model}]: {it + 1}/{sims}")
    return {name: {k: np.asarray(x) for k, x in d.items()} for name, d in res.items()}


def report(res, budgets):
    base = res["none"][0].mean()
    best = {}
    print(f"no cut: cross-community infected {base:.4f}; cells = % removed (paired 95% CI vs bridge_struct)")
    for name in [k for k in res if k != "none"]:
        cells = []
        for k in budgets:
            rem = 1 - res[name][k] / max(base, 1e-12)
            d = (res["bridge_struct"][k] - res[name][k]) / max(base, 1e-12)
            cells.append(f"{100 * rem.mean():5.1f}% ({100 * d.mean():+5.1f}±{196 * d.std(ddof=1) / np.sqrt(len(d)):4.1f})")
            best[name, k] = rem.mean()
        print(f"  {name:16s} " + "  ".join(cells))
    return best


def synthetic(seed=0, mu=0.3, beta=4.0, plant=0.05, place="hub", n=4000):
    import regime
    ei, text, comm, tc = regime.planted_place(n, mu, plant, place, seed)
    b = views._bundle(ei, text, np.ones(n, bool), comm)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    kc = int(comm.max()) + 1
    aff = np.exp(beta * (tc[:, None] == np.arange(kc)[None, :]))
    return b, s, comm, aff / aff.sum(1, keepdims=True)


def selftest():
    """Two groups; group-1 has 10 'A-minded' receptive nodes linked to group 0.
    Cutting the links into those receptive nodes must cut cross spread more than
    cutting the same number of random cross links."""
    rng = np.random.default_rng(0)
    n = 200
    gt = np.repeat([0, 1], n // 2)
    p = np.where(gt[:, None] == gt[None, :], 0.15, 0.0)
    up = np.triu(rng.random((n, n)) < p, 1)
    recv = np.arange(100, 110)
    for x in recv:
        up[rng.choice(100, 3, replace=False), x] = True
    others = rng.choice(np.arange(110, 200), 30, replace=False)
    for x in others:
        up[rng.choice(100, 1), x] = True
    s_, d_ = np.nonzero(up | up.T)
    b = views._bundle(np.stack([s_, d_]), np.eye(n)[:, :4], np.ones(n, bool), gt)
    tc = gt.copy()
    tc[recv] = 0
    aff = np.exp(6 * (tc[:, None] == np.arange(2)[None, :]))
    aff /= aff.sum(1, keepdims=True)
    up2 = sp.triu(b["A"]).tocoo()
    u, v = up2.row, up2.col
    crossidx = np.flatnonzero(gt[u] != gt[v])
    into_recv = crossidx[np.isin(v[crossidx], recv) | np.isin(u[crossidx], recv)]
    rest = np.setdiff1d(crossidx, into_recv)
    ranked = {"recv": np.r_[into_recv, rest], "random_cross": rng.permutation(crossidx)}
    k = len(into_recv)
    res = simulate_edges(b, u, v, ranked, gt, [k], "ic:3", aff, sims=40, n_seeds=3)
    assert res["recv"][k].mean() < 0.5 * res["random_cross"][k].mean(), (
        res["recv"][k].mean(), res["random_cross"][k].mean())
    print("edges self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", choices=["synthetic", "timme"])
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--model", default="ic:2")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.target:
        selftest()
    elif a.target == "synthetic":
        b, s, comm, aff = synthetic()
        u, v, ranked = edge_scores(b, s)
        budgets = [100, 400, 1600]
        report(simulate_edges(b, u, v, ranked, comm, budgets, a.model, aff, a.sims), budgets)
    else:
        b = views.load_timme()
        d = os.path.join(views.OUT, "timme_views")
        s = dict(np.load(os.path.join(d, "ambiguity_minilm.npz")))
        aff = ambiguity.content_probs(b["glove"], s["t"])
        u, v, ranked = edge_scores(b, s)
        comm = np.where(s["t"] < 0, s["t"].max() + 1, s["t"])
        budgets = [2000, 10000, 40000]
        report(simulate_edges(b, u, v, ranked, comm, budgets, a.model, aff, a.sims), budgets)
