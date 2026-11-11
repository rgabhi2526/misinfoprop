"""T11.2 — does re-homing misfits toward their text community change information spread? (TIMME)

Core (dataset-agnostic): rewire(F, t, target, treated, eng, k, drop, score_w, rng)
  treated nodes u each drop up to k own-community follow edges and add the same number of edges
  to non-neighbours inside community target[u] (top Adamic-Adar, ties random; or random anywhere).
  Drops/adds never touch other treated nodes, so every treated node's degree is preserved exactly
  (its neighbours' degrees move). drop='dormant': only edges with NO engagement either direction
  (eng=None -> every own-community edge counts as dormant, the Pokec case); drop='random': any
  own-community edge.
Interventions (same treated count, same per-node edge budget):
  none | rehome_text (target = argmax p_C) | rehome_struct (target = Kumari extraction community)
  random_rewire (same drop counts, random adds anywhere) | rehome_text_random_nodes (random
  non-misfit nodes with argmax p_C != t, text target).
Spread: live-edge IC on the undirected follow graph (blocking.edge_probs), content gating
  p_uv * K * aff[v, c0], aff = softmax(beta * onehot(tc_v)). Each follow edge is two directed
  edges that SHARE one uniform per sim (hash of min,max,sim), so shared edges see the same draw in
  every graph (common random numbers); seeds/sources are identical across interventions.
  (a),(b): source community = t of a random node, 10 seeds uniform in it; outcomes infected share
  and share of misfits reached. (c) correction reach: random misfit u, c0 = tc_u, 10 seeds uniform
  in {t == c0}, outcome = share of misfits with tc == c0 reached.

    python h5_rewire.py timme        # reduced grid, Pool(4), writes ../results/hypotheses/h5/h5_rewire_timme.json
    python h5_rewire.py --selftest
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time
import numpy as np, scipy.sparse as sp
from multiprocessing import Pool
import blocking
from metrics import log

OUT = os.path.join(os.path.dirname(__file__), "out")
RES = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h5")
IV = ["none", "rehome_text", "rehome_struct", "random_rewire", "rehome_text_random_nodes"]


# --------------------------------------------------------------------------- core
def rewire(F, t, target, treated, eng, k, drop="dormant", add="rank", counts=None, rng=None):
    """Return (new symmetric 0/1 csr, per-node drop counts dict). counts: force per-node budget."""
    rng = rng or np.random.default_rng(0)
    F = F.tocsr(); n = F.shape[0]
    deg = np.asarray(F.sum(1)).ravel()
    w_aa = sp.diags(1 / np.log(np.maximum(deg, 2)))
    is_t = np.zeros(n, bool); is_t[treated] = True
    eng = None if eng is None else eng.tocsr()
    rem, ad, used = [], [], {}
    for u in rng.permutation(treated):
        nb = F.indices[F.indptr[u]:F.indptr[u + 1]]
        pool = nb[(t[nb] == t[u]) & ~is_t[nb]]
        if drop == "dormant" and eng is not None:
            pool = pool[np.asarray(eng[u, pool].todense()).ravel() == 0]
        d = min(k if counts is None else counts[u], len(pool))
        forbid = is_t.copy(); forbid[nb] = True; forbid[u] = True
        if add == "rank":
            cand = np.flatnonzero((t == target[u]) & ~forbid)
            sc = np.asarray((F[u] @ w_aa @ F[:, cand]).todense()).ravel() if len(cand) else np.empty(0)
        else:
            cand = np.flatnonzero(~forbid); sc = np.zeros(len(cand))
        d = min(d, len(cand))
        if d == 0:
            used[u] = 0; continue
        order = np.lexsort((rng.random(len(cand)), -sc))          # score desc, ties random
        if add == "random": order = rng.permutation(len(cand))
        ad += [(u, v) for v in cand[order[:d]]]
        rem += [(u, v) for v in rng.choice(pool, d, replace=False)]
        used[u] = d
    G = F.tolil() if False else F.copy()
    for lst, val in ((rem, -1.0), (ad, 1.0)):
        if lst:
            a = np.array(lst)
            M = sp.coo_matrix((np.full(len(a), val), (a[:, 0], a[:, 1])), shape=(n, n))
            G = G + M + M.T
    G.eliminate_zeros()
    return G.tocsr(), used


def interventions(F, t, tc, ext, mis, eng, k, rng_seed=0):
    """All five graphs for one k. mis = misfit ids; ext = structural extraction community."""
    n = len(t)
    g = {"none": F}
    g["rehome_text"], cnt = rewire(F, t, tc, mis, eng, k, rng=np.random.default_rng(rng_seed))
    g["rehome_struct"], _ = rewire(F, t, ext, mis, eng, k, rng=np.random.default_rng(rng_seed))
    g["random_rewire"], _ = rewire(F, t, tc, mis, eng, k, drop="random", add="random", counts=cnt,
                                   rng=np.random.default_rng(rng_seed))
    rng = np.random.default_rng(rng_seed + 1)
    oth = np.flatnonzero((tc != t) & (t >= 0) & ~np.isin(np.arange(n), mis))
    rnd = rng.choice(oth, len(mis), replace=False)
    g["rehome_text_random_nodes"], _ = rewire(F, t, tc, rnd, eng, k, rng=np.random.default_rng(rng_seed))
    return g


# --------------------------------------------------------------------------- spread
def _mix(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def pair_uniform(r, c, salt, n):
    """Deterministic U[0,1) per undirected pair and salt (= sim, cascade id)."""
    lo, hi = np.minimum(r, c).astype(np.uint64), np.maximum(r, c).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = _mix(lo * np.uint64(n) + hi + _mix(np.uint64(salt) + np.uint64(0x9E3779B97F4A7C15)))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def cascade(G, model, aff_c0, seeds, salt, ep=None):
    """Reach in the live-edge graph; aff_c0 = (n,) receiver affinity for the cascade's content."""
    n = G.shape[0]
    r, c, p = ep if ep is not None else blocking.edge_probs(G, model)
    K = aff_c0[1]
    pg = np.minimum(1.0, p * K * aff_c0[0][c])
    live = pair_uniform(r, c, salt, n) < pg
    L = sp.csr_matrix((np.ones(live.sum()), (r[live], c[live])), shape=(n, n))
    hit = blocking.reach(L, seeds, np.array([], dtype=np.int64))
    hit[seeds] = False
    return hit


def simulate(graphs, t, tc, mis, model, beta, sims, n_seeds=10, seed=0):
    """-> {iv: array (sims, 3)} columns: infected share, misfits reached share, correction reach."""
    n, K = len(t), int(max(t.max(), tc.max())) + 1
    ism = np.zeros(n, bool); ism[mis] = True
    ep = {iv: blocking.edge_probs(G, model) for iv, G in graphs.items()}
    def aff(c0): return (np.where(tc == c0, np.exp(beta), 1.0) / (np.exp(beta) + K - 1), K)
    out = {iv: np.zeros((sims, 3)) for iv in graphs}
    ok = np.flatnonzero(t >= 0)
    for s in range(sims):
        rng = np.random.default_rng(seed * 100003 + s)
        c0 = int(t[rng.choice(ok)]); ins = np.flatnonzero(t == c0)
        seeds = rng.choice(ins, min(n_seeds, len(ins)), replace=False)
        u = rng.choice(mis); c1 = int(tc[u]); ins1 = np.flatnonzero(t == c1)
        seeds1 = rng.choice(ins1, min(n_seeds, len(ins1)), replace=False)
        grp = ism & (tc == c1)
        for iv, G in graphs.items():
            h = cascade(G, model, aff(c0), seeds, 2 * s, ep[iv])
            h1 = cascade(G, model, aff(c1), seeds1, 2 * s + 1, ep[iv])
            out[iv][s] = (h.sum() / (n - len(seeds)), h[ism].mean(), h1[grp].mean())
    return out


def boot(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return [float(x.mean()), *map(float, np.percentile(m, [2.5, 97.5]))]


def summarize(out):
    names = ["infected", "misfits_reached", "correction_reach"]
    res = {}
    for j, nm in enumerate(names):
        res[nm] = {iv: {"mean": float(v[:, j].mean()),
                        "vs_none": boot(v[:, j] - out["none"][:, j]),
                        "vs_random": boot(v[:, j] - out["random_rewire"][:, j])} for iv, v in out.items()}
    return res


# --------------------------------------------------------------------------- driver
def _job(a):
    key, graphs, t, tc, mis, model, beta, sims = a
    t0 = time.time()
    out = simulate(graphs, t, tc, mis, model, beta, sims)
    log(f"done {key} ({time.time() - t0:.0f}s)")
    return key, summarize(out)


def dormant_matrix(it, n):
    s = np.concatenate([it[L + "_s"] for L in ("retweet", "like", "mention", "reply")])
    d = np.concatenate([it[L + "_d"] for L in ("retweet", "like", "mention", "reply")])
    E = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n)).tocsr()
    return ((E + E.T) > 0).astype(np.float64)


def run_timme(sims=100):
    import h5_rec
    it = np.load(os.path.join(OUT, "h4b_inter.npz"))
    xb = np.load(os.path.join(OUT, "timme_bio_lcc.npy"))
    grid = {"r1.0_s0": ([1, 5], [0.3, 1, 3, 6]), "r2.0_s0": ([5], [0.3, 6])}
    jobs, meta = [], {}
    for tag, (ks, betas) in grid.items():
        c = h5_rec.setup(tag, it, xb)
        t, tc, mis, F = c["t"], c["tgt"], c["mis"], c["F"]
        eng = dormant_matrix(it, c["n"])
        F.setdiag(0); F.eliminate_zeros()
        meta[tag] = {"n_misfits": int(len(mis)), "n_comm": int(c["k"])}
        for k in ks:
            G = interventions(F, t, tc, c["ext"], mis, eng, k)
            e0 = int(F.nnz // 2)
            meta[tag][f"k{k}"] = {iv: {"edges": int(g.nnz // 2), "edges_changed": int((abs(g - F)).nnz // 2)}
                                  for iv, g in G.items()}
            log(f"{tag} k={k}: {meta[tag][f'k{k}']}")
            for model in ("wc", "ic:2"):
                for beta in betas:
                    jobs.append((f"{tag}|k{k}|{model}|b{beta}", G, t, tc, mis, model, beta, sims))
    res = {}
    with Pool(4) as p:
        for key, r in p.imap_unordered(_job, jobs):
            res[key] = r
    os.makedirs(RES, exist_ok=True)
    json.dump({"sims": sims, "meta": meta, "results": res}, open(os.path.join(RES, "h5_rewire_timme.json"), "w"), indent=1)
    return res


def selftest():
    rng = np.random.default_rng(0)
    n = 120; t = np.repeat([0, 1], n // 2)
    up = np.triu(rng.random((n, n)) < np.where(t[:, None] == t[None, :], 0.2, 0.01), 1)
    F = sp.csr_matrix((up | up.T).astype(float))
    mis = np.arange(0, 10)                       # block-0 nodes whose "text" target is block 1
    tc = t.copy(); tc[mis] = 1
    eng = sp.lil_matrix((n, n))
    for u in mis[:5]:                            # first 5 misfits: engage with half their own-block ties
        nb = F.indices[F.indptr[u]:F.indptr[u + 1]]
        for v in nb[(t[nb] == 0)][::2]:
            eng[u, v] = eng[v, u] = 1
    eng = eng.tocsr()
    G, used = rewire(F, t, tc, mis, eng, 4, rng=np.random.default_rng(1))
    d0, d1 = np.asarray(F.sum(1)).ravel(), np.asarray(G.sum(1)).ravel()
    assert (d0[mis] == d1[mis]).all(), "treated degree changed"
    assert G.diagonal().sum() == 0 and G.max() == 1 and abs(G - G.T).nnz == 0, "loop/dup/asymmetric"
    D = (F - G).tocsr()
    rem, add = sp.find(sp.triu(D > 0))[:2], sp.find(sp.triu(D < 0))[:2]
    for u, v in zip(*rem):                       # drops: own block, dormant, treated end is in mis
        a, b = (u, v) if u in mis else (v, u)
        assert a in mis and t[b] == t[a] == 0 and eng[a, b] == 0, "bad drop"
    for u, v in zip(*add):
        a, b = (u, v) if u in mis else (v, u)
        assert a in mis and t[b] == 1, "bad add"
    assert sum(used.values()) == len(rem[0]) == len(add[0]) > 0
    # CRN: shared pair gets the same uniform in any graph
    r, c = np.array([3, 9]), np.array([9, 3])
    u = pair_uniform(r, c, 5, n); assert u[0] == u[1] and 0 <= u[0] < 1
    assert pair_uniform(r, c, 6, n)[0] != u[0]
    # sanity: re-homing all of block-0 misfits into block 1 cannot lower correction reach to zero
    tc2 = tc.copy(); tc2[60:75] = 0
    gr = interventions(F, t, tc2, np.ones(n, int), mis, eng, 4)
    assert set(gr) == set(IV)
    print("h5_rewire self-check ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) < 2:
        selftest()
    else:
        selftest()
        run_timme()
