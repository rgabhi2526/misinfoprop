"""H6 — does the misfit score catch targets that hide from community detection by rewiring?

Attacker (content unchanged) rewires only the edges of 200 targets per seed, budget k rewires each
(1 rewire = remove one edge + add one edge, never self-loop/duplicate). Attacks (simplified Kumari 2021 /
Fionda & Pirro):
  kumari    dest = neighbouring community with most of u's edges (other than own; random other if none).
            Step: drop edge to the own-community neighbour with highest internal degree, add edge to the
            non-neighbour in dest with highest internal degree (internal degree = original graph, fixed).
  safeness  drop a random own-community edge, add edge to a random node of a random OTHER community.
  random    matched control: drop a random edge of u, add an edge from u to a random non-neighbour (any community).
  noise     Leiden-instability floor: drop a random edge of u, add a random non-edge anywhere NOT incident to u.
Then Leiden x10 on the attacked graph (resolution 1.0) -> t1 = keep_big(highest-modularity run).
Success = target's t1 differs from the t1 label held by the majority of its original (t0) community-mates
(non-targets). Scores w.r.t. t1: misfit (bio-only cross-fitted p_C), participation, consensus_entropy,
persistence_low (1 - Kumari zeta, ponytail: Ex floored at 1), degree. Detection = ROC-AUC of each score,
targets vs all non-target nodes WITH text; per seed, reported mean +- sd over seeds. Also AUC on the
unattacked graph (t0) with the same targets, and AUC restricted to successful targets.
Targets: t0>=0, degree>=3, has text, not in top-10% misfit on the original graph.

    python h6_deception.py timme | pokec    # Pool(4), writes ../results/hypotheses/h6/h6_<ds>.json
    python h6_deception.py --selftest
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys
import numpy as np, scipy.sparse as sp
import igraph as ig
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score
import ambiguity, views
from metrics import log

OUT = os.path.join(os.path.dirname(__file__), "out")
RES = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h6")
ATTACKS = ("kumari", "safeness", "random", "noise")
SCORES = ("misfit", "participation", "consensus_entropy", "persistence_low", "degree")
D = {}                                                  # per-process dataset globals (fork-shared)


def key(a, b):
    return (a, b) if a < b else (b, a)


# --------------------------------------------------------------------------- attacks
def attack(F, t0, targets, k, kind, rng):
    """-> (G, steps {u: done}, removed set, added set). Every step removes one and adds one edge."""
    F = F.tocsr(); n = F.shape[0]
    ok = t0 >= 0
    intdeg = np.asarray(_same(F, t0).sum(1)).ravel()
    comms = np.unique(t0[ok])
    order = {}
    for c in comms:
        m = np.flatnonzero(t0 == c)
        order[c] = m[np.lexsort((rng.random(len(m)), -intdeg[m]))]
    adj = {}
    def N(u):
        if u not in adj:
            adj[u] = set(F.indices[F.indptr[u]:F.indptr[u + 1]].tolist())
        return adj[u]
    added, removed, steps = set(), set(), {}
    for u in rng.permutation(targets):
        u = int(u); done = 0
        dest = None
        if kind == "kumari":
            cnt = {}
            for w in N(u):
                if t0[w] >= 0 and t0[w] != t0[u]:
                    cnt[t0[w]] = cnt.get(t0[w], 0) + 1
            if cnt:
                dest = max(cnt, key=lambda c: (cnt[c], rng.random()))
            else:
                dest = rng.choice(comms[comms != t0[u]])
        for _ in range(k):
            if kind in ("random", "noise"):
                pool = [w for w in N(u) if key(u, w) not in added]
            else:
                pool = [w for w in N(u) if t0[w] == t0[u] and key(u, w) not in added]
            if not pool:
                continue
            if kind == "kumari":
                best = max(intdeg[w] for w in pool)
                w = int(rng.choice([x for x in pool if intdeg[x] == best]))
                v = next((int(x) for x in order[dest] if x != u and x not in N(u) and key(u, x) not in removed), None)
                a = (u, v)
            elif kind == "safeness":
                w = int(rng.choice(pool)); a = None
                for _try in range(50):
                    c = rng.choice(comms[comms != t0[u]])
                    v = int(rng.choice(order[c]))
                    if v != u and v not in N(u) and key(u, v) not in removed:
                        a = (u, v); break
            elif kind == "random":
                w = int(rng.choice(pool)); a = None
                for _try in range(50):
                    v = int(rng.integers(0, n))
                    if v != u and v not in N(u) and key(u, v) not in removed:
                        a = (u, v); break
            else:
                w = int(rng.choice(pool)); a = None
                for _try in range(50):
                    x, y = (int(z) for z in rng.integers(0, n, 2))
                    if x != y and u not in (x, y) and y not in N(x) and key(x, y) not in removed:
                        a = (x, y); break
            if a is None or a[1] is None:
                continue
            N(u).discard(w); N(w).discard(u)
            removed.add(key(u, w))
            N(a[0]).add(a[1]); N(a[1]).add(a[0])
            added.add(key(*a)); done += 1
        steps[u] = done
    G = F.copy()
    for lst, val in ((removed, -1.0), (added, 1.0)):
        if lst:
            a = np.array(sorted(lst))
            M = sp.coo_matrix((np.full(len(a), val), (a[:, 0], a[:, 1])), shape=(n, n))
            G = G + M + M.T
    G.eliminate_zeros()
    return G.tocsr(), steps, removed, added


def _same(F, t0):
    """F restricted to same-community entries (t0>=0), sparse."""
    c = F.tocoo()
    m = (t0[c.row] == t0[c.col]) & (t0[c.row] >= 0)
    return sp.csr_matrix((c.data[m], (c.row[m], c.col[m])), shape=F.shape)


# --------------------------------------------------------------------------- scores
def persistence_low(A, t):
    """1 - zeta, zeta = 0.5 (I / (max(Ex,1) K) + clustering coeff inside own community)."""
    tt = np.where(t < 0, int(t.max()) + 1, t)
    k = int(tt.max()) + 1
    n = A.shape[0]
    H = (A @ sp.csr_matrix((np.ones(n), (np.arange(n), tt)), shape=(n, k))).toarray()
    I = H[np.arange(n), tt]; K = H.sum(1)
    H[np.arange(n), tt] = 0; Ex = H.max(1)
    Ai = _same(A, tt)
    tri = np.asarray(Ai.multiply(Ai @ Ai).sum(1)).ravel() / 2
    pairs = I * (I - 1) / 2
    cc = np.where(pairs > 0, tri / np.maximum(pairs, 1), 0.0)
    zeta = 0.5 * (I / (np.maximum(Ex, 1) * np.maximum(K, 1)) + cc)
    return 1 - zeta


def score_all(A, g, text, t, runs, ref_raw, seed=0):
    p_c = ambiguity.content_probs(text, t, seed=seed)
    tt = np.where(t < 0, 0, t)
    mis = np.where(t < 0, 1 - p_c.max(1), 1 - p_c[np.arange(len(t)), tt])
    return {"misfit": mis, "participation": ambiguity.participation(A, t),
            "consensus_entropy": ambiguity.consensus_entropy(runs, ref_raw),
            "persistence_low": persistence_low(A, t),
            "degree": np.asarray(A.sum(1)).ravel()}


def graph_of(A):
    up = sp.triu(A).tocoo()
    return ig.Graph(n=A.shape[0], edges=np.c_[up.row, up.col])


def detect(S, targets, pool, succ, lowmis=None):
    """AUC per score. all/succ: targets (successful targets) vs non-target nodes in pool.
    all_low: all targets vs non-targets in pool below the original top-10% misfit (targets' own population).
    fail: successful vs unsuccessful targets (same attack)."""
    n = len(S["degree"]); y = np.zeros(n, bool); y[targets] = True
    neg = pool & ~y
    out = {}
    for s, v in S.items():
        r = {}
        cases = [("all", y, neg), ("succ", y & succ, neg)]
        if lowmis is not None:
            cases.append(("all_low", y, neg & lowmis))
        cases.append(("succ_vs_fail", y & succ, y & ~succ))
        for nm, pos, ng in cases:
            if pos.sum() >= 2 and ng.sum() >= 2:
                m = pos | ng
                r[nm] = float(roc_auc_score(pos[m], v[m]))
        out[s] = r
    return out


def success_rate(t0, t1, targets):
    ist = np.zeros(len(t0), bool); ist[targets] = True
    maj = {}
    for c in np.unique(t0[t0 >= 0]):
        lab = t1[(t0 == c) & ~ist]
        maj[c] = np.bincount(lab + 1).argmax() - 1
    succ = np.array([t1[u] != maj[t0[u]] for u in targets])
    return succ


# --------------------------------------------------------------------------- pipeline
def pick_targets(seed, n_t=200):
    ok = (D["t0"] >= 0) & D["has"] & (D["deg"] >= 3) & (D["mis0"] < D["mis_q"])
    return np.sort(np.random.default_rng(seed).choice(np.flatnonzero(ok), n_t, replace=False))


def job(args):
    seed, kind, k = args
    F, t0 = D["F"], D["t0"]
    targets = pick_targets(seed)
    G, steps, rem, add = attack(F, t0, targets, k, kind, np.random.default_rng(1000 * seed + 17 * k + ATTACKS.index(kind)))
    g = graph_of(G)
    runs = views.leiden_runs(g, runs=10, resolution=1.0, seed=100 * seed)
    ref = views.best_run(runs, g)
    t1 = ambiguity.keep_big(ref)
    succ_t = success_rate(t0, t1, targets)
    succ = np.zeros(len(t0), bool); succ[targets[succ_t]] = True
    S = score_all(G, g, D["text"], t1, runs, ref, seed=0)
    res = detect(S, targets, D["has"], succ, D["low"])
    # unattacked baseline for the SAME successful targets (original graph, t0)
    un = detect(D["S0"], targets, D["has"], succ, D["low"])
    for sc in res:
        res[sc]["succ_unatt"] = un[sc].get("succ", np.nan)
    return (seed, kind, k), {"success_rate": float(succ_t.mean()), "n_succ": int(succ_t.sum()),
                             "rewires_done": int(sum(steps.values())), "rewires_budget": int(k * len(targets)),
                             "edges_removed": len(rem), "edges_added": len(add),
                             "n_comm_t1": int(t1.max()) + 1, "auc": res}


def setup(ds):
    if ds == "timme":
        xb = np.load(os.path.join(OUT, "timme_bio_lcc.npy"))
        it = np.load(os.path.join(OUT, "h4b_inter.npz"))
        t0 = np.load(os.path.join(OUT, "h4b_scores_r1.0_s0.npz"))["t"]
        n = len(t0)
        F = sp.coo_matrix((np.ones(len(it["follow_s"])), (it["follow_s"], it["follow_d"])), shape=(n, n)).tocsr()
        text, has = xb, xb[:, -1] > 0
    else:
        b = views.load_pokec("pokec_knm")
        F = b["A"].tocsr(); text, has = b["text"], b["has"]
        t0 = np.load(os.path.join(OUT, "pokec_knm_views", "ambiguity.npz"))["t"]
    F = ((F + F.T) > 0).astype(np.float64).tocsr()
    F.setdiag(0); F.eliminate_zeros()
    assert F.shape[0] == len(t0) == text.shape[0]
    D.update(F=F, t0=t0, text=text, has=has, deg=np.asarray(F.sum(1)).ravel())
    g = graph_of(F)
    runs = views.leiden_runs(g, runs=10, resolution=1.0, seed=999)
    S0 = score_all(F, g, text, t0, runs, views.best_run(runs, g), seed=0)
    D["S0"] = S0; D["mis0"] = S0["misfit"]
    pool = (t0 >= 0) & has
    D["mis_q"] = np.quantile(S0["misfit"][pool], 0.9)
    D["low"] = S0["misfit"] < D["mis_q"]
    return S0


def _init(d):
    D.update(d)


def run(ds, seeds=(0, 1, 2), ks=(2, 5)):
    S0 = setup(ds)
    log(f"{ds}: n={len(D['t0'])} comms={int(D['t0'].max()) + 1} edges={int(D['F'].nnz // 2)}")
    un = {}
    for s in seeds:
        tg = pick_targets(s)
        un[s] = detect(S0, tg, D["has"], np.ones(len(D["t0"]), bool), D["low"])
    jobs = [(s, a, k) for s in seeds for a in ATTACKS for k in ks]
    res = {}
    with Pool(4, initializer=_init, initargs=(D,)) as p:
        for kk, r in p.imap_unordered(job, jobs):
            res[kk] = r; log(f"{ds} {kk}: success={r['success_rate']:.3f} misfit_auc={r['auc']['misfit']['all']:.3f}")
    table = {}
    for a in ATTACKS:
        for k in ks:
            row = {"success_rate": ms([res[(s, a, k)]["success_rate"] for s in seeds]),
                   "rewires_done": [res[(s, a, k)]["rewires_done"] for s in seeds],
                   "rewires_budget": res[(seeds[0], a, k)]["rewires_budget"],
                   "n_succ": [res[(s, a, k)]["n_succ"] for s in seeds], "ddof": 0, "auc": {}}
            for sc in SCORES:
                g = lambda nm: ms([res[(s, a, k)]["auc"][sc].get(nm, np.nan) for s in seeds])
                row["auc"][sc] = {
                    "all": g("all"), "succ": g("succ"), "all_low": g("all_low"),
                    "succ_vs_fail": g("succ_vs_fail"), "succ_unatt": g("succ_unatt"),
                    "unattacked": ms([un[s][sc]["all"] for s in seeds]),
                    "unattacked_low": ms([un[s][sc]["all_low"] for s in seeds])}
            table[f"{a}|k{k}"] = row
    os.makedirs(RES, exist_ok=True)
    json.dump({"ds": ds, "seeds": list(seeds), "n_targets": 200, "n": int(len(D["t0"])),
               "n_comm_t0": int(D["t0"].max()) + 1, "table": table,
               "per_seed": {f"{s}|{a}|{k}": v for (s, a, k), v in res.items()}},
              open(os.path.join(RES, f"h6_{ds}.json"), "w"), indent=1)
    for kk, row in table.items():
        b = row["auc"]
        log(f"{ds} {kk}: succ={row['success_rate'][0]:.3f} " +
            " ".join(f"{sc}={b[sc]['all'][0]:.3f}" for sc in SCORES))


def ms(x):
    x = np.array(x, float)
    return [float(np.nanmean(x)), float(np.nanstd(x))]


# --------------------------------------------------------------------------- selftest
def selftest():
    rng = np.random.default_rng(0)
    n = 200; t0 = np.repeat([0, 1], n // 2)
    up = np.triu(rng.random((n, n)) < np.where(t0[:, None] == t0[None, :], 0.3, 0.03), 1)
    F = sp.csr_matrix((up | up.T).astype(float))
    text = np.eye(2)[t0] * 2 + rng.normal(0, 0.7, (n, 2))
    text = np.c_[text, rng.normal(size=(n, 6))]
    targets = []                                                # 8 block-0 nodes, mutually non-adjacent
    for u in range(80):
        if len(targets) < 8 and not any(F[u, v] for v in targets):
            targets.append(u)
    targets = np.array(targets)
    for kind in ATTACKS:
        k = 20
        G, steps, rem, add = attack(F, t0, targets, k, kind, np.random.default_rng(1))
        assert G.diagonal().sum() == 0 and G.max() == 1 and abs(G - G.T).nnz == 0, kind
        d = (G - F)
        assert (d > 0).nnz // 2 == len(add) and (d < 0).nnz // 2 == len(rem), kind
        assert len(rem) == len(add) == sum(steps.values()), kind
        if kind in ("kumari", "safeness", "random"):
            for u in targets:                                   # exactly k edges changed per target
                assert steps[int(u)] == k, (kind, u, steps[int(u)])
                assert int((d[u] < 0).sum()) == k and int((d[u] > 0).sum()) >= k, (kind, u)   # random may also hit another target
    G, steps, _, _ = attack(F, t0, targets, 20, "kumari", np.random.default_rng(1))
    g = graph_of(G)
    runs = views.leiden_runs(g, runs=5, resolution=1.0)
    ref = views.best_run(runs, g)
    t1 = ambiguity.keep_big(ref)
    S = score_all(G, g, text, t1, runs, ref)
    r = S["misfit"].argsort().argsort() / (n - 1)
    D.update(F=F, t0=t0)
    sr = success_rate(t0, t1, targets)
    assert r[targets].mean() > 0.5, r[targets]
    assert sr.mean() > 0.5, sr
    print(f"h6 selftest ok: target misfit rank {r[targets].mean():.2f} (>0.5), kumari success {sr.mean():.2f}, edge counts exact")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest()
    else:
        run(sys.argv[1])
