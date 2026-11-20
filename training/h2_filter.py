"""H2: misfit as FILTER / RE-RANKER over a centrality shortlist (synthetic planted graphs).

Shortlist = top-M by a centrality (pagerank / betweenness / degree / union). From it choose
exactly k nodes by one of: misfit, rankavg, prod, band, rand (control), or the pure centrality
top-k (baseline); mthenc = top-M by misfit, then top-k by centrality.
Outcome per cascade: infected share (col 0) and cross-community share (col 1), content-aware
IC (ic:2), common random numbers across all candidate sets. Gain = removed(strategy) -
removed(best of PR/btw top-k), removed = 1 - infected/mean(unblocked infected), per seed.

    python h2_filter.py [--seeds 8] [--sims 40] [--quick] -> ../results/hypotheses/h2/synth.json
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
import ambiguity, blocking, regime, views

PCTK = [0.001, 0.005, 0.01, 0.02, 0.05]
OUTD = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h2")


def rk(x):  # rank 0 = best (highest)
    r = np.empty(len(x)); r[np.argsort(-x, kind="stable")] = np.arange(len(x)); return r


def shortlist(cent, M):
    return np.argsort(-cent, kind="stable")[:M]


def pick(short, cent, mis, k, how, rng, bandw=None):
    c, m = cent[short], mis[short]
    if how == "cent": o = np.argsort(-c, kind="stable")
    elif how == "misfit": o = np.argsort(-m, kind="stable")
    elif how == "rand": o = rng.permutation(len(short))
    elif how == "rankavg": o = np.argsort(rk(c) + rk(m), kind="stable")
    elif how == "prod": o = np.argsort(-(c / c.mean()) * m, kind="stable")
    elif how == "band":   # centrality bands of width k, misfit order inside each band
        cr = rk(c); band = (cr // max(len(short) // 5, 1)); o = np.lexsort((-m, band))
    return short[o[:k]]


def build_sets(n, cents, mis, rng, full=True):
    """name -> node array. name = filter|M|k|how."""
    sets = {}
    for pct in PCTK:
        k = max(2, int(round(pct * n)))
        for f, cent in cents.items():
            sets[f"{f}|-|{pct}|cent"] = shortlist(cent, k)
            Ms = {"2k": 2 * k, "3k": 3 * k, "5k": 5 * k, "10k": 10 * k, "20%": n // 5}
            if not (full and f == "pr"):
                Ms = {a: Ms[a] for a in ("3k", "10k", "20%")}
            hows = ("misfit", "rand", "rankavg", "prod", "band", "mthenc") if (full and f == "pr") \
                else ("misfit", "rand", "prod")
            for ml, M in Ms.items():
                M = max(M, k)
                sh = shortlist(cent, M)
                for how in hows:
                    if how == "mthenc":
                        top = shortlist(mis, M)
                        sets[f"{f}|{ml}|{pct}|{how}"] = pick(top, cent, mis, k, "cent", rng)
                    else:
                        sets[f"{f}|{ml}|{pct}|{how}"] = pick(sh, cent, mis, k, how, rng)
        for f in ("pr", "btw"):   # un-shortlisted product (the regime.py misfit_x_pr)
            c = cents[f]; sets[f"{f}_x_misfit|-|{pct}|prod"] = np.argsort(-(c / c.mean()) * mis, kind="stable")[:k]
        sets[f"misfit_alone|-|{pct}|misfit"] = shortlist(mis, k)
        sets[f"random|-|{pct}|rand"] = rng.choice(n, k, replace=False)
    return sets


def cascades(b, sets, comm, aff, sims, model, seed):
    """Per-set mean removed fraction for infected and cross, plus base."""
    rng = np.random.default_rng(seed)
    r, c, p = blocking.edge_probs(b["A"], model)
    n = b["n"]; kc = aff.shape[1] if aff is not None else 0
    uniq = {}
    for nm, a in sets.items():
        uniq.setdefault(tuple(sorted(a.tolist())), []).append(nm)
    keys = list(uniq)
    blocked_any = np.unique(np.concatenate([np.array(k) for k in keys]))
    pool = np.setdiff1d(np.arange(n), blocked_any)
    base = []; out = {k: [] for k in keys}
    for it in range(sims):
        if aff is None:
            live_e = rng.random(len(p)) < p
            seeds = rng.choice(pool, 10, replace=False)
            src = np.unique(comm[seeds])
        else:
            src = min(int(comm[rng.choice(pool)]), kc - 1)
            live_e = rng.random(len(p)) < np.minimum(1.0, p * kc * aff[c, src])
            inside = pool[comm[pool] == src]
            seeds = rng.choice(inside, min(10, len(inside)), replace=False)
        live = sp.csr_matrix((np.ones(live_e.sum()), (r[live_e], c[live_e])), shape=(n, n))
        def oc(bl):
            hit = blocking.reach(live, seeds, bl); hit[seeds] = False
            cr = hit & ~np.isin(comm, src)
            return hit.sum() / (n - len(seeds)), cr.sum() / (n - len(seeds))
        base.append(oc(np.array([], dtype=np.int64)))
        for k in keys:
            out[k].append(oc(np.array(k)))
    base = np.array(base)
    res = {}
    for k, nms in uniq.items():
        arr = np.array(out[k])
        # per-cascade reduction in infected share, in units of mean unblocked share
        red = (1 - arr / np.maximum(base.mean(0), 1e-12))     # (sims, 2)
        for nm in nms:
            res[nm] = red
    return res, base.mean(0), len(pool)


def job(args):
    mu, place, seed, betas, sims, n, plant, full = args
    t0 = time.time()
    ei, text, comm, tc = regime.planted_place(n, mu, plant, place, seed)
    b = views._bundle(ei, text, np.ones(n, bool), comm)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    rng = np.random.default_rng(seed + 7)
    piv = rng.choice(n, 500, replace=False).tolist()
    btw = np.asarray(b["g"].betweenness(sources=piv), float)
    pr = s["pagerank"]; deg = b["deg"].astype(float); mis = s["misfit"]
    un = np.maximum(rk(pr), 0) ; un = -np.minimum(rk(pr), rk(btw))  # union score: best rank in either
    cents = {"pr": pr, "btw": btw, "deg": deg, "union": un.astype(float)}
    sets = build_sets(n, cents, mis, rng, full)
    kc = int(comm.max()) + 1
    rows = []
    for beta in betas:
        aff = np.exp(beta * (tc[:, None] == np.arange(kc)[None, :])); aff /= aff.sum(1, keepdims=True)
        res, base, npool = cascades(b, sets, comm, aff, sims, "ic:2", seed * 100 + int(beta))
        for col, cn in ((0, "inf"), (1, "cross")):
            ps = {nm: res[nm][:, col] for nm in res}
            best = max(("pr", "btw"), key=lambda f: ps[f"{f}|-|{PCTK[2]}|cent"].mean())  # pick at 1%
            for nm in res:
                pct = float(nm.split("|")[2])
                bst = max(("pr", "btw"), key=lambda f: ps[f"{f}|-|{pct}|cent"].mean())
                d = ps[nm] - ps[f"{bst}|-|{pct}|cent"]
                rows.append({"mu": mu, "place": place, "seed": seed, "beta": beta, "metric": cn,
                             "name": nm, "removed": float(ps[nm].mean()), "gain": float(d.mean()),
                             "best_struct": bst, "base": float(base[col]), "pool": npool})
    print(f"mu={mu} place={place} seed={seed} done {time.time()-t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    betas = [0, 1, 3, 6, 10]
    mus = [0.3] if a.quick else [0.1, 0.3, 0.5]
    places = ["hub"] if a.quick else ["random", "hub"]
    jobs = [(mu, pl, sd, betas, a.sims, a.n, 0.05, True) for mu in mus for pl in places for sd in range(a.seeds)]
    if a.quick: jobs = jobs[:1]
    os.makedirs(os.path.join(OUTD, "jobs"), exist_ok=True)
    todo = []
    for j in jobs:
        f = os.path.join(OUTD, "jobs", f"synth_{j[0]}_{j[1]}_{j[2]}.json")
        if not os.path.exists(f): todo.append((j, f))
    with Pool(a.procs) as P:
        for (j, f), rr in zip(todo, P.imap(job, [t[0] for t in todo])):
            json.dump(rr, open(f, "w"))
    rows = []
    for j in jobs:
        rows += json.load(open(os.path.join(OUTD, "jobs", f"synth_{j[0]}_{j[1]}_{j[2]}.json")))
    json.dump(rows, open(os.path.join(OUTD, "synth_quick.json" if a.quick else "synth.json"), "w"))
