"""H2 for LINKS: shortlist = top-M cross-community links by structural bridge score (PR_u*PR_v);
filter/re-rank by receptivity (max_dir p_C(receiver)[sender community]), endpoint misfit, or combos.
Synthetic planted graph, hub-placed mismatches, mu=0.3. Gain = removed cross share minus
bridge_struct top-k (paired over cascades, per seed).
    python h2_edges.py -> ../results/hypotheses/h2/edges.json"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
import ambiguity, edges, regime, views
import h2_filter as H

def job(a):
    seed, mu, place, betas, sims, n = a
    ei, text, comm, tc = regime.planted_place(n, mu, place and place, 0.05, seed) if False else regime.planted_place(n, mu, 0.05, place, seed)
    b = views._bundle(ei, text, np.ones(n, bool), comm)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    u, v, _ = edges.edge_scores(b, s, seed)
    pr = s["pagerank"] / s["pagerank"].mean(); t, pc, mis = s["t"], s["p_C"], s["misfit"]
    tu, tv = np.where(t[u] < 0, 0, t[u]), np.where(t[v] < 0, 0, t[v])
    cross = np.flatnonzero((t[u] != t[v]) & (t[u] >= 0) & (t[v] >= 0))
    struct = pr[u] * pr[v]
    recv = np.maximum(pc[v, tu], pc[u, tv])
    emis = np.maximum(mis[u], mis[v])
    rng = np.random.default_rng(seed)
    kc = int(comm.max()) + 1
    rows = []
    for k in (40, 200, 800):
        ranked = {"struct": cross[np.argsort(-struct[cross], kind="stable")]}
        ranked["bridge_content"] = cross[np.argsort(-(np.maximum(pr[u] * pc[v, tu], pr[v] * pc[u, tv]))[cross] if False else
                                                    -np.maximum(pr[u] * pc[v, tu], pr[v] * pc[u, tv])[cross], kind="stable")]
        for ml, M in (("3k", 3 * k), ("10k", 10 * k), ("all", len(cross))):
            M = min(M, len(cross)); sh = ranked["struct"][:M]
            sc, rc, mc = struct[sh], recv[sh], emis[sh]
            r_ = lambda x: H.rk(x)
            opts = {"recv": -rc, "misfit": -mc, "rankavg": r_(sc) + r_(rc), "prod": -(sc / sc.mean()) * rc,
                    "prod_misfit": -(sc / sc.mean()) * mc, "rand": rng.random(M)}
            for how, key in opts.items():
                ranked[f"{ml}|{how}"] = sh[np.argsort(key, kind="stable")]
        for beta in betas:
            aff = np.exp(beta * (tc[:, None] == np.arange(kc)[None, :])); aff /= aff.sum(1, keepdims=True)
            res = edges.simulate_edges(b, u, v, ranked, comm, [k], "ic:2", aff, sims, seed=seed * 10 + int(beta))
            base = res["none"][0].mean(); st = res["struct"][k]
            for nm in ranked:
                x = res[nm][k]
                rows.append({"seed": seed, "beta": beta, "k": k, "name": nm, "removed": float(1 - x.mean() / base),
                             "gain": float(((st - x) / base).mean()), "base": float(base)})
    print("seed", seed, "done", flush=True)
    return rows

if __name__ == "__main__":
    jobs = [(sd, 0.3, "hub", [1, 3, 6, 10], 40, 4000) for sd in range(8)]
    with Pool(6) as P:
        rows = [r for rr in P.imap_unordered(job, jobs) for r in rr]
    json.dump(rows, open(os.path.join(H.OUTD, "edges.json"), "w"))
