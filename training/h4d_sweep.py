"""H4d: extended node-level regime map (beta up to 20, 10 seeds). CPU only.
Scores/graph built once per (mu,plant,place,seed); beta only changes spread, so 5 betas share them.
python h4d_sweep.py --seeds 10 --workers 4 -> ../results/hypotheses/h4d/h4d_raw.jsonl"""
import argparse, itertools, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
import ambiguity, blocking, views
from regime import planted_place

BETAS = [4, 6, 8, 12, 20]
BUD = [40, 80, 200]            # 1%,2%,5% of n=4000
MIS = ("misfit", "misfit_x_pr", "gateway", "misfit_rerank")
STR = ("pagerank", "betweenness", "degree", "collective_influence", "adaptive_degree")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h4d", "h4d_raw.jsonl")

def job(args):
    mu, plant, place, seed, sims = args
    n = 4000
    ei, text, comm, tc = planted_place(n, mu, plant, place, seed)
    b = views._bundle(ei, text, np.ones(n, bool), comm)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    ranked = blocking.strategies(b, s, max(BUD), seed=seed)
    pr = s["pagerank"]
    ranked = {k: ranked[k] for k in MIS[:3] + STR}
    kc = int(comm.max()) + 1
    rows = []
    for beta in BETAS:
        aff = np.exp(beta * (tc[:, None] == np.arange(kc)[None, :])); aff /= aff.sum(1, keepdims=True)
        # misfit_rerank differs per budget -> simulate per budget with its own ranking
        res = blocking.simulate(b, ranked, comm, BUD, "ic:2", sims=sims, seed=seed, affinity=aff, raw=True)
        base = res["none"][0]            # (sims, 2): infected, cross
        rr = {}
        for k in BUD:
            pool = np.argsort(-pr)[: 3 * k]
            r2 = dict(ranked); r2 = {"misfit_rerank": pool[np.argsort(-s["misfit"][pool])]}
            # rerank only ever blocks first k nodes of its list; simulate with budget k
            rr[k] = blocking.simulate(b, r2, comm, [k], "ic:2", sims=sims, seed=seed, affinity=aff, raw=True)
        rec = {"mu": mu, "plant": plant, "place": place, "seed": seed, "beta": beta,
               "base_inf": float(base[:, 0].mean()), "base_cross": float(base[:, 1].mean()), "rem": {}}
        for k in BUD:
            d = {}
            for name in ranked:
                v = res[name][k]
                d[name] = [float(base[:, 0].mean() - v[:, 0].mean()), float(base[:, 1].mean() - v[:, 1].mean())]
            v = rr[k]["misfit_rerank"][k]
            d["misfit_rerank"] = [float(base[:, 0].mean() - v[:, 0].mean()), float(base[:, 1].mean() - v[:, 1].mean())]
            rec["rem"][k] = d    # absolute drops in infected share / cross share
        rows.append(rec)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--sims", type=int, default=60); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=int, default=0)
    a = ap.parse_args()
    jobs = [(mu, pl, pc, sd, a.sims) for sd in range(a.seeds) for mu in (0.3, 0.5)
            for pl in (0.01, 0.05) for pc in ("hub", "random")]
    if a.only: jobs = jobs[:a.only]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with Pool(a.workers) as p, open(OUT, "a") as f:
        for i, rows in enumerate(p.imap_unordered(job, jobs)):
            for r in rows: f.write(json.dumps(r) + "\n")
            f.flush(); print(f"done {i+1}/{len(jobs)}", flush=True)
