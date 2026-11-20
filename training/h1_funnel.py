"""H1: synthetic radicalisation funnel. Does content-aware (misfit) blocking find the
recruiter nodes when agreement-gated spread and hub-placed mismatches hold?
Defensive study: which accounts to throttle to stop recruitment messages.

Groups: M mainstream communities, 1 sympathiser fringe, 1 radical core.
Recruiters: sit in a mainstream community (label) but send frac_core of their edges to
the core; text mixed (rec_txt share of core), acceptance of core message = rec_acc.
Decoys: high-degree news/celebrity accounts, edges across mainstream, pure mainstream text.
Spread: live-edge IC (blocking.edge_probs), message starts at 10 core nodes, edge u->v
transmits with p * K*exp(beta*a_v)/(e^beta+K-1)  (a_v = receiver's affinity to the
core message; beta as in regime.py, K=8). Harm = share of mainstream nodes reached.
Pipeline (views/ambiguity/blocking.strategies) reused unchanged; scores use OBSERVED
noisy text, spread uses TRUE affinity.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v,'1')
import numpy as np
import scipy.sparse as sp
import ambiguity, blocking, views

BUD = (0.005, 0.01, 0.02, 0.05)
STRAT = ("random", "degree", "pagerank", "betweenness", "misfit", "misfit_x_pr", "gateway", "oracle_recruiters")
STRUCT = ("degree", "pagerank", "betweenness")
BASE = dict(n=4000, M=6, beta=3.0, vuln=0.15, fringe=0.10, core=0.05, rec=0.01, rec_hub=10.0,
            rec_core=0.4, rec_txt=0.3, rec_acc=1.0, decoy=0.0, leak=0.01, fr_out=0.05, fr_aff=0.5, model="ic:2")


def build(P, seed):
    rng = np.random.default_rng(seed)
    n, M = P["n"], P["M"]
    nf, nc, nr, nd = int(P["fringe"] * n), int(P["core"] * n), int(P["rec"] * n), int(P["decoy"] * n)
    nm = n - nf - nc - nr - nd
    grp = np.r_[rng.integers(0, M, nm), np.full(nf, M), np.full(nc, M + 1),
                rng.integers(0, M, nr), rng.integers(0, M, nd)]       # recruiters/decoys: mainstream label
    kind = np.r_[np.zeros(nm), np.ones(nf), 2 * np.ones(nc), 3 * np.ones(nr), 4 * np.ones(nd)].astype(int)
    deg = np.minimum((rng.pareto(1.5, n) + 1) * 8, 300)
    hubd = np.minimum((rng.pareto(1.5, n) + 1) * 8 * P["rec_hub"], 600)
    deg[kind == 3] = hubd[kind == 3] if P["rec_hub"] > 1 else deg[kind == 3]
    deg[kind == 4] = np.maximum(hubd[kind == 4], 8 * 10)               # decoys are always hubs
    deg = deg.astype(int)
    G = M + 2
    mix = np.zeros((G, G))
    for g in range(M):
        mix[g, g] = 0.87 - P["leak"]; mix[g, :M] += 0.13 / M; mix[g, M] = P["leak"]
    mix[M, :M] = P['fr_out'] / M; mix[M, M + 1] = 0.30; mix[M, M] = 0.70 - P['fr_out']
    mix[M + 1, M + 1] = 0.9; mix[M + 1, M] = 0.1
    mix /= mix.sum(1, keepdims=True)
    # target weights: ordinary degree, but hubs are not over-selected as targets (attach via own stubs)
    tw = np.where(kind >= 3, 1.0, deg).astype(float)
    byg = [np.flatnonzero((grp == g) & (kind < 3)) for g in range(G)]
    src, dst = [], []
    for u in range(n):
        d = deg[u]
        if kind[u] == 3:
            probs = np.zeros(G); probs[grp[u]] = 1 - P["rec_core"]; probs[M + 1] = P["rec_core"]
        elif kind[u] == 4:
            probs = np.zeros(G); probs[:M] = 1.0 / M
        else:
            probs = mix[grp[u]]
        gs = rng.choice(G, d, p=probs / probs.sum())
        for g in np.unique(gs):
            c = (gs == g).sum(); pool = byg[g]
            w = tw[pool] / tw[pool].sum()
            dst.append(rng.choice(pool, c, p=w)); src.append(np.full(c, u))
    s, t = np.concatenate(src), np.concatenate(dst)
    keep = s != t
    ei = np.stack([np.r_[s[keep], t[keep]], np.r_[t[keep], s[keep]]])
    # affinity to the core message (true) and text
    a = np.zeros(n)
    sh = P["vuln"] * 5
    mm = kind == 0
    a[mm] = rng.beta(sh, 5 - sh, mm.sum())
    a[kind == 1] = P["fr_aff"]; a[kind == 2] = 1.0; a[kind == 3] = P["rec_acc"]; a[kind == 4] = 0.0
    tmix = a.copy(); tmix[kind == 3] = P["rec_txt"]                      # what the text shows
    cent = np.random.default_rng(seed + 7).normal(size=(G, 64))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    own = np.where(kind == 2, M + 1, grp)
    base = cent[own]; base[kind == 2] = cent[M + 1]
    sig = (1 - tmix)[:, None] * cent[np.where(kind == 2, M + 1, grp)] + tmix[:, None] * cent[M + 1]
    sig = sig + rng.normal(size=(n, 64)) / 2
    sig /= np.linalg.norm(sig, axis=1, keepdims=True)
    topics = cent[:0]
    tp = rng.normal(size=(10, 64)); tp /= np.linalg.norm(tp, axis=1, keepdims=True); tp = tp[rng.integers(0, 10, n)]
    sd = rng.normal(size=(1, 64)); sd /= np.linalg.norm(sd)
    style = sd * (np.log(np.maximum(deg, 1)) - np.log(np.maximum(deg, 1)).mean())[:, None]
    text = np.hstack([sig, 1.5 * (tp + style)]); text /= np.linalg.norm(text, axis=1, keepdims=True)
    return ei, text, grp, kind, a


def run(P, seed, sims=40):
    from sklearn.metrics import roc_auc_score
    ei, text, grp, kind, a = build(P, seed)
    n = P["n"]
    b = views._bundle(ei, text, np.ones(n, bool), grp)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    kmax = int(max(BUD) * n)
    ranked = blocking.strategies(b, s, kmax, seed=seed)
    ranked = {k: ranked[k] for k in STRAT if k in ranked}
    rec = np.flatnonzero(kind == 3)
    ranked["oracle_recruiters"] = np.r_[rec, np.setdiff1d(np.arange(n), rec)]
    K = 8
    mult = K * np.exp(P["beta"] * a) / (np.exp(P["beta"]) + K - 1)
    r, c, p = blocking.edge_probs(b["A"], P["model"])
    p = np.minimum(1.0, p * mult[c])
    budgets = [int(round(x * n)) for x in BUD]
    blk = np.unique(np.concatenate([v[:max(budgets)] for k, v in ranked.items() if k != "oracle_recruiters"]))
    pool = np.setdiff1d(np.flatnonzero(kind == 2), blk)
    rng = np.random.default_rng(seed + 99)
    main = kind == 0
    out = {k: {kk: [] for kk in budgets} for k in ranked}; none = []
    for _ in range(sims):
        live_e = rng.random(len(p)) < p
        seeds = rng.choice(pool, min(10, len(pool)), replace=False)
        live = sp.csr_matrix((np.ones(live_e.sum()), (r[live_e], c[live_e])), shape=(n, n))
        def harm(bl):
            return (blocking.reach(live, seeds, bl) & main).sum() / main.sum()
        none.append(harm(np.array([], dtype=np.int64)))
        for k, o in ranked.items():
            for kk in budgets:
                out[k][kk].append(harm(o[:kk]))
    h0 = max(np.mean(none), 1e-12)
    removed = {k: {kk: 1 - np.mean(v) / h0 for kk, v in d.items()} for k, d in out.items()}
    isrec = kind == 3
    det = {"auc_misfit_recruiter": float(roc_auc_score(isrec, s["misfit"])) if isrec.any() else None,
           "auc_degree_recruiter": float(roc_auc_score(isrec, b["deg"])) if isrec.any() else None}
    topk = {k: {kk: float(isrec[ranked[k][:kk]].mean()) for kk in budgets} for k in ("degree", "pagerank", "misfit", "misfit_x_pr", "gateway")}
    return {"removed": {k: {str(kk): float(x) for kk, x in d.items()} for k, d in removed.items()},
            "base_harm": float(h0), "det": det, "rec_precision": {k: {str(kk): x for kk, x in d.items()} for k, d in topk.items()}}


def _task(args):
    name, P, seed, sims = args
    t = time.time()
    r = run(P, seed, sims)
    return name, seed, r, time.time() - t


def tci(x):
    from scipy import stats
    x = np.asarray(x); m = x.mean()
    if len(x) < 2: return float(m), float("nan")
    return float(m), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarize(name, P, res):
    seeds = sorted(res)
    n = P["n"]; bud = [str(int(round(x * n))) for x in BUD]
    row = {"name": name, "params": P, "n_seeds": len(seeds), "base_harm": float(np.mean([res[s]["base_harm"] for s in seeds])),
           "det": {k: float(np.mean([res[s]["det"][k] for s in seeds])) for k in ("auc_misfit_recruiter", "auc_degree_recruiter")},
           "budgets": {}}
    for f, kk in zip(BUD, bud):
        R = {k: np.array([res[s]["removed"][k][kk] for s in seeds]) for k in STRAT}
        best = max(STRUCT, key=lambda k: R[k].mean())
        cell = {"best_struct": best, "best_struct_removed": float(R[best].mean()),
                "removed": {k: float(R[k].mean()) for k in STRAT}}
        for k in ("misfit", "misfit_x_pr", "gateway", "random", "oracle_recruiters"):
            cell[k] = tci(R[k] - R[best])
        cell["rec_precision"] = {k: float(np.mean([res[s]["rec_precision"][k][kk] for s in seeds])) for k in res[seeds[0]]["rec_precision"]}
        row["budgets"][f"{100*f:g}%"] = cell
    return row


def settings():
    S = [("base", {})]
    for beta in (0, 1.5, 3, 6):
        S.append((f"variant=hub beta={beta}", dict(beta=beta)))
        S.append((f"variant=nonhub beta={beta}", dict(beta=beta, rec_hub=1.0)))
        S.append((f"variant=decoy beta={beta}", dict(beta=beta, decoy=0.01)))
        if beta in (3, 6): S.append((f"variant=hub+decoy3x beta={beta}", dict(beta=beta, decoy=0.03)))
    for v in (0.05, 0.3): S.append((f"vuln={v}", dict(vuln=v)))
    for x in (0.003, 0.02): S.append((f"rec={x}", dict(rec=x)))
    for x in (3.0, 30.0): S.append((f"rec_hub={x}", dict(rec_hub=x)))
    for x in (0.1, 0.8): S.append((f"rec_core={x}", dict(rec_core=x)))
    for x in (0.0, 0.6, 1.0): S.append((f"rec_txt={x}", dict(rec_txt=x)))
    for x in (0.3,): S.append((f"rec_acc={x}", dict(rec_acc=x)))
    for x in (0.15, 0.30): S.append((f"fr_out={x}", dict(fr_out=x)))
    for m in ("ic:4",): S.append((f"model={m}", dict(model=m)))
    seen, out = set(), []
    for nme, d in S:
        P = {**BASE, **d}; key = json.dumps(P, sort_keys=True)
        if key in seen: continue
        seen.add(key); out.append((nme, P))
    return out


def fmt(row):
    ls = [f"{row['name']}  base_harm={row['base_harm']:.3f} AUCmisfit->rec={row['det']['auc_misfit_recruiter']:.2f} AUCdeg->rec={row['det']['auc_degree_recruiter']:.2f}"]
    for f, c in row["budgets"].items():
        ls.append(f"   {f:5s} best_struct={c['best_struct']}({c['best_struct_removed']:.3f}) | " + " ".join(
            f"{k}={c[k][0]:+.3f}±{c[k][1]:.3f}" for k in ("misfit", "misfit_x_pr", "gateway", "oracle_recruiters")))
    return "\n".join(ls)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--only", default=None); ap.add_argument("--out", default="../results/hypotheses/h1")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = [x for x in settings() if not a.only or a.only in x[0]]
    tasks = [(n, P, sd, a.sims) for n, P in S for sd in range(a.seeds)]
    res = {n: {} for n, _ in S}; rows = {}
    Pm = dict(S)
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        for name, sd, r, dt in ex.map(_task, tasks):
            res[name][sd] = r
            print(f"[{time.strftime('%H:%M:%S')}] {name} seed {sd} done {dt:.0f}s", file=sys.stderr, flush=True)
            if len(res[name]) == a.seeds:
                rows[name] = summarize(name, Pm[name], res[name])
                print(fmt(rows[name]), flush=True)
                json.dump(list(rows.values()), open(os.path.join(a.out, "h1_results.json"), "w"), indent=1)
