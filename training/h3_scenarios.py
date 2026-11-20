"""H3: where could misfit-based targeting genuinely help? Scenario tests.
Run from training/:  python h3_scenarios.py [--seeds 8] [--sims 40] [--only name,name]
Reuses regime/blocking/ambiguity/views machinery; adds degree-regular graphs,
protected hubs, multi-message cascades, delivery (maximise reach), detection."""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, scipy.sparse as sp
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

import ambiguity, blocking, views

OUTD = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h3")
MIS = ("misfit", "misfit_x_pr", "gateway", "misfit_rerank")
STR = ("pagerank", "betweenness", "degree", "collective_influence", "adaptive_degree")


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gen(n, k, mu, plant, place, seed, degmode="pareto"):
    rng = np.random.default_rng(seed)
    if degmode == "pareto":
        deg = np.minimum((rng.pareto(1.5, n) + 1) * 8, 300).astype(int)
    else:   # weak heterogeneity, similar mean degree
        deg = np.clip(rng.normal(24, 5, n), 8, 45).astype(int)
    comm = rng.permutation(np.repeat(np.arange(k), n // k + 1)[:n])
    inter = rng.binomial(deg, mu); intra = deg - inter
    edges = []
    for c in range(k):
        st = np.repeat(np.flatnonzero(comm == c), intra[comm == c]); rng.shuffle(st)
        edges.append(st[: len(st) // 2 * 2].reshape(-1, 2))
    st = np.repeat(np.arange(n), inter); rng.shuffle(st)
    st = st[: len(st) // 2 * 2].reshape(-1, 2)
    edges.append(st[comm[st[:, 0]] != comm[st[:, 1]]])
    e = np.concatenate(edges); e = e[e[:, 0] != e[:, 1]]
    ei = np.concatenate([e, e[:, ::-1]]).T
    r2 = np.random.default_rng(seed + 1000)
    p = deg / deg.sum() if place == "hub" else None
    mism = r2.choice(n, int(plant * n), replace=False, p=p)
    tc = comm.copy(); tc[mism] = (comm[mism] + r2.integers(1, k, len(mism))) % k
    cent = normalize(np.random.default_rng(seed).normal(size=(k, 64)))
    sig = normalize(cent[tc] + r2.normal(size=(n, 64)) / 2)
    topics = normalize(r2.normal(size=(10, 64)))[r2.integers(0, 10, n)]
    style = normalize(r2.normal(size=(1, 64))) * (np.log(deg) - np.log(deg).mean())[:, None]
    text = normalize(np.hstack([sig, 1.5 * (topics + style)]))
    return ei, text, comm, tc, mism


_CACHE = {}


def setup(cfg, seed):
    key = (cfg.get("plant", 0.05), cfg.get("place", "random"), cfg.get("deg", "pareto"), cfg.get("mu", 0.3), seed)
    if key not in _CACHE:
        _CACHE.clear() if len(_CACHE) > 12 else None
        _CACHE[key] = _setup(cfg, seed)
    return _CACHE[key]


def _setup(cfg, seed):
    t0 = time.time()
    n, k = cfg.get("n", 4000), 8
    ei, text, comm, tc, mism = gen(n, k, cfg.get("mu", 0.3), cfg.get("plant", 0.05),
                                   cfg.get("place", "random"), seed, cfg.get("deg", "pareto"))
    b = views._bundle(ei, text, np.ones(n, bool), comm)
    say("graph built", time.time() - t0)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    say("setup done", time.time() - t0)
    return b, s, comm, tc, mism


def rankings(b, s, kmax, seed):
    r = blocking.strategies(b, s, kmax, seed=seed)
    pr = s["pagerank"]
    pool = np.argsort(-pr)[: 3 * kmax]
    r["misfit_rerank"] = pool[np.argsort(-s["misfit"][pool])]
    return r


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2: return [float(x.mean()), float("nan")]
    return [float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))]


def pick(per_seed, names):
    """per_seed: {name: array over seeds}. best by mean."""
    return max(names, key=lambda m: np.mean(per_seed[m]))


def summarize(per_seed, base_mis=MIS, base_str=STR):
    bm, bs = pick(per_seed, base_mis), pick(per_seed, base_str)
    gain = np.asarray(per_seed[bm]) - np.asarray(per_seed[bs])
    out = {"best_misfit": bm, "best_struct": bs, "misfit_val": ci(per_seed[bm]),
           "struct_val": ci(per_seed[bs]), "gain": ci(gain),
           "wins": int((gain > 0).sum()), "n_seeds": len(gain),
           "all": {k: ci(v) for k, v in per_seed.items()}}
    # fixed (no selection) comparison: misfit_x_pr vs best structural
    g2 = np.asarray(per_seed.get("misfit_x_pr", per_seed[bm])) - np.asarray(per_seed[bs])
    out["gain_fixed_misfit_x_pr"] = ci(g2)
    return out


# ---------------------------------------------------------------- blocking sim
def sim_block(b, ranked, comm, tc, budgets, beta, sims, seed, n_msgs=1, n_seeds=10,
              outcome="cross", protected=None, fixed_src=None):
    """Returns {strategy:{budget: removed fraction (mean over sims)}}.
    Each run = n_msgs concurrent messages from distinct source communities; message m
    is accepted by node v with weight exp(beta*[tc_v == src_m]) (normalised, x k)."""
    n, kc = b["n"], int(comm.max()) + 1
    rng = np.random.default_rng(seed)
    if protected is not None:
        ranked = {k: np.asarray([int(x) for x in v if int(x) not in protected], dtype=np.int64) for k, v in ranked.items()}
    r, c, p = blocking.edge_probs(b["A"], "ic:2")
    kmax = max(budgets)
    pool = np.setdiff1d(np.arange(n), np.unique(np.concatenate([v[:kmax] for v in ranked.values()])))
    tot = {nm: {k: 0.0 for k in budgets} for nm in list(ranked) + ["none"]}
    for it in range(sims):
        srcs = [fixed_src] if fixed_src is not None else \
            list(rng.choice(kc, n_msgs, replace=False))
        msgs = []
        for src in srcs:
            w = np.exp(beta * (tc[c] == src)); den = (np.exp(beta) + kc - 1) / kc
            live_e = rng.random(len(p)) < np.minimum(1.0, p * w / den)
            inside = pool[comm[pool] == src]
            seeds = rng.choice(inside, min(n_seeds, len(inside)), replace=False)
            live = sp.csr_matrix((np.ones(live_e.sum()), (r[live_e], c[live_e])), shape=(n, n))
            msgs.append((src, seeds, live))

        def out(blocked):
            tot_ = 0.0
            for src, seeds, live in msgs:
                hit = blocking.reach(live, seeds, blocked); hit[seeds] = False
                if outcome == "cross": tot_ += (hit & (comm != src)).sum()
                else: tot_ += (hit & (tc == src)).sum()      # harm: audience of message
            return tot_ / n
        base = out(np.array([], dtype=np.int64)); tot["none"][0] = tot["none"].get(0, 0) + base
        for nm, o in ranked.items():
            for k in budgets:
                tot[nm][k] += out(o[:k])
    base = tot["none"][0]
    return {nm: {k: 1 - tot[nm][k] / max(base, 1e-12) for k in budgets} for nm in ranked}


def run_block(name, cfg, seeds, sims, budgets):
    per = {k: {} for k in budgets}
    for sd in range(seeds):
        b, s, comm, tc, _ = setup(cfg, sd)
        ranked = rankings(b, s, max(budgets), sd)
        prot = None
        if cfg.get("protect"):
            prot = set(np.argsort(-b["deg"])[: int(cfg["protect"] * b["n"])].tolist())
        res = sim_block(b, ranked, comm, tc, budgets, cfg["beta"], sims, sd,
                        cfg.get("msgs", 1), outcome=cfg.get("outcome", "cross"),
                        protected=prot, fixed_src=cfg.get("fixed_src"))
        for k in budgets:
            for nm in MIS + STR:
                per[k].setdefault(nm, []).append(res[nm][k])
        say(name, "seed", sd, " ".join(f"{nm}={res[nm][budgets[-1]]:.3f}" for nm in ("misfit_x_pr", "pagerank", "betweenness")))
    return {f"k={k}": summarize(per[k]) for k in budgets}


# ---------------------------------------------------------------- delivery (b)
def run_delivery(name, cfg, seeds, sims, budgets):
    """Seed k nodes outside target community B (hard to reach) to maximise reach into
    audience = nodes whose content is B (they accept; others accept at ~e^-beta)."""
    per = {k: {} for k in budgets}
    beta = cfg["beta"]
    for sd in range(seeds):
        b, s, comm, tc, _ = setup(cfg, sd)
        n, kc = b["n"], 8
        Bp = 0
        Bd = int(np.bincount(s["t"][comm == Bp][s["t"][comm == Bp] >= 0]).argmax())
        pr = s["pagerank"]; deg = b["deg"]
        btw = np.asarray(b["g"].betweenness(sources=np.random.default_rng(sd).choice(n, 500, replace=False).tolist()), float)
        inB = (b["A"] @ (s["t"] == Bd).astype(float))
        pc, pt = s["p_C"][:, Bd], s["p_T"][:, Bd]
        cands = {"pagerank": pr, "degree": deg, "betweenness": btw, "edges_into_B": inB,
                 "misfit": s["misfit"], "misfit_x_pr": s["misfit"] * pr / pr.mean(),
                 "misfit_toB": pc * (1 - pt), "misfit_toB_x_edges": pc * (1 - pt) * inB,
                 "text_toB": pc, "text_toB_x_edges": pc * inB, "random": np.random.default_rng(sd).random(n)}
        allowed = comm != Bp
        r, c, p = blocking.edge_probs(b["A"], cfg.get("model", "ic:2"))
        den = (np.exp(beta) + kc - 1) / kc
        aud = tc == Bp
        res = {nm: {k: [] for k in budgets} for nm in cands}
        rng = np.random.default_rng(sd)
        sets = {nm: {k: np.flatnonzero(allowed)[np.argsort(-sc[allowed])[:k]] for k in budgets} for nm, sc in cands.items()}
        for it in range(sims):
            live_e = rng.random(len(p)) < np.minimum(1.0, p * np.exp(beta * (tc[c] == Bp)) / den)
            live = sp.csr_matrix((np.ones(live_e.sum()), (r[live_e], c[live_e])), shape=(n, n))
            for nm in cands:
                for k in budgets:
                    hit = blocking.reach(live, sets[nm][k], np.array([], dtype=np.int64))
                    res[nm][k].append((hit & aud).sum() / aud.sum())
        for k in budgets:
            for nm in cands: per[k].setdefault(nm, []).append(float(np.mean(res[nm][k])))
        say(name, "seed", sd, " ".join(f"{nm}={np.mean(res[nm][budgets[-1]]):.3f}" for nm in ("misfit_toB_x_edges", "edges_into_B", "pagerank", "text_toB")))
    out = {}
    for k in budgets:
        d = per[k]
        o = summarize(d, ("misfit", "misfit_x_pr", "misfit_toB", "misfit_toB_x_edges"),
                      ("pagerank", "degree", "betweenness", "edges_into_B"))
        o["gain_fixed_misfit_x_pr"] = o["gain_fixed_misfit_x_pr"]
        g = np.asarray(d[o["best_misfit"]]) - np.asarray(d["text_toB"])
        o["gain_vs_text_toB"] = ci(g)
        g = np.asarray(d[o["best_misfit"]]) - np.asarray(d[pick(d, ("text_toB", "text_toB_x_edges"))])
        o["gain_vs_best_text"] = ci(g)
        out[f"k={k}"] = o
    return out


# ---------------------------------------------------------------- detection (e)
def run_detect(name, cfg, seeds, budgets):
    per = {k: {} for k in budgets}; auc = {}
    for sd in range(seeds):
        b, s, comm, tc, mism = setup(cfg, sd)
        n = b["n"]; truth = np.zeros(n, bool); truth[mism] = True
        truth = tc != comm
        btw = np.asarray(b["g"].betweenness(sources=np.random.default_rng(sd).choice(n, 500, replace=False).tolist()), float)
        sc = {"misfit": s["misfit"], "disagreement": s["disagreement"], "degree": b["deg"].astype(float),
              "pagerank": s["pagerank"], "betweenness": btw,
              "participation": s["participation"], "content_outlier": s["content_outlier"],
              "consensus_entropy": s["consensus_entropy"], "random": np.random.default_rng(sd).random(n)}
        for nm, v in sc.items():
            auc.setdefault(nm, []).append(roc_auc_score(truth, v))
            for k in budgets:
                top = np.argsort(-v)[:k]
                per[k].setdefault(nm, []).append(float(truth[top].mean()))
        say(name, "seed", sd, "prec@%d misfit=%.2f" % (budgets[-1], per[budgets[-1]]["misfit"][-1]))
    out = {"auc": {k: ci(v) for k, v in auc.items()}}
    for k in budgets:
        d = per[k]
        o = summarize(d, ("misfit", "disagreement"), ("pagerank", "betweenness", "degree"))
        nonmis = pick(d, ("participation", "content_outlier", "consensus_entropy"))
        o["best_other_nonstruct"] = nonmis
        o["gain_vs_best_other_nonstruct"] = ci(np.asarray(d["misfit"]) - np.asarray(d[nonmis]))
        out[f"prec@{k}"] = o
    return out


SCEN = {
    "ctl_realistic_hub_b1.4": dict(kind="block", beta=1.4, place="hub"),
    "ctl_regime_hub_b6": dict(kind="block", beta=6, place="hub"),
    "a_gated_random_b10": dict(kind="block", beta=10, place="random"),
    "a_gated_hub_b10": dict(kind="block", beta=10, place="hub"),
    "c_niche_harm_random_b10": dict(kind="block", beta=10, place="random", outcome="harm", fixed_src=0, plant=0.10),
    "c_niche_harm_hub_b10": dict(kind="block", beta=10, place="hub", outcome="harm", fixed_src=0, plant=0.10),
    "d_protected_hub_b1.4": dict(kind="block", beta=1.4, place="hub", protect=0.02),
    "d_protected_hub_b6": dict(kind="block", beta=6, place="hub", protect=0.02),
    "d_protected_random_b3": dict(kind="block", beta=3, place="random", protect=0.02),
    "f_regular_random_b1.4": dict(kind="block", beta=1.4, place="random", deg="regular"),
    "f_regular_random_b6": dict(kind="block", beta=6, place="random", deg="regular"),
    "f_regular_hub_b6": dict(kind="block", beta=6, place="hub", deg="regular"),
    "g_multi3_random_b3": dict(kind="block", beta=3, place="random", msgs=3),
    "g_multi3_hub_b6": dict(kind="block", beta=6, place="hub", msgs=3),
    "b_delivery_b6": dict(kind="deliver", beta=6, place="random"),
    "b_delivery_b10": dict(kind="deliver", beta=10, place="random", plant=0.10),
    "b_delivery_lowT0.2_b6": dict(kind="deliver", beta=6, place="random", model="ic:0.2"),
    "b_delivery_lowT0.1_b6": dict(kind="deliver", beta=6, place="random", model="ic:0.1"),
    "b_delivery_lowT0.2_b10_plant10": dict(kind="deliver", beta=10, place="random", plant=0.10, model="ic:0.2"),
    "e_detect_random": dict(kind="detect", place="random", plant=0.03),
    "e_detect_hub": dict(kind="detect", place="hub", plant=0.03),
    "e_detect_regular": dict(kind="detect", place="random", plant=0.03, deg="regular"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    jp = os.path.join(OUTD, "h3_results.json")
    allr = json.load(open(jp)) if os.path.exists(jp) else {}
    budgets = [20, 40]
    for nm, cfg in SCEN.items():
        if a.only and nm not in a.only.split(","): continue
        if nm in allr and not a.only: continue
        t0 = time.time()
        if cfg["kind"] == "block": r = run_block(nm, cfg, a.seeds, a.sims, budgets)
        elif cfg["kind"] == "deliver": r = run_delivery(nm, cfg, a.seeds, a.sims, budgets)
        else: r = run_detect(nm, cfg, a.seeds, budgets)
        allr[nm] = {"cfg": cfg, "res": r}
        json.dump(allr, open(jp, "w"), indent=1)
        say(f"DONE {nm} in {time.time()-t0:.0f}s")
