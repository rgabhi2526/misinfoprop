"""T11.1 — misfit re-homing as link recommendation (TIMME, cross-layer held-out).

Graph = FOLLOW edges (communities t from h4b_scores, Leiden on follows). Held-out = engagement
ties (retweet/like/mention) u->v where u does NOT follow v: ties the follow graph misses.
Users: misfits (top-10% bio-only misfit, has bio, argmax p_C != t) vs controls matched on
(degree decile x participation decile).

Recommenders (score every candidate v not already followed by u):
  pop            PageRank(v)                                  popularity only
  cn / aa        common neighbours / Adamic-Adar on follows   structure-only link prediction
  struct_comm    p_T(u)[t_v] * PR(v)                          community weighting from STRUCTURE
  kumari         [t_v == extraction community of u] * PR(v)   Kumari et al. structural target
  text_sim       cos(bio_u, bio_v)                            pure content
  ours           p_C(u)[t_v] * PR(v)                          community weighting from TEXT (bio-only)
  ours+aa        rank-average(ours, aa)
ours vs struct_comm isolates text vs structure with popularity held fixed.

Claim A ("text beats structure"): on misfits, ours > struct_comm, kumari, cn, aa.
Claim B ("misfits under-served"): share of a user's held-out ties landing in argmax p_C, misfits vs controls;
       and own-community share of held-out ties.
Community level: which community gets most of u's held-out ties — argmax p_C vs argmax p_T vs kumari.

    python h5_rec.py --robust   # T11.1b: 27 settings (9 communities x 3 bio-classifier seeds), paired aa_x_text vs aa / aa_x_struct
    python h5_rec.py            # needs out/h4b_inter.npz, out/h4b_scores_*.npz, out/timme_bio.npy (h5_probe.py)
"""
import json, os, numpy as np, scipy.sparse as sp
import ambiguity
from metrics import log

OUT = os.path.join(os.path.dirname(__file__), "out")
ENG = ["retweet", "like", "mention"]


def prec_rec(scores, pos, k):
    """scores: (m, n) with excluded candidates at -inf; pos: list of index arrays. -> P@k, R@k per row."""
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    p, r = np.zeros(len(pos)), np.zeros(len(pos))
    for i, P in enumerate(pos):
        h = np.isin(top[i], P).sum()
        p[i], r[i] = h / k, h / max(len(P), 1)
    return p, r


def rankavg(*s):
    return sum(np.argsort(np.argsort(x, axis=1), axis=1) for x in s).astype(float)


def boot(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return x.mean(), *np.percentile(m, [2.5, 97.5])


def setup(tag, it, xb, seed=0):
    """Shared construction: communities, p_C, follow graph F, held-out E, misfits, matched controls."""
    z = np.load(os.path.join(OUT, f"h4b_scores_{tag}.npz"))
    t, p_t, n = z["t"], z["p_T"], len(z["t"])
    ok = t >= 0
    has = xb[:, -1] > 0
    p_c = ambiguity.content_probs(xb, t, seed=seed)                       # bio-only, cross-fitted
    tt = np.where(ok, t, 0)
    mis_s = 1 - p_c[np.arange(n), tt]
    tgt = p_c.argmax(1)
    F = sp.coo_matrix((np.ones(len(it["follow_s"])), (it["follow_s"], it["follow_d"])), shape=(n, n)).tocsr()
    F = ((F + F.T) > 0).astype(np.float64)
    deg = np.asarray(F.sum(1)).ravel()
    k = p_c.shape[1]
    H = (F @ sp.csr_matrix((np.ones(n), (np.arange(n), tt)), shape=(n, k))).toarray()
    Hx = H.copy(); Hx[np.arange(n), tt] = -1; ext = Hx.argmax(1)
    # held-out engagement ties not in the follow graph
    s = np.concatenate([it[L + "_s"] for L in ENG]); d = np.concatenate([it[L + "_d"] for L in ENG])
    E = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n)).tocsr()
    E = ((E > 0).astype(np.float64) - E.multiply(F)).tocsr(); E.eliminate_zeros()
    E.setdiag(0); E.eliminate_zeros()
    ne = np.diff(E.indptr)
    pool = ok & has & (ne >= 5)                                 # need enough held-out ties to score
    q = np.quantile(mis_s[pool], 0.9)
    mis = np.flatnonzero(pool & (mis_s >= q) & (tgt != t))
    # matched controls: same degree decile x participation decile, low misfit
    dq = np.searchsorted(np.quantile(deg[pool], np.linspace(0, 1, 11)[1:-1]), deg)
    pq = np.searchsorted(np.quantile(z["participation"][pool], np.linspace(0, 1, 11)[1:-1]), z["participation"])
    cell = dq * 10 + pq
    rng = np.random.default_rng(0)
    cand = pool & (mis_s < np.median(mis_s[pool]))
    ctl = []
    for u in mis:
        c = np.flatnonzero(cand & (cell == cell[u]))
        if len(c): ctl.append(rng.choice(c))
    ctl = np.array(ctl)
    log(f"{tag}: k={k} misfits {len(mis)} matched controls {len(ctl)}")
    pr = z["pagerank"]
    return dict(z=z, t=t, p_t=p_t, n=n, p_c=p_c, tt=tt, tgt=tgt, F=F, deg=deg, k=k, ext=ext, E=E, mis=mis, ctl=ctl, pr=pr, xb=xb)


def run(tag, it, xb):
    c = setup(tag, it, xb)
    z, t, p_t, n, p_c, tt, tgt, F, deg, k, ext, E, mis, ctl, pr = (c[x] for x in
        "z t p_t n p_c tt tgt F deg k ext E mis ctl pr".split())
    Xn = xb[:, :-1] / np.maximum(np.linalg.norm(xb[:, :-1], axis=1, keepdims=True), 1e-12)
    w_aa = sp.diags(1 / np.log(np.maximum(deg, 2)))
    res = {}
    for name, U in [("misfit", mis), ("control", ctl)]:
        pos = [E.indices[E.indptr[u]:E.indptr[u + 1]] for u in U]
        excl = F[U].toarray() > 0
        excl[np.arange(len(U)), U] = True
        S = {
            "pop": np.tile(pr, (len(U), 1)),
            "cn": (F[U] @ F).toarray(),
            "aa": (F[U] @ w_aa @ F).toarray(),
            "struct_comm": p_t[U][:, tt] * pr,
            "kumari": (tt[None, :] == ext[U][:, None]) * pr,
            "text_sim": Xn[U] @ Xn.T,
            "ours": p_c[U][:, tt] * pr,
        }
        S["ours+aa"] = rankavg(S["ours"], S["aa"])
        # fair account-level test: re-weight Adamic-Adar by community, text vs structure
        S["aa_x_text"] = S["aa"] * p_c[U][:, tt]
        S["aa_x_struct"] = S["aa"] * p_t[U][:, tt]
        S["aa+text_sim"] = rankavg(S["aa"], S["text_sim"])
        r = {}
        for m, sc in S.items():
            sc = np.where(excl, -np.inf, sc.astype(float))
            p10, _ = prec_rec(sc, pos, 10)
            _, r100 = prec_rec(sc, pos, 100)
            r[m] = {"P@10": boot(p10), "R@100": boot(r100)}
        # community level: which community receives most of u's held-out ties
        best = np.array([np.bincount(tt[P], minlength=k).argmax() for P in pos])
        r["comm_acc"] = {"text argmax p_C": float((tgt[U] == best).mean()),
                         "struct argmax p_T": float((p_t[U].argmax(1) == best).mean()),
                         "kumari extraction": float((ext[U] == best).mean()),
                         "own follow community": float((tt[U] == best).mean())}
        r["share_in_text_target"] = boot(np.array([(tt[P] == tgt[u]).mean() for u, P in zip(U, pos)]))
        r["share_in_own"] = boot(np.array([(tt[P] == tt[u]).mean() for u, P in zip(U, pos)]))
        res[name] = r
    return res


def paired_ci(a, b, B=2000, seed=0):
    """Paired per-user difference a-b: (mean, lo, hi, per-user diff). Bootstrap over users."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return (*boot(d, B, seed), d)


def sign(lo, hi):
    return "pos" if lo > 0 else "neg" if hi < 0 else "span"


def robust_setting(tag, seed, it, xb):
    """Per-user P@10 / R@100 of aa, aa_x_text, aa_x_struct on misfits and controls for one setting."""
    c = setup(tag, it, xb, seed)
    F, deg, E, tt, p_c, p_t = c["F"], c["deg"], c["E"], c["tt"], c["p_c"], c["p_t"]
    w_aa = sp.diags(1 / np.log(np.maximum(deg, 2)))
    out = {"n_misfit": len(c["mis"]), "n_control": len(c["ctl"])}
    for name, U in [("misfit", c["mis"]), ("control", c["ctl"])]:
        pu = {m: {"P@10": [], "R@100": []} for m in ["aa", "aa_x_text", "aa_x_struct"]}
        for i in range(0, len(U), 250):                      # chunk users: dense (users x n) blocks are big
            V = U[i:i + 250]
            pos = [E.indices[E.indptr[u]:E.indptr[u + 1]] for u in V]
            excl = F[V].toarray() > 0
            excl[np.arange(len(V)), V] = True
            aa = (F[V] @ w_aa @ F).toarray()
            S = {"aa": aa, "aa_x_text": aa * p_c[V][:, tt], "aa_x_struct": aa * p_t[V][:, tt]}
            for m, sc in S.items():
                sc = np.where(excl, -np.inf, sc.astype(float))
                pu[m]["P@10"].append(prec_rec(sc, pos, 10)[0]); pu[m]["R@100"].append(prec_rec(sc, pos, 100)[1])
        pu = {m: {q: np.concatenate(v[q]) for q in v} for m, v in pu.items()}
        o = {"mean": {m: {q: float(v[q].mean()) for q in v} for m, v in pu.items()}, "diff": {}, "_d": {}}
        for q in ["P@10", "R@100"]:
            for ref in ["aa", "aa_x_struct"]:
                mu, lo, hi, d = paired_ci(pu["aa_x_text"][q], pu[ref][q])
                o["diff"][f"text-{ref} {q}"] = [mu, lo, hi]
                o["_d"][f"text-{ref} {q}"] = d
        out[name] = o
    return out


def gain_ci(dm, dc, B=2000, seed=0):
    """Misfit-specific gain: mean(dm)-mean(dc), groups bootstrapped independently."""
    rng = np.random.default_rng(seed)
    g = np.array([dm[rng.integers(0, len(dm), len(dm))].mean() - dc[rng.integers(0, len(dc), len(dc))].mean()
                  for _ in range(B)])
    return [float(dm.mean() - dc.mean()), *map(float, np.percentile(g, [2.5, 97.5]))]


def robust(procs=2):
    from multiprocessing import Pool
    it = np.load(os.path.join(OUT, "h4b_inter.npz"))
    xb = np.load(os.path.join(OUT, "timme_bio_lcc.npy"))
    jobs = [(f"r{r}_s{s}", sd, dict(it), xb) for r in ["0.5", "1.0", "2.0"] for s in range(3) for sd in range(3)]
    with Pool(procs) as p:
        done = p.imap(_job_k, jobs)
        res = {}
        for key, r in done:
            res[key] = r; log(f"done {key}")
    return res


def _job_k(a):
    return f"{a[0]}_b{a[1]}", robust_setting(*a)


def summarize(res):
    keys = [f"text-{ref} {q}" for q in ["P@10", "R@100"] for ref in ["aa", "aa_x_struct"]]
    summ = {}
    for key in keys:
        cnt = {g: {"pos": 0, "neg": 0, "span": 0} for g in ["misfit", "control", "gain"]}
        for r in res.values():
            for g in ["misfit", "control"]:
                cnt[g][sign(*r[g]["diff"][key][1:])] += 1
            gn = gain_ci(r["misfit"]["_d"][key], r["control"]["_d"][key])
            r.setdefault("gain", {})[key] = gn
            cnt["gain"][sign(*gn[1:])] += 1
        summ[key] = cnt
    return summ


def selftest():
    x = np.random.default_rng(1).random(50)
    mu, lo, hi, d = paired_ci(x, x)
    assert mu == lo == hi == 0 and not d.any()
    assert sign(0, 0) == "span"


def verdict(summ):
    k = "text-aa P@10"; k2 = "text-aa_x_struct P@10"
    parts = []
    for kk in (k, k2):
        m, g = summ[kk]["misfit"], summ[kk]["gain"]
        parts.append(f"{kk}: misfit CI>0 in {m['pos']}/27, <0 in {m['neg']}/27; misfit-specific gain >0 in {g['pos']}/27, <0 in {g['neg']}/27")
    return "; ".join(parts)


def fmt(res):
    for g, r in res.items():
        print(f"  [{g}]")
        for m, v in r.items():
            if isinstance(v, dict) and "P@10" in v:
                print(f"    {m:12s} P@10 {v['P@10'][0]:.3f} [{v['P@10'][1]:.3f},{v['P@10'][2]:.3f}]"
                      f"  R@100 {v['R@100'][0]:.3f} [{v['R@100'][1]:.3f},{v['R@100'][2]:.3f}]")
        print("    community acc:", {a: round(b, 3) for a, b in r["comm_acc"].items()})
        print(f"    held-out share in text target {r['share_in_text_target'][0]:.3f}, in own {r['share_in_own'][0]:.3f}")


if __name__ == "__main__":
    import sys
    selftest()
    # ponytail: no self-test fixture; prec_rec is checked inline, the rest reuses tested ambiguity code
    p, r = prec_rec(np.array([[3., 2, 1, 0]]), [np.array([0, 3])], 2)
    assert p[0] == 0.5 and r[0] == 0.5
    if "--robust" in sys.argv:
        res = robust()
        summ = summarize(res)
        for r in res.values():
            for g in ["misfit", "control"]:
                r[g].pop("_d")
        d = os.path.join(os.path.dirname(__file__), "../results/hypotheses/h5")
        json.dump({"settings": res, "summary": summ}, open(os.path.join(d, "h5_robust.json"), "w"), indent=1, default=float)
        print(json.dumps(summ, indent=1)); print(verdict(summ)); sys.exit()
    it = np.load(os.path.join(OUT, "h4b_inter.npz"))
    xb = np.load(os.path.join(OUT, "timme_bio_lcc.npy"))      # written by h5_probe.py
    allres = {}
    for tag in ["r1.0_s0", "r1.0_s1", "r2.0_s0"]:
        allres[tag] = run(tag, it, xb)
        print(tag); fmt(allres[tag])
    d = os.path.join(os.path.dirname(__file__), "../results/hypotheses/h5")
    os.makedirs(d, exist_ok=True)
    json.dump(allres, open(os.path.join(d, "h5_rec.json"), "w"), indent=1, default=float)
