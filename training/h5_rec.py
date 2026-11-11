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


def run(tag, it, xb):
    z = np.load(os.path.join(OUT, f"h4b_scores_{tag}.npz"))
    t, p_t, n = z["t"], z["p_T"], len(z["t"])
    ok = t >= 0
    has = xb[:, -1] > 0
    p_c = ambiguity.content_probs(xb, t)                       # bio-only, cross-fitted
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
    # ponytail: no self-test fixture; prec_rec is checked inline, the rest reuses tested ambiguity code
    p, r = prec_rec(np.array([[3., 2, 1, 0]]), [np.array([0, 3])], 2)
    assert p[0] == 0.5 and r[0] == 0.5
    it = np.load(os.path.join(OUT, "h4b_inter.npz"))
    xb = np.load(os.path.join(OUT, "timme_bio_lcc.npy"))      # written by h5_probe.py
    allres = {}
    for tag in ["r1.0_s0", "r1.0_s1", "r2.0_s0"]:
        allres[tag] = run(tag, it, xb)
        print(tag); fmt(allres[tag])
    d = os.path.join(os.path.dirname(__file__), "../results/hypotheses/h5")
    os.makedirs(d, exist_ok=True)
    json.dump(allres, open(os.path.join(d, "h5_rec.json"), "w"), indent=1, default=float)
