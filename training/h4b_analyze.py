"""H4b: do misfit nodes matter for REAL spread (TIMME interaction edges)? Communities/misfit from FOLLOW edges only,
outcomes from retweet/mention/reply/like edges only.   python h4b_analyze.py -> ../results/hypotheses/h4b/results.json"""
import json, os, sys, numpy as np, scipy.sparse as sp, statsmodels.api as sm
from multiprocessing import Pool
import ambiguity, views, h4b_data, h4b_scores
OUTD = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h4b")
OUTC = h4b_scores.OUTC
GROUPS = {"spread": ["retweet", "mention", "reply"], "retweet": ["retweet"], "mention": ["mention"],
          "reply": ["reply"], "like": ["like"]}
PRIMARY = dict(res=1.0, seed=0, score="misfit", group="spread")
NPERM = 3000

def outcomes(inter, t, group):
    """per node: cross/all distinct partner-edges (out, in), distinct directed edges, both ends in a community."""
    n = len(t); ss, dd, cc = [], [], []
    for k in GROUPS[group]:
        s, d, c = inter[k]; ss.append(s); dd.append(d); cc.append(c)
    s, d, c = map(np.concatenate, (ss, dd, cc))
    ok = (t[s] >= 0) & (t[d] >= 0); s, d, c = s[ok], d[ok], c[ok]
    key = s.astype(np.int64) * n + d
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=c)
    s, d = uk // n, uk % n
    x = t[s] != t[d]
    r = {"cross_out": np.bincount(s[x], minlength=n), "cross_in": np.bincount(d[x], minlength=n),
         "all_out": np.bincount(s, minlength=n), "all_in": np.bincount(d, minlength=n),
         "wcross_out": np.bincount(s[x], weights=cnt[x], minlength=n), "wcross_in": np.bincount(d[x], weights=cnt[x], minlength=n)}
    r["cross_n"] = r["cross_out"] + r["cross_in"]; r["all_n"] = r["all_out"] + r["all_in"]
    with np.errstate(all="ignore"):
        r["cross_share"] = np.where(r["all_n"] >= 5, r["cross_n"] / r["all_n"], np.nan)
        r["in_cross_share"] = np.where(r["all_in"] >= 3, r["cross_in"] / r["all_in"], np.nan)
        r["out_cross_share"] = np.where(r["all_out"] >= 3, r["cross_out"] / r["all_out"], np.nan)
    return r

def strata_of(deg, pop, q=10):
    e = np.quantile(deg[pop], np.linspace(0, 1, q + 1)[1:-1])
    return np.digitize(deg, e)

def matched_test(score, y, deg, pop, rng, top=0.10, ctrl=0.5, nperm=NPERM, boot=300, strata=None):
    """top-`top` by score vs bottom-`ctrl` by score, within follow-degree deciles. diff = high-weighted mean of stratum diffs."""
    ok = pop & ~np.isnan(y)
    sc = score[ok]; yy = y[ok]; st = (strata_of(deg, pop) if strata is None else strata)[ok]
    hi = sc >= np.quantile(score[pop], 1 - top); lo = sc <= np.quantile(score[pop], ctrl)
    keep = hi | lo; yy, st, hi = yy[keep], st[keep], hi[keep]
    S = [np.flatnonzero(st == s) for s in np.unique(st)]
    S = [i for i in S if hi[i].any() and (~hi[i]).any()]
    w = np.array([hi[i].sum() for i in S], float); w /= w.sum()
    def stat(h_by_s):
        return sum(wi * (yy[i][h].mean() - yy[i][~h].mean()) for wi, i, h in zip(w, S, h_by_s))
    obs = stat([hi[i] for i in S])
    ctrl_mean = sum(wi * yy[i][~hi[i]].mean() for wi, i in zip(w, S))
    null = np.empty(nperm)
    for p in range(nperm):
        null[p] = stat([rng.permutation(hi[i]) for i in S])
    pval = (1 + (null >= obs).sum()) / (nperm + 1); p2 = (1 + (np.abs(null) >= abs(obs)).sum()) / (nperm + 1)
    bs = np.empty(boot)
    for b in range(boot):
        v = 0
        for wi, i in zip(w, S):
            h = i[hi[i]]; l = i[~hi[i]]
            v += wi * (yy[rng.choice(h, len(h))].mean() - yy[rng.choice(l, len(l))].mean())
        bs[b] = v
    return {"diff": float(obs), "ctrl_mean": float(ctrl_mean), "lift": float(obs / ctrl_mean + 1) if ctrl_mean > 0 else None,
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p_one": float(pval), "p_two": float(p2),
            "n_high": int(hi.sum()), "n_ctrl": int((~hi).sum())}

def z(x): return (x - np.nanmean(x)) / np.nanstd(x)

def regress(sc, y, deg, pr, btw, part, pop, count):
    ok = pop & ~np.isnan(y)
    cols = {"score": z(sc), "logdeg": z(np.log(deg + 1)), "logpr": z(np.log(pr)), "logbtw": z(np.log1p(btw)), "part": z(part)}
    out = {}
    for name, use in (("deg+pr+btw", ["score", "logdeg", "logpr", "logbtw"]), ("+participation", ["score", "logdeg", "logpr", "logbtw", "part"])):
        X = sm.add_constant(np.column_stack([cols[c][ok] for c in use])); yy = y[ok]
        m = sm.GLM(yy, X, family=sm.families.Poisson()).fit(cov_type="HC1") if count else sm.OLS(yy, X).fit(cov_type="HC3")
        ci = m.conf_int()
        out[name] = {c: {"coef": float(m.params[i + 1]), "ci": [float(ci[i + 1, 0]), float(ci[i + 1, 1])], "p": float(m.pvalues[i + 1])} for i, c in enumerate(use)}
    return out

def load_cfg(res, seed, b, bf, btw_f):
    s = dict(np.load(os.path.join(OUTC, f"h4b_scores_r{res}_s{seed}.npz")))
    t = s["t"]; deg = s["deg_follow"]
    s["participation"] = ambiguity.participation(bf["A"], t)
    s["btw"] = btw_f; s["degree"] = deg.astype(float)
    pop = (t >= 0) & (deg >= 1)
    return s, t, deg, pop

def grid_job(a):
    res, seed = a
    b = views.load_timme(); bf = h4b_scores.follow_bundle(b); inter = h4b_data.load_inter()
    s, t, deg, pop = load_cfg(res, seed, b, bf, np.load(os.path.join(OUTC, "h4b_btw_follow.npy")))
    rng = np.random.default_rng(1)
    rows = []
    for g in GROUPS:
        o = outcomes(inter, t, g)
        for sn in ("misfit", "disagreement", "pagerank", "btw", "participation", "ambiguity_T", "content_outlier"):
            for on in ("cross_n", "cross_share", "wcross_n"):
                y = o["wcross_out"] + o["wcross_in"] if on == "wcross_n" else o[on]
                r = matched_test(s[sn], y, deg, pop, rng, nperm=2000, boot=100)
                rows.append({"res": res, "seed": seed, "group": g, "score": sn, "outcome": on, "k": int(t.max() + 1), **r})
    print("grid", a, flush=True)
    return rows

def primary():
    rng = np.random.default_rng(7)
    b = views.load_timme(); bf = h4b_scores.follow_bundle(b); inter = h4b_data.load_inter()
    btw = np.load(os.path.join(OUTC, "h4b_btw_follow.npy"))
    s, t, deg, pop = load_cfg(PRIMARY["res"], PRIMARY["seed"], b, bf, btw)
    res = {"n_pop": int(pop.sum()), "k": int(t.max() + 1), "sizes": np.bincount(t[t >= 0]).tolist(),
           "spearman": {}, "q1q2q3": {}}
    from scipy.stats import spearmanr
    for sn in ("pagerank", "btw", "degree", "participation", "disagreement"):
        res["spearman"][f"misfit~{sn}"] = float(spearmanr(s["misfit"][pop], s[sn][pop])[0])
    for g in GROUPS:
        o = outcomes(inter, t, g)
        res["q1q2q3"][g] = {}
        for sn in ("misfit", "disagreement", "pagerank", "btw", "participation"):
            d = {}
            for on in ("cross_n", "cross_share", "in_cross_share", "out_cross_share", "cross_in", "cross_out"):
                d[on] = matched_test(s[sn], o[on], deg, pop, rng, nperm=20000 if (sn == "misfit" and g == "spread") else 5000, boot=500)
            # also matched on interaction activity (all_n) rather than follow degree: cross share is the cleaner one
            res["q1q2q3"][g][sn] = d
        res["q1q2q3"][g]["regression"] = {
            "cross_n_poisson(misfit)": regress(s["misfit"], o["cross_n"].astype(float), deg, s["pagerank"], s["btw"], s["participation"], pop, True),
            "cross_share_ols(misfit)": regress(s["misfit"], o["cross_share"], deg, s["pagerank"], s["btw"], s["participation"], pop, False),
            "cross_share_ols(disagreement)": regress(s["disagreement"], o["cross_share"], deg, s["pagerank"], s["btw"], s["participation"], pop, False)}
        res["q1q2q3"][g]["activity"] = {"mean_all_n_misfit_top": float(o["all_n"][pop & (s["misfit"] >= np.quantile(s["misfit"][pop], .9))].mean()),
                                        "mean_all_n_all": float(o["all_n"][pop].mean()),
                                        "mean_cross_share_all": float(np.nanmean(o["cross_share"][pop]))}
    # circular variant: pooled-graph communities & scores (what the earlier sessions used)
    sp_ = dict(np.load(os.path.join(OUTC, "timme_views", "ambiguity.npz"))); tp = sp_["t"]
    popp = (tp >= 0) & (b["deg"] >= 1)
    res["circular_pooled"] = {}
    for g in ("spread", "retweet"):
        o = outcomes(inter, tp, g)
        res["circular_pooled"][g] = {on: matched_test(sp_["misfit"], o[on], b["deg"], popp, rng, nperm=5000, boot=300)
                                      for on in ("cross_n", "cross_share")}
    return res

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "primary":
        json.dump(primary(), open(os.path.join(OUTD, "primary.json"), "w"), indent=1)
    elif mode == "grid":
        cfgs = [(r, s) for r in (0.5, 1.0, 2.0) for s in (0, 1, 2)]
        with Pool(4) as P: rows = [r for rr in P.imap_unordered(grid_job, cfgs) for r in rr]
        json.dump(rows, open(os.path.join(OUTD, "grid.json"), "w"))
