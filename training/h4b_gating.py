"""H4b Q5: agreement gating on real interactions. Case-control: observed directed interaction (u,v) vs degree-preserving
shuffled-target pairs; logit on text-agreement controlling for degrees and T-community-pair fixed effects.
beta_hat = coefficient on agreement (log odds ratio)  vs the synthetic gate exp(beta) (aff ~ exp(beta*match)).
    python h4b_gating.py -> ../results/hypotheses/h4b/gating.json"""
import json, os, numpy as np, statsmodels.api as sm
import views, h4b_data, h4b_scores
OUTD = h4b_scores.OUTC and os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h4b")

def fit(s, d, neg_d, t, tc, pc, deg, same_T, ntake=400000, rng=None):
    """rows: positives (s,d) + negatives (s,neg_d); keep only pairs with same_T == (t[s]==t[d]) and both t>=0."""
    S = np.concatenate([s, s]); D = np.concatenate([d, neg_d]); y = np.r_[np.ones(len(s)), np.zeros(len(s))]
    ok = (t[S] >= 0) & (t[D] >= 0) & ((t[S] == t[D]) == same_T) & (S != D)
    S, D, y = S[ok], D[ok], y[ok]
    if len(S) > ntake: i = rng.choice(len(S), ntake, replace=False); S, D, y = S[i], D[i], y[i]
    K = int(t.max()) + 1
    pair = t[S] * K + t[D]
    up = np.unique(pair)[1:]
    X = [np.log(deg[S] + 1), np.log(deg[D] + 1)] + [(pair == p).astype(float) for p in up]
    out = {}
    for name, f in (("agree_C", (tc[S] == tc[D]).astype(float)), ("receptivity_pC", pc[D, np.where(t[S] < 0, 0, t[S])]),):
        Xm = sm.add_constant(np.column_stack([f] + X))
        m = sm.GLM(y, Xm, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": S})
        ci = m.conf_int()
        out[name] = {"coef": float(m.params[1]), "ci": [float(ci[1, 0]), float(ci[1, 1])], "p": float(m.pvalues[1]),
                     "feature_mean_pos": float(f[y == 1].mean()), "feature_mean_neg": float(f[y == 0].mean())}
    out["n_pos"], out["n_neg"] = int(y.sum()), int((1 - y).sum())
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    b = views.load_timme(); inter = h4b_data.load_inter()
    C = np.load(os.path.join(h4b_scores.OUTC, "timme_views", "C_leiden_runs.npy"))
    res = {}
    for cfg in ((1.0, 0), (1.0, 1), (2.0, 0)):
        s = dict(np.load(os.path.join(h4b_scores.OUTC, f"h4b_scores_r{cfg[0]}_s{cfg[1]}.npz")))
        if not os.path.exists(os.path.join(h4b_scores.OUTC, f"h4b_scores_r{cfg[0]}_s{cfg[1]}.npz")): continue
        t, pc, deg = s["t"], s["p_C"], s["deg_follow"]
        for ci in (0, 1):
            tc = C[ci]
            for typ in ("retweet", "mention", "reply", "like", "follow"):
                a, d, _ = inter[typ]
                r = {}
                for same in (False, True):
                    negs = [fit(a, d, rng.permutation(d), t, tc, pc, deg, same, rng=rng)]
                    r["cross_T" if not same else "same_T"] = negs[0]
                res[f"r{cfg[0]}_s{cfg[1]}|C{ci}|{typ}"] = r
                x = r["cross_T"]["agree_C"]
                print(cfg, ci, typ, "cross agree beta %.2f [%.2f,%.2f]" % (x["coef"], *x["ci"]),
                      "same-T %.2f" % r["same_T"]["agree_C"]["coef"], flush=True)
    json.dump(res, open(os.path.join(OUTD, "gating.json"), "w"), indent=1)
