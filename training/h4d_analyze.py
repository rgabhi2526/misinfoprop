"""Analyse h4d_raw.jsonl -> h4d_cells.json, h4d_table.md, h4d_surface.txt"""
import json, os, itertools
import numpy as np
from scipy import stats
D = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h4d")
MIS = ("misfit", "misfit_x_pr", "gateway", "misfit_rerank")
STR = ("pagerank", "betweenness", "degree", "collective_influence", "adaptive_degree")
rows = [json.loads(l) for l in open(os.path.join(D, "h4d_raw.jsonl"))]
cells = {}
for r in rows:
    cells.setdefault((r["mu"], r["plant"], r["place"], r["beta"]), []).append(r)
out = []
def ci(x):
    x = np.asarray(x) * 100; n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(.975, n - 1); p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else 1.0
    return float(m), float(t * se), float(p)
for key, rs in sorted(cells.items()):
    rs.sort(key=lambda r: r["seed"])
    for bud in ("40", "80", "200"):
        for mi, mname in ((0, "infected"), (1, "cross")):
            M = {s: np.array([r["rem"][bud][s][mi] for r in rs]) for s in MIS + STR}
            bs = max(STR, key=lambda s: M[s].mean()); bm = max(MIS, key=lambda s: M[s].mean())
            g_best = M[bm] - M[bs]; g_fix = M["misfit_x_pr"] - M[bs]
            m1 = ci(g_best); m2 = ci(g_fix)
            out.append(dict(mu=key[0], plant=key[1], place=key[2], beta=key[3], budget=int(bud), metric=mname,
                n=len(rs), best_mis=bm, best_str=bs, mis_removed=float(M[bm].mean() * 100),
                str_removed=float(M[bs].mean() * 100), base=float(np.mean([r["base_inf" if mi == 0 else "base_cross"] for r in rs]) * 100),
                gain=m1[0], gain_ci=m1[1], gain_p=m1[2], fix_gain=m2[0], fix_ci=m2[1], fix_p=m2[2],
                seed_gains=(g_fix * 100).tolist()))
json.dump(out, open(os.path.join(D, "h4d_cells.json"), "w"), indent=1)

def verdict(c, g="fix"):
    m, h = c[g + "_gain"] if g == "gain" else c["fix_gain"], c[g + "_ci"] if g == "gain" else c["fix_ci"]
    return "WIN" if m - h > 0 else ("LOSS" if m + h < 0 else "TIE")
L = []
for metric in ("infected", "cross"):
    for g, gl in (("gain", "best misfit-based (selected) vs best structural"), ("fix", "fixed misfit_x_pr vs best structural")):
        cs = [c for c in out if c["metric"] == metric]
        pk = g + "_p" if g == "fix" else "gain_p"
        mk = "fix_gain" if g == "fix" else "gain"; ck = "fix_ci" if g == "fix" else "gain_ci"
        win = [c for c in cs if c[mk] - c[ck] > 0]; loss = [c for c in cs if c[mk] + c[ck] < 0]
        N = len(cs)
        # Holm on one-sided positive p
        ps = np.array([(c[pk] / 2 if c[mk] > 0 else 1 - c[pk] / 2) for c in cs]); o = np.argsort(ps); holm = 0
        for i, j in enumerate(o):
            if ps[j] <= 0.05 / (N - i): holm += 1
            else: break
        L.append(f"[{metric}] {gl}: cells={N}; CI>0: {len(win)} (chance under null ~{N*0.025:.1f} one-sided); "
                 f"CI<0: {len(loss)}; Holm-surviving wins (alpha .05): {holm}")
open(os.path.join(D, "h4d_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

# tables: metric=infected, fixed & selected
T = ["# h4d cell table (infected-share points; gain = misfit-based minus best structural, 95% t-CI over seeds)\n"]
for metric in ("infected", "cross"):
    T.append(f"\n## metric: {metric}\n")
    T.append("| mu | plant | place | beta | budget | base% | struct removed (best) | misfit removed (best) | gain best-sel | gain fixed misfit_x_pr | verdict(fixed) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for c in out:
        if c["metric"] != metric: continue
        T.append(f"| {c['mu']} | {c['plant']} | {c['place']} | {c['beta']} | {c['budget']} | {c['base']:.0f} | {c['str_removed']:.1f} ({c['best_str']}) | {c['mis_removed']:.1f} ({c['best_mis']}) | {c['gain']:+.2f}±{c['gain_ci']:.2f} | {c['fix_gain']:+.2f}±{c['fix_ci']:.2f} | {verdict(c)} |")
open(os.path.join(D, "h4d_table.md"), "w").write("\n".join(T) + "\n")

# response surface: OLS on seed-level fixed-gain, infected metric, cluster-robust by (cell-seed)
import numpy.linalg as la
def surface(metric, target):
    X, y, grp = [], [], []
    for c in out:
        if c["metric"] != metric: continue
        for sd, gv in enumerate(c["seed_gains"]):
            hub = c["place"] == "hub"; lb = np.log(c["beta"] / 4)
            X.append([1, lb, hub, hub * lb, c["plant"] / .05, c["mu"] > .4, c["budget"] == 80, c["budget"] == 200, hub * (c["plant"] / .05)])
            y.append(gv); grp.append(sd)
    X = np.array(X, float); y = np.array(y); grp = np.array(grp)
    names = ["const", "log(beta/4)", "hub", "hub*log(beta/4)", "plant5%", "mu0.5", "bud2%", "bud5%", "hub*plant5%"]
    b = la.lstsq(X, y, rcond=None)[0]; res = y - X @ b
    XtXi = la.inv(X.T @ X); meat = np.zeros((X.shape[1],) * 2)
    for g in np.unique(grp):   # cluster by seed (graph shared across cells)
        i = grp == g; s = X[i].T @ res[i]; meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi; se = np.sqrt(np.diag(V))
    r2 = 1 - res.var() / y.var()
    s = [f"response surface ({metric}, fixed misfit_x_pr minus best structural, points; seed-clustered SE), R2={r2:.2f}, n={len(y)}"]
    s += [f"  {n:18s} {bb:+7.2f} +- {1.96*e:5.2f}" for n, bb, e in zip(names, b, se)]
    # predicted gain
    for place in ("hub", "random"):
        for beta in (1.44, 4, 6, 8, 12, 20):
            hub = place == "hub"; lb = np.log(beta / 4)
            x = np.array([1, lb, hub, hub * lb, 1, 0, 0, 0, hub * 1.0])
            s.append(f"  pred {place:6s} beta={beta:5.2f} plant5% mu0.3 bud1%: {x@b:+.2f}")
    return "\n".join(s)
txt = surface("infected", "fix") + "\n\n" + surface("cross", "fix")
open(os.path.join(D, "h4d_surface.txt"), "w").write(txt + "\n"); print(txt)
