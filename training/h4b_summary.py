import json, os, numpy as np
from collections import defaultdict
D = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h4b")
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
def bh(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 1
    for r in range(m - 1, -1, -1): i = o[r]; run = min(run, p[i] * m / (r + 1)); adj[i] = run
    return adj
out = []
P = json.load(open(os.path.join(D, "primary.json")))
out.append(f"PRIMARY follow-only T: k={P['k']} sizes={P['sizes']} n={P['n_pop']}\nspearman {P['spearman']}")
for g, d in P["q1q2q3"].items():
    out.append(f"\n== group {g} (activity: {d['activity']})")
    for sn in ("misfit", "disagreement", "pagerank", "btw", "participation"):
        for on in ("cross_n", "cross_share", "in_cross_share", "cross_in"):
            r = d[sn][on]
            out.append(f"{sn:13s} {on:15s} diff {r['diff']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] ctrl {r['ctrl_mean']:.3f} p1 {r['p_one']:.4f} p2 {r['p_two']:.4f}")
    for k, v in d["regression"].items():
        for mname, m in v.items():
            out.append(f"  reg {k} [{mname}]: " + "; ".join(f"{c} {x['coef']:+.3f}[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]" for c, x in m.items()))
out.append("\nCIRCULAR pooled: " + json.dumps({g: {o: (round(r['diff'], 4), [round(x, 4) for x in r['ci']], r['p_two']) for o, r in v.items()} for g, v in P["circular_pooled"].items()}))
fn = os.path.join(D, "grid.json")
if os.path.exists(fn):
    G = json.load(open(fn)); p = [r["p_two"] for r in G]
    for r, a, b in zip(G, holm(p), bh(p)): r["holm"], r["bh"] = a, b
    out.append(f"\nGRID {len(G)} tests; nominal p2<0.05: {sum(x<0.05 for x in p)}, BH<0.05: {sum(r['bh']<0.05 for r in G)}, Holm<0.05: {sum(r['holm']<0.05 for r in G)}")
    for sn in ("misfit", "disagreement", "pagerank", "btw", "participation", "ambiguity_T", "content_outlier"):
        for on in ("cross_n", "cross_share"):
            rs = [r for r in G if r["score"] == sn and r["outcome"] == on and r["res"] >= 1.0]
            pos = [r for r in rs if r["diff"] > 0 and r["ci"][0] > 0]; neg = [r for r in rs if r["diff"] < 0 and r["ci"][1] < 0]
            out.append(f"{sn:15s} {on:12s} (res>=1, n={len(rs)}): CI>0 {len(pos)}, CI<0 {len(neg)}, median diff {np.median([r['diff'] for r in rs]):+.4f}, BH<.05 {sum(r['bh']<0.05 for r in rs)}")
    for r in sorted([r for r in G if r["score"] in ("misfit", "disagreement") and r["res"] >= 1.0], key=lambda r: r["p_two"])[:12]:
        out.append(f"  top: r{r['res']} s{r['seed']} {r['group']} {r['score']} {r['outcome']} diff {r['diff']:+.4f} {r['ci']} p2 {r['p_two']:.4f} holm {r['holm']:.3f}")
G2 = json.load(open(os.path.join(D, "gating.json"))) if os.path.exists(os.path.join(D, "gating.json")) else {}
out.append("\nGATING (cross-T agree_C log-OR; same-T for comparison; rec = receptivity p_C coef)")
for k, v in G2.items():
    x, s, r = v["cross_T"]["agree_C"], v["same_T"]["agree_C"], v["cross_T"]["receptivity_pC"]
    out.append(f"{k:28s} cross {x['coef']:+.3f}[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}] same {s['coef']:+.3f} rec {r['coef']:+.3f}[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] n+ {v['cross_T']['n_pos']}")
open(os.path.join(D, "summary.txt"), "w").write("\n".join(out)); print("\n".join(out))
