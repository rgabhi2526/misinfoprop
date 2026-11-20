import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, views, ambiguity, edges
K = ("bridge_struct", "bridge_content", "bridge_struct_x_recv")
for beta in (0.0, 4.0):
    g = {k: [] for k in K[1:]}
    for seed in range(4):
        b, s, comm, aff = edges.synthetic(seed=seed, beta=beta)
        u, v, ranked = edges.edge_scores(b, s, seed)
        res = edges.simulate_edges(b, u, v, {k: ranked[k] for k in K}, comm, [1600], "ic:2", aff, sims=60, seed=seed)
        base = res["none"][0].mean()
        for k in K[1:]: g[k].append((res["bridge_struct"][1600] - res[k][1600]) / base)
    for k in K[1:]:
        x = np.concatenate(g[k]); print(f"synthetic beta={beta} {k}: {100*x.mean():+.1f} +- {196*x.std(ddof=1)/np.sqrt(len(x)):.1f} vs bridge_struct", flush=True)
b = views.load_timme(); d = os.path.join(views.OUT, "timme_views")
s = dict(np.load(os.path.join(d, "ambiguity_minilm.npz"))); aff = ambiguity.content_probs(b["glove"], s["t"])
u, v, ranked = edges.edge_scores(b, s); t = s["t"]; pr = s["pagerank"] / s["pagerank"].mean()
orc2 = np.where(t[u] != t[v], pr[u] * pr[v] * np.maximum(aff[v, t[u]], aff[u, t[v]]), -1.0)
r = {k: ranked[k] for k in K}; r["bridge_struct_x_recv_ORACLE"] = np.argsort(-orc2, kind="stable")
print("TIMME:"); edges.report(edges.simulate_edges(b, u, v, r, t, [10000, 40000], "ic:2", aff, sims=60), [10000, 40000])
