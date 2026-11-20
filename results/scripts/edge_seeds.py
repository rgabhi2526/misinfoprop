import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, edges
for beta in (0.0, 4.0):
    gains = []
    for seed in range(4):
        b, s, comm, aff = edges.synthetic(seed=seed, beta=beta)
        u, v, ranked = edges.edge_scores(b, s, seed)
        ranked = {k: ranked[k] for k in ("bridge_struct", "bridge_content")}
        res = edges.simulate_edges(b, u, v, ranked, comm, [1600], "ic:2", aff, sims=60, seed=seed)
        base = res["none"][0].mean()
        d = (res["bridge_struct"][1600] - res["bridge_content"][1600]) / base
        gains.append(d)
        print(f"beta={beta} seed={seed}: content - struct = {100*d.mean():+.1f} +- {196*d.std(ddof=1)/np.sqrt(len(d)):.1f} points", flush=True)
    g = np.concatenate(gains)
    print(f"== beta={beta} pooled over 4 seeds: {100*g.mean():+.1f} +- {196*g.std(ddof=1)/np.sqrt(len(g)):.1f} points", flush=True)
