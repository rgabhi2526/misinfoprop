import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, views, ambiguity, edges
b = views.load_timme(); d = os.path.join(views.OUT, "timme_views")
s = dict(np.load(os.path.join(d, "ambiguity_minilm.npz")))
aff = ambiguity.content_probs(b["glove"], s["t"])
t = s["t"]; pc = s["p_C"]
# how well does MiniLM receptivity track the GloVe acceptance used by the spread?
from scipy.stats import spearmanr
print("node-level spearman(p_C minilm, affinity glove), all (node,community) pairs:",
      round(spearmanr(pc.ravel(), aff.ravel())[0], 3))
print("argmax agreement:", round((pc.argmax(1) == aff.argmax(1)).mean(), 3))
u, v, ranked = edges.edge_scores(b, s)
pr = s["pagerank"] / s["pagerank"].mean()
cross = (t[u] != t[v])
orc = np.where(cross, np.maximum(pr[u] * aff[v, t[u]], pr[v] * aff[u, t[v]]), -1.0)
ranked = {"bridge_struct": ranked["bridge_struct"], "bridge_content": ranked["bridge_content"],
          "bridge_oracle": np.argsort(-orc, kind="stable")}
comm = t
res = edges.simulate_edges(b, u, v, ranked, comm, [10000, 40000], "ic:2", aff, sims=60)
edges.report(res, [10000, 40000])
