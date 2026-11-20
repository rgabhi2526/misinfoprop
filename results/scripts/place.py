import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, views, ambiguity
from scipy.stats import rankdata
b = views.load_timme(); d = os.path.join(views.OUT, "timme_views")
s = dict(np.load(os.path.join(d, "ambiguity_minilm.npz"))); t = s["t"]; k = t.max() + 1
aff = ambiguity.content_probs(b["glove"], t)
own = aff[np.arange(len(t)), t]
# effective beta: synthetic own-community acceptance = e^b / (e^b + k - 1)  ->  b = log(own*(k-1)/(1-own))
beta_eff = np.log(np.median(own) * (k - 1) / (1 - np.median(own)))
print(f"TIMME acceptance of own community: median {np.median(own):.3f} -> effective agreement effect beta ~ {beta_eff:.2f} (map tested 0,1,3,6)")
# who is mismatched (top 5% misfit) and how connected are they?
mis = s["misfit"] >= np.quantile(s["misfit"], 0.95)
pct = rankdata(b["deg"]) / len(t)
print(f"top-5% misfit nodes: median degree percentile {np.median(pct[mis]):.2f} (0.5 = typical; 'hub' placement in the map ~ {0.0:.0f}+)")
# synthetic reference: degree percentile of planted nodes under hub placement
import regime
ei, text, comm, tc = regime.planted_place(4000, 0.3, 0.05, "hub", 0)
dg = np.bincount(ei[0], minlength=4000); pp = rankdata(dg) / 4000
print(f"synthetic hub placement: planted median degree percentile {np.median(pp[tc != comm]):.2f}; random placement ~0.50")
