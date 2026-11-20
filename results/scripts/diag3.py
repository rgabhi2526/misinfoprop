import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score as nmi
import train_timme, metrics
data, y = train_timme.load_data(); n = data.num_nodes; ei = data.edge_index.numpy(); m = y >= 0
A = sp.coo_matrix((np.ones(ei.shape[1]), ei), (n, n)).tocsr(); A = ((A + A.T) > 0)
ld = np.log1p(np.asarray(A.sum(1)).ravel())
for mdl in ["bgrl", "dgi", "mvgrl"]:
    e = torch.load(f"../timme/timme_{mdl}_emb.pt").numpy().astype(np.float64)
    e = e - e.mean(0); L = np.c_[np.ones(n), ld]
    r = e - L @ np.linalg.lstsq(L, e, rcond=None)[0]          # regress out log-degree
    out = []
    for k in (2, 6):
        for name, z in (("raw", e), ("-deg", r)):
            p, _ = metrics.cluster(z, k)
            out.append(f"k={k} {name}: Q={metrics.modularity(data.edge_index, p):+.3f} NMI={nmi(y[m], p[m]):.3f}")
    print(mdl, " | ".join(out))
