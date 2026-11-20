import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
import scipy.sparse as sp
import train_timme, metrics
data, y = train_timme.load_data()
n = data.num_nodes; ei = data.edge_index.numpy()
x = data.x.numpy(); has = x[:, -1]
A = sp.coo_matrix((np.ones(ei.shape[1]), ei), (n, n)).tocsr(); A = ((A + A.T) > 0).astype(float)
deg = np.asarray(A.sum(1)).ravel(); ld = np.log1p(deg)
m = y >= 0; cv = StratifiedKFold(5, shuffle=True, random_state=0)
probe = lambda z: cross_val_score(LogisticRegression(max_iter=3000), z[m], y[m], cv=cv).mean()
print(f"labeled {m.sum()}  R share {y[m].mean():.2f}  (majority baseline {max(y[m].mean(),1-y[m].mean()):.3f})")
print(f"deg of labeled median {np.median(deg[m]):.0f} vs all {np.median(deg):.0f}; has_text labeled {has[m].mean():.2f} all {has.mean():.2f}")
# graph-only baseline: spectral 64-d of normalized adjacency
d = 1/np.sqrt(np.maximum(deg,1)); An = sp.diags(d) @ A @ sp.diags(d)
u, s, _ = svds(An, k=64); spec = u * s
nbr_vote = np.zeros(n)
yl = np.where(m, y*2-1, 0).astype(float); nbr_vote = A @ yl  # +R -D neighbour votes (leaky, info only)
print("probe text-only       ", round(probe(x), 3))
print("probe spectral(graph) ", round(probe(spec), 3))
print("probe logdeg only     ", round(probe(ld[:, None]), 3))
for mdl in ["bgrl", "dgi", "mvgrl"]:
    e = torch.load(f"../timme/timme_{mdl}_emb.pt").numpy()
    z = normalize(e)
    p2, _ = metrics.cluster(e, 2)
    p6, _ = metrics.cluster(e, 6)
    r = lambda a, b: abs(np.corrcoef(a, b)[0, 1])
    print(f"\n{mdl}: probe {probe(e):.3f} | probe[emb+text] {probe(np.hstack([z, x])):.3f}")
    print(f"  k=2 split vs logdeg |r|={r(p2, ld):.2f}  vs has_text |r|={r(p2, has):.2f}  vs party NMI={nmi(y[m], p2[m]):.3f}")
    print(f"  k=2 mean logdeg per cluster {[round(ld[p2==c].mean(),2) for c in (0,1)]}, has_text {[round(has[p2==c].mean(),2) for c in (0,1)]}")
    print(f"  k=6: sizes {np.bincount(p6).tolist()} Q={metrics.modularity(data.edge_index, p6):.3f} party NMI={nmi(y[m], p6[m]):.3f}")
    print(f"  top-3 PCs |r| with logdeg: {[round(r(pc, ld),2) for pc in (np.linalg.svd(z - z.mean(0), full_matrices=False)[0][:, :3]).T]}")
