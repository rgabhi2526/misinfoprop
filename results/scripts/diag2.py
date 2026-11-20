import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch, scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import cross_val_score, StratifiedKFold
import train_timme
data, y = train_timme.load_data()
n = data.num_nodes; ei = data.edge_index.numpy(); x = data.x.numpy()
print("x last col unique (first 5):", np.unique(x[:, -1])[:5], " x col0 mean/std", x[:,0].mean().round(3), x[:,0].std().round(3))
A = sp.coo_matrix((np.ones(ei.shape[1]), ei), (n, n)).tocsr(); A = ((A + A.T) > 0).astype(float)
deg = np.asarray(A.sum(1)).ravel()
d = 1/np.sqrt(np.maximum(deg,1)); u, s, _ = svds(sp.diags(d) @ A @ sp.diags(d), k=64); spec = u
m = y >= 0; cv = StratifiedKFold(5, shuffle=True, random_state=0)
def probe(z, C=1.0, std=True):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)) if std else LogisticRegression(C=C, max_iter=5000)
    return round(cross_val_score(clf, z[m], y[m], cv=cv).mean(), 3)
# neighbour-label baseline, computed out-of-fold (no leakage)
acc = []
for tr, te in cv.split(np.where(m)[0], y[m]):
    idx = np.where(m)[0]; yl = np.zeros(n); yl[idx[tr]] = y[idx[tr]]*2-1
    v = A @ yl; acc.append(((v[idx[te]] > 0).astype(int) == y[idx[te]]).mean())
print("neighbour-vote (graph, out-of-fold)", round(np.mean(acc), 3))
print("text std-probe", probe(x), "| spectral64 std-probe", probe(spec))
for mdl in ["bgrl", "dgi", "mvgrl"]:
    e = torch.load(f"../timme/timme_{mdl}_emb.pt").numpy()
    print(f"{mdl}: raw C=1 nostd {probe(e,std=False)} | L2 C=1 nostd (=report) {probe(normalize(e),std=False)} | std {probe(e)} | std emb+text {probe(np.hstack([e,x]))}")
