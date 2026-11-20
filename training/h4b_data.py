"""H4b data: TIMME interaction edges (retweet/mention/reply/like + counts) mapped to the LCC node index
used by views.load_timme(); follow-only graph bundle."""
import os, numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import views
D = os.path.join(os.path.dirname(__file__), "../data_final/timme/repo/data/P_all")
FILES = {"follow": "friend_list.csv", "like": "favorite_list.csv", "mention": "mention_list.csv",
         "reply": "reply_list.csv", "retweet": "retweet_list.csv"}
CACHE = os.path.join(os.path.dirname(__file__), "out", "h4b_inter.npz")

def load_inter():
    """returns dict type -> (src,dst,count) in LCC index; also checks LCC reproduces views bundle."""
    if os.path.exists(CACHE):
        z = np.load(CACHE); return {k: (z[k+"_s"], z[k+"_d"], z[k+"_c"]) for k in FILES}
    ids = pd.read_csv(os.path.join(D, "all_twitter_ids.csv"), header=None, dtype=str).iloc[:, 0].str.strip()
    if not ids.iloc[0].lstrip("-").isdigit(): ids = ids.iloc[1:]
    id2 = pd.Series(np.arange(len(ids)), index=ids.to_numpy()); n = len(ids)
    raw = {}
    for k, fn in FILES.items():
        df = pd.read_csv(os.path.join(D, fn), sep="\t", dtype=str)
        a, b = df.iloc[:, 0].str.strip().map(id2), df.iloc[:, 1].str.strip().map(id2)
        c = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(1).to_numpy()
        ok = (a.notna() & b.notna()).to_numpy()
        raw[k] = (a[ok].to_numpy(np.int64), b[ok].to_numpy(np.int64), c[ok])
    s = np.concatenate([raw[k][0] for k in raw]); d = np.concatenate([raw[k][1] for k in raw])
    m = s != d
    A = sp.coo_matrix((np.ones(m.sum()), (s[m], d[m])), shape=(n, n)).tocsr(); A = A + A.T
    _, comp = connected_components(A, directed=False)
    keep = np.flatnonzero(comp == np.bincount(comp).argmax())
    lut = np.full(n, -1); lut[keep] = np.arange(len(keep))
    out, save = {}, {}
    for k, (a, b, c) in raw.items():
        a2, b2 = lut[a], lut[b]; ok = (a2 >= 0) & (b2 >= 0) & (a2 != b2)
        out[k] = (a2[ok], b2[ok], c[ok])
        save[k+"_s"], save[k+"_d"], save[k+"_c"] = out[k]
    save["n"] = len(keep)
    np.savez_compressed(CACHE, **save)
    return out

if __name__ == "__main__":
    b = views.load_timme(); it = load_inter()
    print("n views", b["n"], "edges", b["g"].ecount())
    for k, (s, d, c) in it.items(): print(k, len(s), "sum count", c.sum(), "max idx", max(s.max(), d.max()))
    # consistency: pooled sym graph from files == views A
    n = b["n"]; s = np.concatenate([v[0] for v in it.values()]); d = np.concatenate([v[1] for v in it.values()])
    A = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(float)
    print("A diff nnz", abs(A - b["A"]).nnz)
