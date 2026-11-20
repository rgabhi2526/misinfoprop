"""Phase 1 views of the make-it-work plan: two CLEAN partitions of the same nodes.

  T (topology)  edges only: Leiden on the social graph, many seeds; plus a
                regularized spectral embedding (Qin & Rohe 2013 / Zhang & Rohe 2018)
  C (content)   text only: Leiden on a cosine kNN graph of the raw MiniLM vectors,
                many seeds. Never built from GNN output: a content view that has
                seen the graph makes T-vs-C disagreement meaningless.

Every partition is scored the same way (`score`): modularity on the SOCIAL graph,
party NMI/ARI on TIMME (reported, never used to select), degree_eta2 (is it just
sorting by degree?), cluster count/size, and seed stability across runs.

    python views.py timme            # TIMME P_all, writes out/timme_views/
    python views.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import os
import random

import igraph as ig
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from metrics import degree_eta2, log

OUT = os.path.join(os.path.dirname(__file__), "out")


# --------------------------------------------------------------------------- #
# data
def load_timme():
    """Symmetric binary adjacency (LCC, same node order as train_timme), RAW
    text vectors (unit MiniLM, not standardized), has_text flag, party labels.
    Cached: the full loader re-reads 6.5M relation rows."""
    import train_timme
    cache = os.path.join(train_timme.DEFAULT_DIR, "timme_views.npz")
    if not os.path.exists(cache):
        data, y = train_timme.load_data()
        raw = np.load(os.path.join(train_timme.DEFAULT_DIR, "timme_text.npy"))
        raw = raw[data.orig_idx.numpy()]
        glove = np.load(os.path.join(train_timme.DEFAULT_DIR, "tweet_features.npz"))
        glove = glove["tweets_average"][data.orig_idx.numpy()]
        np.savez_compressed(cache, ei=data.edge_index.numpy(), text=raw[:, :-1],
                            has=raw[:, -1], y=y, glove=glove.astype(np.float32))
    c = np.load(cache)
    if "glove" not in c.files:          # cache from before GloVe was added
        os.remove(cache)
        return load_timme()
    b = _bundle(c["ei"], c["text"], c["has"].astype(bool), c["y"])
    # second content source: GloVe average over ALL of a user's tweets (shipped
    # with TIMME). Party probe 0.84 vs 0.70 for MiniLM on bio + latest tweet.
    b["glove"] = c["glove"]
    return b


POKEC_SLICES = {"pokec_knm": "zilinsky kraj, kysucke nove mesto"}


def load_pokec(tag="pokec_knm"):
    """Pokec region slice: friendship graph (LCC) + RAW interest-text vectors.
    No labels (y = -1). Region is deliberately NOT part of the content view
    (inside one district it is constant anyway). Cached like TIMME."""
    import train_pokec
    region = POKEC_SLICES[tag]
    cache = os.path.join(train_pokec.DEFAULT_DIR, f"{tag}_views.npz")
    if not os.path.exists(cache):
        data, _ = train_pokec.load_data(region=region)
        key = region.replace(" ", "_").replace(",", "")
        raw = np.load(os.path.join(train_pokec.DEFAULT_DIR, f"pokec_text_{key}.npy"))
        raw = raw[data.orig_idx.numpy()]
        np.savez_compressed(cache, ei=data.edge_index.numpy(), text=raw[:, :-1],
                            has=raw[:, -1])
    c = np.load(cache)
    return _bundle(c["ei"], c["text"], c["has"].astype(bool), np.full(len(c["has"]), -1))


def _bundle(ei, text, has, y):
    n = len(y)
    a = sp.coo_matrix((np.ones(ei.shape[1]), ei), (n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64)
    a.setdiag(0)
    a.eliminate_zeros()
    up = sp.triu(a).tocoo()
    g = ig.Graph(n=n, edges=np.c_[up.row, up.col])
    return {"A": a, "g": g, "deg": np.asarray(a.sum(1)).ravel(), "text": text,
            "has": has, "y": y, "n": n}


# --------------------------------------------------------------------------- #
# views
def leiden_runs(g, runs=100, resolution=1.0, seed=0):
    """`runs` Leiden partitions (modularity objective), one per RNG seed."""
    out = []
    for r in range(runs):
        ig.set_random_number_generator(random.Random(seed + r))
        out.append(g.community_leiden(objective_function="modularity",
                                      resolution=resolution, n_iterations=-1).membership)
    return np.asarray(out, dtype=np.int64)


def reg_spectral(a, dim=64, tau=None):
    """Regularized spectral embedding: eigenvectors of (D+tau)^-1/2 A (D+tau)^-1/2,
    trivial top vector dropped, rows L2-normalized. tau = mean degree (Qin & Rohe)
    stops hubs and dangling followers from owning whole eigenvectors; row
    normalization removes the degree 'star' that broke the GNN embeddings."""
    deg = np.asarray(a.sum(1)).ravel()
    tau = deg.mean() if tau is None else tau
    d = sp.diags(1 / np.sqrt(deg + tau))
    w, u = eigsh(d @ a @ d, k=dim + 1, which="LA")
    u = u[:, np.argsort(-w)][:, 1:]
    return normalize(u)


def knn_graph(z, k=15):
    """Undirected cosine kNN graph (union of kNN lists)."""
    nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(z)
    idx = nb.kneighbors(z, return_distance=False)[:, 1:]
    g = ig.Graph(n=len(z), edges=np.c_[np.repeat(np.arange(len(z)), k), idx.ravel()])
    g.simplify()
    return g


# --------------------------------------------------------------------------- #
# scoring
def stability(runs, pairs=50, seed=0):
    """Mean ARI between random pairs of runs (1 = identical every seed)."""
    rng = np.random.default_rng(seed)
    pr = list(itertools.combinations(range(len(runs)), 2))
    pick = rng.choice(len(pr), min(pairs, len(pr)), replace=False)
    return float(np.mean([adjusted_rand_score(runs[pr[i][0]], runs[pr[i][1]]) for i in pick]))


def score(p, b):
    """Selection-rule numbers for one partition p on bundle b."""
    p = np.asarray(p)
    sizes = np.bincount(p)
    m = b["y"] >= 0
    return {
        "k": int((sizes > 0).sum()),
        "k_ge_1pct": int((sizes >= 0.01 * b["n"]).sum()),
        "largest": float(sizes.max() / b["n"]),
        "Q": float(b["g"].modularity(p.tolist())),
        "nmi": float(normalized_mutual_info_score(b["y"][m], p[m])) if m.any() else None,
        "ari": float(adjusted_rand_score(b["y"][m], p[m])) if m.any() else None,
        "degree_eta2": degree_eta2(p, b["deg"]),
    }


def best_run(runs, g):
    """Highest-modularity run (on graph g): the representative partition."""
    return runs[int(np.argmax([g.modularity(r.tolist()) for r in runs]))]


def fmt(name, s, stab=None):
    st = "" if stab is None else f" stab={stab:.2f}"
    nm = "" if s["nmi"] is None else f" NMI={s['nmi']:.3f} ARI={s['ari']:.3f}"
    return (f"{name:38s} k={s['k']:3d} (>=1%: {s['k_ge_1pct']:2d}) largest={s['largest']:.2f} "
            f"Q={s['Q']:+.3f}{nm} eta2={s['degree_eta2']:.2f}{st}")


# --------------------------------------------------------------------------- #
def run(b, tag, runs=100):
    from sklearn.cluster import KMeans
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    save = lambda k, v: np.save(os.path.join(OUT, tag, f"{k}.npy"), v)

    log(f"T view: Leiden x{runs} on social graph ({b['g'].ecount()} edges)")
    t_runs = leiden_runs(b["g"], runs)
    t = best_run(t_runs, b["g"])
    save("T_leiden_runs", t_runs)
    print(fmt("T  Leiden (best of runs)", score(t, b), stability(t_runs)), flush=True)
    k = int((np.bincount(t) >= 0.01 * b["n"]).sum())

    log("T view: regularized spectral embedding")
    z = reg_spectral(b["A"])
    save("T_spectral", z.astype(np.float32))
    km = [KMeans(k, n_init=10, random_state=s).fit_predict(z) for s in range(5)]
    print(fmt(f"T  reg-spectral + k-means k={k}", score(km[0], b), stability(km)), flush=True)

    log(f"C view: text-only kNN graph + Leiden x{runs} ({int(b['has'].sum())} nodes with text)")
    gc = knn_graph(b["text"])
    c_runs = leiden_runs(gc, runs)
    c = best_run(c_runs, gc)
    save("C_leiden_runs", c_runs)
    print(fmt("C  text kNN Leiden (best of runs)", score(c, b), stability(c_runs)), flush=True)
    kmc = [KMeans(k, n_init=10, random_state=s).fit_predict(b["text"]) for s in range(5)]
    print(fmt(f"C  text k-means k={k}", score(kmc[0], b), stability(kmc)), flush=True)

    s = score(c, b)
    print(f"\nT vs C: NMI={normalized_mutual_info_score(t, c):.3f} "
          f"ARI={adjusted_rand_score(t, c):.3f}  (low = the views genuinely differ)")
    return t_runs, c_runs


def selftest():
    """Planted case: 4 graph communities; text follows the graph except for a
    20-node group whose text says community 0 while their edges say community 1.
    T must recover the graph blocks, C the text blocks, and they must differ."""
    rng = np.random.default_rng(0)
    n, kc = 400, 4
    gt = np.repeat(np.arange(kc), n // kc)
    p = np.where(gt[:, None] == gt[None, :], 0.15, 0.005)
    up = np.triu(rng.random((n, n)) < p, 1)
    s, d = np.nonzero(up | up.T)
    tt = gt.copy()
    mism = np.flatnonzero(gt == 1)[:20]
    tt[mism] = 0
    centers = normalize(rng.normal(size=(kc, 32)))
    text = normalize(centers[tt] + 0.15 * rng.normal(size=(n, 32)))
    b = _bundle(np.stack([s, d]), text, np.ones(n, bool), gt)
    t = best_run(leiden_runs(b["g"], 5), b["g"])
    gc = knn_graph(text, 10)
    c = best_run(leiden_runs(gc, 5), gc)
    assert adjusted_rand_score(gt, t) > 0.95, adjusted_rand_score(gt, t)
    assert adjusted_rand_score(tt, c) > 0.9, adjusted_rand_score(tt, c)
    # the mismatched nodes sit with community 0 in C, with community 1 in T
    assert np.mean(c[mism] == np.bincount(c[gt == 0]).argmax()) > 0.9
    assert np.mean(t[mism] == np.bincount(t[gt == 1]).argmax()) > 0.9
    z = reg_spectral(b["A"], dim=8)
    assert np.allclose(np.linalg.norm(z, axis=1), 1)
    print("views self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=["timme", *POKEC_SLICES])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.dataset:
        selftest()
    else:
        b = load_timme() if a.dataset == "timme" else load_pokec(a.dataset)
        run(b, f"{a.dataset}_views", a.runs)
