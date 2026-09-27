"""Embedding evaluation report. numpy/sklearn/networkx only (no PyG) so it runs
on saved embeddings anywhere.

Sections:
  health      per-dim std, effective rank, norm-degree corr, cluster sizes
  labeled     linear probe acc/F1, NMI, ARI, kNN label agreement, silhouette(true)
  unlabeled   modularity, conductance, silhouette/DB (diagnostic only),
              edge-reconstruction AUC, seed stability (ARI across seeds)
  task        ARI vs graph partition, boundary-node vs betweenness overlap

Caveat baked into the output: TIMME is 2 classes, 1,206 labeled of 20,811 nodes,
so label metrics are computed on a small labeled subset and cannot separate
models on their own.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score, f1_score,
                             normalized_mutual_info_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SAMPLE = 10_000     # cap for O(n^2)-ish metrics (silhouette, kNN, eff. rank)
# ponytail: networkx is pure python; graph-algorithm metrics are skipped above
# this size. Swap in igraph/cuGraph if pokec/gab need them.
GRAPH_MAX = 200_000


def cluster(emb, k, seed=0):
    """k-means on L2-normalized embeddings (cosine geometry; raw vectors let a few
    high-norm outliers grab whole clusters). Returns (labels, unit centroids)."""
    z = normalize(np.asarray(emb, dtype=np.float64))
    if len(z) > 50_000:
        km = MiniBatchKMeans(k, n_init=3, batch_size=4096, random_state=seed)
    else:
        km = KMeans(k, n_init=10, random_state=seed)
    pred = km.fit_predict(z)
    return pred, normalize(km.cluster_centers_)


def modularity(edge_index, comm):
    """Newman modularity on a symmetrized edge list (both directions stored,
    so src.size == 2m)."""
    src, dst = np.asarray(edge_index)
    two_m = src.size
    if two_m == 0:
        return 0.0
    deg = np.bincount(src, minlength=comm.shape[0]).astype(np.float64)
    intra = (comm[src] == comm[dst]).sum()   # 2 * intra-community edges
    dsum = np.bincount(comm, weights=deg, minlength=comm.max() + 1)
    return float(intra / two_m - ((dsum / two_m) ** 2).sum())


def conductance(edge_index, comm):
    """Per-cluster cut / min(vol, 2m - vol); returns (mean, max). Lower = better."""
    src, dst = np.asarray(edge_index)
    k = comm.max() + 1
    vol = np.bincount(comm[src], minlength=k).astype(np.float64)
    cut = np.bincount(comm[src][comm[src] != comm[dst]], minlength=k)
    den = np.minimum(vol, src.size - vol)
    c = np.divide(cut, den, out=np.ones(k), where=den > 0)
    return float(c.mean()), float(c.max())


def _sample(n, cap, rng):
    return np.arange(n) if n <= cap else rng.choice(n, cap, replace=False)


def _health(emb, z, deg, pred, rng):
    s = np.linalg.svd(emb[_sample(len(emb), SAMPLE, rng)] - emb.mean(0),
                      compute_uv=False)
    p = s / s.sum()
    std = emb.std(0)
    return {
        "dim_std_min": float(std.min()), "dim_std_mean": float(std.mean()),
        "effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
        "dims": emb.shape[1],
        "corr_norm_logdeg": float(np.corrcoef(np.linalg.norm(emb, axis=1),
                                              np.log1p(deg))[0, 1]),
        "cluster_sizes": np.bincount(pred).tolist(),
    }


def _labeled(z, y, pred, seed, rng):
    m = y >= 0
    zl, yl = z[m], y[m]
    idx = _sample(len(yl), 20_000, rng)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    yp = cross_val_predict(LogisticRegression(max_iter=2000), zl[idx], yl[idx], cv=cv)
    kn = 10
    q = _sample(len(yl), SAMPLE, rng)
    nbr = NearestNeighbors(n_neighbors=kn + 1, metric="cosine").fit(zl)
    _, nb = nbr.kneighbors(zl[q])
    sil_idx = _sample(len(yl), SAMPLE, rng)
    out = {
        "n_labeled": int(m.sum()), "n_classes": int(len(np.unique(yl))),
        "probe_acc": float((yp == yl[idx]).mean()),
        "probe_f1_macro": float(f1_score(yl[idx], yp, average="macro")),
        "nmi": float(normalized_mutual_info_score(yl, pred[m])),
        "ari": float(adjusted_rand_score(yl, pred[m])),
        "knn10_label_agreement": float((yl[nb[:, 1:]] == yl[q, None]).mean()),
        "silhouette_true_labels": float(silhouette_score(zl[sil_idx], yl[sil_idx],
                                                         metric="cosine")),
    }
    if out["n_classes"] <= 2 or out["probe_acc"] > 0.95:
        out["warning"] = ("near-ceiling / few classes: label metrics cannot "
                          "separate models on their own")
    return out


def _edge_auc(z, edge_index, rng, n_pairs=100_000):
    src, dst = np.asarray(edge_index)
    keep = src < dst
    src, dst = src[keep], dst[keep]
    i = _sample(src.size, n_pairs, rng)
    # ponytail: random pairs used as negatives without a non-edge check; the
    # graphs are sparse enough that collisions are negligible.
    a, b = rng.integers(0, len(z), (2, i.size))
    pos = (z[src[i]] * z[dst[i]]).sum(1)
    neg = (z[a] * z[b]).sum(1)
    return float(roc_auc_score(np.r_[np.ones(i.size), np.zeros(i.size)], np.r_[pos, neg]))


def graph_stats(edge_index, n, seed=0):
    """Embedding-independent graph side of the task metrics: Louvain partition
    + sampled betweenness. Compute once per dataset and pass to report()."""
    import networkx as nx

    src, dst = np.asarray(edge_index)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(zip(src[src < dst].tolist(), dst[src < dst].tolist()))
    gp = np.zeros(n, dtype=np.int64)
    for c, s in enumerate(nx.community.louvain_communities(g, seed=seed)):
        gp[list(s)] = c
    # ponytail: sampled betweenness (500 pivots); exact is O(VE)
    btw = nx.betweenness_centrality(g, k=min(n, 500), seed=seed)
    return {"partition": gp, "betweenness": np.array([btw[i] for i in range(n)])}


def _task(z, edge_index, pred, centers, y, graph):
    gp, btw = graph["partition"], graph["betweenness"]
    out = {
        "graph_partition_k": int(gp.max() + 1),
        "graph_partition_modularity": modularity(edge_index, gp),
        "ari_vs_graph_partition": float(adjusted_rand_score(gp, pred)),
    }
    if y is not None and (y >= 0).any():
        m = y >= 0
        out["graph_partition_nmi"] = float(normalized_mutual_info_score(y[m], gp[m]))

    # boundary = small gap between nearest and 2nd-nearest centroid (cosine)
    if len(centers) >= 2:
        sims = np.sort(z @ centers.T, axis=1)
        margin = sims[:, -1] - sims[:, -2]
        top = max(10, len(z) // 100)
        a = set(np.argsort(margin)[:top])
        b = set(np.argsort(-btw)[:top])
        out.update({
            "spearman_boundary_vs_betweenness": float(spearmanr(-margin, btw)[0]),
            f"jaccard_top{top}_boundary_vs_betweenness": len(a & b) / len(a | b),
        })
    return out


def report(emb, edge_index, k, labels=None, seed=0, n_seeds=5, graph=None):
    """Full evaluation of one embedding. edge_index: (2, E) symmetrized, numpy or
    torch. labels: -1 = unlabeled, or None. graph: graph_stats() output, reused
    across models; computed here if None."""
    emb = np.asarray(emb, dtype=np.float64)
    edge_index = np.asarray(edge_index)
    y = None if labels is None else np.asarray(labels)
    rng = np.random.default_rng(seed)
    n = len(emb)
    z = normalize(emb)
    pred, centers = cluster(emb, k, seed)
    deg = np.bincount(edge_index[0], minlength=n)

    s = _sample(n, SAMPLE, rng)
    seeds = [cluster(emb, k, seed + i)[0] for i in range(n_seeds)]
    mc, xc = conductance(edge_index, pred)
    rep = {
        "health": _health(emb, z, deg, pred, rng),
        "unlabeled": {
            "k": k,
            "modularity": modularity(edge_index, pred),
            "conductance_mean": mc, "conductance_max": xc,
            "edge_reconstruction_auc": _edge_auc(z, edge_index, rng),
            "seed_stability_ari": float(np.mean([adjusted_rand_score(a, b) for a, b
                                                 in itertools.combinations(seeds, 2)])),
            # diagnostic only: rewards whatever shape k-means already assumes
            "silhouette_diag": (float(silhouette_score(z[s], pred[s], metric="cosine"))
                                if len(np.unique(pred[s])) > 1 else None),
            "davies_bouldin_diag": (float(davies_bouldin_score(z[s], pred[s]))
                                    if len(np.unique(pred[s])) > 1 else None),
        },
    }
    if y is not None and (y >= 0).any():
        rep["labeled"] = _labeled(z, y, pred, seed, rng)
    if n > GRAPH_MAX:
        rep["task"] = {"skipped": f"n={n} > {GRAPH_MAX} (networkx too slow)"}
    else:
        rep["task"] = _task(z, edge_index, pred, centers, y,
                            graph or graph_stats(edge_index, n, seed))
    return rep


if __name__ == "__main__":
    # self-check: planted 2-community graph, embedding = community one-hot + noise
    import networkx as nx

    rng = np.random.default_rng(0)
    n = 400
    y = np.repeat([0, 1], n // 2)
    p = np.where(y[:, None] == y[None, :], 0.10, 0.005)
    a = np.triu(rng.random((n, n)) < p, 1)
    s, d = np.nonzero(a | a.T)
    ei = np.stack([s, d])
    emb = np.eye(2)[y] * 3 + rng.normal(0, 0.5, (n, 2))
    emb = np.c_[emb, rng.normal(0, 0.1, (n, 14))]

    g = nx.Graph(list(zip(s[s < d].tolist(), d[s < d].tolist())))
    assert abs(modularity(ei, y) - nx.community.modularity(
        g, [set(np.flatnonzero(y == c)) for c in (0, 1)])) < 1e-9
    r = report(emb, ei, 2, labels=y)
    assert r["labeled"]["probe_acc"] > 0.95, r
    assert r["labeled"]["nmi"] > 0.9 and r["labeled"]["ari"] > 0.9, r
    assert r["unlabeled"]["modularity"] > 0.35, r
    # 2 equal communities: half of random pairs are same-community -> AUC ceiling ~0.75
    assert r["unlabeled"]["edge_reconstruction_auc"] > 0.65, r
    assert r["unlabeled"]["seed_stability_ari"] > 0.99, r
    assert r["unlabeled"]["conductance_mean"] < 0.15, r
    assert "warning" in r["labeled"]
    print("metrics self-check ok")
