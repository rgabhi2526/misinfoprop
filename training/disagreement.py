"""Clustering views + disagreement regions (TASKS T4, T5; AAS paper §3.1-A).

View A = HDBSCAN (density; `-1` = noise), view B = FINCH (first-neighbour,
parameter-free), both on L2-normalized, PCA-reduced embeddings. Clusters from the
two views that partially overlap (0 < IoU < 1) are linked; connected groups of
linked clusters are "regions of uncertainty". A node is uncertain if it sits in a
region or is HDBSCAN noise (S-06).

    python disagreement.py timme                       # bgrl embedding by default
    python disagreement.py pokec --region "zilinsky kraj, kysucke nove mesto" --tag pokec_knm

Writes out/<tag>/<tag>_<model>_regions.npz (+ .json summary).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def reduce(emb, dim=32, seed=0):
    """Cosine geometry + PCA: density is meaningless in 256-d (S-03)."""
    z = normalize(np.asarray(emb, dtype=np.float64))
    if z.shape[1] > dim:
        z = PCA(dim, random_state=seed).fit_transform(z)
    return normalize(z)


def hdbscan(z, min_cluster_size=None):
    # ponytail: size knob scales with n; tune if noise share looks absurd
    mcs = min_cluster_size or max(10, len(z) // 500)
    return HDBSCAN(min_cluster_size=mcs).fit_predict(z)


def finch(z):
    """FINCH (Sarfraz et al., CVPR'19): link every point to its first nearest
    neighbour; clusters = connected components; repeat on cluster means.
    Returns the partitions, finest first."""
    parts, lab, x = [], np.arange(len(z)), z
    while True:
        n = len(x)
        nn = NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x, return_distance=False)[:, 1]
        k, comp = connected_components(coo_matrix((np.ones(n), (np.arange(n), nn)), shape=(n, n)),
                                       directed=False)
        if k == n:                          # nothing merged: hierarchy done
            break
        lab = comp[lab]
        parts.append(lab.copy())
        if k <= 2:
            break
        x = np.zeros((k, z.shape[1]))
        np.add.at(x, lab, z)                # next level runs on cluster means
        x = normalize(x)
    return parts


def regions(a, b):
    """a: view-A labels (-1 = noise), b: view-B labels. Returns region id per node
    (-1 = both views agree on it, or it's noise). IoU is computed over non-noise
    nodes so that noise alone never makes two clusters 'disagree'."""
    m = a >= 0
    ka, kb = a.max() + 1, b.max() + 1
    cont = coo_matrix((np.ones(m.sum()), (a[m], b[m])), shape=(ka, kb)).tocsr().tocoo()
    r, c, inter = cont.row, cont.col, cont.data
    sa, sb = np.bincount(a[m], minlength=ka), np.bincount(b[m], minlength=kb)
    part = inter / (sa[r] + sb[c] - inter) < 1          # 0 < IoU < 1
    g = coo_matrix((np.ones(part.sum()), (r[part], ka + c[part])), shape=(ka + kb,) * 2)
    _, comp = connected_components(g, directed=False)
    linked = np.zeros(ka + kb, bool)
    linked[r[part]] = True
    linked[ka + c[part]] = True

    region = np.full(len(a), -1)
    idx = np.flatnonzero(m)
    hit = linked[a[m]]                                   # node's A-cluster is in a region
    region[idx[hit]] = comp[a[m]][hit]
    ids = np.unique(region[region >= 0])
    region[region >= 0] = np.searchsorted(ids, region[region >= 0])
    return region


def run(emb, dim=32, min_cluster_size=None, seed=0):
    z = reduce(emb, dim, seed)
    a = hdbscan(z, min_cluster_size)
    ka = int(a.max() + 1)
    levels = finch(z)
    # FINCH level whose cluster count is closest to HDBSCAN's
    li = int(np.argmin([abs(int(p.max() + 1) - ka) for p in levels]))
    b = levels[li]
    reg = regions(a, b)
    noise = a < 0
    sizes = np.bincount(reg[reg >= 0]) if (reg >= 0).any() else np.array([0])
    summary = {
        "n": len(a), "hdbscan_k": ka, "hdbscan_noise": float(noise.mean()),
        "finch_levels_k": [int(p.max() + 1) for p in levels], "finch_level": li,
        "finch_k": int(b.max() + 1),
        "n_regions": int((reg >= 0).any() and reg.max() + 1),
        "in_region": float((reg >= 0).mean()),
        "largest_region_share": float(sizes.max() / len(a)),
        "uncertain": float(((reg >= 0) | noise).mean()),
    }
    return {"hdbscan": a, "finch": b, "region": reg, "noise": noise}, summary


def enrichment(labels, b, region, noise, seed=0):
    """T6.1 (S-08): is a node 'misgrouped' (its label != its FINCH cluster's
    majority label) more often inside uncertain areas than outside? Returns rates
    plus a random-set baseline of the same size."""
    m = labels >= 0
    y, cl = labels[m], b[m]
    maj = np.zeros(cl.max() + 1, dtype=int)
    for c in np.unique(cl):
        maj[c] = np.bincount(y[cl == c]).argmax()
    wrong = maj[cl] != y
    unc = ((region >= 0) | noise)[m]
    rng = np.random.default_rng(seed)
    rand = np.zeros_like(unc)
    rand[rng.choice(len(unc), unc.sum(), replace=False)] = True
    rate = lambda s: float(wrong[s].mean()) if s.any() else None
    return {"labeled": int(m.sum()), "uncertain_labeled": int(unc.sum()),
            "misgrouped_in_uncertain": rate(unc), "misgrouped_elsewhere": rate(~unc),
            "misgrouped_random_same_size": rate(rand), "misgrouped_overall": float(wrong.mean())}


if __name__ == "__main__":
    import argparse
    import importlib
    import json
    import os

    import torch

    HERE = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--model", default="bgrl")
    p.add_argument("--region", default=None, help="pokec only: region slice")
    p.add_argument("--tag", default=None, help="out/<tag>/ (default: dataset)")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--min-cluster-size", type=int, default=None)
    a = p.parse_args()
    tag = a.tag or a.dataset

    loader = importlib.import_module(f"train_{a.dataset}")
    data, labels = (loader.load_data(region=a.region) if a.region else loader.load_data())
    emb = torch.load(os.path.join(HERE, "out", tag, f"{tag}_{a.model}_emb.pt"),
                     map_location="cpu", weights_only=True).numpy()
    assert len(emb) == data.num_nodes, "embedding predates the current loader — retrain"
    out, summary = run(emb, a.dim, a.min_cluster_size)
    if labels is not None and (labels >= 0).any():
        summary["enrichment"] = enrichment(labels, out["finch"], out["region"], out["noise"])
    base = os.path.join(HERE, "out", tag, f"{tag}_{a.model}_regions")
    np.savez_compressed(base + ".npz", orig_idx=data.orig_idx.numpy(), **out)
    json.dump(summary, open(base + ".json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
