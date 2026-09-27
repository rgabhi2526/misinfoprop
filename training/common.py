"""Data plumbing shared by the per-dataset loaders: id remapping, symmetrizing,
feature standardizing, CLI args. Keeps each train_<dataset>.py to just the parts
that are actually dataset-specific (which files, which columns).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, subgraph, to_undirected


def remap_ids(src, dst):
    """Map arbitrary node ids (str/int) in two parallel arrays to contiguous
    0..N-1 indices. Returns (edge_index[2,E], id2idx dict, idx2id list)."""
    src = np.asarray(src)
    dst = np.asarray(dst)
    uniq, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    e = inv.reshape(2, -1)
    edge_index = torch.from_numpy(e).long()
    id2idx = {v: i for i, v in enumerate(uniq)}
    return edge_index, id2idx, uniq


def make_data(edge_index, x, num_nodes=None, lcc=True):
    """Symmetrize edges (embedding stage is undirected — docs/MODELS.md §3), coalesce
    duplicates, drop self-loops, keep only the largest connected component, wrap
    in a PyG Data. Directed edges stay in the raw files for downstream
    diffusion/centrality; we only undirect the copy used here.

    data.orig_idx maps each kept node back to its loader index — index labels
    with it (labels[data.orig_idx]). Isolated / tiny components carry no
    community or bridge signal and get embedded from features alone (40% of
    reddit was degree-0 and formed its own UMAP blob)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = num_nodes if num_nodes is not None else int(edge_index.max()) + 1
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=n)   # also coalesces
    keep = torch.arange(n)
    if lcc:
        s, d = edge_index.numpy()
        _, comp = connected_components(coo_matrix((np.ones(s.size), (s, d)), shape=(n, n)),
                                       directed=False)
        keep = torch.from_numpy(np.flatnonzero(comp == np.bincount(comp).argmax()))
        edge_index, _ = subgraph(keep, edge_index, relabel_nodes=True, num_nodes=n)
        print(f"largest connected component: {keep.numel()}/{n} nodes kept")
    data = Data(x=standardize(x[keep]), edge_index=edge_index, num_nodes=keep.numel())
    data.orig_idx = keep
    return data


ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"   # one encoder for all datasets (English + Slovak)


def encode_texts(texts, cache=None):
    """Sentence-embed per-node texts -> float32 tensor [N, 384 + 1].

    Blank text -> zero vector; the last column is the `has_metadata` flag, so the
    model can tell "no metadata" from "metadata near the origin" (R4: nodes
    without metadata are kept and embedded mostly from topology).
    ponytail: the encoder truncates at 128 tokens; put the most informative text first.
    `cache` (.npy path): reuse a previous encoding — delete it to rebuild."""
    import os

    if cache and os.path.exists(cache):
        return torch.from_numpy(np.load(cache))
    from sentence_transformers import SentenceTransformer

    has = np.array([bool(t and t.strip()) for t in texts])
    model = SentenceTransformer(ENCODER, device="cuda" if torch.cuda.is_available() else "cpu")
    x = np.zeros((len(texts), model.get_sentence_embedding_dimension() + 1), np.float32)
    x[has, :-1] = model.encode([t for t, h in zip(texts, has) if h], batch_size=256,
                               show_progress_bar=True, convert_to_numpy=True,
                               normalize_embeddings=True)
    x[:, -1] = has
    if cache:
        np.save(cache, x)
    return torch.from_numpy(x)


def standardize(x):
    x = x.float()
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True)
    return (x - mu) / (sd + 1e-8)


def base_args(dataset, default_dir, clusters=2, epochs=500):
    p = argparse.ArgumentParser(description=f"Train BGRL/DGI/MVGRL on {dataset}")
    p.add_argument("--data-dir", default=default_dir)
    import os   # anchor to training/, where eval/plot/disagreement read
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", dataset))
    p.add_argument("--models", nargs="+", default=["bgrl", "dgi", "mvgrl"])
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clusters", type=int, default=clusters)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="cap #rows/edges read (quick sanity runs)")
    return p


def cfg_from_args(a):
    from gclib import Cfg
    kw = dict(dim=a.dim, hidden=a.dim, layers=a.layers, epochs=a.epochs,
              lr=a.lr, clusters=a.clusters)
    if a.device:
        kw["device"] = a.device
    return Cfg(**kw)
