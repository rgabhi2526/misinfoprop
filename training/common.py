"""Data plumbing shared by the per-dataset loaders: id remapping, symmetrizing,
feature standardizing, CLI args. Keeps each train_<dataset>.py to just the parts
that are actually dataset-specific (which files, which columns).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce


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


def make_data(edge_index, x, num_nodes=None):
    """Symmetrize edges (embedding stage is undirected — MODELS.md §3), coalesce
    duplicates, wrap in a PyG Data. Directed edges stay in the raw files for
    downstream diffusion/centrality; we only undirect the copy used here."""
    n = num_nodes if num_nodes is not None else int(edge_index.max()) + 1
    edge_index = to_undirected(edge_index, num_nodes=n)
    edge_index = coalesce(edge_index, num_nodes=n)
    x = standardize(x)
    return Data(x=x, edge_index=edge_index, num_nodes=n)


def standardize(x):
    x = x.float()
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True)
    return (x - mu) / (sd + 1e-8)


def degree_features(edge_index, n, dim=64):
    """Fallback node features when a dataset ships none: log-degree bucketed into
    a small random projection. ponytail: cheap, deterministic, good enough as a
    structural prior; swap for real ideology features when available."""
    deg = torch.bincount(edge_index.reshape(-1), minlength=n).float()
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(1, dim, generator=g)
    return torch.log1p(deg).unsqueeze(1) * proj


def base_args(dataset, default_dir, clusters=2, epochs=500):
    p = argparse.ArgumentParser(description=f"Train BGRL/DGI/MVGRL on {dataset}")
    p.add_argument("--data-dir", default=default_dir)
    p.add_argument("--out-dir", default=f"./out/{dataset}")
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
