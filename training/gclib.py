"""Shared graph-contrastive library: GCN encoder + BGRL / DGI / MVGRL + train/eval.

Same encoder backbone and training harness feed all three models so the only
thing that differs across a run is the objective family (see MODELS.md). Each
per-dataset script builds a PyG `Data` (symmetrized edges + ideology features)
and hands it here.

ponytail: DGI reuses torch_geometric.nn.DeepGraphInfomax; only BGRL/MVGRL are
hand-rolled because PyG core ships no equivalent. No custom trainer class,
config dataclass, or plugin registry — a dict of hyperparams and three train_*
functions is the whole thing.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch_geometric.utils import dropout_edge


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    dim: int = 256            # embedding dimension (fixed across all 3 models)
    hidden: int = 256
    layers: int = 2           # encoder depth (fixed across all 3 models)
    epochs: int = 500
    lr: float = 1e-3
    wd: float = 1e-5
    # augmentation strengths (BGRL / MVGRL)
    drop_edge: float = 0.2
    drop_feat: float = 0.2
    ema: float = 0.99         # BGRL target-encoder momentum
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clusters: int = 2         # kmeans k for downstream eval; override per dataset
    seed: int = 0
    log_every: int = 50


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        d = in_dim
        for i in range(layers):
            o = out_dim if i == layers - 1 else hidden
            self.convs.append(GCNConv(d, o))
            self.acts.append(nn.PReLU(o))
            d = o

    def forward(self, x, edge_index, edge_weight=None):
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x, edge_index, edge_weight))
        return x


def _drop_feat(x, p):
    if p <= 0:
        return x
    mask = torch.empty(x.size(1), device=x.device).bernoulli_(1 - p)
    return x * mask


def _augment(x, edge_index, cfg):
    ei, _ = dropout_edge(edge_index, p=cfg.drop_edge)
    return _drop_feat(x, cfg.drop_feat), ei


# --------------------------------------------------------------------------- #
# BGRL — bootstrapped, no negatives (anchor baseline)
# --------------------------------------------------------------------------- #
class BGRL(nn.Module):
    def __init__(self, in_dim, cfg: Cfg):
        super().__init__()
        self.online = GCNEncoder(in_dim, cfg.hidden, cfg.dim, cfg.layers)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim), nn.PReLU(cfg.dim), nn.Linear(cfg.dim, cfg.dim)
        )
        self.ema = cfg.ema

    @torch.no_grad()
    def update_target(self):
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.data = self.ema * pt.data + (1 - self.ema) * po.data

    def embed(self, x, edge_index):
        return self.online(x, edge_index)


def train_bgrl(data, cfg: Cfg):
    m = BGRL(data.num_features, cfg).to(cfg.device)
    opt = torch.optim.AdamW(
        list(m.online.parameters()) + list(m.predictor.parameters()),
        lr=cfg.lr, weight_decay=cfg.wd,
    )
    x, ei = data.x.to(cfg.device), data.edge_index.to(cfg.device)
    m.train()
    for ep in range(1, cfg.epochs + 1):
        opt.zero_grad()
        x1, e1 = _augment(x, ei, cfg)
        x2, e2 = _augment(x, ei, cfg)
        q1 = m.predictor(m.online(x1, e1))
        q2 = m.predictor(m.online(x2, e2))
        with torch.no_grad():
            t1 = m.target(x1, e1)
            t2 = m.target(x2, e2)
        # symmetric cosine loss (2 - 2cos)
        loss = 2 - (F.cosine_similarity(q1, t2.detach(), -1).mean()
                    + F.cosine_similarity(q2, t1.detach(), -1).mean())
        loss.backward()
        opt.step()
        m.update_target()
        _log("BGRL", ep, loss, cfg)
    m.eval()
    with torch.no_grad():
        return m.embed(x, ei).cpu()


# --------------------------------------------------------------------------- #
# DGI — InfoMax (contrast point vs BGRL)
# --------------------------------------------------------------------------- #
def _corrupt(x, edge_index):
    return x[torch.randperm(x.size(0), device=x.device)], edge_index


def train_dgi(data, cfg: Cfg):
    enc = GCNEncoder(data.num_features, cfg.hidden, cfg.dim, cfg.layers)
    m = DeepGraphInfomax(
        hidden_channels=cfg.dim,
        encoder=enc,
        summary=lambda z, *a, **k: torch.sigmoid(z.mean(0)),
        corruption=_corrupt,
    ).to(cfg.device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    x, ei = data.x.to(cfg.device), data.edge_index.to(cfg.device)
    m.train()
    for ep in range(1, cfg.epochs + 1):
        opt.zero_grad()
        pos, neg, summ = m(x, ei)
        loss = m.loss(pos, neg, summ)
        loss.backward()
        opt.step()
        _log("DGI", ep, loss, cfg)
    m.eval()
    with torch.no_grad():
        z, _, _ = m(x, ei)
        return z.cpu()


# --------------------------------------------------------------------------- #
# MVGRL — multi-view contrastive (adjacency view + diffusion view)
# --------------------------------------------------------------------------- #
class MVGRL(nn.Module):
    def __init__(self, in_dim, cfg: Cfg):
        super().__init__()
        self.enc_a = GCNEncoder(in_dim, cfg.hidden, cfg.dim, cfg.layers)
        self.enc_d = GCNEncoder(in_dim, cfg.hidden, cfg.dim, cfg.layers)
        self.disc = nn.Bilinear(cfg.dim, cfg.dim, 1)

    @staticmethod
    def _readout(h):
        return torch.sigmoid(h.mean(0))

    def disc_scores(self, h, s):
        # score every node against a graph summary s
        s = s.expand_as(h)
        return self.disc(h, s).squeeze(-1)

    def forward(self, x, ei_a, ei_d, ew_d):
        ha = self.enc_a(x, ei_a)
        hd = self.enc_d(x, ei_d, ew_d)
        xc = x[torch.randperm(x.size(0), device=x.device)]
        ha_c = self.enc_a(xc, ei_a)
        hd_c = self.enc_d(xc, ei_d, ew_d)
        sa, sd = self._readout(ha), self._readout(hd)
        # cross-view: summary of one view discriminates nodes of the other
        pos = torch.cat([self.disc_scores(ha, sd), self.disc_scores(hd, sa)])
        neg = torch.cat([self.disc_scores(ha_c, sd), self.disc_scores(hd_c, sa)])
        return pos, neg

    def embed(self, x, ei_a, ei_d, ew_d):
        return self.enc_a(x, ei_a) + self.enc_d(x, ei_d, ew_d)


def train_mvgrl(data, cfg: Cfg):
    assert hasattr(data, "diff_edge_index"), \
        "MVGRL needs a diffusion view — call add_diffusion(data) first"
    m = MVGRL(data.num_features, cfg).to(cfg.device)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    x = data.x.to(cfg.device)
    ei_a = data.edge_index.to(cfg.device)
    ei_d = data.diff_edge_index.to(cfg.device)
    ew_d = data.diff_edge_weight.to(cfg.device)
    bce = nn.BCEWithLogitsLoss()
    m.train()
    for ep in range(1, cfg.epochs + 1):
        opt.zero_grad()
        pos, neg = m(x, ei_a, ei_d, ew_d)
        logits = torch.cat([pos, neg])
        lbl = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = bce(logits, lbl)
        loss.backward()
        opt.step()
        _log("MVGRL", ep, loss, cfg)
    m.eval()
    with torch.no_grad():
        return m.embed(x, ei_a, ei_d, ew_d).cpu()


def add_diffusion(data, alpha=0.15, eps=1e-4):
    """Attach a sparse PPR diffusion view (MODELS.md: use GDC, never dense N^2).

    Cache the returned edges to disk per dataset — this is CPU-bound preprocessing.
    """
    from torch_geometric.transforms import GDC
    gdc = GDC(
        self_loop_weight=1,
        normalization_in="sym",
        normalization_out="col",
        diffusion_kwargs=dict(method="ppr", alpha=alpha, eps=eps),
        sparsification_kwargs=dict(method="threshold", eps=eps),
        exact=False,  # sparse/approximate — required at 1.9M-node scale
    )
    d = gdc(data.clone())
    data.diff_edge_index = d.edge_index
    data.diff_edge_weight = d.edge_weight if d.edge_weight is not None \
        else torch.ones(d.edge_index.size(1))
    return data


# --------------------------------------------------------------------------- #
# eval — cluster embeddings, score against graph (modularity) + labels (NMI)
# --------------------------------------------------------------------------- #
def evaluate(emb, edge_index, n_clusters, labels=None, seed=0):
    import numpy as np
    from sklearn.cluster import KMeans

    emb = emb.numpy()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = km.fit_predict(emb)

    out = {"modularity": _modularity(edge_index, pred)}
    if labels is not None:
        from sklearn.metrics import normalized_mutual_info_score
        labels = np.asarray(labels)
        keep = labels >= 0            # -1 marks unlabeled nodes
        if keep.any():
            out["nmi"] = float(
                normalized_mutual_info_score(labels[keep], pred[keep])
            )
    return out


def _modularity(edge_index, comm):
    """Newman modularity of a partition on an undirected edge list."""
    import numpy as np

    src, dst = edge_index.numpy()
    m = src.size
    if m == 0:
        return 0.0
    n = comm.shape[0]
    deg = np.bincount(np.concatenate([src, dst]), minlength=n).astype(np.float64)
    same = comm[src] == comm[dst]
    intra = same.sum()                       # 2*edges within communities (both dirs)
    # sum of (deg_c)^2 over communities
    k = comm.max() + 1
    dsum = np.bincount(comm, weights=deg, minlength=k)
    return float(intra / (2 * m) - ((dsum / (2 * m)) ** 2).sum())


# --------------------------------------------------------------------------- #
# dispatch + helpers
# --------------------------------------------------------------------------- #
TRAINERS = {"bgrl": train_bgrl, "dgi": train_dgi, "mvgrl": train_mvgrl}


def run_all(data, cfg: Cfg, labels=None, models=("bgrl", "dgi", "mvgrl")):
    """Train each model on the same data; return {name: (emb, metrics)}."""
    torch.manual_seed(cfg.seed)
    results = {}
    for name in models:
        if name == "mvgrl" and not hasattr(data, "diff_edge_index"):
            print("[mvgrl] computing diffusion view (one-time, CPU)…")
            add_diffusion(data)
        t0 = time.time()
        emb = TRAINERS[name](data, cfg)
        metrics = evaluate(emb, data.edge_index, cfg.clusters, labels, cfg.seed)
        metrics["seconds"] = round(time.time() - t0, 1)
        print(f"[{name}] {metrics}")
        results[name] = (emb, metrics)
    return results


def _log(tag, ep, loss, cfg):
    if ep == 1 or ep % cfg.log_every == 0 or ep == cfg.epochs:
        print(f"[{tag}] epoch {ep:4d}/{cfg.epochs}  loss {loss.item():.4f}")


def save(results, out_dir, dataset):
    import json
    import os
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for name, (emb, metrics) in results.items():
        torch.save(emb, os.path.join(out_dir, f"{dataset}_{name}_emb.pt"))
        summary[name] = metrics
    with open(os.path.join(out_dir, f"{dataset}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved embeddings + metrics to {out_dir}")
    return summary
