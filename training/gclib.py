"""Shared graph-contrastive library: GCN encoder + BGRL / DGI / MVGRL + train/eval.

Same encoder backbone and training harness feed all three models so the only
thing that differs across a run is the objective family (see docs/MODELS.md). Each
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
from torch_geometric.utils import (add_remaining_self_loops, dropout_edge,
                                   to_torch_csr_tensor)

import metrics as emb_metrics
from metrics import log


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
    # periodic eval / early stopping / collapse guard
    eval_every: int = 25      # evaluate embedding quality every N epochs
    patience: int = 5         # stop after this many eval windows with no improvement (0 = off)
    collapse_std: float = 1e-3  # warn if embedding std drops below this (BGRL can collapse)
    # checkpoints: dir (e.g. on Google Drive) for resumable training. Every eval window
    # saves model+optimizer+best embedding; each finished model saves its result, and a
    # rerun of run_all skips finished models and resumes the unfinished one. None = off.
    ckpt_dir: str | None = None


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
            # self-loops are added in _adj (sparse input would otherwise get
            # add_self_loops, which stacks +1 on the diffusion view's own loops)
            self.convs.append(GCNConv(d, o, add_self_loops=False))
            self.acts.append(nn.PReLU(o))
            d = o

    def forward(self, x, adj):
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x, adj))
        return x


def _adj(edge_index, n, edge_weight=None):
    """Edge list -> sparse CSR A^T (row = target, col = source) for GCNConv.

    Same numbers as GCNConv(edge_index, edge_weight): add_remaining_self_loops
    is exactly what gcn_norm does on the edge-list path. Sparse matmul avoids
    the E x C message tensor (TIMME: ~12 GB/layer as an edge list).
    """
    if edge_weight is None:   # explicit ones: coalesce would dedupe, not sum, duplicates
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    ei, ew = add_remaining_self_loops(edge_index, edge_weight, 1., n)
    return to_torch_csr_tensor(ei.flip(0), ew, size=(n, n))


def _drop_feat(x, p):
    if p <= 0:
        return x
    mask = torch.empty(x.size(1), device=x.device).bernoulli_(1 - p)
    return x * mask


def _augment(x, edge_index, cfg):
    # force_undirected: drop both directions of an edge together, so the
    # augmented view stays a symmetric graph like the input
    ei, _ = dropout_edge(edge_index, p=cfg.drop_edge, force_undirected=True)
    return _drop_feat(x, cfg.drop_feat), _adj(ei, x.size(0))


# --------------------------------------------------------------------------- #
# shared training harness: periodic eval + best-embedding + early stop + collapse
# --------------------------------------------------------------------------- #
def _save(obj, path):
    import os
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)     # atomic: a disconnect mid-write keeps the old file


def _fit(tag, params, step_fn, embed_fn, data, cfg, labels, post_step=None, module=None):
    """Run the epoch loop for any of the three models.

    step_fn()  -> scalar loss (one forward/backward's worth; grads already needed)
    embed_fn() -> current node embedding tensor (eval mode, no grad)
    post_step  -> optional callback after opt.step() (BGRL uses it for the EMA)

    Every cfg.eval_every epochs we cluster the current embedding and score it by
    modularity (unsupervised), keep the best-scoring embedding, and
    stop early after cfg.patience windows without improvement. Also flags a
    collapsed embedding (near-zero std) — the classic BGRL failure mode.

    Returns (best_embedding_cpu, best_metrics).
    """
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    log(f"[{tag}] training: up to {cfg.epochs} epochs, eval every {cfg.eval_every}, "
        f"patience {cfg.patience}, device {cfg.device}")
    t0 = time.time()
    best = {"score": float("-inf"), "emb": None, "epoch": 0, "metrics": {}}
    wait, start = 0, 1
    ckpt = None
    if cfg.ckpt_dir and module is not None:
        import os
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        ckpt = os.path.join(cfg.ckpt_dir, f"{tag.lower()}_train.pt")
        if os.path.exists(ckpt):
            c = torch.load(ckpt, map_location=cfg.device, weights_only=False)
            module.load_state_dict(c["model"])
            opt.load_state_dict(c["opt"])
            best, wait, start = c["best"], c["wait"], c["epoch"] + 1
            log(f"[{tag}] resumed from {ckpt} at epoch {c['epoch']} "
                f"(best ep {best['epoch']}, score {best['score']:.4f})")
    for ep in range(start, cfg.epochs + 1):
        opt.zero_grad()
        loss = step_fn()
        loss.backward()
        opt.step()
        if post_step is not None:
            post_step()
        _log(tag, ep, loss, cfg, t0)

        if ep % cfg.eval_every == 0 or ep == cfg.epochs:
            emb = embed_fn().detach().cpu()
            std = float(emb.std())
            metrics = evaluate(emb, data.edge_index, cfg.clusters, labels, cfg.seed)
            metrics["emb_std"] = round(std, 4)
            # select on modularity only: selecting on NMI tunes the embedding on
            # the same labels it is evaluated against (label leakage). NMI is
            # still logged. Report modularity at best epoch as selection-biased.
            score = metrics["modularity"]
            flag = "  ⚠ possible collapse" if std < cfg.collapse_std else ""
            log(f"[{tag}] eval ep {ep:4d}: {metrics} score={score:.4f}{flag}")
            if score > best["score"]:
                best.update(score=score, emb=emb, epoch=ep, metrics=metrics)
                wait = 0
            else:
                wait += 1
            if ckpt:
                _save({"model": module.state_dict(), "opt": opt.state_dict(),
                       "best": best, "wait": wait, "epoch": ep}, ckpt)
            if cfg.patience and wait >= cfg.patience:
                log(f"[{tag}] early stop at ep {ep} "
                    f"(best ep {best['epoch']}, score {best['score']:.4f})")
                break
    best["metrics"]["best_epoch"] = best["epoch"]
    log(f"[{tag}] done: best epoch {best['epoch']}, modularity {best['score']:.4f}")
    return best["emb"], best["metrics"]


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

    def embed(self, x, adj):
        return self.online(x, adj)


def train_bgrl(data, cfg: Cfg, labels=None):
    m = BGRL(data.num_features, cfg).to(cfg.device)
    x, ei = data.x.to(cfg.device), data.edge_index.to(cfg.device)
    a = _adj(ei, x.size(0))
    m.train()

    def step():
        x1, e1 = _augment(x, ei, cfg)
        x2, e2 = _augment(x, ei, cfg)
        q1 = m.predictor(m.online(x1, e1))
        q2 = m.predictor(m.online(x2, e2))
        with torch.no_grad():
            t1 = m.target(x1, e1)
            t2 = m.target(x2, e2)
        # symmetric cosine loss (2 - 2cos)
        return 2 - (F.cosine_similarity(q1, t2.detach(), -1).mean()
                    + F.cosine_similarity(q2, t1.detach(), -1).mean())

    def embed():
        m.eval()
        with torch.no_grad():
            z = m.embed(x, a)
        m.train()
        return z

    params = list(m.online.parameters()) + list(m.predictor.parameters())
    return _fit("BGRL", params, step, embed, data, cfg, labels,
                post_step=m.update_target, module=m)


# --------------------------------------------------------------------------- #
# DGI — InfoMax (contrast point vs BGRL)
# --------------------------------------------------------------------------- #
def _corrupt(x, adj):
    return x[torch.randperm(x.size(0), device=x.device)], adj


def train_dgi(data, cfg: Cfg, labels=None):
    enc = GCNEncoder(data.num_features, cfg.hidden, cfg.dim, cfg.layers)
    m = DeepGraphInfomax(
        hidden_channels=cfg.dim,
        encoder=enc,
        summary=lambda z, *a, **k: torch.sigmoid(z.mean(0)),
        corruption=_corrupt,
    ).to(cfg.device)
    x = data.x.to(cfg.device)
    a = _adj(data.edge_index.to(cfg.device), x.size(0))
    m.train()

    def step():
        pos, neg, summ = m(x, a)
        return m.loss(pos, neg, summ)

    def embed():
        m.eval()
        with torch.no_grad():
            z, _, _ = m(x, a)
        m.train()
        return z

    return _fit("DGI", m.parameters(), step, embed, data, cfg, labels, module=m)


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

    def forward(self, x, a, d):
        ha = self.enc_a(x, a)
        hd = self.enc_d(x, d)
        xc = x[torch.randperm(x.size(0), device=x.device)]
        ha_c = self.enc_a(xc, a)
        hd_c = self.enc_d(xc, d)
        sa, sd = self._readout(ha), self._readout(hd)
        # cross-view: summary of one view discriminates nodes of the other
        pos = torch.cat([self.disc_scores(ha, sd), self.disc_scores(hd, sa)])
        neg = torch.cat([self.disc_scores(ha_c, sd), self.disc_scores(hd_c, sa)])
        return pos, neg

    def embed(self, x, a, d):
        return self.enc_a(x, a) + self.enc_d(x, d)


def train_mvgrl(data, cfg: Cfg, labels=None):
    assert hasattr(data, "diff_edge_index"), \
        "MVGRL needs a diffusion view — call add_diffusion(data) first"
    m = MVGRL(data.num_features, cfg).to(cfg.device)
    x = data.x.to(cfg.device)
    n = x.size(0)
    a = _adj(data.edge_index.to(cfg.device), n)
    # diffusion weights are asymmetric (normalization_out="col"): _adj's flip
    # keeps GCNConv's source->target reading of diff_edge_index
    d = _adj(data.diff_edge_index.to(cfg.device), n,
             data.diff_edge_weight.to(cfg.device))
    bce = nn.BCEWithLogitsLoss()
    m.train()

    def step():
        pos, neg = m(x, a, d)
        logits = torch.cat([pos, neg])
        lbl = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        return bce(logits, lbl)

    def embed():
        m.eval()
        with torch.no_grad():
            z = m.embed(x, a, d)
        m.train()
        return z

    return _fit("MVGRL", m.parameters(), step, embed, data, cfg, labels, module=m)


def add_diffusion(data, alpha=0.15, eps=1e-4, exact_max_nodes=50000):
    """Attach a sparse PPR diffusion view (docs/MODELS.md: use GDC, never dense N^2).

    The approximate (scalable) PPR path needs a working `numba` (missing, or a
    version PyG's kernel fails to compile under). On failure we fall back to the
    exact dense PPR for small graphs; for large graphs the dense matrix is
    infeasible, so we re-raise with a clear message instead.

    Cache the returned edges to disk per dataset — this is CPU-bound preprocessing.
    """
    from torch_geometric.transforms import GDC

    def gdc(exact):
        return GDC(
            self_loop_weight=1,
            normalization_in="sym",
            normalization_out="col",
            diffusion_kwargs=dict(method="ppr", alpha=alpha, eps=eps),
            sparsification_kwargs=dict(method="threshold", eps=eps),
            exact=exact,
        )

    t0 = time.time()
    log(f"[mvgrl] diffusion view: approximate PPR (alpha={alpha}, eps={eps}) on "
        f"{data.num_nodes} nodes / {data.edge_index.size(1)} edges")
    try:
        d = gdc(False)(data.clone())          # sparse/approx — required at scale
    except Exception as e:   # ImportError, or numba failing to compile get_ppr
        n = data.num_nodes
        if n > exact_max_nodes:
            raise RuntimeError(
                "GDC approximate PPR failed — needs a numba compatible with this "
                f"torch_geometric. ({n} nodes is too large for the exact fallback.)"
            ) from e
        log(f"[mvgrl] approx PPR failed ({type(e).__name__}: {e}) → exact dense "
            f"PPR fallback ({n} nodes, ~{n * n * 4 / 2**30:.1f} GB per dense copy)")
        d = gdc(True)(data.clone())

    data.diff_edge_index = d.edge_index
    data.diff_edge_weight = d.edge_weight if d.edge_weight is not None \
        else torch.ones(d.edge_index.size(1))
    log(f"[mvgrl] diffusion view ready: {d.edge_index.size(1)} edges "
        f"in {time.time() - t0:.0f}s")
    return data


# --------------------------------------------------------------------------- #
# eval — cluster embeddings, score against graph (modularity) + labels (NMI)
# --------------------------------------------------------------------------- #
def evaluate(emb, edge_index, n_clusters, labels=None, seed=0):
    """Cheap per-epoch score for best-epoch selection. Full report: emb_metrics.report."""
    import numpy as np
    from sklearn.metrics import normalized_mutual_info_score

    pred, _ = emb_metrics.cluster(emb.numpy(), n_clusters, seed)
    out = {"modularity": emb_metrics.modularity(edge_index.numpy(), pred)}
    if labels is not None:
        labels = np.asarray(labels)
        keep = labels >= 0            # -1 marks unlabeled nodes
        if keep.any():
            out["nmi"] = float(normalized_mutual_info_score(labels[keep], pred[keep]))
    return out


# --------------------------------------------------------------------------- #
# dispatch + helpers
# --------------------------------------------------------------------------- #
TRAINERS = {"bgrl": train_bgrl, "dgi": train_dgi, "mvgrl": train_mvgrl}


def run_all(data, cfg: Cfg, labels=None, models=("bgrl", "dgi", "mvgrl")):
    """Train each model on the same data; return {name: (emb, metrics)}."""
    torch.manual_seed(cfg.seed)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    log(f"run_all: models {list(models)} | {data.num_nodes} nodes, "
        f"{data.edge_index.size(1)} edges, {data.num_features} features | "
        f"device {cfg.device} (GPU: {gpu})")
    ei = data.edge_index.cpu().numpy()
    graph = (emb_metrics.graph_stats(ei, data.num_nodes, cfg.seed)
             if data.num_nodes <= emb_metrics.GRAPH_MAX else None)
    import os
    results = {}
    for name in models:
        done = cfg.ckpt_dir and os.path.join(cfg.ckpt_dir, f"{name}_result.pt")
        if done and os.path.exists(done):
            results[name] = torch.load(done, weights_only=False)
            log(f"[{name}] already finished — loaded {done}")
            continue
        if name == "mvgrl" and not hasattr(data, "diff_edge_index"):
            add_diffusion(data)
        # per model, after the diffusion step: --models dgi alone == dgi after bgrl
        torch.manual_seed(cfg.seed)
        t0 = time.time()
        emb, metrics = TRAINERS[name](data, cfg, labels)  # best emb + its metrics
        metrics["seconds"] = round(time.time() - t0, 1)
        log(f"[{name}] trained in {metrics['seconds']}s — computing full report")
        metrics["report"] = emb_metrics.report(
            emb.numpy(), ei, cfg.clusters, labels, cfg.seed, graph=graph)
        r = metrics["report"]
        log(f"[{name}] report: modularity {r['unlabeled']['modularity']:.3f}, "
            f"seed-stability ARI {r['unlabeled']['seed_stability_ari']:.3f}, "
            f"edge AUC {r['unlabeled']['edge_reconstruction_auc']:.3f}"
            + (f", probe acc {r['labeled']['probe_acc']:.3f}" if "labeled" in r else ""))
        results[name] = (emb, metrics)
        if done:
            _save(results[name], done)
    return results


def _log(tag, ep, loss, cfg, t0):
    if ep == 1 or ep % cfg.log_every == 0 or ep == cfg.epochs:
        log(f"[{tag}] epoch {ep:4d}/{cfg.epochs}  loss {loss.item():.4f}  "
            f"({(time.time() - t0) / ep:.2f}s/epoch)")


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
    log(f"saved embeddings + metrics to {out_dir}")
    return summary
