"""Phase 2 of the make-it-work plan: per-node structure-content disagreement.

  1. Label space = the topology view's communities t (views.py, Leiden on the graph).
  2. p_T = soft STRUCTURAL membership: label spreading of t over the social graph
     (Zhou et al. 2004; a random walk with restart, alpha = 0.5). Interior nodes
     stay crisp, boundary nodes get mixed rows.
  3. p_C = soft CONTENT membership: a cross-fitted text-only classifier predicting
     t (Newman & Clauset 2016's "what community does the metadata predict?").
     Every node is scored by a model trained on the other folds.
  4. Scores per node:
       misfit        1 - p_C[t_i]: how little the text supports the node's own
                     graph community (best detector on the synthetic benchmark)
       disagreement  Jensen-Shannon divergence of p_T and p_C, in [0, 1]
       ambiguity_T   normalized entropy of p_T (structural mixing only)

Rejected on 2026-10-03 (benchmark.py, logged in the plan doc): absorbing random
walks into 'core' nodes (the CellRank transplant). With 20 cores per community the
walk mixes before absorption: mean max p_T 0.24 for 8 communities (~uniform),
AUC 0.64 vs 0.92 for misfit. Known failure of harmonic propagation with few
anchors on large graphs (Nadler et al. 2009). Raw-text kNN clusters were also
rejected as the content view (modularity ~0.01 on TIMME and Pokec).

Baselines the scores must beat (Phase 3), all computed here:
  consensus_entropy      structure-only instability across Leiden runs (Riolo & Newman)
  participation          participation coefficient w.r.t. t
  content_outlier        Gao et al. 2010-style: text far from own community's text
                         centroid (a cheap stand-in for ONE, AAAI 2019)

    python ambiguity.py timme | pokec_knm   # needs out/<ds>_views/ from views.py
    python ambiguity.py --selftest
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

import views
from metrics import log


def onehot(p, k=None):
    k = int(p.max()) + 1 if k is None else k
    return sp.csr_matrix((np.ones(len(p)), (np.arange(len(p)), p)), shape=(len(p), k))


def keep_big(t, min_share=0.01):
    """Relabel t so communities under min_share of nodes become -1 (outside the
    label space: never a training target, no spreading seed)."""
    sizes = np.bincount(t)
    big = np.flatnonzero(sizes >= min_share * len(t))
    lut = np.full(len(sizes), -1)
    lut[big] = np.arange(len(big))
    return lut[t]


def spread(a, t, alpha=0.5, iters=40):
    """Label spreading (Zhou et al. 2004): F <- alpha * S F + (1 - alpha) * Y,
    S = D^-1/2 A D^-1/2, Y = one-hot(t) (zero rows for t < 0). Row-normalized."""
    k = int(t.max()) + 1
    y = np.zeros((len(t), k))
    ok = t >= 0
    y[np.flatnonzero(ok), t[ok]] = 1.0
    d = np.asarray(a.sum(1)).ravel()
    dm = sp.diags(1 / np.sqrt(np.maximum(d, 1)))
    s = dm @ a @ dm
    f = y.copy()
    for _ in range(iters):
        f = alpha * (s @ f) + (1 - alpha) * y
    row = f.sum(1, keepdims=True)
    return np.where(row > 1e-12, f / np.maximum(row, 1e-12), 1.0 / k)


def content_probs(text, t, folds=5, seed=0):
    """Cross-fitted text -> community classifier. Each node is scored by a model
    trained on the OTHER folds, so its own label never informs its own p_C.
    Nodes with t < 0 are never training targets but are scored (fold average)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    k = int(t.max()) + 1
    f = np.zeros((len(t), k))
    lab, unl = np.flatnonzero(t >= 0), np.flatnonzero(t < 0)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in cv.split(lab, t[lab]):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        clf.fit(text[lab[tr]], t[lab[tr]])
        f[lab[te]] = clf.predict_proba(text[lab[te]])
        if len(unl):
            f[unl] += clf.predict_proba(text[unl]) / folds
    return f


def entropy(p):
    q = np.clip(p, 1e-12, 1)
    return -(q * np.log(q)).sum(1) / np.log(p.shape[1])


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: (np.clip(a, 1e-12, 1) * np.log(np.clip(a, 1e-12, 1) / np.clip(b, 1e-12, 1))).sum(1)
    return np.clip((kl(p, m) + kl(q, m)) / 2 / np.log(2), 0, 1)


# --------------------------------------------------------------------------- #
# baselines
def consensus_entropy(runs, ref):
    """Align every run's labels to `ref` (Hungarian on overlap counts), then the
    normalized entropy of each node's label across runs (0 = same every run)."""
    k = int(ref.max()) + 1
    counts = np.zeros((len(ref), k))
    for r in runs:
        kr = int(r.max()) + 1
        ov = (onehot(r, kr).T @ onehot(ref, k)).toarray()
        rows, cols = linear_sum_assignment(-ov)
        lut = np.full(kr, -1)
        lut[rows] = cols
        lab = lut[r]
        ok = lab >= 0
        np.add.at(counts, (np.flatnonzero(ok), lab[ok]), 1)
    p = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    return entropy(p) if k > 1 else np.zeros(len(ref))


def participation(a, t):
    tt = np.where(t < 0, int(t.max()) + 1, t)
    h = (a @ onehot(tt)).toarray()
    deg = np.maximum(h.sum(1), 1)
    return 1 - ((h / deg[:, None]) ** 2).sum(1)


def content_outlier(text, t):
    tt = np.where(t < 0, int(t.max()) + 1, t)
    cent = normalize((onehot(tt).T @ text) / np.maximum(np.bincount(tt), 1)[:, None])
    return 1 - (normalize(text) * cent[tt]).sum(1)


# --------------------------------------------------------------------------- #
def compute(b, t_runs, text=None, alpha=0.5):
    """All Phase 2 scores. b: views bundle; t_runs: Leiden runs on the social graph;
    text: content features (default b['text'])."""
    text = b["text"] if text is None else text
    t_best = views.best_run(t_runs, b["g"])
    t = keep_big(t_best)
    k = int(t.max()) + 1
    log(f"ambiguity: {k} communities, {int((t < 0).sum())} nodes in tiny ones")
    p_t = spread(b["A"], t, alpha)
    p_c = content_probs(text, t)
    tt = np.where(t < 0, 0, t)
    pr = np.asarray(b["g"].pagerank(), dtype=np.float64)
    out = {
        "t": t, "p_T": p_t, "p_C": p_c,
        "misfit": np.where(t < 0, 1 - p_c.max(1), 1 - p_c[np.arange(len(t)), tt]),
        "disagreement": js(p_t, p_c),
        "ambiguity_T": entropy(p_t),
        "consensus_entropy": consensus_entropy(t_runs, t_best),
        "participation": participation(b["A"], t),
        "content_outlier": content_outlier(text, t),
        "pagerank": pr,
    }
    # content informativeness: does text predict the graph community at all?
    ok = t >= 0
    acc = float((p_c[ok].argmax(1) == t[ok]).mean())
    base = float(np.bincount(t[ok]).max() / ok.sum())
    out["content_acc"], out["content_majority"] = acc, base
    log(f"ambiguity: text predicts graph community with acc {acc:.3f} "
        f"(majority-class baseline {base:.3f})")
    return out


def selftest():
    """Planted case: 20 nodes whose text says community 0 while their edges say
    community 1 must top the misfit and disagreement rankings; the structure-only
    baseline must not see them."""
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
    b = views._bundle(np.stack([s, d]), text, np.ones(n, bool), gt)
    out = compute(b, views.leiden_runs(b["g"], 10))
    hits = lambda sc: len(set(np.argsort(-sc)[:20]) & set(mism))
    # classifier is trained on rows where 20 of ~120 community-0-text nodes are
    # labeled community 1, so it partly learns the mismatch as normal: >= 14 of 20
    assert hits(out["misfit"]) >= 14, hits(out["misfit"])
    assert hits(out["disagreement"]) >= 14, hits(out["disagreement"])
    assert hits(out["consensus_entropy"]) <= 5, hits(out["consensus_entropy"])
    assert np.allclose(out["p_T"].sum(1), 1) and np.allclose(out["p_C"].sum(1), 1)
    print("ambiguity self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", choices=["timme", *views.POKEC_SLICES])
    ap.add_argument("--content", default="auto", help="timme: minilm | glove | both")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.dataset:
        selftest()
    else:
        b = views.load_timme() if a.dataset == "timme" else views.load_pokec(a.dataset)
        text = b["text"]
        if a.dataset == "timme" and a.content in ("auto", "both"):
            text = np.hstack([b["text"], normalize(b["glove"])])
        elif a.dataset == "timme" and a.content == "glove":
            text = b["glove"]
        d = os.path.join(views.OUT, f"{a.dataset}_views")
        out = compute(b, np.load(os.path.join(d, "T_leiden_runs.npy")), text)
        name = "ambiguity.npz" if a.content == "auto" else f"ambiguity_{a.content}.npz"
        np.savez_compressed(os.path.join(d, name), **out)
        log(f"ambiguity: saved {d}/{name}")
