"""Phase 3 real-data check on TIMME: do high-scoring nodes sit in the 'wrong'
community for their party?

Target: a labeled node is a party misfit when its party differs from the
majority party of its topology community. Only communities whose labeled
majority is >= 80% one party count (the two party cores); mixed communities
have no 'wrong' side. Each score's AUC for predicting misfits is reported
overall and within degree terciles.

Null: shuffle the text rows across nodes, recompute misfit/disagreement; any
real score must clear its own shuffled version.

    python validate.py
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

import ambiguity
import views

SCORES = ("misfit", "disagreement", "ambiguity_T", "consensus_entropy",
          "participation", "content_outlier")


def party_misfits(t, y, purity=0.8):
    """(mask of nodes counted, misfit flag) for labeled nodes in party cores."""
    use = np.zeros(len(t), bool)
    mis = np.zeros(len(t), bool)
    for c in np.unique(t[t >= 0]):
        lab = (t == c) & (y >= 0)
        if lab.sum() < 20:
            continue
        r = y[lab].mean()
        if max(r, 1 - r) >= purity:
            use |= lab
            mis[lab] = y[lab] != int(r >= 0.5)
    return use, mis


def auc_table(scores, use, mis, deg):
    terc = np.digitize(deg, np.quantile(deg[use], [1 / 3, 2 / 3]))
    out = {}
    for k, s in scores.items():
        within = [roc_auc_score(mis[use & (terc == q)], s[use & (terc == q)])
                  for q in range(3) if 0 < mis[use & (terc == q)].sum() < (use & (terc == q)).sum()]
        # one-sided Mann-Whitney p: are misfits scored higher than non-misfits?
        pv = mannwhitneyu(s[use & mis], s[use & ~mis], alternative="greater").pvalue
        out[k] = (roc_auc_score(mis[use], s[use]), float(np.mean(within)) if within else float("nan"), pv)
    return out


def main():
    b = views.load_timme()
    d = os.path.join(views.OUT, "timme_views")
    s = dict(np.load(os.path.join(d, "ambiguity.npz")))
    use, mis = party_misfits(s["t"], b["y"])
    print(f"party cores: {int(use.sum())} labeled nodes, {int(mis.sum())} party misfits")
    scores = {k: s[k] for k in SCORES}
    scores["degree"] = b["deg"].astype(float)
    scores["pagerank"] = s["pagerank"]
    rng = np.random.default_rng(0)
    text = np.hstack([b["text"], normalize(b["glove"])])
    sh = ambiguity.compute(b, np.load(os.path.join(d, "T_leiden_runs.npy")), text[rng.permutation(b["n"])])
    scores["misfit_SHUFFLED_TEXT"] = sh["misfit"]
    scores["disagreement_SHUFFLED_TEXT"] = sh["disagreement"]
    for k, (o, w, pv) in sorted(auc_table(scores, use, mis, b["deg"]).items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:28s} AUC {o:.3f}   within-degree {w:.3f}   one-sided p {pv:.3f}")


if __name__ == "__main__":
    main()
