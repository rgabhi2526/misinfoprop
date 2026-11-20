"""Phase 3 synthetic benchmark: can a score find nodes whose TEXT says one
community while their EDGES say another?

Generator (planted DC-SBM, LFR-like): power-law degrees (exponent 2.5), K equal
communities, mixing mu = share of each node's edge stubs that go to other
communities (configuration-model stub matching). Text = unit(centroid[c] + sigma *
noise) in 64 dims. A share `plant` of nodes get the centroid of a DIFFERENT
community: those are the ground-truth mismatches.

Scores compared by AUC for recovering the planted nodes, overall and within
degree terciles (so 'degree in disguise' can't win):
  misfit, disagreement (ambiguity.py),
  consensus_entropy, participation, content_outlier, degree.

    python benchmark.py               # grid over mu x sigma, 3 seeds each
    python benchmark.py --quick
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

import ambiguity
import views
from metrics import log


def planted(n=4000, k=8, mu=0.3, sigma=1.0, plant=0.03, dim=64, seed=0, nuisance=0.0):
    """nuisance > 0 appends a `dim`-wide block that real text has and community-blind
    scores get distracted by (TIMME/Pokec text clusters follow topic + style, not
    community): 10 random topics unrelated to community, plus a style direction
    proportional to log-degree (politicians write differently from followers)."""
    rng = np.random.default_rng(seed)
    deg = np.minimum((rng.pareto(1.5, n) + 1) * 8, 300).astype(int)   # exponent 2.5, k_min 8
    comm = rng.permutation(np.repeat(np.arange(k), n // k + 1)[:n])
    inter = rng.binomial(deg, mu)
    intra = deg - inter
    edges = []
    for c in range(k):                                  # within-community stubs
        stubs = np.repeat(np.flatnonzero(comm == c), intra[comm == c])
        rng.shuffle(stubs)
        stubs = stubs[: len(stubs) // 2 * 2]
        edges.append(stubs.reshape(-1, 2))
    stubs = np.repeat(np.arange(n), inter)              # between-community stubs
    rng.shuffle(stubs)
    stubs = stubs[: len(stubs) // 2 * 2].reshape(-1, 2)
    edges.append(stubs[comm[stubs[:, 0]] != comm[stubs[:, 1]]])
    e = np.concatenate(edges)
    e = e[e[:, 0] != e[:, 1]]
    ei = np.concatenate([e, e[:, ::-1]]).T
    tc = comm.copy()
    mism = rng.choice(n, int(plant * n), replace=False)
    tc[mism] = (comm[mism] + rng.integers(1, k, len(mism))) % k
    cent = normalize(rng.normal(size=(k, dim)))
    text = normalize(cent[tc] + sigma * rng.normal(size=(n, dim)) / np.sqrt(dim) * 4)
    if nuisance > 0:
        topics = normalize(rng.normal(size=(10, dim)))[rng.integers(0, 10, n)]
        style = normalize(rng.normal(size=(1, dim))) * (np.log(deg) - np.log(deg).mean())[:, None]
        text = normalize(np.hstack([text, nuisance * (topics + style)]))
    truth = np.zeros(n, bool)
    truth[mism] = True
    return ei, text, comm, truth


def aucs(scores, truth, deg):
    out = {}
    terc = np.digitize(deg, np.quantile(deg, [1 / 3, 2 / 3]))
    for name, s in scores.items():
        overall = roc_auc_score(truth, s)
        within = [roc_auc_score(truth[terc == q], s[terc == q]) for q in range(3)
                  if 0 < truth[terc == q].sum() < (terc == q).sum()]
        out[name] = (float(overall), float(np.mean(within)))
    return out


def one(mu, sigma, seed, n, nuisance=0.0):
    ei, text, comm, truth = planted(n=n, mu=mu, sigma=sigma, seed=seed, nuisance=nuisance)
    b = views._bundle(ei, text, np.ones(len(comm), bool), comm)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    t_ari = views.adjusted_rand_score(comm, s["t"])
    sc = {k: s[k] for k in ("misfit", "disagreement", "consensus_entropy",
                            "participation", "content_outlier")}
    sc["degree"] = b["deg"].astype(float)
    return t_ari, aucs(sc, truth, b["deg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    mus, sigmas, seeds = ([0.3], [1.0], [0]) if a.quick else ([0.1, 0.3, 0.5], [0.5, 1.0, 2.0], [0, 1, 2])
    rows = []
    for nu in (0.0, 1.5):
        for mu in mus:
            for sg in sigmas:
                res = [one(mu, sg, sd, a.n, nu) for sd in seeds]
                agg = {k: np.mean([r[1][k] for r in res], axis=0).round(3).tolist() for k in res[0][1]}
                rows.append({"nuisance": nu, "mu": mu, "sigma": sg,
                             "T_ari": round(float(np.mean([r[0] for r in res])), 3), "auc": agg})
                line = " ".join(f"{k}={v[0]:.2f}/{v[1]:.2f}" for k, v in agg.items())
                print(f"nu={nu} mu={mu} sigma={sg} T_ari={rows[-1]['T_ari']:.2f} | "
                      f"AUC overall/within-degree: {line}", flush=True)
    out = os.path.join(views.OUT, "benchmark.json")
    json.dump(rows, open(out, "w"), indent=1)
    log(f"benchmark: saved {out}")


if __name__ == "__main__":
    main()
