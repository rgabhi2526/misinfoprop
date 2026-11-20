"""Regime map: WHEN does mismatch-aware targeting beat the best structural
targeting? Controlled synthetic networks (benchmark.planted) with
content-aware spread, sweeping the knobs real data cannot turn:

  mu        community mixing (0.1 strong .. 0.5 weak communities)
  beta      how much sharing depends on agreement: a node accepts a message from
            community A with weight exp(beta * [its TRUE content is A]); beta=0 is
            content-blind spread
  plant     share of mismatched nodes (content of another community)
  place     'random' or 'hub' (mismatched nodes drawn proportional to degree)

Scores are computed from OBSERVED noisy text (sigma=1, with nuisance) through
the real pipeline (views + ambiguity); spread uses the TRUE content. Outcome per
setting: paired gain = share of cross-community spread removed by each
content-aware strategy minus the best of PageRank / betweenness, over the same
simulated cascades (common random numbers), with a 95% CI.

    python regime.py              # full sweep -> out/regime.json
    python regime.py --quick
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

import ambiguity
import benchmark
import blocking
import views
from metrics import log

CONTENT = ("misfit", "misfit_x_pr", "gateway", "misfit_rerank")
STRUCT = ("pagerank", "betweenness")


def planted_place(n, mu, plant, place, seed, k=8):
    """benchmark.planted, but with control over WHO is mismatched."""
    ei, text, comm, truth = benchmark.planted(n=n, k=k, mu=mu, sigma=1.0, plant=0.0,
                                              seed=seed, nuisance=1.5)
    rng = np.random.default_rng(seed + 1000)
    deg = np.bincount(ei[0], minlength=n).astype(float)
    pr = deg / deg.sum() if place == "hub" else None
    mism = rng.choice(n, int(plant * n), replace=False, p=pr)
    tc = comm.copy()
    tc[mism] = (comm[mism] + rng.integers(1, k, len(mism))) % k
    # rebuild observed text from the true content tc (same generator settings)
    cent = np.random.default_rng(seed).normal(size=(k, 64))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    noise = rng.normal(size=(n, 64)) / 8 * 4
    sig = cent[tc] + noise
    sig /= np.linalg.norm(sig, axis=1, keepdims=True)
    text = np.hstack([sig, text[:, 64:] * np.linalg.norm(text[:, :64], axis=1, keepdims=True)])
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    return ei, text, comm, tc


def one(mu, beta, plant, place, seed, n=4000, sims=60, model="ic:2"):
    ei, text, comm, tc = planted_place(n, mu, plant, place, seed)
    b = views._bundle(ei, text, np.ones(n, bool), comm)
    s = ambiguity.compute(b, views.leiden_runs(b["g"], 10, seed=seed))
    kbud = n // 100
    ranked = blocking.strategies(b, s, kbud, seed=seed)
    pr = s["pagerank"]
    pool = np.argsort(-pr)[: 3 * kbud]                    # misfit re-rank inside top-3k PageRank
    ranked["misfit_rerank"] = pool[np.argsort(-s["misfit"][pool])]
    ranked = {k: ranked[k] for k in CONTENT + STRUCT + ("gateway_struct",)}
    kc = int(comm.max()) + 1
    aff = np.exp(beta * (tc[:, None] == np.arange(kc)[None, :]))
    aff /= aff.sum(1, keepdims=True)
    res = blocking.simulate(b, ranked, comm, [kbud], model, sims=sims, seed=seed,
                            affinity=aff, raw=True)
    base = res["none"][0][:, 1]
    removed = {k: 1 - res[k][kbud][:, 1] / max(base.mean(), 1e-12) for k in ranked}
    best = max(STRUCT, key=lambda k: removed[k].mean())
    from sklearn.metrics import roc_auc_score
    out = {"best_struct": best, "best_struct_removed": float(removed[best].mean()),
           "detect_auc": float(roc_auc_score(tc != comm, s["misfit"]))}
    for k in CONTENT + ("gateway_struct",):
        d = removed[k] - removed[best]
        out[k] = [float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    grid = ([0.3], [0.0, 4.0], [0.05], ["hub"]) if a.quick else \
        ([0.1, 0.3, 0.5], [0.0, 1.0, 3.0, 6.0], [0.01, 0.05, 0.10], ["random", "hub"])
    rows = []
    path = os.path.join(views.OUT, "regime.json")
    for mu, beta, plant, place in itertools.product(*grid):
        reps = [one(mu, beta, plant, place, sd) for sd in range(a.seeds)]
        row = {"mu": mu, "beta": beta, "plant": plant, "place": place,
               "detect_auc": float(np.mean([r["detect_auc"] for r in reps])),
               "best_struct_removed": float(np.mean([r["best_struct_removed"] for r in reps]))}
        for k in CONTENT + ("gateway_struct",):
            row[k] = [float(np.mean([r[k][0] for r in reps])),
                      float(np.sqrt(np.sum([r[k][1] ** 2 for r in reps])) / len(reps))]
        rows.append(row)
        print(f"mu={mu} beta={beta} plant={plant} place={place} auc={row['detect_auc']:.2f} "
              f"best_struct={row['best_struct_removed']:.2f} | "
              + " ".join(f"{k}={row[k][0]:+.3f}±{row[k][1]:.3f}" for k in CONTENT + ("gateway_struct",)),
              flush=True)
        json.dump(rows, open(path, "w"), indent=1)
    log(f"regime: saved {path}")


if __name__ == "__main__":
    main()
