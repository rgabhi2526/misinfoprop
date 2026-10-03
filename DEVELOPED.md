# DEVELOPED — what exists, how it works, why

The point of truth for **what has been built** and **how it behaves**. Scope and status
live in `TASKS.md` (T-ids), options and decisions in `SUGGESTIONS.md` (S-ids),
cross-session disputes in `DISCUSSION.md`, dataset facts in `data_final/DATASETS.md`.
**Update this file whenever something major changes** (rule in `CLAUDE.md`).
Last updated: 2026-10-03

---

## Repository map

```
Major project /
├── CLAUDE.md  TASKS.md  SUGGESTIONS.md  DEVELOPED.md  DISCUSSION.md   living docs
├── docs/        MODELS.md · ideas/project-idea*.md · papers/active_learning.pdf
├── scripts/     download_data.sh (fetches data_final/)
├── training/    pipeline code: common.py gclib.py metrics.py train_<ds>.py (+ .ipynb
│                wrappers) eval_embeddings.py plot_umap.py disagreement.py · out/ = generated
├── archive/     not maintained: early notebooks, VGAE outputs, reserve loaders (training/)
├── data/        Pokec, Orkut, SNAP ego nets (raw, gitignored)
└── data_final/  TIMME, Gab, Reddit, VoterFraud (raw, gitignored) + DATASETS.md
```

---

## The idea in one paragraph
Messages spread inside groups of people who can reach each other and think alike, and
jump between groups through **bridge nodes**. We (1) embed every node from what it says
(metadata) and who it's tied to (topology), (2) cluster the embedding with two
different kinds of clustering, (3) find where they **disagree** — those "regions of
uncertainty" (AAS paper, `docs/papers/active_learning.pdf`) are nodes whose group is ambiguous,
our candidate bridges, (4) later: refine with active learning and test by simulating a
fake message and blocking it at the candidates (Influence Blocking Maximization).
Success = spread blocked, not label accuracy.

## Pipeline

```
raw files ──► loader (train_<ds>.py) ──► common.make_data ──► gclib.run_all ──► out/<ds>/*_emb.pt
               text → encode_texts        symmetrize, LCC,     BGRL | DGI | MVGRL      + metrics.json
               (+ has_metadata)           standardize          best epoch by modularity
                                                                        │
      metrics.report / eval_embeddings.py / plot_umap.py ◄──────────────┘
                                                                        │
  ░░ NOT BUILT ░░  clustering views (T4) ──► disagreement regions (T5) ──► validation (T6)
                   ──► bridges + IBM simulation (T7) ──► active learning (T8, gated on T7)
```

| Stage | Status | Tasks |
|---|---|---|
| Datasets | **Final: TIMME `P_all` + Pokec** | T0 ✅, T1 |
| Loaders + features | built, smoke-tested | T0.6, T1 |
| Embedding models | built; **all saved embeddings stale → retrain** | T2 (T2.6 open) |
| Embedding evaluation | built | T3 |
| Clustering views | not built — **next** | T4 |
| Disagreement (core) | not built | T5 |
| Validation / IBM / AL | not built (IBM before AL, S-21) | T6–T8 |

---

## 1. Data layer — `training/common.py`, `training/train_<ds>.py` (T0, T1)

**How it works**
- Each loader returns `(data, labels)`: a PyG `Data` plus per-node labels (`-1` = none,
  `None` for unlabeled datasets).
- `common.encode_texts(texts, cache)` — one multilingual sentence encoder
  (`paraphrase-multilingual-MiniLM-L12-v2`, 384-d) for every dataset; blank text → zero
  vector; last column = `has_metadata` flag. Cached as `.npy` next to the data.
- `common.make_data` — removes self-loops, makes edges two-way, keeps the **largest
  connected component**, standardizes features; `data.orig_idx` maps kept nodes back to
  loader order (loaders already index labels with it).

**Why**
- Nodes without metadata are **kept** (R4): dropping them deletes edges, possibly the
  very bridges we want. The flag lets the model tell "missing" from "says little";
  their embedding comes mostly from neighbours.
- LCC only: a node with no path to the main graph can't bridge anything.
- Two-way edges only for embedding; directed raw edges stay on disk for diffusion (T8).
- One encoder everywhere → features are comparable across datasets.

**Datasets as loaded** (details: `data_final/DATASETS.md`)
| Loader | Nodes (LCC) | Features | Labels |
|---|---|---|---|
| `train_timme.py` (`P_all`) | 20,811 | bio + latest tweet → 384 + flag (98% have text) | 1,206 R/D (bonus only) |
| `train_pokec.py` | 1.63M (full) · `--region` slice | interest text → 384 + flag, gender, age + known flag, region one-hot | none |
| Gab, Reddit (reserve) | loaders archived in `archive/training/` | — | — |

## 2. Embedding models — `training/gclib.py` (T2; background `docs/MODELS.md`)

**How it works**
- Shared 2-layer GCN encoder (256-d) for all three; only the training objective differs:
  - **BGRL** — two augmented views; online net predicts the slow-moving (EMA) target
    net's embedding. No negatives.
  - **DGI** — node embeddings vs a whole-graph summary; negatives = shuffled features.
  - **MVGRL** — adjacency view vs diffusion (PPR) view, cross-view discrimination.
    Exact dense PPR fallback for small graphs when numba fails.
- Augmentation: drop 20% of edges (both directions together) and 20% of feature columns.
- GCN runs on a **sparse CSR adjacency** (`gclib._adj`, self-loops added there), not an
  edge list: PyG's edge-list path builds an E×256 message tensor (TIMME 8.2M edges ≈
  12 GB/layer). Same numbers — `test_gclib.py` checks outputs + grads match.
- Seed reset before each model, so `--models dgi` alone == DGI in a full run.
- `_fit`: every 25 epochs cluster the embedding (k-means on L2-normalized vectors),
  keep the best epoch **by modularity**, early-stop after 5 evals without improvement,
  warn if embedding std collapses.
- `run_all` then attaches the full `metrics.report` to `<ds>_metrics.json`.

**Why**
- Three different objective families, same encoder and features → differences are due
  to the objective.
- Best epoch by modularity, never NMI: selecting on labels you then report is leakage.
  Best-epoch modularity is therefore optimistic; don't headline it.
- L2-normalize before k-means: raw vectors let a few high-norm nodes grab whole
  clusters (old TIMME collapse [580, 3]).

## 3. Evaluation — `training/metrics.py`, `eval_embeddings.py`, `plot_umap.py` (T3)

`metrics.report(emb, edge_index, k, labels, graph)` → four sections:
| Section | Measures | Read it as |
|---|---|---|
| health | per-dim std, effective rank, norm↔degree corr, cluster sizes | did training collapse / get hub-dominated? |
| labeled | linear probe acc/F1, NMI, ARI, kNN label agreement, silhouette(true) | TIMME only; near-ceiling → can't rank models (auto warning) |
| unlabeled | modularity, conductance, edge-reconstruction AUC, seed-stability ARI | structure kept? clusters real or noise? |
| task | ARI vs Louvain partition, boundary-node vs betweenness overlap | graph-only agreement; bridge hint (skipped >200k nodes) |

- `eval_embeddings.py <ds>` scores saved embeddings; refuses stale ones (node-count check).
- `plot_umap.py <ds>` — UMAP grid (models × colourings). Self-checks: `python metrics.py`,
  `python test_gclib.py`.
- Outputs live in `training/out/<tag>/<tag>_<model>_emb.pt`; `--tag` (default = dataset)
  and `--region` (Pokec slice) must match across train / eval / plot / disagreement.

**Intuition to keep:** seed stability matters most for us — if a clustering changes
with the random seed, "disagreement" between two clusterings is partly noise (S-07).

## 4. Not built yet — intended behaviour
- **T4 views:** HDBSCAN (density) and FINCH (first-neighbour) on L2-normalized,
  dimension-reduced embeddings; optional Louvain on the graph (S-01…S-06). Both are
  deterministic → no seed noise.
- **T5 disagreement:** link clusters from the two views that partly overlap
  (0 < IoU < 1); connected groups of linked clusters = regions of uncertainty; per-node
  flag + region id + inlier/outlier type. Plus (S-22) a continuous **ambiguity score**:
  co-membership entropy across a partition ensemble (HDBSCAN, FINCH levels, Louvain
  seeds) — Nepusz-style bridgeness. HDBSCAN noise = outlier, never counted as a bridge.
- **T6 validation:** are region nodes more often mis-grouped (TIMME party) than
  others, and better than random / low-margin / low-degree baselines?
- **T7 IBM (go/no-go):** rumor source **unknown** (random / degree-weighted seeds);
  weighted cascade (p = 1/in-degree, assumed) + uniform-p sweep on directed edges; remove
  k nodes; compare random / degree / PageRank / betweenness / participation coefficient /
  modular centrality / graph-only bridgeness / ambiguity / ambiguity × PageRank, greedy
  ceiling on the slice. Success = spread blocked. Why this shape: S-22 (literature).
- **T8 active learning (only if T7 shows a gain):** AAS pair sampling + oracle + NP3.

## 5. Evidence so far (read with care)
All pre-2026-09-27 numbers came from broken loaders or selection — **superseded**.
- PureP smoke run (50 epochs, fixed pipeline): NMI 0.72 / 0.76 / 0.72
  (BGRL/DGI/MVGRL). Tiny, easy dataset — can't rank models.
- Earlier "MVGRL doesn't cluster" and "boundary ≠ betweenness" — **withdrawn**, rerun after T2.6.
- **Modularity (T7.0, `check_modularity.py`, Louvain vs degree-preserving rewired null):**
  | Graph | Nodes | Mean deg | Q (real) | Q (null) | Cross-community edges |
  |---|---|---|---|---|---|
  | TIMME pooled | 20,811 | 395 | 0.26 | 0.04 | 50% (null 84%) |
  | TIMME per relation | 15.8–20.8k | — | 0.26 (mention) – 0.34 (like) | 0.05–0.08 | — |
  | Pokec slice (KNM) | 5,599 | 16 | 0.49 | 0.21 | 40% (null 67%) |
  | Pokec full | 1,632,803 | 27 | **0.72** | 0.12 | 22% (null 71%) |
  Real communities in all (Q far above null). TIMME is **moderate**: half its edges cross
  communities, so bridges are not bottlenecks and hubs are likely competitive; pooling
  relations costs ~0.05–0.08 Q vs likes/retweets alone. Full Pokec is **strongly** modular
  (the regime where bridge-targeting beat hubs in the literature), but its 43 communities
  are probably mostly geographic — region is also a node feature, so check that the
  ambiguity score isn't just "has friends in two regions" (T7.1).

## 6. Environment
- **Training runs on the user's Linux CUDA box**, not the Mac (8 GB RAM). Setup:
  `training/requirements.txt` (includes a numba/GDC check — base's numba 0.60 fails on
  PyG 2.8, so MVGRL at scale needs a numba that passes it).
- Mac (dev only) Python: `/opt/anaconda3/bin/python` (conda base: torch 2.5, torch_geometric 2.8,
  networkx 3.7, sentence-transformers, umap-learn). `conda run` drops stdin.
- Base's numba can't compile PyG's approximate PPR → MVGRL uses the exact fallback on
  small graphs only.
- networkx 3.7 breaks `checkov`'s pin in base (S-19).
- GPU: rented, ₹2k budget (~23 h A100). Path has a trailing space — quote it.

## 7. Change log (major changes only)
- **2026-10-03** Colab RAM crash on full Pokec (during text encoding): `encode_texts`
  now encodes in 100k-row chunks saved as `<cache>.partNNN.npy` (resumes after a crash);
  Pokec loader frees the edge frames + 23 interest string columns before encoding and
  takes `cache_dir`. Resumable training: `Cfg.ckpt_dir` saves model+optimizer+best emb
  every eval window and each finished model's result; rerunning `run_all` skips finished
  models and resumes the current one. Notebook mounts Drive and caches the built graph.
- **2026-09-28** `graph_stats` moved from networkx to igraph (TIMME: hours → ~25 s; new
  dependency `igraph`); timestamped, flushed progress logs (`metrics.log`) in all scripts;
  text capped at 1000 chars before tokenizing (identical embeddings, ~2x faster).
- **2026-09-27** Literature review (S-22): ambiguity score, outliers ≠ bridges (S-06
  rejected), source-unknown blocking, stronger baselines; modularity measured (T7.0).
- **2026-09-27** Plan reordered (S-21): IBM simulation + baselines is now T7 and gates
  active learning (now T8).
- **2026-09-27** Pipeline check before GPU training: sparse-adjacency GCN (fixes OOM,
  identical outputs), per-model seed, `--tag`/`--region` wiring for Pokec slices,
  output dir anchored to `training/out/`, requirements completed for a Linux CUDA box.
- **2026-09-27** Reserve loaders (gab, reddit) + rejected `test_disagreement.py` moved to
  `archive/training/`; `training/` now holds only the TIMME + Pokec pipeline.
- **2026-09-27** Repo reorganized: docs/, scripts/, archive/; living docs at root;
  `train_pokec.ipynb` added, `train_voterfraud.ipynb` + stale `inst.txt` removed.
- **2026-09-27** Datasets final: TIMME `P_all` + Pokec; VoterFraud loader deleted.
  One multilingual encoder + `has_metadata`; TIMME switched to raw bio/tweet text;
  new Pokec loader. Nodes without metadata kept; LCC filter. Epoch selection by
  modularity. Gab edges fixed + cached. `metrics.py` report added.
- **2026-09-26** Modularity formula fixed; k-means on L2-normalized embeddings.
- **≤2026-09-23** BGRL/DGI/MVGRL training harness, early stopping, dataset downloads.
