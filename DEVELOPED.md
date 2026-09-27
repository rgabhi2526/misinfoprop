# DEVELOPED — what exists, how it works, why

The point of truth for **what has been built** and **how it behaves**. Scope and status
live in `TASKS.md` (T-ids), options and decisions in `SUGGESTIONS.md` (S-ids),
cross-session disputes in `DISCUSSION.md`, dataset facts in `data_final/DATASETS.md`.
**Update this file whenever something major changes** (rule in `CLAUDE.md`).
Last updated: 2026-09-27

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
                   ──► active learning (T7) ──► bridges + IBM simulation (T8)
```

| Stage | Status | Tasks |
|---|---|---|
| Datasets | **Final: TIMME `P_all` + Pokec** | T0 ✅, T1 |
| Loaders + features | built, smoke-tested | T0.6, T1 |
| Embedding models | built; **all saved embeddings stale → retrain** | T2 (T2.6 open) |
| Embedding evaluation | built | T3 |
| Clustering views | not built — **next** | T4 |
| Disagreement (core) | not built | T5 |
| Validation / AL / IBM | not built | T6–T8 |

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
- `plot_umap.py <ds>` — UMAP grid (models × colourings). Self-check: `python metrics.py`.

**Intuition to keep:** seed stability matters most for us — if a clustering changes
with the random seed, "disagreement" between two clusterings is partly noise (S-07).

## 4. Not built yet — intended behaviour
- **T4 views:** HDBSCAN (density) and FINCH (first-neighbour) on L2-normalized,
  dimension-reduced embeddings; optional Louvain on the graph (S-01…S-06). Both are
  deterministic → no seed noise.
- **T5 disagreement:** link clusters from the two views that partly overlap
  (0 < IoU < 1); connected groups of linked clusters = regions of uncertainty; per-node
  flag + region id + inlier/outlier type.
- **T6 validation:** are region nodes more often mis-grouped (TIMME party) than
  others, and better than random / low-margin / low-degree baselines?
- **T7/T8:** AAS pair sampling + oracle + NP3 refinement; then IBM with an assumed
  weighted-cascade spread model (p = 1/in-degree), fixed blocking budget vs baselines.

## 5. Evidence so far (read with care)
All pre-2026-09-27 numbers came from broken loaders or selection — **superseded**.
- PureP smoke run (50 epochs, fixed pipeline): NMI 0.72 / 0.76 / 0.72
  (BGRL/DGI/MVGRL). Tiny, easy dataset — can't rank models.
- Earlier "MVGRL doesn't cluster" and "boundary ≠ betweenness" — **withdrawn**, rerun after T2.6.

## 6. Environment
- Python: `/opt/anaconda3/bin/python` (conda base: torch 2.5, torch_geometric 2.8,
  networkx 3.7, sentence-transformers, umap-learn). `conda run` drops stdin.
- Base's numba can't compile PyG's approximate PPR → MVGRL uses the exact fallback on
  small graphs only.
- networkx 3.7 breaks `checkov`'s pin in base (S-19).
- GPU: rented, ₹2k budget (~23 h A100). Path has a trailing space — quote it.

## 7. Change log (major changes only)
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
