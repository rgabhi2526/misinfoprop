# DEVELOPED — what exists, how it works, why

The point of truth for **what has been built** and **how it behaves**. Scope and status
live in `TASKS.md` (T-ids), options and decisions in `SUGGESTIONS.md` (S-ids),
cross-session disputes in `DISCUSSION.md`, dataset facts in `data_final/DATASETS.md`.
**Update this file whenever something major changes** (rule in `CLAUDE.md`).
Last updated: 2026-11-18

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
**2026-11-18 (H2 correction, H6, final_results).** *H2* (`results/hypotheses/h2/README.md`) verdict is now "mixed": infected share 0/6900 filter configs CI>0 (unchanged); synthetic cross-community share 764/6900 CI>0, almost all beta>=6, median +4.4 pts, random-from-shortlist control 0/2250; real data still noise (TIMME 7/1840, Pokec 1/920). *H6* community-deception detection (`results/hypotheses/h6/README.md`, commits 3ea4736, fdfb58f): NOT SUPPORTED. Misfit is a weak, mostly non-specific signal: +0.04..+0.10 AUC over the unattacked same-population baseline under kumari/safeness, +0.01..+0.03 under matched random rewire. The earlier "misfit detects successful kumari hiding" was circular (success = label flip; a matched random control gives the same succ AUC). participation / persistence_low detect rewired targets better but also fire on random rewiring. *final_results/* (commits 99306e6, f81c104, ef14696) is the single-script ground truth for all numbers: `build_results.py` -> `results_long.csv` (14063 rows), `results_summary.csv` (22 experiments E01-E22), `metrics_glossary.csv` (273), `FINAL_RESULTS.md`, `_gaps.md`. Regenerate with `cd final_results && /opt/anaconda3/bin/python build_results.py`. `h5_probe.py` now saves `results/hypotheses/h5/h5_probe.json`. Dashboard: https://claude.ai/artifact/5iP76RnE7TAZZuSEfMQHcz. H5 caveat: random_rewire's gain over none is +0.4..+3.5 pp for model wc at beta>=3 only; for ic:2 it is about 0 (r2.0 k5 b6 -0.34 pp).

**2026-11-11 (T11 re-homing, `results/hypotheses/h5/README.md`).** Misfits under-served by follow graph (supported); community-level text target wins (k=4: 0.54 vs 0.14 struct vs 0.41 Kumari). Account-level text recommender NOT robust over 27 settings (P@10 vs AA: misfit CI>0 12/27, <0 9/27; R@100 CI>0 24/27). Rewiring simulation (correction reach): TIMME weakly supported (targeting residual over random rewire CI>0 in 4/7 beta>=3 cells, null at real beta 0.3); Pokec supported at beta>=3 (+4.0..+8.7 pp over random, CI>0 in 8/8 cells; beta is swept, not measured). Re-homing helps only under selective exposure.

**2026-10-04 (S-23, T10) supersedes the GNN-based plan.** Pipeline now: `views.py` (Leiden
topology view, text-only content view) → `ambiguity.py` (label-spread p_T, cross-fitted text
classifier p_C, misfit/disagreement + baselines) → `benchmark.py` (planted mismatches) /
`validate.py` (TIMME party check) → `blocking.py` (IC, content-blind and content-aware) →
`regime.py` (synthetic regime map). Key numbers: GNN clusters Q ≤ 0; Leiden TIMME Q 0.264;
text→community acc TIMME 0.62 (chance 0.32), Pokec slice 0.28 (0.25); planted detection AUC
0.73–1.00 vs 0.53–0.85 baseline under nuisance; TIMME blocking: betweenness/PageRank beat every
content-aware strategy (content-weighted gateway +1–2 points, inside noise). Regime map (`regime.py`): node-level
content targeting beats centrality only at strong agreement effect + hub-placed mismatches; link
friction (`edges.py`, structural bridge score × receptivity) +14.9 ± 1.9 points on synthetic,
tie on TIMME (outside the regime: agreement effect ≈ 1.4). Full log in the plan tracker.

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
- **2026-11-18** H2 verdict corrected to "mixed" (synthetic cross-community share 764/6900 CI>0, beta>=6; infected share 0/6900); H6 deception-detection experiment recorded, NOT SUPPORTED (circular success result retracted); `final_results/` ground-truth tables added (`build_results.py`, E01-E22); `h5_probe.py` writes `h5_probe.json`; H5 random_rewire range qualified to model wc. TASKS T11.4/T11.5; dashboard artifact linked.
- **2026-11-11** T11.1b/T11.2: `training/h5_rec.py --robust` (27-setting account-level robustness, `h5_robust.json`) and `training/h5_rewire.py` (degree-preserving rewiring + content-aware IC on TIMME and Pokec; `h5_rewire_timme.json`, `h5_rewire_pokec.json`). Results in `results/hypotheses/h5/README.md`.
- **2026-11-11** Direction change to misfit re-homing (TASKS T11, advisor). `training/h5_probe.py`:
  misfits follow one community but engage with their text-predicted one (retweet hit 0.60-0.66 vs
  0.40 chance, k=4); survives bio-only text. Bio embeddings cached at `training/out/timme_bio.npy`.
- **2026-11-01** Hypothesis testing H1-H4 (see TASKS T10.13, `results/hypotheses/`): misfit-guided
  blocking not supported in radicalisation funnels, as a filter over centrality, or on real TIMME
  interactions (real agreement beta ~0.3 vs >=6 needed); detection of non-structural infiltrators
  and campaigns works on synthetic data; rumor-detection feature untested/weak (PHEME). New files:
  `training/h1_funnel.py, h2_*.py, h3_*.py, h4a_*.py, h4b_*.py, h4c_pheme.py, h4d_*.py`.
  Downloads: `data_final/rumor_ext/` (PHEME, Twitter15/16). Thesis framing: detector/screening
  tool + regime boundary; blocking is a negative-with-conditions extension.
- **2026-10-04** GNN backbone dropped (S-23): two clean views + cross-fitted misfit score;
  synthetic benchmark, TIMME validation, content-aware blocking, regime map added
  (`training/views.py, ambiguity.py, benchmark.py, validate.py, blocking.py, regime.py`).
  `metrics._labeled` probe now standardizes inputs (was underfitting by 15–20 points);
  `degree_eta2` added. Blocking gate failed on TIMME; thesis reframed (detector + regime map).
- **2026-10-03** BGRL CUDA OOM on full Pokec (A30, 24 GB): feature dropout now masks the
  first layer's weight columns instead of copying `x` per view (identical outputs/grads,
  checked in `test_gclib.py`; saves ~7 GB).
- **2026-10-03** Colab RAM crash on full Pokec (during text encoding): `encode_texts`
  now encodes in 100k-row chunks saved as `<cache>.partNNN.npy` (resumes after a crash);
  Pokec loader frees the edge frames + 23 interest string columns before encoding. Resumable training: `Cfg.ckpt_dir` saves model+optimizer+best emb
  every eval window and each finished model's result; rerunning `run_all` skips finished
  models and resumes the current one. Notebook caches the built graph + checkpoints in
  `out/pokec_ckpt/` (local GPU box; Colab/Drive dropped).
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
