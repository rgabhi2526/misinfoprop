# TASKS — single source of truth

**Goal (current phase):** embed nodes from metadata + topology → cluster them with
two complementary methods → **find where the clusterings disagree** (regions of
uncertainty, per the AAS active-learning paper, `docs/papers/active_learning.pdf`). Disagreement
is the core deliverable; embeddings are the current input to it, not the point.

**Later phase:** use disagreement to drive active learning (pair queries + oracle),
then bridge-node / intervention work (see `docs/ideas/project-idea-1.md`).

Status: `[x]` done · `[~]` in progress · `[ ]` todo · `[!]` blocked · `[-]` dropped
Suggestions and decisions for each task: `SUGGESTIONS.md` (keyed by task ID).
Last updated: 2026-09-27

---

## T0 — Finalize datasets (≥2 that fully meet requirements)  ← BLOCKS EVERYTHING
Datasets must not change after this. Discussion log: `DISCUSSION.md` Topic 1.

- [x] T0.1 Pin down requirements with the user (Q&A, 2026-09-27):
  - **R1 Human-like structure** — real social network; groups form by reachability +
    like-mindedness (homophily). Topic irrelevant: the propagated message is synthetic.
  - **R2 Node = any identity that can pass a message on** (user, or e.g. a subreddit,
    if it can plausibly act as a bridge).
  - **R3 Rich per-node metadata reflecting thoughts/interests** (text, interests,
    topics). Used **as embedding features only**; clusters explained by inspecting members.
  - **R4 Keep nodes without metadata** (dropping them deletes edges, possibly bridges):
    empty features + a binary `has_metadata` flag; topology fills in via the GCN.
    Largest-connected-component filter still applies (unreachable nodes can't bridge).
  - **R5 Any social tie is an edge** (flow or relationship); direction not required.
  - **R6 Labels not required** — success = spread blocked in the IBM simulation.
  - **R7 Timestamps not required** — spread probability from an assumed model
    (weighted cascade, p = 1/in-degree) by default; data-estimated is a bonus.
  - **R8 Offline-available** (no API hydration).
  - **R9 Size: start small** (≲100k nodes); large is fine later.
    Compute: ₹2k GPU budget (~23 h A100 / ~50 h A30–L4) — not a binding constraint.
  - **R10 ≥2 datasets**, no diversity constraint between them.
- [x] T0.2 Inventory every dataset on disk (→ `DISCUSSION.md` Topic 2, 2026-09-27): nodes, edges, direction, node type,
      metadata fields, labels, timestamps, coverage — verified from files, not docs
  - candidates: TIMME, Gab, Reddit, VoterFraud (`data_final/`);
    Pokec, Orkut, SNAP Twitter-ego, SNAP Facebook-ego (`data/`)
- [x] T0.3 Score vs R1–R10: pass TIMME, Twitter-ego, Gab, Pokec · fail VoterFraud, Orkut, Facebook-ego · weak Reddit (2026-09-27)
- [x] T0.4 **Final: TIMME `P_all` + Pokec** (2026-09-27). Gab, Reddit reserve; VoterFraud,
      Orkut, Facebook-ego dropped. Features: TIMME bio + latest tweet text; Pokec interest
      text + gender/age/region; one multilingual encoder for all text.
  - [x] T0.4a Develop T4–T8 on TIMME + a Pokec region slice; full Pokec for the final run (2026-09-27)
- [x] T0.5 Rewrite `data_final/DATASETS.md` (2026-09-27)
- [~] T0.6 Loaders: `common.encode_texts` (+ `has_metadata`), TIMME text features,
      new `train_pokec.py` (`--region`), Gab on shared encoder, VoterFraud loader deleted

---

## T1 — Datasets & loaders
Reference: `data_final/DATASETS.md`, `training/train_<ds>.py`

| Dataset | Metadata | Topology | Labels | Role |
|---|---|---|---|---|
| TIMME `P_all` | GloVe tweets | 5 relations | 1,206 R/D | **Primary** — only one that can validate |
| Gab | post text | repost/reply/quote | none | Scale demo, unvalidated |
| Reddit | subreddit emb. | hyperlinks | none | Secondary (nodes = subreddits) |
| VoterFraud | degree + status only | retweets | graph-derived | Weak fit — see S-12 |

- [x] T1.1 Download all datasets (`scripts/download_data.sh`)
- [x] T1.2 Keep only the largest connected component; `data.orig_idx` aligns labels (2026-09-27)
- [x] T1.3 TIMME → `P_all` + `additional_labels`; independents unlabeled (2026-09-27)
- [x] T1.4 Reddit: drop 20.5k edgeless subreddits (via T1.2) (2026-09-27)
- [x] T1.5 VoterFraud: remove community-derived (label-leaking) features (2026-09-27)
- [x] T1.6 Gab: fix repost/reply/quote edges, credit text to author, cache parse (2026-09-27)
- [ ] T1.7 Gab: full 48 GB parse + encode → `gab_cache_all.pt` (~20–40 min, one-off)
- [ ] T1.8 Decide VoterFraud's fate — S-12
- [ ] T1.9 Decide Reddit's fate — S-13

## T2 — Embedding models (BGRL / DGI / MVGRL)
Reference: `docs/MODELS.md`, `training/gclib.py`, `training/README.md`

- [x] T2.1 Shared GCN encoder + 3 objectives, same features for all
- [x] T2.2 Early stopping, best-epoch tracking, collapse guard
- [x] T2.3 Select best epoch by **modularity**, not NMI (label leakage) (2026-09-27)
- [x] T2.4 Symmetric edge dropout in augmentations (2026-09-27)
- [x] T2.5 MVGRL exact-PPR fallback when numba fails (2026-09-27)
- [x] T2.5a Sparse-adjacency GCN (edge list OOMs: TIMME ~12 GB/layer, Pokec ~46 GB);
  numerically identical, `training/test_gclib.py`. Per-model seed reset. (2026-09-27)
- [ ] T2.6 **Retrain on the Linux CUDA box (user runs)** — every `.pt` in `training/out/` is stale
  - [ ] preflight: `python test_gclib.py` on cuda + numba/GDC check in `requirements.txt`
  - [ ] timme (P_all)  - [ ] pokec slice (`--tag pokec_knm`)  - [ ] pokec full
- [ ] T2.7 Cache MVGRL diffusion to disk — S-18

## T3 — Embedding evaluation
Reference: `training/metrics.py`, `training/eval_embeddings.py`, `training/plot_umap.py`

- [x] T3.1 Fix `_modularity` (degree double-count, m off by 2×) (2026-09-26)
- [x] T3.2 L2-normalize before k-means (raw vectors collapsed to [580, 3]) (2026-09-26)
- [x] T3.3 `metrics.report`: health / labeled / unlabeled / task sections + self-check (2026-09-27)
- [x] T3.4 UMAP plots (timme, reddit) (2026-09-27)
- [ ] T3.5 Rerun `eval_embeddings.py` + `plot_umap.py` after T2.6
- [ ] T3.6 Pick the embedding(s) to feed T4 — by probe/NMI on TIMME **and** seed stability
- [ ] T3.7 Graph metrics for >200k nodes (VoterFraud, Gab) — S-17

## T4 — Clustering views  ← PAUSED until T2.6 (user trains the models)
Draft: `training/disagreement.py` (T4 + T5), uncommitted, untested, not reviewed.
Self-check draft parked in `archive/training/test_disagreement.py`. Resume only after the
user has trained TIMME + Pokec-slice embeddings and asks to continue.
Two complementary clusterers on the same embedding (paper: density + nearest-neighbour).

- [ ] T4.1 View A: HDBSCAN (density) on L2-normalized, dimension-reduced embeddings — S-02, S-03
- [ ] T4.2 View B: FINCH (first-neighbour, parameter-free) — S-05
- [x] T4.3 Noise handling decided: HDBSCAN `-1` = uncertain (S-06, 2026-09-27) — (HDBSCAN `-1` = "uncertain", as in the paper) — S-06
- [ ] T4.4 Optional view C: Louvain on the graph (topology-only) — S-01
- [ ] T4.5 Check each view is **stable across seeds** before trusting its disagreement — S-07

## T5 — Disagreement / regions of uncertainty  ← CORE
Paper §3.1-A.

- [ ] T5.1 Cluster-overlap graph: link cA_i–cB_j when 0 < IoU < 1
- [ ] T5.2 Regions S_k = connected components of that graph (transitive closure)
- [ ] T5.3 Per-node output: `in_region`, `region_id`, inlier/outlier type (β1/β2/β3)
- [ ] T5.4 Summary stats: #regions, sizes, % nodes in regions, per dataset
- [ ] T5.5 Self-check on a synthetic case with known disagreement

## T6 — Validate that disagreement means something
- [ ] T6.1 TIMME: error enrichment — are in-region nodes more often in a cluster whose
      majority party ≠ their own? (first real result) — S-08
- [ ] T6.2 Compare against baselines: random nodes, low k-means margin, low-degree nodes — S-09
- [ ] T6.3 Disagreement across embedding models (BGRL vs DGI vs MVGRL): same regions?

## T7 — Bridges & IBM simulation — go/no-go for the idea (S-21)
Moved ahead of active learning (2026-09-27): test the core claim cheaply first.
Reference: `docs/ideas/project-idea-1.md`
- [ ] T7.1 Profile uncertain nodes: share of graph, degree, `has_metadata`; overlap with
      betweenness + participation coefficient — S-04
- [ ] T7.2 Spread model: weighted cascade (p = 1/in-degree) on the raw directed edges;
      IC/LT sensitivity check; Gab cascades as a data-driven estimate — S-14
- [ ] T7.3 Blocking at fixed budget k: none, random, degree, PageRank, betweenness,
      participation coefficient, disagreement, disagreement × betweenness
- [ ] T7.4 Decision: disagreement (× centrality) beats centrality alone → T8;
      otherwise report the negative result and rethink before building T8

## T8 — Active learning loop (paper §3.1-B…E) — only if T7.4 passes
Objective is blocking gain (T7.3), not label quality.
- [ ] T8.1 Over-segmentation pool U_os (region medoids, k_max NN, s_min)
- [ ] T8.2 Under-segmentation pool U_us (closest cross-cluster pairs ∩ inconsistent pairs)
- [ ] T8.3 Sampling distribution P(Y) (ε, π, ρ, ω)
- [ ] T8.4 Oracle: simulated from TIMME party labels — S-10 (no oracle for "is a bridge")
- [ ] T8.5 NP3 constrained refinement (must-link merge, cannot-link purification)
- [ ] T8.6 Budget curves: blocking gain vs #queries, vs random-pair baseline

## T9 — Housekeeping
- [x] T9.1 Commit the current work — `6189707` (2026-09-27)
- [ ] T9.2 Resolve conda-base `checkov` vs networkx 3.7 conflict — S-19
- [ ] T9.3 Advisor check-in with the narrowed novelty claim — S-15
