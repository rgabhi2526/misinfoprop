# TASKS — single source of truth

> **2026-10-04 — plan replaced by T10 (S-23).** GNN embeddings failed (clusters = degree);
> T2.6/T3.6/T4/T5 are superseded by the two-clean-views pipeline in T10. Live tracker:
> https://claude.ai/code/artifact/26f5235e-05f0-48b8-8b52-230efd38d85f · findings + proposals:
> https://claude.ai/code/artifact/a5516d50-4562-49ec-ba26-7e86d4961699

**Goal (current phase):** embed nodes from metadata + topology → cluster them with
two complementary methods → **find where the clusterings disagree** (regions of
uncertainty, per the AAS active-learning paper, `docs/papers/active_learning.pdf`). Disagreement
is the core deliverable; embeddings are the current input to it, not the point.

**Later phase:** use disagreement to drive active learning (pair queries + oracle),
then bridge-node / intervention work (see `docs/ideas/project-idea-1.md`).

Status: `[x]` done · `[~]` in progress · `[ ]` todo · `[!]` blocked · `[-]` dropped
Suggestions and decisions for each task: `SUGGESTIONS.md` (keyed by task ID).
Last updated: 2026-11-11

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
- [x] T4.3 Noise handling: HDBSCAN `-1` = **outlier**, kept separate from bridges (S-22 overturns S-06, 2026-09-27)
- [ ] T4.4 Partition ensemble: HDBSCAN + all FINCH levels + Louvain over seeds (S-22; was optional view C, S-01)
- [ ] T4.5 Check each view is **stable across seeds** before trusting its disagreement — S-07

## T5 — Disagreement / regions of uncertainty  ← CORE
Paper §3.1-A.

- [ ] T5.1 Cluster-overlap graph: link cA_i–cB_j when 0 < IoU < 1
- [ ] T5.2 Regions S_k = connected components of that graph (transitive closure)
- [ ] T5.3 Per-node output: `in_region`, `region_id`, inlier/outlier type (β1/β2/β3)
- [ ] T5.6 Continuous **ambiguity score** per node: entropy of co-membership across the
      partition ensemble (Nepusz bridgeness with our embeddings); outlier flag separate — S-22
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
- [x] T7.0 Measure modularity vs a rewired null (`training/check_modularity.py`) — S-22
      (2026-09-27): TIMME Q 0.26 (null 0.04), Pokec slice 0.49 (null 0.21), Pokec full 0.72 (null 0.12) — DEVELOPED §5
- [ ] T7.1 Profile uncertain nodes: share of graph, degree, `has_metadata`; overlap with
      betweenness + participation coefficient — S-04
- [ ] T7.2 Spread model: **source unknown** (random / degree-weighted seeds), node removal;
      weighted cascade (p = 1/in-degree) on the raw directed edges + a sweep of uniform p;
      IC/LT sensitivity; Gab cascades as a data-driven estimate — S-14, S-22
- [ ] T7.3 Blocking at fixed budget k: none, random, degree, PageRank, betweenness,
      participation coefficient, Comm/modular centrality, graph-only bridgeness (Louvain
      ensemble, no metadata — isolates what metadata adds), ambiguity, ambiguity × PageRank;
      greedy upper bound on the Pokec slice — S-22
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

## T10 — Make-it-work pipeline (S-23, 2026-10-03)
Two clean views (graph only / text only) → mismatch scores → validation → blocking.
- [x] T10.1 Fair probe + `degree_eta2` in `metrics.py`
- [x] T10.2 `views.py`: Leiden topology view (TIMME Q 0.264, 5 comms; Pokec slice Q 0.505); raw-text
      clusters rejected (Q ≈ 0.01)
- [x] T10.3 `ambiguity.py`: label-spreading p_T + cross-fitted text classifier p_C → misfit,
      disagreement; absorbing walk rejected (AUC 0.64)
- [x] T10.4 `benchmark.py`: planted mismatches — misfit AUC 0.73–1.00 vs content-outlier 0.53–0.85
      under nuisance; structure-only 0.50
- [x] T10.5 Content informativeness: TIMME 0.62 vs 0.32; Pokec 0.28 vs 0.25 (profile+text 0.36)
- [x] T10.6 `validate.py`: TIMME party check, misfit AUC 0.68 vs 0.54 shuffled, only 11 positives
- [x] T10.7 `blocking.py`: content-blind + content-aware IC (GloVe affinity, MiniLM scores),
      gateway score — **gate failed**: betweenness/PageRank win on TIMME and Pokec slice
- [x] T10.8 `regime.py`: node-level regime map, 72 settings × 2 seeds — content targeting wins only
      at strong agreement effect (beta 6) AND hub-placed mismatches (6/72, +1.4..+8.9 points)
- [x] T10.9 TIMME on the map: agreement effect ≈ 1.4, mismatches at 56th degree percentile → losing region
- [x] T10.11 `edges.py`: link friction, structural bridge score × receptivity: +14.9 ± 1.9 synthetic
      (control −7.3 ± 2.9), tie on TIMME even with oracle receptivity
- [x] T10.13 Hypothesis tests H1-H4 (2026-11-01; user asked "does it work anywhere?"). Details: `results/hypotheses/*/README.md`
  - H1 radicalisation funnel (`h1_funnel.py`): misfit-blocking NOT supported (0/120 cells CI>0; misfit finds recruiters AUC 0.59-0.65 vs degree 0.95)
  - H2 misfit as filter over PageRank/betweenness (`h2_*.py`): **mixed** (corrected 2026-11-18). Infected share: 0/6900 filter configs CI>0 (unchanged). Synthetic cross-community share: 764/6900 CI>0, almost all beta>=6, median +4.4 pts; random-from-shortlist control 0/2250. Real data still noise (TIMME 7/1840, Pokec 1/920). Details: `results/hypotheses/h2/README.md`
  - H3 scenario ranking (`h3_scenarios.py`): only *detection* of mismatched nodes clearly works (precision@20 +36..+76 pts); blocking needs beta>=6 + hub-placed/no-hub
  - H4a infiltrator/campaign detection with realistic text (`h4a_*.py`): works for non-structural infiltrators (gain AUC +0.1..+0.4 at text acc >=0.46, mix >=0.4); loses to degree/PageRank for recruiters/hubs/peripheral
  - H4b real TIMME interactions (`h4b_*.py`): misfit nodes NOT special bridges (cross-edge count ~0 after controls; receive less); real agreement beta ~0.3 vs >=6 needed
  - H4c rumor detection feature on PHEME (`h4c_pheme.py`): no gain (dAUC -0.001); test weak (no follow graph, misfit ~constant)
  - H4d extended regime sweep (`h4d_*.py`): beta up to 20, 12 seeds: fixed misfit_x_pr loses in 120/120 cells; best-selected wins 25/120 but <= +0.06 pts (negligible); blocking stays negative
  - Thesis consequence: misfit = screening/detection tool; blocking claim is a tested, mostly negative extension with a stated regime boundary.
- [ ] T10.12 Edge-level regime map; write up the reframed thesis
- [ ] T10.10 Extension (pending advisor): active learning with disagreement-chosen queries

## T11 — Misfit re-homing (S-25, 2026-11-11; advisor direction)
Misfits = nodes placed in the wrong community; re-home them to argmax p_C via (b) link
recommendation or (c) degree-preserving rewire (drop dormant old-community follows). Headline =
recommendation framing. Reference: Kumari et al. 2021 (`misfit.md`, community deception; its MPL
relabels only and its target is structural, so only the rewiring mechanics carry over).
- [x] T11.0 Probe (`training/h5_probe.py`): follow-graph communities, held-out retweet/like/mention.
      Top-10% misfits' cross-community retweets land in argmax p_C 0.66 (k=4) vs 0.40 chance vs 0.53
      Kumari extraction target; only 13% of their retweets stay in own follow community (controls 69%).
      **Bio-only text (leak control: 34% of latest tweets are RTs)**: 0.60 vs 0.40 (k=4), 0.28 vs 0.12 (k=15) -> survives.
- [x] T11.1 M1 (`training/h5_rec.py`, `results/hypotheses/h5/README.md`): misfits under-served SUPPORTED
      (held-out ties 0.43 text target vs 0.18 own; AA P@10 0.090 vs 0.138 controls); community-level text
      wins (0.54 vs 0.14 struct vs 0.41 Kumari, k=4); account-level MIXED (k=15 AA x p_C +60% over AA;
      k=4 +18% not misfit-specific; k=5 loses)
- [x] T11.1b Account-level robustness (`h5_rec.py --robust`, 27 settings = 9 communities x 3 bio seeds; settings share users, not independent): **NOT ROBUST** on P@10 -
      AA x p_C beats AA with misfit CI>0 in 12/27, loses in 9/27; R@100 favourable (CI>0 in 24/27). Gain is concentrated at r2.0. Details: `results/hypotheses/h5/README.md`
- [x] T11.2 M2 degree-preserving rewire + content-aware IC (`training/h5_rewire.py`, TIMME + Pokec): correction reach vs random rewire.
      TIMME: WEAKLY SUPPORTED - targeting residual over random rewire CI>0 in 4/7 beta>=3 cells, null at real beta 0.3.
      Pokec: SUPPORTED at beta>=3 (swept, no measured beta) - +4.0..+8.7 pp over random rewire, CI>0 in all 8 cells. Details: `results/hypotheses/h5/README.md`
- [x] T11.4 H6 community-deception detection (Kumari/safeness/random attacks, TIMME + Pokec; commits 3ea4736, fdfb58f; `results/hypotheses/h6/README.md`): **NOT SUPPORTED**. Misfit is a weak, mostly non-specific signal (+0.04..+0.10 AUC over unattacked same-population baseline under kumari/safeness, +0.01..+0.03 under matched random rewire). Earlier "misfit detects successful kumari hiding" was circular (success = label flip; matched random control gives the same succ AUC). participation / persistence_low detect rewired targets better but also fire on random rewiring.
- [x] T11.5 `final_results/` ground truth (commits 99306e6, f81c104, ef14696): `build_results.py` -> `results_long.csv` (14063 rows), `results_summary.csv` (22 experiments E01-E22), `metrics_glossary.csv` (273), `FINAL_RESULTS.md`, `_gaps.md`; `h5_probe.py` now writes `results/hypotheses/h5/h5_probe.json`. Regenerate: `cd final_results && /opt/anaconda3/bin/python build_results.py`. Source of truth for report numbers. Dashboard artifact (navigates final_results): https://claude.ai/artifact/5iP76RnE7TAZZuSEfMQHcz
- [ ] T11.3 M3 structure-content alignment (NMI / modularity), secondary only
- Thesis consequence: the detection / "wrongly placed" claim is supported on real data; re-homing helps information reach misfits only under selective exposure (beta>=3), not at TIMME's measured beta (~0.3).

## T9 — Housekeeping
- [x] T9.1 Commit the current work — `6189707` (2026-09-27)
- [ ] T9.2 Resolve conda-base `checkov` vs networkx 3.7 conflict — S-19
- [ ] T9.3 Advisor check-in with the narrowed novelty claim — S-15
- [~] T9.4 Mid-term report (LaTeX, `report/`): intro, related work, methodology, setup; no
      results. University template applied + review fixes (2026-10-14); final PDF built, no TODOs.
      Plan: full Pokec on its largest SCC; everything done by 2026-11-30 (report Table 5.1)
