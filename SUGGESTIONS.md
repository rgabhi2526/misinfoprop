# SUGGESTIONS — ideas, trade-offs, decisions

Each entry maps to a task in `TASKS.md`. Status:
**open** (needs a call) · **accepted** (do it) · **done** · **rejected** (with reason).
When a suggestion is accepted, add/adjust the task in `TASKS.md` and link back here.
Last updated: 2026-09-27

| ID | Task | Suggestion | Status |
|---|---|---|---|
| S-01 | T4.4 | Louvain on the graph as a third view | open |
| S-02 | T4.1 | HDBSCAN instead of DBSCAN | open (recommended) |
| S-03 | T4.1 | Cosine + dimension reduction before density clustering | open (recommended) |
| S-04 | T8.1 | Participation coefficient alongside betweenness | open |
| S-05 | T4.2 | FINCH: own ~30-line implementation vs `finch-clust` | open |
| S-06 | T4.3 | Treat HDBSCAN noise as "uncertain", not "no community" | open (recommended) |
| S-07 | T4.5 | Separate real disagreement from seed instability | accepted |
| S-08 | T6.1 | Error enrichment as the first validation metric | open (recommended) |
| S-09 | T6.2 | Baselines for "is disagreement informative" | open |
| S-10 | T7.4 | Oracle design for TIMME | open |
| S-11 | T4 | Expect fine-grained, not R-vs-D, disagreement | open |
| S-12 | T1.8 | VoterFraud: get text, or demote/drop | open |
| S-13 | T1.9 | Reddit: demote to secondary or drop | open |
| S-14 | T8.2 | Use Gab repost cascades to estimate spread | open |
| S-15 | T9.3 | Pitch the narrowed novelty claim to the advisor | open |
| S-16 | T2.3 | Select best epoch on modularity, not NMI | done |
| S-17 | T3.7 | igraph / cuGraph for graph metrics >200k nodes | open |
| S-18 | T2.7 | Cache MVGRL diffusion to disk | open |
| S-19 | T9.2 | Isolate the project env from conda base | open |
| S-20 | T1.7 | Random-user sampling for Gab smoke runs | open |

---

### S-01 — Louvain as a third view (T4.4)
The paper's two views both look at the embedding. Louvain looks only at topology,
so it is a genuinely different signal and matches "disagreement irrespective of
embeddings". **Risk:** it is a departure from the paper — keep A/B = HDBSCAN/FINCH
as the faithful baseline, add C only as an ablation (A∩B vs A∩C vs B∩C).
Earlier evidence: graph-only clustering hit NMI 0.82 on PureP vs 0.70 for
embeddings — needs re-checking on P_all after retraining.

### S-02 — HDBSCAN over DBSCAN (T4.1)
One global `eps` cannot fit power-law graphs whose density varies by orders of
magnitude. HDBSCAN has no `eps`, ships in sklearn ≥1.3 (no new dependency), and
still labels noise. The paper used DBSCAN; say so and justify the swap.

### S-03 — Cosine + reduce dims before density clustering (T4.1)
Density is near-meaningless in 256-d. L2-normalize (already done for k-means —
raw vectors collapsed to [580, 3]), then PCA/UMAP to ~16–50 d. The paper's SpCL
setup uses k-reciprocal Jaccard distance; a faithful option, costlier.

### S-04 — Participation coefficient for bridges (T8.1)
Measures how evenly a node's edges spread across communities — closer to "bridge"
than betweenness, and O(E) instead of O(VE). Earlier boundary-vs-betweenness
numbers were computed on stale embeddings; redo after T2.6.

### S-05 — FINCH implementation (T4.2)
FINCH = link each point to its first nearest neighbour, take connected
components, recurse on cluster means → a hierarchy of partitions. ~30 lines with
sklearn `NearestNeighbors` + scipy `connected_components`, vs a pip dependency.
Lean: write it, plus a self-check. Pick the partition level whose cluster count
is closest to HDBSCAN's, or report several levels.

### S-06 — Noise = uncertain (T4.3)
The paper uses DBSCAN outliers explicitly (inlier/outlier pair types β1–β3).
Expect a large noise share on low-degree nodes; decide before interpreting results.

### S-07 — Instability ≠ disagreement (T4.5) — accepted
If view A itself changes across random seeds, A-vs-B "disagreement" is partly
noise. k-means seed stability was only 0.5–0.6 ARI on (contaminated) Reddit.
Measure each view's self-agreement first; HDBSCAN and FINCH are deterministic,
which is a point in their favour.

### S-08 — Error enrichment (T6.1)
On TIMME: for each node, is its cluster's majority party ≠ its own party? Compare
that rate inside vs outside regions of uncertainty. If regions are enriched for
errors, disagreement is informative → first real result.

### S-09 — Baselines for disagreement (T6.2)
Regions must beat: random node sets of equal size, lowest k-means margin nodes,
lowest-degree nodes. Otherwise "disagreement" is just "hard nodes".

### S-10 — Oracle for TIMME (T7.4)
Simulated oracle answers "same party?" for a pair. Only 2 classes → very coarse
must-link / cannot-link. Only 1,206 of 20,811 nodes are labeled, so queries are
restricted to labeled pairs. State this limitation up front.

### S-11 — Granularity mismatch with the paper (T4)
The paper's clusters are individual animals (hundreds of tight clusters); ours are
communities (few, large, overlapping). Disagreement at k=2 is tiny. HDBSCAN/FINCH
pick their own cluster counts — expect disagreement among sub-communities.

### S-12 — VoterFraud (T1.8)
After removing leaky features it has only degree + active status → fails the
"metadata" half of the idea, and its labels come from the authors' own graph
clustering. Options: hydrate tweet text (API access risky in 2026), keep as a
topology-only ablation, or drop.

### S-13 — Reddit (T1.9)
Nodes are subreddits, not users; no labels; 40% of nodes had no edges. Useful as
an unlabeled scale demo at best.

### S-14 — Gab cascades (T8.2)
Timestamped repost chains → estimate IC/LT transmission probabilities from data
instead of assuming them. Unique among our datasets.

### S-15 — Advisor pitch (T9.3)
Both halves (community-aware intervention, uncertainty + centrality) exist in the
literature (`docs/ideas/project-idea-1.md`). Narrow claim: uncertainty about a node's
community/bridging role, from clustering disagreement, under a labeling budget.

### S-16 — Modularity for epoch selection (T2.3) — done
NMI against the evaluation labels leaked them into model selection. Best-epoch
modularity is now selection-biased; report other metrics as the result.

### S-17 — Scalable graph metrics (T3.7)
networkx Louvain/betweenness is skipped above 200k nodes. igraph (CPU, fast) or
cuGraph (A100) if Gab/VoterFraud task metrics are needed.

### S-18 — Cache MVGRL diffusion (T2.7)
GDC PPR is CPU-bound and recomputed per run; save `diff_edge_index/weight` per dataset.

### S-19 — Project env (T9.2)
Base now has networkx 3.7, which breaks `checkov`'s pin (<2.7). A dedicated
conda env for the project avoids fighting base.

### S-20 — Gab sampling (T1.7)
The dump is grouped by user timeline, so `--limit N` = a few users' full
histories. For representative smoke runs, sample users first, then stream.
