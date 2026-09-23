# Embedding Models

Purpose: produce node embeddings that jointly encode topology + ideological/stance signal, for later clustering (community detection) and bridge-node identification. Clustering is a separate downstream step — these models only need to produce good embeddings, not cluster assignments directly.

Hardware: NVIDIA A100 (40GB assumed — confirm if 80GB, changes headroom but not the approach below).

Three models, chosen to compare different objective families rather than three variants of the same idea:

| Model | Objective family | Role in comparison |
|---|---|---|
| **BGRL** | Bootstrapped (no negative sampling) | Most scale-robust, cheapest to trust — anchor baseline |
| **DGI** | Mutual-information maximization (InfoMax) | Contrast point against BGRL — different training signal, similar cost |
| **MVGRL** | Multi-view contrastive (adjacency view + diffusion view) | Highest quality potential — diffusion view captures structure BGRL/DGI's neighborhood-based views miss |

Dropped from consideration: plain GAE (strictly weaker clustering quality than DGI for the same engineering cost — no reason to burn a comparison slot on it), VGAE (its main advantage — a probabilistic latent giving free uncertainty estimates — doesn't matter once clustering/active-learning uncertainty is handled as a separate task), DMoN/DAEGC/SDCN (end-to-end clustering models — out of scope since clustering stays a separate step).

---

## Constraints and how each is handled

### 1. Scale — VoterFraud2020 (1.89M nodes, 16.7M edges) is the stress case
- **BGRL:** built and benchmarked by its own authors at million-node scale using neighbor sampling. No changes needed — use as-is with a `NeighborLoader`-style subgraph sampler if memory pressure shows up, but likely runs full-batch on A100 given no negative-sampling overhead.
- **DGI:** off-the-shelf implementations (e.g. PyG's example) are full-batch and assume dense adjacency-style handling, which breaks past ~100–200k nodes on typical hardware. **On A100 with sparse adjacency (PyG's SpMM ops, not a dense matrix), full-batch training at 1.9M nodes is feasible.** Fallback if memory is still tight: retrofit with Cluster-GCN-style subgraph sampling (well-trodden path, not novel engineering).
- **MVGRL:** GPU compute isn't the constraint — the diffusion matrix (personalized PageRank) is. A dense PPR matrix at 1.9M nodes is not just infeasible, it's meaningless to attempt (N² blowup). **Fix: use PyG's `GDC` transform**, which computes a sparse/truncated approximation of the diffusion matrix instead of the literal dense one. This is the standard way MVGRL is run at scale, not a workaround.

### 2. MVGRL's diffusion preprocessing is a real CPU cost, separate from GPU training time
- Computing truncated/sparse PPR for 1.9M nodes takes real wall-clock time (minutes, not seconds) and runs on CPU before any GPU training starts.
- **Handling:** budget for this explicitly as a preprocessing stage in the pipeline, run once per dataset and cache the result to disk — don't recompute it per training run.

### 3. Directed graphs, symmetrized for embedding
- All target graphs (VoterFraud2020 retweets, Gab follows, TIMME follow/retweet/mention, Reddit-hyperlinks) are directed. All three models (BGRL, DGI, MVGRL) assume undirected input in their standard form.
- **Handling:** symmetrize for the embedding/clustering stage only. The original directed edges are kept and used separately downstream for diffusion simulation (IC/LT) and betweenness/in-degree centrality — direction is not lost from the project, just not used at the embedding step. (Decided earlier: embedding is for grouping only, bridge-detection/intervention runs on the directed graph.)

### 4. Node features must carry ideology signal, not just generic metadata
- Group A datasets (VoterFraud2020, TIMME) ship explicit or pre-derived ideology labels/cluster signal — usable close to directly as node features.
- Group B datasets (Gab, Reddit-hyperlinks) ship only raw metadata (post text, hashtags, LIWC features) — ideology signal must be derived first (hashtag/URL co-occurrence, text embedding, retweet-of-known-elites propagation) before it's fed into any of the three models as node features.
- **Handling:** this derivation is a preprocessing step shared across all three models — same feature matrix goes into BGRL, DGI, and MVGRL, so the comparison stays fair (differences in output are attributable to the model, not to different input features).

### 5. Fair comparison across three different training procedures
- BGRL, DGI, and MVGRL have different augmentation/corruption schemes (BGRL: two augmented views; DGI: corrupted negative graph; MVGRL: adjacency + diffusion views), which makes naive "same hyperparameters for all three" comparisons misleading.
- **Handling:** fix what should be fixed (embedding dimension, encoder depth/width, train/eval split, downstream clustering algorithm) and let each model use its own standard augmentation/corruption recipe from its original paper rather than forcing identical training mechanics — comparing embedding quality on downstream clustering metrics (e.g. modularity, NMI against Group A ground truth), not training-loss curves.

---

## Open questions / not yet decided

- A100 VRAM variant (40GB vs 80GB) — affects how much headroom exists before DGI/MVGRL need sampling fallbacks, not the overall approach.
- Exact ideology-signal derivation method for Group B datasets (hashtag co-occurrence vs text embedding vs elite-propagation) — to be decided before feature matrices are built.
- Final embedding dimension and encoder architecture (GCN vs GraphSAGE-style encoder) — to be fixed once first runs start, so all three models use the same encoder backbone for a clean comparison.
