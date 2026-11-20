# Raw results

Unedited outputs of every experiment in the project, 2026-09-27 → 2026-10-04.
Files ending `_rerun` in the table were re-run on 2026-10-13 because the original
run only printed to the console; same scripts, same data, same seeds.

Scripts referenced as `training/x.py` live in `training/`; one-off analysis
scripts are copied into `results/scripts/`. Run them from `training/` with
`/opt/anaconda3/bin/python`.

## gnn_embeddings/
| File | What | Produced by |
|---|---|---|
| `timme_gnn_metrics_2026-09-28.json` | BGRL / DGI / MVGRL on TIMME: modularity, NMI, probe (old, biased probe), health, task metrics | `training/train_timme.py` (GPU run, 2026-09-28) |

## views/  (graph-only and text-only partitions — large arrays, not copied)
Leiden runs and spectral embeddings stay in `training/out/timme_views/` and
`training/out/pokec_knm_views/` (`T_leiden_runs.npy`, `C_leiden_runs.npy`,
`T_spectral.npy`, 1–16 MB each). Their summary numbers are in `logs/views_*.log`.

## scores/  (per-node scores, NumPy `.npz`)
Keys: `t` (topology community), `p_T`, `p_C` (memberships), `misfit`,
`disagreement`, `ambiguity_T`, `consensus_entropy`, `participation`,
`content_outlier`, `pagerank`, `content_acc`, `content_majority`.
| File | Text used for p_C | Produced by |
|---|---|---|
| `timme_scores_minilm_glove.npz` | MiniLM (bio + latest tweet) + GloVe (all tweets) | `training/ambiguity.py timme` |
| `timme_scores_minilm_only.npz` | MiniLM only (used with content-aware spread) | `training/ambiguity.py timme --content minilm` |
| `pokec_slice_scores.npz` | Interest text (MiniLM) | `training/ambiguity.py pokec_knm` |

## detection/
| File | What | Produced by |
|---|---|---|
| `synthetic_benchmark.json` | AUC (overall, within-degree) per score, 18 settings × 3 seeds | `training/benchmark.py` |

## blocking/  (spread simulations)
Each cell = `[mean infected share, mean cross-community share, SE infected, SE cross]`
where present (older files: first two only); key `none` = no blocking.
| File | Spread model | Produced by |
|---|---|---|
| `timme_content_blind.json` | IC: weighted cascade, 0.5–4× threshold; 100 sims | `training/blocking.py timme` |
| `timme_content_aware.json` | Agreement-driven (GloVe acceptance), MiniLM scores; 100 sims | `training/blocking.py timme --scores ambiguity_minilm.npz --affinity glove` |
| `pokec_slice_content_blind.json` | IC as above; 200 sims | `training/blocking.py pokec_knm` |

## regime/
| File | What | Produced by |
|---|---|---|
| `node_regime_map.json` | 72 settings × 2 seeds: paired gain (mean, 95% CI) of each content strategy over the best of PageRank / betweenness; detection AUC | `training/regime.py` |

## logs/  (console output, verbatim)
| File | What |
|---|---|
| `modularity_timme.log` | Louvain modularity vs rewired null, TIMME (re-run) |
| `modularity_pokec_slice.log` | Same, Pokec district slice (re-run) |
| `modularity_pokec_full.log` | Same, full Pokec (original run, 2026-09-27) |
| `gnn_degree_diagnostics.log` | GNN embeddings: k=2 split vs log-degree, k=6, top-PC correlations (re-run, `scripts/diag.py`) |
| `gnn_cluster_after_removing_degree.log` | Clustering GNN embeddings after regressing out degree (re-run, `scripts/diag3.py`) |
| `gnn_fair_probe_and_baselines.log` | Fair probe (standardized), spectral + neighbour-vote baselines (re-run, `scripts/diag2.py`) |
| `views_timme.log`, `views_pokec.log` | Topology / text-only views: k, modularity, NMI, stability |
| `text_only_clustering_timme.log` | MiniLM vs GloVe vs both, text-only clustering (re-run, `scripts/content_src.py`) |
| `amb_minilm.log` | MiniLM-only text → community accuracy |
| `bench.log` | Synthetic benchmark, console view of `synthetic_benchmark.json` |
| `timme_party_check.log` | Party-misfit AUC + one-sided p, shuffled-text null (re-run, `training/validate.py`) |
| `pokec_profile_fields.log` | Age/gender/text → Pokec community accuracy (re-run, `scripts/pokec_meta.py`) |
| `block_timme.log`, `block_timme_aware.log` | TIMME blocking tables (content-blind, content-aware) |
| `block_gateway.log` | Gateway vs structural gateway, content-aware |
| `regime.log` | Regime sweep, one line per setting |
| `timme_position_on_regime_map.log` | TIMME's agreement effect and mismatch degree percentile (re-run, `scripts/place.py`) |
| `edges_timme.log` | Link friction on TIMME |
| `edge_seeds.log` | Link friction, synthetic, 4 seeds + content-blind control |
| `edge_oracle.log` | TIMME link friction with true receptivity |
| `edge_v2.log` | Structural link score × receptivity: synthetic + TIMME (+ oracle) |
| `pokec_text_cap_check.log` | 1000-character text cap: tokens changed on 300k Pokec profiles (re-run, `scripts/chk.py`) |

Note: `gnn_degree_diagnostics.log` prints a `has_text` column computed on standardized
features; it is not a 0/1 share and should be ignored (noted at the time of the first run).

## hypotheses/  (2026-11-01, `training/h1_*, h2_*, h3_*, h4*_*`)
Each subfolder has its own `README.md` with the verdict.
| Folder | What |
|---|---|
| `h1/` | Radicalisation-funnel blocking (not supported) |
| `h2/` | Misfit as filter over PageRank/betweenness; synthetic + TIMME + Pokec slice (not supported); `synth.json` is 35 MB |
| `h3/` | Scenario ranking a-g (only detection works) |
| `h4a/` | Infiltrator/campaign/recruiter detection with realistic text |
| `h4b/` | Real TIMME interactions: bridges + agreement-gating estimate |
| `h4c/` | PHEME rumor-detection feature test |
| `h4d/` | Extended node-level regime sweep (beta 4-20) |
