# Final results: offline ground truth

Last updated 2026-11-18. Every number below is a value in `results_summary.csv` or `results_long.csv`
(rounded for reading), cited by experiment ID (E01 to E22). Where a number is not in the CSVs it is marked.

## 1. Summary

The project asks whether a text-versus-graph disagreement score ("misfit": one minus the text-predicted
probability of a node's own graph community) finds accounts that sit between communities, and whether that
helps with four jobs: finding bad actors, blocking misinformation spread, re-homing (recommending or rewiring
links toward the text-predicted community), and detecting community-deception attacks.

What holds. (1) The graphs have real community structure (E01). (2) On TIMME, text predicts a node's graph
community well above chance (0.62 vs 0.32, E04), and cross-community retweets land in the text-predicted community
more often than chance even with bio-only text (0.60 vs 0.40, E17). (3) On synthetic graphs with planted
text/graph mismatches, misfit detects them far better than centrality scores (E05, E12, E13), with conditions.
(4) In simulation, rewiring toward the text-predicted community increases correction reach on Pokec (+10.9 pp vs
no rewiring, E21) and weakly on TIMME (+4.72 pp, E20), at strong content gating only.

What failed. (1) Self-supervised GNN embeddings (BGRL/DGI/MVGRL) gave no usable clusters (E02). (2) Misfit-based
blocking does not beat PageRank or betweenness on the real graphs (E07), nor in the synthetic radicalisation
funnel (E10) or the extended regime sweep (E16); it wins only when content gating is strong (beta 6 or more) and
mismatches are hubs, while TIMME's measured gating is about 1.4 (E08, E14). (3) Misfit as a filter over centrality
shortlists does not improve infected share (E11). (4) Misfit adds nothing to misinformation detection on PHEME
(E15). (5) Misfit does not detect deception attacks (E22). (6) Link recommendation gains are not robust (E19).

Most positive results are on synthetic data with planted effects and are partly circular (the planted signal is
what misfit measures). Real-data evidence is mostly negative or small. Nothing here was pre-registered.

## 2. How to read this

Files in `final_results/`:

| File | Content |
|---|---|
| `results_summary.csv` | One row per experiment (22 rows): question, dataset, our method and result, best competitor and result, verdict, caveat. |
| `results_long.csv` | One row per number (14063 rows): experiment, dataset, setting, method, method_role (ours / baseline / control / reference), metric, value, ci_low, ci_high, uncertainty type, n, source file and key. |
| `metrics_glossary.csv` | One-line definition of every method, baseline, control and metric name used in the long file (273 rows). |
| `_gaps.md` | Numbers not traced to a raw file, and mismatches with the READMEs. |
| `_inventory.md` | Which raw result files feed which experiment. |
| `_build_report.md` | Row counts, hand spot-checks, build decisions. |
| `build_results.py` | Generator. Reads raw JSON/npz/log files under the repo root; asserts no duplicate rows and valid verdicts. |

Regenerate (about 10 s, CPU only): `cd final_results && /opt/anaconda3/bin/python build_results.py`

Terms. "failed" means the pipeline or method was broken (E02: embeddings carry no community signal). "not supported" means the
hypothesis was tested properly and rejected. The two are kept separate.

Conventions. Shares, AUC, P@10, R@100 and success rates are fractions (0 to 1). Anything called gain, removed
or `_pts` / `_pp` is in percentage points. Verdicts are one of: supported, partly supported, mixed, not supported,
failed, descriptive. Verdict wording comes from the project READMEs and TASKS.md; for E06 and E09 the verdict
is the builder's mapping (see section 8).

## 3. Scoreboard

"Ours" is the project's method. "Best competitor" is the strongest baseline in the same setting, as chosen in
`results_summary.csv`; where it is a control or a null this is stated.

| ID | Use case | Dataset | Our method and result | Best competitor and result | Verdict |
|---|---|---|---|---|---|
| E01 | Community structure | TIMME, Pokec slice, Pokec full | Louvain Q: 0.256 / 0.49 / 0.715 | Degree-preserving null Q: 0.035 / 0.206 / 0.124 | descriptive |
| E02 | GNN embeddings | TIMME | BGRL / DGI / MVGRL k-means Q: -0.00681 / -0.00934 / -0.015 | spectral64 probe acc 0.961; neighbour-vote acc 0.928 | failed |
| E03 | View construction | TIMME, Pokec slice | T-Leiden Q 0.264 (TIMME), 0.505 (Pokec) | Text kNN-Leiden Q 0.005; T-vs-C NMI 0.066 / 0.016 | descriptive |
| E04 | Content informativeness | TIMME, Pokec slice | Text classifier p_C acc 0.622 (TIMME), 0.278 (Pokec) | Majority class 0.318 / 0.249 | descriptive |
| E05 | Misfit detection, planted | Synthetic DC-SBM | Misfit AUC 0.74 (nu 1.5, mu 0.5, sigma 2) | content_outlier AUC 0.529; degree 0.508 | supported |
| E06 | Misfit detection, party labels | TIMME | Misfit AUC 0.682 (11 positives) | Shuffled-text null AUC 0.541; degree 0.361 | partly supported |
| E07 | Blocking gate | TIMME, Pokec slice | Gateway removes 9.22 pct infected (wc, k=50) | Betweenness 13.99 pct | not supported |
| E08 | Blocking regime map | Synthetic; TIMME placement | misfit_x_pr gain +1.38 pts in one favourable cell | Best structural removes 33.0 pts; TIMME beta 1.44 | partly supported |
| E09 | Edge-level blocking | Synthetic; TIMME | bridge_struct_x_recv +14.9 pts (synthetic beta 4) | Same on TIMME 40000 links: -0.5 pts; synthetic beta 0: -7.3 | partly supported |
| E10 | Radicalisation funnel | Synthetic funnel | misfit: 0 of 120 cells CI>0, 120 CI<0 | Best of degree/PageRank/betweenness | not supported |
| E11 | Misfit as filter | Synthetic, TIMME, Pokec slice | Infected: 0 of 6900 synthetic configs CI>0. Cross share: 764 of 6900 synthetic, 6 of 920 TIMME, 1 of 460 Pokec | Best of PageRank/betweenness top-k; random-from-shortlist 0 of 2250 | mixed |
| E12 | Scenario map | Synthetic | Detection precision@20: misfit 0.787 | PageRank 0.0312 | partly supported |
| E13 | Detection of infiltrators | Synthetic | Misfit AUC 0.772 (text acc 0.6) | Betweenness AUC 0.508 | partly supported |
| E14 | Bridge validation | TIMME | Cross-community count diff -1.6 vs matched controls; real agreement beta 0.29 | misfit~pagerank Spearman -0.0231 | not supported |
| E15 | Misinformation detection | PHEME | Adding misfit: delta AUC -0.000574 | Text + structure AUC 0.88 | not supported |
| E16 | Extended blocking sweep | Synthetic | misfit_x_pr fixed: 0 of 120 cells CI>0, 120 CI<0 | Best structural; best-selected misfit has 25 cells CI>0 (selection biased) | not supported |
| E17 | Re-homing motivation | TIMME | argmax p_C hit share 0.601 (bio-only, k=4) | Chance 0.401 | supported |
| E18 | Link recommendation | TIMME | aa_x_text P@10 0.122 (misfits, k=15) | Adamic-Adar P@10 0.0755 | mixed |
| E19 | Recommendation robustness | TIMME | aa_x_text vs aa: 12 of 27 settings CI>0 | 9 settings CI<0 | not supported |
| E20 | Rewiring, correction reach | TIMME | rehome_text +4.72 pp vs none [3.05, 6.91] (r1.0, k5, wc, beta 6); residual over random_rewire +1.26 [-0.33, 2.22], not distinguishable from generic rewiring | random_rewire +3.46 pp | partly supported |
| E21 | Rewiring, correction reach | Pokec slice | rehome_text +10.9 pp vs none (k5, wc, beta 6) | random_rewire +2.92 pp | supported |
| E22 | Deception detection | TIMME, Pokec slice | Misfit AUC 0.487 (kumari k=2, TIMME, all targets) | Participation AUC 0.541 | not supported |

Tally: supported 3 (E05, E17, E21); partly supported 6 (E06, E08, E09, E12, E13, E20); mixed 2 (E11, E18);
not supported 7 (E07, E10, E14, E15, E16, E19, E22); failed 1 (E02); descriptive 3 (E01, E03, E04). Only 3 of 22 experiments are clean positives, and one of those (E05) is synthetic and
partly circular.

## 4. Results by use case

### 4.1 Graph structure and embeddings (E01, E02, E03)

Question. Do the graphs have community structure, and do self-supervised GNN embeddings recover it?

Setup. Louvain modularity Q against a degree-preserving rewired null (E01). BGRL, DGI and MVGRL embeddings clustered
with k-means k=2 on TIMME, single run (E02). Topology (T) partition by 100 Leiden runs on the social graph versus a text
(C) partition (E03).

| E01 dataset | Real Q | Null Q | Cross-community edges, real | Cross-community edges, null |
|---|---|---|---|---|
| TIMME | 0.256 | 0.035 | 49.3 % | 83.1 % |
| Pokec slice | 0.49 | 0.206 | 39.7 % | 67.3 % |
| Pokec full | 0.715 | 0.124 | 21.8 % | 70.8 % |

| E02 method (TIMME) | k-means Q | Standardised (fair) probe acc |
|---|---|---|
| BGRL | -0.00681 | 0.951 |
| DGI | -0.00934 | 0.958 |
| MVGRL | -0.015 | 0.774 |
| spectral64 probe (graph-only baseline) | n/a | 0.961 |
| neighbour-vote (graph-only baseline) | n/a | 0.928 |
| text-only probe | n/a | 0.699 |
| majority class | n/a | 0.534 |

| E03 view | TIMME Q on social graph | Pokec Q |
|---|---|---|
| T Leiden (best of runs) | 0.264 | 0.505 |
| T regularised spectral + k-means (k=5 TIMME, k=11 Pokec) | 0.059 | 0.341 |
| C text kNN Leiden | 0.005 | not tabulated here |
| C text k-means k=5 | -0.002 | not tabulated here |

T-vs-C NMI is 0.066 on TIMME and 0.016 on Pokec: the two views differ.

Verdict. E01 descriptive: real communities exist. E02 failed: embedding clusters sort by degree (Q at or below 0,
NMI about 0, correlation of embedding norm with log degree 0.77 to 0.85); a fair probe is no better than graph-only
baselines. E03 descriptive: raw text clusters have no graph structure.

Caveats. The GNN embeddings are the phase-1 method and are an abandoned phase; they stay listed as "ours". E02 is a single run with no interval; the old probe was biased and the study was superseded by T10 (S-23). The
Reddit GNN has modularity only and is not included (`_gaps.md` 11). Pokec full was not run through the T10 pipeline.

### 4.2 Content informativeness and misfit detection (E04, E05, E06, E12 detection part, E13)

Question. Does text carry information about graph community, and does misfit find nodes whose text and graph disagree?

Setup. A cross-fitted text classifier p_C predicts the graph community (E04). Misfit is one minus p_C of the node's own
community. Detection is scored by AUC against planted mismatches on a DC-SBM (E05, E12, E13) or against party labels
on TIMME (E06).

E04, accuracy of text to graph community:

| Dataset | Predictor | Accuracy | Majority |
|---|---|---|---|
| TIMME | p_C (MiniLM bio + GloVe tweets) | 0.622 | 0.318 |
| TIMME | p_C (MiniLM only) | 0.496 | 0.318 |
| TIMME | p_C (GloVe only) | 0.637 | not stored for this row |
| Pokec slice | p_C (interest text) | 0.278 | 0.249 |
| Pokec slice | age + gender + completion | 0.314 | 0.249 |
| Pokec slice | text + profile | 0.356 | 0.249 |

E05, synthetic AUC overall at nu 1.5, mu 0.5, sigma 2 (mean of 3 seeds, no spread stored):

| Method | AUC |
|---|---|
| misfit (ours) | 0.74 |
| disagreement (ours) | 0.716 |
| content_outlier | 0.529 |
| degree | 0.508 |
| participation | 0.505 |
| consensus_entropy | 0.503 |

E05 baseline added: Radar (Li et al., IJCAI 2017, residual analysis on text attributes plus graph Laplacian), same planted graphs and seeds. Paper-style defaults (alpha=beta=gamma=1) collapse the residual to about 1e-9 on this data (n 4000, d 64 or 128, so attributes are fully reconstructable) and give AUC 0.494 in the cell above. With parameters picked using the planted labels on that cell (alpha 10, beta 0.001, gamma 100, so favourable to Radar), Radar AUC is 0.488 to 0.541 in all 18 cells and 0.512 at nu 1.5, mu 0.5, sigma 2, against misfit 0.74. Radar does not beat misfit in any cell (misfit 0.73 to 0.999). Source: results/detection/synthetic_benchmark_radar.json.

E06, TIMME party misfits (11 positives among 595 labelled nodes):

| Method | AUC | One-sided p |
|---|---|---|
| misfit | 0.682 | 0.019 |
| disagreement | 0.666 | 0.029 |
| participation | 0.622 | 0.083 |
| consensus_entropy | 0.616 | 0.015 |
| misfit, shuffled text (null) | 0.541 | 0.323 |
| content_outlier | 0.541 | 0.322 |
| ambiguity_T | 0.464 | 0.66 |
| degree | 0.361 | 0.943 |
| pagerank | 0.36 | 0.945 |

E12 detection (planted, random placement, precision@20, 8 seeds): misfit 0.787 [0.729, 0.846]; disagreement 0.6375;
content_outlier 0.0625; consensus_entropy 0.0375; PageRank 0.0312; participation 0.0312; degree 0.025; betweenness
0.025; random 0.0187. With hub placement: misfit 0.725, PageRank 0.369, betweenness 0.3625, degree 0.3563.

E13 infiltrators (random placement, text accuracy 0.61, mu 0.2), AUC:

| Method | AUC [95% CI] |
|---|---|
| misfit | 0.772 [0.752, 0.792] |
| disagreement | 0.765 [0.745, 0.785] |
| misfit_nonhub | 0.721 |
| misfit_x_lowdeg | 0.657 |
| topo_ambT | 0.509 |
| betweenness | 0.508 |
| pagerank | 0.501 |
| degree | 0.501 |
| low_degree | 0.499 |
| knn_dup | 0.459 |
| text_iso | 0.448 |

Verdict. Detection of planted mismatches is the one clearly supported capability (E05, E12, E13). On real TIMME
party labels it is partly supported: 0.682 beats its shuffled-text null (0.541) but rests on 11 positives.

Caveats. Synthetic and partly circular: the planted text mismatch is what misfit measures. E06 positives are few and
within-degree AUC shows participation (0.769) above misfit (0.744). E13 works for non-structural infiltrators and
campaigns when text accuracy is about 0.45 or higher; recruiters and hub or peripheral cases lose to PageRank or
low-degree (rows in the long file). Pokec text is nearly uninformative (E04), so misfit on Pokec is weak.

### 4.3 Misinformation blocking and spread (E07, E08, E09, E10, E11, E12 blocking part, E14, E16)

Question. Does targeting misfit nodes (or content-weighted cross-community links) reduce simulated misinformation
spread better than centrality-based blocking?

Setup. Block k nodes (or cut links) and compare the reduction in infected share, from stored simulations on TIMME and
the Pokec slice (E07), on planted DC-SBM graphs (E08, E10, E12, E16), and on links (E09). H2 (E11) shortlists nodes by
PageRank or betweenness and reorders by a misfit rule. Beta is the strength of text agreement gating in the cascade
model; TIMME's effective beta is 1.44 (E08), and real agreement beta from interactions is 0.20 to 0.29 (E14).

E07, percent of infected removed, TIMME, wc model, k=50:

| Method | Content-aware model | Content-blind model |
|---|---|---|
| random | 1.49 | 2.50 |
| degree | 8.83 | 35.54 |
| pagerank | 9.71 | 29.48 |
| betweenness | 13.99 | 32.98 |
| participation_x_pr | 11.70 | 27.54 |
| gateway (ours) | 9.22 | not run |
| gateway_struct (ablation) | 11.46 | not run |
| misfit_x_pr (ours) | 10.81 | 34.35 |
| disagreement_x_pr (ours) | not run | 41.52 |
| misfit (ours, alone) | not run | 1.11 |

Pokec slice, content-blind, wc, k=50: degree 25.10, pagerank 24.85, betweenness 23.77, misfit_x_pr 25.61,
disagreement_x_pr 19.28, misfit 8.61, random 4.85. There is no Pokec content-aware blocking file.

E08 regime map (planted, mu 0.5, beta 6, plant 10 percent, hub placement, 2 seeds), gain over the best structural
baseline in points: misfit_x_pr +1.38 [0.404, 2.36]; gateway_struct +0.027; gateway -3.31; misfit_rerank -3.21; misfit
alone -22.8. The best structural baseline removes 33.0 pts of cross-community spread. Misfit detection AUC in that
cell is 0.9106.

E09 edge-level (cross-community infection removed, difference vs bridge_struct, points):

| Method | Synthetic beta 4 | Synthetic beta 0 (control) | TIMME 40000 links |
|---|---|---|---|
| bridge_struct_x_recv (ours) | +14.9 [13, 16.8] | -7.3 [-10.2, -4.4] | -0.5 [-1.2, 0.2] |
| bridge_content | +6.1 [4.2, 8] | -10.4 [-13.4, -7.4] | -3.8 [-5.3, -2.3] |
| pr_product | not tabulated | not tabulated | +3.9 [-2.6, 10.4] |
| degree_product | not tabulated | not tabulated | -0.1 [-0.8, 0.6] |
| random | not tabulated | not tabulated | -4.6 [-6.1, -3.1] |

E10 funnel: recruiter-detection AUC, hub case: misfit 0.6486 vs degree 0.9502; non-hub case: misfit 0.5901 vs degree
0.2218 (degree is below chance there). Of 120 cells, 0 have misfit CI above 0 and 120 below.

E11 (H2). Gain is reduction in infected share relative to the best of PageRank/betweenness top-k; a "config" is one
filter rule, shortlist size and budget (plus model, mode and, synthetically, mu, placement, beta), and a positive config
has a 95 % t-interval over seeds above 0. Counts are recomputed from the raw files.

| Dataset | Metric | Configs | CI>0 | CI<0 |
|---|---|---|---|---|
| Synthetic (8 seeds) | Infected share | 6900 | 0 | 6044 |
| Synthetic (8 seeds) | Cross-community share | 6900 | 764 | 3900 |
| TIMME | Infected share | 920 | 1 | 564 |
| TIMME | Cross-community share | 920 | 6 | 559 |
| Pokec slice | Infected share | 460 | 0 | 273 |
| Pokec slice | Cross-community share | 460 | 1 | 250 |

Synthetic cross-community positives by filter family: misfit 159 of 2250, prod 351 of 2400, band 63 of 750, rankavg
141 of 750, mthenc 50 of 750. By beta: 10 gives 481, 6 gives 282, 3 gives 1. By placement: hub 595, random 169. The
median gain of the positives is +4.43 pts, the maximum +13.28 pts. Controls on the same metric: random-from-shortlist 0
of 2250; centrality-only 1 of 600. About 170 false positives would be expected by chance at a one-sided 2.5 % rate (this
expectation is the analyst's arithmetic, not a CSV value). On real data the positive rate is below chance
expectation: noise.

E12 blocking (planted, misfit_x_pr, gain vs best structural, points): realistic hub, beta 1.4: -2.15 [-2.59, -1.70];
hub, beta 6, k=2: +4.46 [3.34, 5.58]; gated hub, beta 10, k=20: +5.35 [4.16, 6.54]. Misfit helps only when gating is strong.

E14 bridge validation on TIMME (res 1.0, seed 0): misfit has a cross-community count difference of -1.6 [-17.5, 12.3] vs
matched controls (no effect), but a higher cross-community share (+0.286 [0.273, 0.299], descriptive). Spearman of misfit
with pagerank -0.0231, betweenness 0.005, degree -0.024, participation 0.2423, disagreement 0.9243. Real agreement beta
is 0.29 (retweet), 0.23 (mention), 0.20 (reply), against 6 or more needed for the blocking gains.

E16 extended sweep (planted, 120 cells, 12 seeds), infected share: misfit_x_pr (fixed) 0 cells CI>0, 120 CI<0; mean gain
by beta from -0.378 (beta 4) to -0.275 (beta 20) pts. Best-selected misfit strategy: 25 cells CI>0, 54 CI<0, maximum
cell gain +0.0647 pts [0.0211, 0.108]. Cross-community share: misfit_x_pr 48 CI>0 and 30 CI<0, mean gains at most +0.0072
pts. Holm-corrected winners on cross share: 38 (fixed) and 36 (best-selected); on infected share: 0 and 10.

Verdict. E07, E10, E14, E16: not supported. E08, E09, E12: partly supported, only at strong gating in synthetic data. E11:
mixed (see above). The blocking gate failed on real data, so the project does not support "misfit blocking beats
centrality blocking" for TIMME or Pokec.

Caveats. Gateway differences on TIMME are within noise (SE about 15 points in E07). E07 removed-percent values are
derived from stored means (older files have no SE). E08 and E05 use 2 and 3 seeds. E16 best-selected numbers are
selection biased; the fixed rule is the honest one. E09 budgets were inferred from calls in `edges.py`.
In the content-blind model disagreement_x_pr (41.52) is the top TIMME entry. This is a content-blind structural result: it does not
test misfit (misfit alone removes 1.11) and is not a positive for our method. H2 has no pre-registration (section 7).

### 4.4 Misinformation detection (E15)

Question. Does adding misfit features improve a PHEME rumour classifier?

Setup. Thread classifiers on text, structure and misfit features, stratified 5-fold and leave-event-out.

| Feature set | AUC stratified 5-fold | AUC leave-event-out |
|---|---|---|
| text only | 0.8788 | 0.6932 |
| text + structure | 0.8799 | 0.7075 |
| text + structure + misfit (ours) | 0.8793 | 0.7065 |
| misfit alone | 0.6044 | 0.4556 |

Paired delta AUC from adding misfit to text + structure: -0.0006 [-0.00125, 0.000137] (stratified); -0.001
[-0.0022, 0.000178] (leave-event-out). Univariate thread AUCs of misfit aggregates are 0.459 to 0.469, below 0.5.

Verdict. Not supported. Caveats. The test is weak: no follow graph is available, and misfit is near-constant (the count of
threads with defined misfit correlates 0.7233 with log thread size).

### 4.5 Re-homing and link recommendation (E17, E18, E19)

Question. Do cross-community links of misfit nodes land in the text-predicted community, and can that improve link
recommendation for misfits?

Setup (E17). Top-10 % misfits on TIMME follow communities (resolution 1.0, seed 0, k=4); share of held-out cross-community
retweets landing in the argmax p_C community, against size-weighted chance. A bio-only text run controls for leakage.

| Target | Hit share |
|---|---|
| argmax p_C, bio-only text, retweets | 0.601 |
| chance (size-weighted) | 0.401 |
| Leaky text (bio + latest tweet + GloVe), argmax p_C, retweets (first row): leak control, not a valid result, role=reference | 0.664 |
| Leaky text, Kumari extraction target (hand-entered from TASKS.md): leak control, not a valid result, role=reference | 0.53 |

For k=15 (r2.0), bio-only retweets: 0.278 vs chance 0.122. Ties staying in the node's own community: 0.173 for k=4 bio-only
retweets.

Graph-only targets (E17 addendum, same misfits and same held-out cross-community ties, follow layer only; engagement ties are never
used to build a predictor). Hit share, bio-only text, retweets / likes / mentions; chance in brackets. argmax p_C restricted to
non-own communities is identical to argmax p_C (misfits are defined by a text target different from their own community).

| k (run) | argmax p_C | most-linked | label prop (1 step) | argmax p_T | Adamic-Adar | chance |
|---|---|---|---|---|---|---|
| 4 (r1.0_s0) | 0.601 / 0.584 / 0.510 | 0.496 / 0.515 / 0.438 | 0.576 / 0.599 / 0.508 | 0.487 / 0.506 / 0.436 | 0.403 / 0.403 / 0.367 | 0.401 / 0.402 / 0.399 |
| 5 (r1.0_s1) | 0.543 / 0.524 / 0.451 | 0.431 / 0.448 / 0.369 | 0.509 / 0.540 / 0.443 | 0.441 / 0.455 / 0.381 | 0.345 / 0.348 / 0.301 | 0.356 / 0.357 / 0.353 |
| 15 (r2.0_s0) | 0.278 / 0.248 / 0.242 | 0.189 / 0.186 / 0.165 | 0.258 / 0.243 / 0.216 | 0.336 / 0.320 / 0.267 | 0.304 / 0.285 / 0.245 | 0.122 / 0.122 / 0.123 |

Text beats the Kumari-style most-linked target in every cell (paired user-bootstrap 95 % CI of the difference above 0: +0.062 to
+0.113). Text also beats AA and p_T at k=4 and k=5 (differences +0.07 to +0.20, CIs above 0). The text advantage is NOT universal:
(i) one-step label propagation is statistically tied with text at k=4 and k=5 for likes and mentions (likes k=4: text minus LP
-0.015 [-0.042, +0.011]; LP is nominally ahead on likes); text leads LP only for retweets (+0.025 [0.000, 0.053] k=4;
+0.034 [0.003, 0.064] k=5); (ii) at k=15, argmax p_T beats text for retweets (+0.336 vs 0.278; text minus p_T -0.058
[-0.096, -0.019]) and likes (-0.072 [-0.108, -0.036]), and Adamic-Adar is close to or ahead of text (likes -0.037 [-0.074, -0.002]).
So the claim that misfits' engagement lands in the text-predicted community is supported against structural targets at coarse
resolution, but at fine resolution (k=15) structure matches or beats text; part of the original 0.60 vs 0.40 gap is simply that
engagement ties follow follow-graph proximity.

Setup (E18, E19). Recommend links for held-out ties on TIMME (r2.0, seed 0, k=15 shown). Misfits compared with matched controls.

| Recommender | Misfits P@10 | Misfits R@100 | Controls P@10 |
|---|---|---|---|
| aa_x_text (ours) | 0.1223 | 0.0456 | not shown |
| aa+text_sim | 0.0822 | 0.0399 | not shown |
| aa (Adamic-Adar) | 0.0755 | 0.036 | 0.1389 |
| cn (common neighbours) | 0.068 | 0.0338 | 0.1306 |
| ours+aa | 0.0652 | 0.042 | not shown |
| aa_x_struct | 0.062 | 0.0196 | not shown |
| text_sim | 0.0436 | 0.021 | 0.1029 |
| struct_comm | 0.037 | 0.0238 | 0.0574 |
| ours | 0.0352 | 0.0282 | 0.0464 |
| kumari | 0.0317 | 0.0308 | 0.0416 |
| pop (PageRank) | 0.0141 | 0.0117 | 0.0227 |

Community-level top-1 hit (misfits, k=15): text target 0.265, kumari 0.153, structural community 0.0779, own community
0.0779.

Robustness (E19, 27 settings, aa_x_text minus aa on P@10): misfit group has 12 settings with CI>0, 9 with CI<0, 6
spanning 0. The control group has 15 CI>0 and 0 CI<0; so a gain also appears for non-misfits and the misfit-specific
gain group has 3 settings CI>0 and 11 CI<0. In one setting (r2.0, seed 0, b0) the paired P@10 difference is +0.0467 for
misfits and +0.0376 for controls.

Verdict. E17 supported. E18 mixed: misfits are under-served by plain Adamic-Adar (0.0755 vs 0.1389 for controls) and
the community-level text target wins, but the account-level gain is not robust. E19: not supported on P@10.

Caveats. The settings in E19 share users and are not independent; the gain is concentrated at r2.0; R@100 is favourable.
In E18 the gain is +60 percent at k=15, not misfit-specific at k=4 and a loss at k=5 (from the E18 caveat).
Kumari 0.53 and matched controls 0.69 are quoted from TASKS.md (`_gaps.md` 1).

### 4.6 Rewiring simulation (E20, E21)

Question. If dormant own-community follows are rewired toward the text-predicted community, does correction content
reach more nodes?

Setup. Cascade simulation on TIMME (r1.0, seed 0) and the Pokec slice, wc model, k=5 rewires per misfit, content-gating beta
swept (0.3, 1, 3, 6); "correction reach" is the share of nodes reached by correcting content, in points. Contrasts are
paired.

Correction reach at beta 6 (wc, k=5) and contrasts, points:

| Method | TIMME reach | TIMME vs none [95% CI] | Pokec reach | Pokec vs none [95% CI] |
|---|---|---|---|---|
| none | 34.54 | 0 | 50.34 | 0 |
| rehome_text (ours) | 39.25 | +4.72 [3.05, 6.91] | 61.26 | +10.92 [9.76, 12.1] |
| rehome_struct | 37.55 | +3.01 [1.52, 5.08] | 52.81 | +2.46 [1.69, 3.29] |
| random_rewire | 38.00 | +3.46 [1.59, 5.92] | 53.27 | +2.92 [2.36, 3.51] |
| rehome_text on random nodes | 33.93 | -0.61 [-2.11, 0.169] | 50.98 | +0.63 [0.0474, 1.31] |

rehome_text minus random_rewire at beta 6: TIMME +1.26 [-0.326, 2.22] (spans 0); Pokec +8.00 [6.87, 9.1].
At beta 0.3 the effect vanishes on TIMME (rehome_text vs none +0.104 [-0.365, 0.559]); on Pokec it is +1.48 [0.449, 2.52] vs
none but +0.394 [-0.882, 1.65] vs random_rewire. Total infected share is essentially unchanged (rehome_text vs none
+0.068 pp TIMME, +0.110 pp Pokec at beta 6).

Verdict. E21 supported at beta 3 or more. E20 partly supported. Its headline is the gain versus no rewiring (+4.72 pp), reported as a headline only because its interval excludes 0. The part attributable to text targeting, rehome_text minus random_rewire (+1.26 [-0.326, 2.22]), is not distinguishable from generic rewiring in this cell.

Caveats. Beta is swept, not measured; selective exposure is assumed. TIMME's measured agreement beta (0.20 to 0.29, E14) is
in the null region (beta 0.3). The TIMME text-over-random-rewire residual in the headline cell spans 0; the README counts
CI>0 in 4 of the 7 beta>=3 cells shown. Intervals exclude rewiring-draw variance (`_gaps.md` 12). One TIMME resolution and
seed in the headline.

### 4.7 Deception detection (E22)

Question. Does misfit flag nodes whose edges were rewired to hide them from community detection (simplified Kumari 2021
attack, safeness attack, random and noise controls)?

Setup. 200 targets per seed, 3 seeds, rewiring budget k of 2 or 5, TIMME and Pokec slice. Score: ROC-AUC of target versus non-target.

TIMME, kumari k=2, AUC [95% CI]:

| Method | All non-targets | Successful vs failed targets |
|---|---|---|
| misfit | 0.487 [0.475, 0.498] | 0.713 [0.702, 0.724] |
| participation | 0.541 [0.523, 0.559] | 0.560 [0.524, 0.596] |
| persistence_low | 0.542 [0.532, 0.551] | 0.378 [0.36, 0.396] |
| consensus_entropy | 0.505 [0.504, 0.506] | 0.648 [0.622, 0.674] |
| degree | 0.534 [0.51, 0.558] | 0.217 [0.208, 0.225] |

Misfit AUC with negatives restricted to successful targets (`negatives=succ`): 0.640; with `negatives=succ_unatt`: 0.465.
Matched controls, misfit AUC on successful targets (`negatives=succ`), all from `h6_timme.json` / `h6_pokec.json` (rows in E22):

| Attack | TIMME k=2 | TIMME k=5 | Pokec k=2 | Pokec k=5 |
|---|---|---|---|---|
| kumari | 0.640 | 0.655 | 0.603 | 0.598 |
| random attack (matched control) | 0.663 | 0.679 | 0.623 | 0.660 |
| noise attack | 0.587 | 0.452 | 0.566 | 0.462 |

The random control matches or exceeds kumari in all four cells, so there is no attack-specific signal. Successful-vs-failed (`succ_vs_fail`) gives the same pattern: TIMME k=2 kumari 0.713 vs random 0.736; Pokec k=2 0.672 vs 0.686.
Attack success rates, TIMME k=2: kumari 0.232, safeness 0.187, random 0.087, noise 0.057 (Pokec k=2: 0.373, 0.277, 0.158, 0.143).

Verdict. Not supported (README, revised). The apparent success on successful targets is circular: success is a label flip, and
misfit rises when the label changes. The matched random-rewire control gives the same misfit AUC (table above). See
`results/hypotheses/h6/README.md`.

Caveats. Simplified attacker, one resolution, 3 seeds sharing one t0; mean and interval over 3 seeds only.

## 5. Methods compared

Names are as in `metrics_glossary.csv` (kind not "metric"). Kinds: method = ours or a tested approach, baseline,
control, term. The glossary has 273 rows in total.

Ours and tested methods.
- misfit: one minus the text-predicted probability of a node's own graph community; high means text disagrees with placement.
- disagreement: Jensen-Shannon divergence between structural and text community membership. disagreement_x_pr and misfit_x_pr multiply by PageRank.
- misfit_rerank: PageRank shortlist reordered by misfit. misfit_nonhub: misfit restricted to non-hub accounts. misfit_x_lowdeg: misfit times an inverse-degree term.
- gateway: node score of cross-community links weighted by receiver text membership (gateway_struct is its no-text ablation).
- text classifier p_C: cross-fitted text-only classifier giving community membership; variants by text source (MiniLM, GloVe, interest text).
- T Leiden: topology partition, best of 100 Leiden runs. C text kNN Leiden: Leiden on a text kNN graph.
- aa_x_text: Adamic-Adar times text membership p_C. ours: p_C times PageRank recommender. ours+aa: rank average. aa+text_sim: Adamic-Adar plus bio similarity.
- bridge_struct_x_recv: structural bridge score times receptivity. bridge_content: bridge score with content-based receptivity.
- rehome_text: rewire dormant own-community follows toward the text-predicted community (degree-preserving).
- BGRL, DGI, MVGRL: self-supervised GNN embeddings (bootstrapped latents; local-global infomax; adjacency and PPR-diffusion views).
- H2 filter rules (misfit, prod, band, rankavg, mthenc): shortlist by centrality, then choose by the named rule. Link-level variants are named `link <size>|<rule>`.
- PHEME feature sets: text only, text + structure, text + structure + misfit, misfit alone.

Baselines.
- Centrality: degree, pagerank, betweenness (btw), collective_influence, adaptive_degree (iteratively remove highest degree), low_degree.
- Structural scores: participation, participation_x_pr, consensus_entropy (and _x_pr), ambiguity_T (and _x_pr), persistence_low, topo_ambT, topo_participation.
- Text scores: content_outlier (text far from own-community centroid), content_outlier_x_pr, text_iso, text_sim, knn_dup.
- Link scores: aa (Adamic-Adar), cn (common neighbours), pop (PageRank), degree_product, pr_product, struct, struct_comm, aa_x_struct, bridge_struct, kumari (neighbouring community with most of the node's edges, times PageRank), text_toB, edges_into_B and related H3 delivery strategies.
- Partitions and probes: T regularised spectral + k-means, C text k-means, spectral probe, spectral64 probe, neighbour-vote, log-degree-only probe, text-only probe, structural community (argmax p_T), own follow community, age + gender + completion, interest text, text+profile.
- Rewiring: rehome_struct (rewire toward the structural/Kumari community).
- Upper bounds (reference): oracle_recruiters, bridge_oracle, bridge_struct_x_recv_ORACLE, real graph.

Controls.
- random; no blocking / none; majority class; chance (size-weighted); matched controls (low-misfit nodes matched on degree and participation decile); random-from-shortlist; random_rewire (same drop counts, random adds); rehome_text_random_nodes; rewired null; misfit_SHUFFLED_TEXT and disagreement_SHUFFLED_TEXT; random attack, noise attack (Leiden-instability floor), safeness attack, kumari attack (attack models in H6).

## 6. Metrics

- modularity Q: edges inside communities minus the random expectation; higher means stronger structure.
- NMI, ARI: agreement of a clustering with reference labels (NMI normalised mutual information, ARI chance-corrected Rand). nmi_T_vs_C and ari_T_vs_C compare the topology and text partitions.
- AUC: probability a random positive outranks a random negative; 0.5 is chance. auc_within_degree and _terciles score inside degree strata so degree in disguise cannot win.
- precision_at_k, P@10, R@100: share of top-k recommended items that are true ties; recall of held-out ties in the top 100.
- text_to_community_acc: accuracy of a text-only classifier at predicting graph community; compared with majority class.
- infected_share, infected_removed_pct: fraction of nodes reached by a simulated cascade; percent reduction versus no blocking.
- gain_vs_best_structural_pts, removed_pts, diff_vs_bridge_struct_pts: percentage-point differences against a structural baseline, paired over common random numbers.
- cross_community_infected_share, cross_spread_removed_pts, cross_community_edge_share_pct: spread or edges across community boundaries.
- n_configs, n_cells, n_settings and their _ci_above_0 / _ci_below_0 counts: how many configurations, cells or settings have a 95 % interval entirely above or below 0. Holm-surviving counts apply a multiple-comparison correction.
- agreement_beta: log-odds coefficient of text agreement in a case-control logit of observed interactions; effective_agreement_beta is the TIMME-implied gating.
- correction_reach_pts and *_contrast_vs_{none, random_rewire}_pp: rewiring outcomes and paired contrasts in points.
- deception_success_rate: fraction of attacked targets whose community label changed.
- delta_auc_paired: change in AUC from adding a feature, paired bootstrap.
- cross_ties_in_target_share, community_top1_hit: share of held-out cross-community ties (or ties) landing in the target community.
- Collapse and structure diagnostics for embeddings: emb_std, effective_rank, degree_eta2 (share of log-degree variance explained by clusters), corr_norm_logdeg, conductance_mean, knn10_label_agreement, silhouette_true_labels, seed_stability_ari.

How uncertainty is reported (the `uncertainty` column of the long file).
- `ci95_t_over_seeds`: two-sided 95 % t-interval over seeds (t critical values hard-coded for 1 to 30 degrees of freedom). Used for H2, H3, H4a, H4d, E10 and others. Seed counts range from 2 to 12 and are small; intervals with 2 or 3 seeds are wide and fragile.
- Paired-bootstrap or permutation intervals where the source stored them (E06, E14, E15, E18, E19, E20, E21).
- `se_plus_minus_1`: value plus or minus one standard error (E07 blocking).
- none: single run or seed-averaged values with no stored spread (E02, E05, E08 means, E01 non-bootstrapped cells).
- A "CI above 0" count is a count of intervals, not a corrected hypothesis test; with many configurations some positives are expected by chance (about 2.5 % one-sided).

## 7. Known gaps and caveats

From `_gaps.md`, with two corrections noted.

1. H2 correction (E11). The H2 README previously said all synthetic cross-community CIs include 0. That was wrong: 764 of
   6900 synthetic configs have CI above 0 on cross-community share (0 of 6900 on infected share). The README's "0 of 1550"
   count could not be reproduced and is a narrower count of the same infected-share conclusion. README corrected 2026-11-18;
   `build_results.py` now carries per-family, per-beta, per-placement counts and the two controls. The pooled
   README verdict is now: not supported on infected share; synthetic cross-community share improves at strong content gating
   (beta 6 or more); real data: noise. The earlier TIMME "7 of 1840" figure is the sum of 1 (infected) and 6 (cross) of 920 each.
2. H6 revision (E22). The first H6 reading (misfit detects hidden targets among successful ones) was circular, since success
   is defined as a label flip. The revised verdict is: not supported. See `results/hypotheses/h6/README.md`.
3. Hand-entered numbers: E17 Kumari 0.53 and matched-control 0.69 are quoted from TASKS.md (`h5_probe.py` does not compute them).
4. E11 link-level variant (`edges.json`) has no README verdict; rows are aggregates only.
5. E14: agreement-gating raw JSON (`gating.json`) was never saved; values were parsed from `gating.log`. The extra, extra2 and grid files are aggregated (median, CI counts), not row-for-row.
6. E07: no Pokec content-aware blocking file exists; older content-blind files have means only (no SE).
7. E09: only text logs exist; budgets and the synthetic edge model (ic:2) were inferred from `edges.py`.
8. E05, E08: seed-averaged values, no per-seed spread stored. E10: per-seed arrays not stored. E02: single run.
9. E20, E21 intervals exclude rewiring-draw variance; E19 settings share users and are not independent.
10. Not run: T2.6, T3.5 to T3.7, T7.1 to T7.4, T8, T10.10, T10.12, T11.3, and the full-Pokec T10 pipeline.
11. Units: gains are in points. The H2 README quotes +4.6 +- 3.5 and +0.26 +- 0.09 where raw JSON gives 4.6 +- 3.8 and 0.3 +- 0.1 (rounding or interval method).
12. Statistical practice: no pre-registration anywhere; many configurations and settings per experiment; most synthetic results use planted effects that match what the method measures; few real-data positives; seeds 2 to 12.
13. Code was not audited line by line (stated for E14).

## 8. Open questions for the user

1. E06 and E09 verdicts (partly supported) are the builder's mapping; TASKS.md has no explicit label. Should E06 be "supported" given p 0.019, or "mixed" given 11 positives?
2. Which definition of an H2 "config" did the original H2 author use? The "1550" total is unexplained and not recoverable from the files; this document uses 6900 per metric (and 2300 averaged over mu).
