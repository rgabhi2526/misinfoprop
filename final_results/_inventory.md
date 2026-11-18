# Experiment inventory (read-only; ground-truth for final_results table)

Paths are relative to repo root `Major project /`. "CI" conventions are per experiment. Verdicts are copied from README/TASKS, not reinterpreted.
Shared terminology: **misfit** = 1 - p_C[t_i] (text-predicted probability of node's own graph community); **t** = topology community (Leiden); **p_C** = cross-fitted text-only classifier membership; **p_T** = label-spreading structural membership. See glossary.
Common data: TIMME `P_all` (Twitter, 20,811 nodes LCC, real), Pokec slice `pokec_knm` (5,599 nodes, 11 comms, real), planted DC-SBM synthetic (n=4000, 8 comms; synthetic), PHEME (real reply threads).
Files `results/hypotheses/h*/` are authoritative; `training/out/*.json` duplicate some `results/*` copies.

---
## E01  T7.0 Modularity vs rewired null (structure check)
- Question: do these graphs have real community structure (Q far above degree-preserving rewired null)?  Use case: dataset characterisation.
- Data: TIMME pooled + per relation, Pokec slice, full Pokec (1.63M); real.
- Method: Louvain modularity vs degree-preserving rewired null (`training/check_modularity.py`). No competitors.
- Metrics: modularity Q (real vs null), cross-community edge share, #communities. Uncertainty: +- (sd over runs) in log.
- Files: `results/logs/modularity_timme.log`, `modularity_pokec_slice.log`, `modularity_pokec_full.log` (text logs, parse lines "real Q = x ± y ... cross-community edges z%"). Numbers also in DEVELOPED.md section 5 table.
- Headline: TIMME Q 0.26 (null 0.04), cross-edges 50% (null 84%); Pokec slice 0.49 (0.21), 40%/67%; Pokec full 0.72 (0.12), 22%/71%. "Real communities in all."

## E02  T2/T3 GNN embeddings BGRL / DGI / MVGRL (embedding quality; superseded)
- Question: do GNN self-supervised embeddings (BGRL/DGI/MVGRL, shared 2-layer GCN) give meaningful clusters / disagreement input?  Use case: embedding quality.
- Data: TIMME (GPU run 2026-09-28; real). Reddit also trained (`training/out/reddit/`, modularity only). Pokec never trained (T2.6 stale/open). Gab/VoterFraud not run.
- Methods: BGRL, DGI, MVGRL (+ diagnostics: spectral64 probe, neighbour-vote, L2/standardised probe, degree regress-out: `results/scripts/diag*.py`).
- Metrics: modularity Q of k-means clusters (k=2), NMI/ARI vs party, linear probe acc/F1 macro, kNN10 label agreement, silhouette(true), conductance, edge-reconstruction AUC, seed-stability ARI, emb_std, effective rank, norm-degree corr, degree eta^2. No CIs (single run).
- Files: `results/gnn_embeddings/timme_gnn_metrics_2026-09-28.json` = {bgrl|dgi|mvgrl}: {modularity, nmi, emb_std, best_epoch, seconds, report:{health,unlabeled,labeled,task}}; `report.labeled.{probe_acc,probe_f1_macro,nmi,ari,knn10_label_agreement,silhouette_true_labels}`, `report.unlabeled.{modularity,conductance_mean,edge_reconstruction_auc,seed_stability_ari}`. Copies: `training/out/timme/timme_metrics.json`, `timme_report.json`, `training/out/reddit/reddit_metrics.json`, `reddit_report.json`. Logs: `results/logs/gnn_degree_diagnostics.log`, `gnn_cluster_after_removing_degree.log`, `gnn_fair_probe_and_baselines.log`.
- Headline: modularity BGRL -0.007 / DGI -0.009 / MVGRL -0.015, NMI ~0 (probe acc 0.744/0.802/0.748 with old biased probe). Fair probe (standardised) 0.95/0.96/0.77 vs neighbour-vote 0.928, spectral64 0.961 (=> not better than graph baselines). "GNN clusters Q <= 0 / clusters = degree"; plan replaced by T10 (S-23). All pre-2026-09-27 numbers superseded. Caveat: old probe biased (metrics._labeled fixed later).

## E03  T10.1/T10.2 Two clean views (topology vs text)
- Question: can a graph-only partition (T) and a text-only partition (C) both be built cleanly, and do they differ?  Use case: view construction.
- Data: TIMME, Pokec slice; real.
- Methods: T = Leiden x100 on social graph ("best of runs"), regularised spectral + k-means; C = text kNN-graph Leiden x100, text k-means. (`training/views.py`)
- Metrics: modularity Q on social graph, party NMI/ARI (TIMME only, not used to select), degree_eta2, #clusters/largest share, seed stability (ARI across runs), T-vs-C NMI/ARI.
- Files: `results/logs/views_timme.log`, `views_pokec.log` (text); arrays `training/out/timme_views/*`, `pokec_knm_views/*` (npy/npz; not in results/).
- Headline: TIMME Leiden k=5 Q 0.264, NMI 0.288, stab 0.77; spectral Q 0.059; text-Leiden Q 0.005 (rejected); T-vs-C NMI 0.066. Pokec Leiden k=17 (11>=1%) Q 0.505; text Q 0.024; T-vs-C NMI 0.016.

## E04  T10.3/T10.5 Content informativeness (text -> graph community) + text source comparison
- Question: how well does text predict graph community? (is the content signal informative?)  Use case: content informativeness.
- Data: TIMME (MiniLM bio+latest tweet, GloVe all tweets, both), Pokec slice (interest text, age/gender profile); real.
- Methods: cross-fitted text classifier p_C (`training/ambiguity.py`); text-only clustering comparison (`results/scripts/content_src.py`); Pokec profile-field probe (`scripts/pokec_meta.py`).
- Metrics: classifier accuracy vs majority-class baseline; probe acc; text clustering Q/NMI/eta2/stab.
- Files: `results/logs/amb_minilm.log` (MiniLM-only acc 0.496 vs 0.318), `block_gateway.log` (GloVe-only acc 0.637), `timme_party_check.log` (acc 0.275 vs 0.318 on a different text config), `text_only_clustering_timme.log` (MiniLM probe 0.700 / GloVe 0.841 / both 0.828; Q ~0.005-0.012), `pokec_profile_fields.log` (age+gender 0.314, interest text 0.289, text+profile 0.356; majority 0.249), `pokec_text_cap_check.log`. Per-node scores: `results/scores/timme_scores_minilm_glove.npz`, `timme_scores_minilm_only.npz`, `pokec_slice_scores.npz` (npz keys: t,p_T,p_C,misfit,disagreement,ambiguity_T,consensus_entropy,participation,content_outlier,pagerank,content_acc,content_majority).
- Headline (TASKS T10.5): TIMME 0.62 vs 0.32 chance; Pokec 0.28 vs 0.25 (profile+text 0.36). NOTE: the "0.62" is not a single log line; logs show 0.496 (MiniLM), 0.637 (GloVe only) -> see Gaps.

## E05  T10.4 Synthetic planted-mismatch benchmark (detection)
- Question: can a score recover nodes whose text says community A while edges say B?  Use case: detection (synthetic).
- Data: planted DC-SBM (power-law deg 2.5, K=8, text = unit(centroid + sigma*noise), 64-d); 18 settings = nuisance {0,1.5} x mu {0.1,0.3,0.5} x sigma {0.5,1,2}; 3 seeds each; synthetic.
- Methods (ours): misfit, disagreement. Baselines: consensus_entropy, participation, content_outlier, degree. (`training/benchmark.py`; a rejected absorbing-random-walk variant is documented in ambiguity.py docstring: AUC 0.64.)
- Metrics: AUC overall and within degree terciles. Uncertainty: pair [value_a, value_b] per score (as stored; see JSON) over seeds.
- Files: `results/detection/synthetic_benchmark.json` (= `training/out/benchmark.json`): list of 18 dicts {nuisance, mu, sigma, T_ari, auc:{misfit|disagreement|consensus_entropy|participation|content_outlier|degree: [overall, within-degree]}}. Log: `results/logs/bench.log`.
- Headline: misfit AUC 0.73-1.00 vs content-outlier 0.53-0.85 under nuisance; structure-only 0.50. (e.g. nu=1.5,mu=0.5,sigma=2: misfit 0.74, content_outlier 0.53.)

## E06  T10.6 TIMME real-data party validation
- Question: do high misfit nodes sit in the "wrong" community for their party (R/D)?  Use case: detection (real).
- Data: TIMME labeled nodes in two party-core communities (595 labeled; 11 party misfits). Real.
- Methods: misfit, disagreement; baselines participation, consensus_entropy, content_outlier, ambiguity_T, degree, pagerank; null = shuffled text (misfit_SHUFFLED_TEXT, disagreement_SHUFFLED_TEXT). (`training/validate.py`)
- Metrics: AUC overall, within-degree AUC, one-sided p.
- Files: `results/logs/timme_party_check.log` (text; lines "name AUC x within-degree y one-sided p z").
- Headline: misfit AUC 0.682 (within-deg 0.744, p 0.019) vs shuffled 0.541; disagreement 0.666; degree 0.361; pagerank 0.360. Only 11 positives (TASKS: "misfit AUC 0.68 vs 0.54 shuffled, only 11 positives").

## E07  T10.7 Blocking on real graphs (content-blind + content-aware IC; gate)
- Question: does blocking high-misfit/disagreement nodes stop spread better than structural targeting? (go/no-go gate)  Use case: blocking / misinformation spread.
- Data: TIMME (100 sims), Pokec slice (200 sims); real graphs, simulated spread.
- Spread: live-edge IC; 'wc' weighted cascade p=1/deg(v); 'ic:c' uniform p = c*threshold (c in 0.5,1,2,4); content-aware = GloVe affinity acceptance with MiniLM scores. Source unknown; common random numbers.
- Strategies: random, degree, adaptive_degree, pagerank, betweenness, collective_influence, participation, participation_x_pr, consensus_entropy_x_pr, ambiguity_T_x_pr, content_outlier_x_pr, disagreement, disagreement_x_pr, misfit, misfit_x_pr (content-blind); aware set adds gateway, gateway_struct. (`training/blocking.py`)
- Metrics: infected share, cross-community infected share, "% of infected removed", budgets k. Uncertainty: SE (cell = [mean infected share, mean cross share, SE infected, SE cross]; older files first two only).
- Files: `results/blocking/timme_content_blind.json` {model(wc|ic:0.5|1|2|4): {strategy: {k("20","50","100","200"): [inf, cross, se_inf, se_cross]}}, "none" key = no blocking}; `timme_content_aware.json` {wc|ic:2: {strategy: {k("50","100","200"): [...]}}}; `pokec_slice_content_blind.json` (k "10","25","50","100"). Logs `block_timme.log`, `block_timme_aware.log`, `block_gateway.log`; copies `training/out/*_views/blocking*.json`.
- Headline: **gate failed**: betweenness/PageRank win on TIMME and Pokec slice; content-weighted gateway +1-2 points, inside noise (e.g. gateway 9.2% vs betweenness 14.0% at k=50, wc, ±~15).

## E08  T10.8/T10.9 Node-level regime map (synthetic) + TIMME placement
- Question: when does mismatch-aware node targeting beat the best of PageRank/betweenness?  Use case: blocking (regime boundary).
- Data: planted DC-SBM; 72 settings x 2 seeds: mu {0.1,0.3,0.5} x beta {0,1,3,6} x plant {1,5,10%} x place {random,hub}; synthetic. TIMME positioned on the map (agreement effect, degree percentile).
- Methods: misfit, misfit_x_pr, gateway, misfit_rerank, gateway_struct vs best structural (PageRank/betweenness). (`training/regime.py`, `results/scripts/place.py`)
- Metrics: paired gain (points) in cross-community spread removed vs best structural, 95% CI; detection AUC.
- Files: `results/regime/node_regime_map.json` (= `training/out/regime.json`): list of 72 dicts {mu,beta,plant,place,detect_auc,best_struct_removed, misfit|misfit_x_pr|gateway|misfit_rerank|gateway_struct: [gain, ci]}. Logs: `regime.log`, `timme_position_on_regime_map.log`.
- Headline: content targeting wins only at strong agreement effect (beta 6) AND hub-placed mismatches (6/72 cells, +1.4..+8.9 points). TIMME: effective beta ~1.44 (docs also quote ~1.4), mismatches at 56th degree percentile -> losing region.

## E09  T10.11 Edge-level link friction
- Question: does adding friction to links (structural bridge score x receptivity) beat structural edge scores?  Use case: blocking (edge-level).
- Data: synthetic planted graph (mu 0.3, 5% hub mismatches, beta 0/4, 4 seeds) and TIMME (content-aware spread, GloVe affinity, MiniLM scores, + ORACLE receptivity); synthetic + real.
- Methods: bridge_struct_x_recv (ours), bridge_content, bridge_oracle. Baselines: random, degree_product, pr_product, bridge_struct. (`training/edges.py`)
- Metrics: % of cross-community infection removed, paired 95% CI vs bridge_struct.
- Files: logs only: `results/logs/edge_seeds.log`, `edge_v2.log`, `edge_oracle.log`, `edges_timme.log` (text). Scripts `results/scripts/edge_oracle.py, edge_seeds.py, edge_v2.py`. No JSON.
- Headline: synthetic beta=4 bridge_struct_x_recv +14.9 +- 1.9 vs bridge_struct (control beta=0: -7.3 +- 2.9; bridge_content +6.1 +- 1.9 / -10.4 +- 3.0); TIMME tie even with oracle receptivity (+0.1 +- 0.6 / -0.5 +- 0.7).

## E10  H1 Radicalisation-funnel blocking (synthetic)
- Question: can misfit find recruiters to block in a radicalisation funnel?  Use case: blocking / deradicalisation defence.
- Data: synthetic, n=4000, 6 mainstream comms + 10% fringe + 5% core; recruiters 1% (hub / non-hub), decoys 1-3%; 30 settings x 4 budgets (0.5/1/2/5%), 8 seeds, 30 cascades; synthetic.
- Methods: misfit, misfit_x_pr, gateway. Baselines: degree, pagerank, betweenness (best-of structural), random; oracle_recruiters (upper bound). (`training/h1_funnel.py`)
- Metrics: harm removed share, gain vs best structural, recruiter-detection AUC (misfit vs degree). CI = t-interval over seeds.
- Files: `results/hypotheses/h1/h1_results.json`: list of 30 {name, params, n_seeds, base_harm, det:{auc_misfit_recruiter, auc_degree_recruiter}, budgets:{"0.5%"|"1%"|"2%"|"5%": {best_struct, best_struct_removed, removed:{strategy:val}, misfit:[gain,ci], misfit_x_pr:[..], gateway:[..], random:[..], oracle_recruiters:[..]}}}. Logs `h1_run.log`, `h1_progress.log`; README.md has tables.
- Headline: NOT SUPPORTED. 0/120 cells CI>0 for misfit; misfit CI<0 in 120/120; recruiter AUC misfit 0.59-0.65 vs degree 0.95 (hub case).

## E11  H2 Misfit as filter/re-ranker over centrality shortlist (synthetic + real, node + edge)
- Question: does misfit improve a PageRank/betweenness shortlist?  Use case: blocking.
- Data: synthetic planted (beta 0-10, hub/random, mu 0.1-0.5), TIMME, Pokec slice; link-level variant synthetic (beta 1,3,6,10; k 40,200,800).
- Methods: shortlist(top-M by pr/btw/degree/union) then choose k by misfit / rankavg / prod / band / rand (control) / mthenc; edge variant: misfit, prod, prod_misfit, rankavg, recv, rand vs struct. Baseline: best of PR/btw top-k. (`training/h2_filter.py, h2_real.py, h2_edges.py, h2_analyze.py`)
- Metrics: gain (points of infected share and cross-community share) vs best structural, 95% CI t-interval over seeds.
- Files: `results/hypotheses/h2/synth.json` (35 MB; list of 156000 {mu,place,seed,beta,metric(inf|cross),name,removed,gain,best_struct,base,pool,...}); `real_timme.json` (15600; {ds,model(wc|ic:2),mode(blind|aware),seed,metric,name,removed,gain,best_struct,base,pool}); `real_pokec_knm.json` (10400); `edges.json` (1920 {seed,beta,k,name,removed,gain,base}); older `synth_v1.json`, `real_pokec_v1.json`, `synth_quick.json`, `jobs/`, `jobs_v1/` (per-job shards); tables `tables_synth_inf.md`, `tables_synth_cross.md`; npy betweenness caches. README.md.
- Headline: NOT SUPPORTED. Synthetic: 0/1550 configs CI>0; real TIMME 7/1840 configs CI>0 (best +4.6±3.5 cross, +2.7±2.1 inf); Pokec 1/920 (+0.26±0.09 cross). Edge variant (`edges.json`) has no README verdict (see Gaps).

## E12  H3 Scenario ranking a-g (synthetic)
- Question: in which scenarios could misfit targeting help (gated, niche harm, protected hubs, regular graphs, multi-source, delivery, detection)?  Use case: blocking + detection + delivery.
- Data: planted DC-SBM n=4000, 8 comms, mu 0.3, 8 paired seeds, 60 sims, budgets k=20,40; synthetic. 22 scenarios.
- Methods: misfit, misfit_x_pr, gateway, misfit_rerank vs pagerank, betweenness, degree, collective_influence, adaptive_degree; detection scenario also content_outlier, disagreement, participation, random. (`training/h3_scenarios.py`, `h3_report.py`)
- Metrics: gain (points removed share) +- 95% CI (t over seeds), "wins x/8", fixed misfit_x_pr gain, detection AUC + precision@20.
- Files: `results/hypotheses/h3/h3_results.json`: {scenario ("a_gated_hub_b10","e_detect_random",...): {cfg:{kind,beta,place,...}, res:{"k=20"|"k=40": {best_misfit,best_struct,misfit_val:[m,ci],struct_val,gain:[m,ci],wins,n_seeds,all:{strategy:[m,ci]}}}}}; detect scenarios: res.auc{score:[m,sd]}, res["prec@20"]{...,gain}. `h3_table.md`, `h3_run.log`, README.md.
- Headline: only detection clearly works (precision@20 +36..+76 pts; AUC 0.92-0.94); blocking needs beta>=6 + hub-placed/no-hub; at realistic beta 1.4 loses (-2.1±0.4). Delivery: structural "edges into target community" wins.

## E13  H4a Infiltrator / campaign / recruiter detection with realistic text (synthetic)
- Question: when does misfit detect bad accounts given realistic noisy text?  Use case: detection.
- Data: DC-SBM n=4000, K=8; text acc calibrated 0.21-0.90; sweep (63 cells x 8 seeds) + main (24 configs x 10 seeds); synthetic.
- Methods: misfit, disagreement, misfit_nonhub, misfit_x_lowdeg. Baselines: degree, pagerank, betweenness, text_iso, topo_participation, topo_ambT, low_degree, knn_dup. Kinds: infil, camp, rec; placement random/hub/peripheral. (`training/h4a_detect.py`, `h4a_report.py`)
- Metrics: AUC (and precision@k), gain = misfit - best baseline with paired CI over seeds.
- Files: `results/hypotheses/h4a/h4a_main.json` {config ("acc0.6|base","acc0.6|kind=camp",...): list of 10 {P,seed,acc,degratio,res:{score:[AUC,prec@k]},sec}}; `h4a_sweep.json` {"s=0.5,m=0.2": list of 8 same}; `tables_main.md`, `tables_sweep.md`, logs, README.md.
- Headline: conditional WORK: infiltrators w/ random placement, gain AUC +0.05..+0.37 (text acc 0.56; needs acc>=~0.45 & mix>=0.4); campaigns +0.06..+0.35; recruiters/hub/peripheral LOSE (pagerank 1.00, topology ambiguity 0.98). Misfit AUC 0.77 vs best baseline 0.51 at acc 0.61.

## E14  H4b Real TIMME interactions: are misfit nodes special bridges?
- Question: do misfit nodes generate more cross-community interactions / does real agreement gating match blocking assumptions (beta)?  Use case: bridge validation, spread realism.
- Data: TIMME; communities from FOLLOW edges only (res 0.5/1/2 x seeds 0-2); outcomes from retweet/mention/reply/like (+pooled "spread"); real.
- Methods: misfit, disagreement vs pagerank, betweenness (btw), participation; matched controls (degree decile, degree x participation); permutation tests; Poisson regression; case-control logit for agreement beta_hat. (`training/h4b_*.py`)
- Metrics: cross_n, cross_share, in_cross_share, cross_in (diff vs matched controls with 95% CI, p_one/p_two), Spearman, log-odds agreement beta (retweet/mention/reply/like/follow).
- Files: `results/hypotheses/h4b/primary.json` {n_pop,k,sizes,spearman:{misfit~pagerank..},q1q2q3:{group:{score:{outcome:{diff,ctrl_mean,lift,ci,p_one,p_two,n_high,n_ctrl}}}},circular_pooled}; `extra.json` (120), `extra2.json` (96), `grid.json` (945 rows {res,seed,group,score,outcome,k,diff,ctrl_mean,lift,ci,p_one,p_two,n_high,n_ctrl}); `gating.log` (agreement beta lines, text), `summary.txt`, `scores.log`; arrays `training/out/h4b_*.npz/npy`. README.md.
- Headline: verdict NO. Misfit not a centrality proxy (Spearman PR -0.02); cross count diff -1.6 [-17.5,+12.3]; cross-in lower (-33 [-45,-19]); cross SHARE higher (+0.29, +0.14 after controls); real agreement beta ~0.18-0.33 (retweet 0.29-0.33) vs >=6 needed (~20x too small).

## E15  H4c Rumour detection feature on PHEME
- Question: does a per-account text-vs-neighbourhood misfit feature help rumour vs non-rumour thread classification?  Use case: misinformation detection.
- Data: PHEME all-rnr-annotated-threads (6425 threads, 9 events, 50,593 users, reply graph 63,820 edges); real. (Twitter15/16, VoterFraud, Gab, Reddit scoped out as unusable.)
- Methods: HistGradientBoosting on (a) source text only, (b) text+structure, (c) +misfit features (7 aggregates), (d) misfit alone; misfit via TF-IDF+SVD(64) cross-fitted centroid softmax. (`training/h4c_pheme.py`)
- Metrics: AUC [95% bootstrap CI over threads, 1000x], F1, paired-bootstrap delta AUC; stratified 5-fold and leave-event-out.
- Files: `results/hypotheses/h4c/h4c_pheme_results.json`: {"stratified5/text"|"/struct"|"/struct+misfit"|"/misfit"|"/delta text->struct"|"/delta struct->struct+misfit", "leave_event_out/...": {auc, auc_ci, f1}; also univariate AUCs, corr_ndefined_logsize, meta}; `h4c_pheme.log`, README.md.
- Headline: NO. delta AUC -0.0006 [-0.0013,+0.0001] (stratified), -0.0010 [-0.0022,+0.0002] (leave-event-out); misfit alone 0.604 / 0.456; baseline leave-event-out AUC 0.708. Test weak (no follow graph, misfit ~constant).

## E16  H4d Extended node-level regime sweep (beta up to 20)
- Question: does extending beta to 4-20 with more seeds/baselines rescue misfit-guided blocking?  Use case: blocking.
- Data: synthetic DC-SBM; mu {0.3,0.5} x plant {1,5%} x place {hub,random} x beta {4,6,8,12,20} x budget {40,80,200} = 120 cells, 12 seeds, 60 sims; synthetic.
- Methods: misfit, misfit_x_pr, gateway, misfit_rerank vs pagerank, betweenness, degree, collective_influence, adaptive_degree. (`training/h4d_sweep.py`, `h4d_analyze.py`)
- Metrics: gain (points) infected / cross-community share, 95% CI, Holm-corrected; best-selected vs fixed misfit_x_pr.
- Files: `results/hypotheses/h4d/h4d_cells.json`: list of 240 {mu,plant,place,beta,budget,metric(infected|cross),n,best_mis,best_str,mis_removed,str_removed,base,gain,gain_ci,gain_p,fix_gain,fix_ci,fix_p,seed_gains}; `h4d_raw.jsonl` (480 records), `h4d_table.md`, `h4d_summary.txt`, `h4d_surface.txt`, crashed-run files (`*_crashed*`), README.md.
- Headline: statistically visible, practically negligible. Fixed misfit_x_pr CI>0 in 0/120, CI<0 in 120/120; best-selected CI>0 in 25/120 but gain <= +0.06 +/- 0.04 points. "Blocking = tested, negative extension."

## E17  T11.0 Re-homing probe (h5_probe)
- Question: do misfits follow one community but engage with their text-predicted one?  Use case: re-homing/recommendation motivation.
- Data: TIMME; follow-graph communities; held-out retweet/like/mention; bio-only MiniLM (leak control: 34% of latest tweets are RTs); real.
- Methods: top-10% misfits vs matched controls; text target argmax p_C vs chance vs Kumari extraction target. (`training/h5_probe.py`)
- Metrics: share of cross-community retweets landing in argmax p_C (k=4: 0.66 vs 0.40 chance vs 0.53 Kumari; bio-only 0.60 vs 0.40; k=15 0.28 vs 0.12); share staying in own follow community 13% (controls 69%).
- Files: **no raw results file found** (see Gaps). Bio embeddings only: `training/out/timme_bio.npy`, `timme_bio_lcc.npy`. Numbers quoted in TASKS.md T11.0 and DEVELOPED.md change log 2026-11-11.
- Headline: survives bio-only text.

## E18  T11.1 Re-homing as link recommendation (h5_rec)
- Question: are misfits under-served by follow graph, and does a text-aware recommender beat structure-only ones?  Use case: recommendation.
- Data: TIMME follow graph, 3 settings r1.0_s0 (k=4), r1.0_s1 (k=5), r2.0_s0 (k=15); held-out engagement ties u->v with u not following v; ~1.5k misfits vs 1:1 controls matched on degree decile x participation decile; real.
- Methods: ours (p_C(u)[t_v]*PR(v)), ours+aa, aa_x_text (AA*p_C), aa+text_sim. Baselines: pop (PageRank), cn (common neighbours), aa (Adamic-Adar), struct_comm (p_T x PR), aa_x_struct, kumari (structural extraction target), text_sim (cosine of bios). (`training/h5_rec.py`)
- Metrics: P@10, R@100; share of held-out ties in text target / own community (claim B); community-level top-1 hit; 95% bootstrap CI over users.
- Files: `results/hypotheses/h5/h5_rec.json`: {setting: {misfit|control: {pop|cn|aa|struct_comm|kumari|text_sim|ours|ours+aa|aa_x_text|aa_x_struct|aa+text_sim: {"P@10":[mean,lo,hi],"R@100":[..]}, comm_acc, share_in_text_target, share_in_own}}}. README.md.
- Headline: Claim B SUPPORTED (ties in text target 0.43 vs own 0.18; AA P@10 0.090 vs 0.138 controls, k=4). Community-level: TEXT WINS (0.54 vs 0.14 struct vs 0.41 Kumari, k=4). Account-level (Claim A): MIXED (k=15 AA x p_C 0.122 vs AA 0.076, +60%; k=4 +18% not misfit-specific; k=5 loses).

## E19  T11.1b Account-level robustness (h5_rec --robust)
- Question: is the account-level text advantage robust across resolutions/seeds?  Use case: recommendation.
- Data: TIMME; 27 settings = 9 communities (r0.5/1.0/2.0 x s0..s2) x bio-classifier seeds 0-2 (settings share users; not independent); real.
- Methods: pre-registered primary aa_x_text vs aa and aa_x_struct. Metrics: paired P@10 (primary), R@100 (secondary); paired per-user diff, 95% bootstrap CI (2000 draws); misfit-specific gain = misfit diff - control diff; counts of settings with CI>0 / <0 / spans 0.
- Files: `results/hypotheses/h5/h5_robust.json`: {settings:{"r0.5_s0_b0":{n_misfit,n_control,misfit:{mean:{aa|aa_x_text|aa_x_struct:{P@10,R@100}},diff:{"text-aa P@10":[m,lo,hi],...}},control:{..},gain:{..}}}, summary:{"text-aa P@10"|"text-aa_x_struct P@10"|"text-aa R@100"|"text-aa_x_struct R@100": {misfit|control|gain:{pos,neg,span}}}}.
- Headline: **NOT ROBUST.** P@10: AA x p_C beats AA with misfit CI>0 in 12/27, <0 in 9/27; misfit-specific gain +ve 3/27, -ve 11/27. R@100 favourable (CI>0 in 24/27). Gain concentrated at r2.0.

## E20  T11.2 Rewiring simulation, TIMME
- Question: does degree-preserving re-homing of misfits toward their text community raise correction reach?  Use case: rewiring / spread (correction reach).
- Data: TIMME follow graph, r1.0_s0 (1541 treated misfits), r2.0_s0 (1464); k in {1,5} (r2.0 only k=5), beta in {0.3,1,3,6}; wc and ic:2; 100 sims; real graph + simulated spread.
- Methods: rehome_text (ours), rehome_struct (Kumari extraction), random_rewire, rehome_text_random_nodes, none. (`training/h5_rewire.py`)
- Metrics: infected share, misfits_reached, correction_reach (pp); paired contrasts vs_none / vs_random / vs_text with 95% bootstrap CI over sims. CIs do not cover rewiring-draw variance (one draw per community,k).
- Files: `results/hypotheses/h5/h5_rewire_timme.json`: {sims, meta, results:{"r1.0_s0|k5|wc|b6": {infected|misfits_reached|correction_reach: {intervention(none|rehome_text|rehome_struct|random_rewire|rehome_text_random_nodes): {mean, vs_none:[d,lo,hi], vs_random:[..], vs_text:[..]}}}}}. Note: `vs_text` is the sign-flipped contrast of the intervention relative to rehome_text.
- Headline: WEAKLY SUPPORTED, small and conditional: text-targeting residual over random rewire CI>0 in 4/7 beta>=3 cells shown; null at real beta 0.3 (+0.4 [-0.2,1.2] pp). (e.g. r1.0 k5 wc b6: none 34.5%, text-none +4.7 [3.1,6.9], text-rand +1.3 [-0.3,2.2]).

## E21  T11.2 Rewiring simulation, Pokec slice
- Question/design as E20 on Pokec. Data: `pokec_knm` (5599 nodes, 45,043 edges, 11 comms; 554 treated misfits = top-10% misfit with argmax p_C != t; no engagement filter, no controls; drops = random own-community edges). k in {1,5}, beta {0.3,1,3,6}, wc & ic:2, 100 sims.
- Methods/metrics as E20 (rehome_struct target = neighbour-majority community).
- Files: `results/hypotheses/h5/h5_rewire_pokec.json`: {sims, meta:{n,n_comm,n_misfits,edges,k1,k5}, results:{"pokec_knm|k5|wc|b6": same structure as E20}}.
- Headline: SUPPORTED on Pokec at beta>=3 (and beta=1 for wc only; beta swept, not measured); text-rand +4.0..+8.7 pp, CI>0 in all 8 cells beta>=3; null at beta 0.3 (+0.4 [-0.9,1.7]). Total infected share essentially unchanged.

## E22  H6 Community-deception detection (Kumari / safeness / random attacks)
- Question: can misfit detect nodes that hide from community detection by rewiring their own edges (bio unchanged)?  Use case: deception detection.
- Data: TIMME follow graph (20,811 nodes, 4 comms) and Pokec slice (5,599, 11 comms); 200 random targets per seed, 3 seeds, k in {2,5} rewires; attacks simulated; real graphs.
- Attacks: kumari (simplified Kumari 2021), safeness (Fionda & Pirro-style), random (control). Detectors: misfit (ours), participation, consensus_entropy, persistence_low (=1-zeta), degree. (`training/h6_deception.py`)
- Metrics: ROC-AUC targets vs non-targets: `all`, `succ` (successfully hidden), `unattacked`; deception success rate. Uncertainty: mean +- sd over 3 seeds (not bootstrap).
- Files: `results/hypotheses/h6/h6_timme.json`, `h6_pokec.json`: {ds,seeds,n_targets,n,n_comm_t0,table:{"kumari|k2"|"kumari|k5"|"safeness|k2"|"safeness|k5"|"random|k2"|"random|k5":{success_rate:[m,sd],rewires_done,rewires_budget,n_succ,auc:{misfit|participation|consensus_entropy|persistence_low|degree:{all:[m,sd],succ:[m,sd],unattacked:[m,sd]}}}},per_seed:{"seed|attack|k":..}}. Logs `run_timme.log`, `run_pokec.log`; README.md. (H6 commit 3ea4736; not yet in TASKS/DEVELOPED.)
- Headline: no as all-target detector (misfit AUC 0.48-0.57; participation/persistence_low 0.58-0.81 under safeness); yes partially for successful kumari moves (misfit 0.60-0.66 best in all 4 kumari cells, beaten in all 4 safeness cells). Success rates TIMME kumari 0.23/0.44, safeness 0.19/0.33; Pokec 0.37/0.67, 0.28/0.49.

---
# Glossary

## Metrics
- **AUC / ROC-AUC**: probability a random positive (planted/target/misfit) node is scored above a random negative; 0.5 = chance.
- **within-degree AUC**: AUC computed inside degree terciles so "degree in disguise" cannot win.
- **precision@k / P@10**: share of the top-k ranked items that are true positives (E12/E13: top-20 nodes; E18/E19: top-10 recommended accounts held-out hits).
- **R@100**: recall of a user's held-out ties within the top-100 recommendations.
- **modularity Q**: fraction of edges inside communities minus that expected at random; higher = stronger community structure (null = degree-preserving rewired graph).
- **NMI / ARI**: agreement between a clustering and reference labels (party or another partition), adjusted/normalised for chance.
- **eta^2 (degree_eta2)**: share of log-degree variance explained by cluster membership; high = clusters just sort by degree.
- **seed stability (ARI)**: agreement of clusterings across random seeds.
- **conductance**: fraction of cluster edges leaving the cluster (1 = no structure).
- **edge-reconstruction AUC**: how well embedding dot products separate real from fake edges.
- **linear probe accuracy / F1 macro**: logistic-regression accuracy predicting party from embeddings (fair probe = standardised inputs).
- **kNN10 label agreement / silhouette(true)**: label purity of 10 nearest neighbours / cluster separation under true labels.
- **infected share**: fraction of nodes reached by a simulated cascade (seeds excluded).
- **cross-community share**: infected nodes outside every seed's community.
- **"% / points removed" (gain)**: reduction in infected (or cross) share due to blocking, relative to unblocked; gain = strategy minus best structural baseline (percentage points), paired over common random numbers; CI = t-interval over seeds.
- **wins x/8**: number of seeds where misfit-based strategy beat best structural.
- **correction reach (pp)**: share of misfits sharing a random misfit's text-community that a cascade seeded in that community reaches (E20/E21); reported as percentage-point difference between interventions.
- **misfits_reached**: share of treated misfits reached by the cascade.
- **success rate (deception)**: fraction of targets whose attacked-graph community differs from their original community-mates' label.
- **Spearman correlation, lift, p_one / p_two**: rank correlation; ratio vs matched controls; one-/two-sided permutation p-values (E14).
- **agreement beta (log-odds)**: coefficient of text-agreement in a case-control logit of observed interactions; compared with simulation gate exp(beta*match).
- **text->community accuracy**: accuracy of cross-fitted text classifier at predicting a node's graph community (vs majority-class baseline).
- **delta AUC (paired bootstrap)**: AUC change from adding misfit features (E15).
- **emb_std / effective rank / norm-degree corr**: embedding health diagnostics (collapse, hub dominance).
- **degree decile matching / matched controls**: controls with same degree (and participation) decile as treated nodes.
- **Holm-surviving**: cells remaining significant after Holm multiple-comparison correction (E16).

## Scores / node-level methods
- **misfit**: 1 - p_C[t_i]; how little the node's own text supports the graph community it sits in (ours).
- **disagreement**: Jensen-Shannon divergence between structural p_T and text p_C membership.
- **ambiguity_T**: normalised entropy of p_T (structure-only mixing).
- **consensus_entropy**: instability of a node's community across Leiden runs (Riolo & Newman).
- **participation (coefficient)**: how spread a node's edges are across communities.
- **content_outlier**: text far from own community's text centroid (Gao et al. 2010-style stand-in for ONE).
- **persistence_low**: 1 - zeta, Kumari's community persistence (I/(Ex K) + clustering), low = easy to move community.
- **degree / low_degree**: node degree / its inverse.
- **pagerank (PR)**, **betweenness (btw)**: standard centralities.
- **collective_influence**: CI(i) = (k_i-1) * sum over ball neighbours (k_j-1) influence heuristic.
- **adaptive_degree**: iteratively remove highest-degree node and recompute.
- **participation_x_pr, misfit_x_pr, disagreement_x_pr, *_x_pr**: score multiplied by PageRank (influential AND ambiguous).
- **misfit_rerank**: top-M PageRank shortlist reordered by misfit.
- **gateway**: edge-based node score of cross-community links weighted by receiver's text-membership; **gateway_struct**: same without text (isolates what text adds).
- **text_iso, knn_dup, misfit_nonhub, misfit_x_lowdeg**: H4a baselines/variants (text isolation, near-duplicate text neighbours, misfit restricted to non-hubs, misfit x low-degree).
- **oracle_recruiters**: blocks the true planted recruiters (upper bound, E10).
- **random**: uniformly random nodes/edges.
- **rankavg / prod / prod_misfit / band / recv / mthenc**: H2 filter rules (rank average, product of centrality and misfit, mid-band, receptivity, misfit-then-centrality).
- **p_T**: soft structural membership via label spreading (Zhou 2004, alpha=0.5). **p_C**: cross-fitted text-only classifier membership.
- **Leiden / Louvain**: modularity-maximising community detection.
- **Regularised spectral embedding + k-means**: topology baseline view (Qin & Rohe 2013).
- **Text kNN-graph Leiden / text k-means**: text-only content views.
- **BGRL / DGI / MVGRL**: self-supervised GNN embeddings (bootstrapped latents without negatives / local-global infomax / multi-view adjacency vs PPR diffusion).
- **spectral64 probe, neighbour-vote**: graph-only baselines in E02 diagnostics.

## Recommender / intervention methods (E18-E21)
- **pop**: rank candidates by PageRank. **cn**: common neighbours. **aa**: Adamic-Adar.
- **struct_comm / aa_x_struct**: p_T(u)[t_v] weighting (community from structure) x PR or AA.
- **ours / aa_x_text / ours+aa / aa+text_sim**: p_C(u)[t_v] (text-community) weighting x PR or AA; rank-average with AA; AA plus bio similarity.
- **kumari**: structural extraction community (neighbouring community with most of u's edges) x PR; Kumari et al. 2021 target.
- **text_sim**: cosine similarity of bio embeddings.
- **rehome_text**: drop dormant own-community follows, add equal number to top-AA non-neighbours in argmax p_C community (degree-preserving). **rehome_struct**: same toward Kumari/neighbour-majority community. **random_rewire**: same drop counts, random adds anywhere. **rehome_text_random_nodes**: same procedure on random non-misfit nodes. **none**: no change.

## Spread models
- **wc**: weighted cascade, p_uv = 1/deg(v). **ic:c**: independent cascade with uniform p = c * epidemic threshold <k>/(<k^2>-<k>). **live-edge IC**: sample edge liveness once per sim (common random numbers across strategies).
- **beta**: agreement gating strength, acceptance weight exp(beta*[content match]); beta=0 content-blind. Real TIMME estimated ~0.3 (interactions) / ~1.4 (GloVe acceptance).
- **mu**: mixing share of edges to other communities. **plant**: share of planted mismatches. **place**: hub or random placement. **sigma/nuisance**: text noise / nuisance dimensions.

## Attack methods (E22)
- **kumari**: move target toward neighbouring community by dropping highest-internal-degree own-community edge, adding edge to highest-internal-degree node in destination. **safeness**: drop random own-community edge, add edge to random node in random other community. **random**: control rewiring anywhere.

---
# Gaps (documented experiments with missing raw numbers / unresolved items)
1. **T11.0 probe (E17)**: no JSON/log with raw numbers on disk (only `training/out/timme_bio*.npy`); figures exist only in TASKS.md / DEVELOPED.md. `training/h5_probe.py` would have to be rerun (training/GPU rule: ask user).
2. **H4b agreement-gating raw JSON**: script writes `gating.json` but only `gating.log` exists in `results/hypotheses/h4b/`.
3. **H6 (E22)** is in README/JSON and commit 3ea4736 but absent from TASKS.md and DEVELOPED.md.
4. **T10.5 headline "TIMME 0.62 vs 0.32; Pokec 0.28 vs 0.25"**: no single log reproduces 0.62; logs show MiniLM 0.496, GloVe-only 0.637 (block_gateway.log), 0.275 (party check run, different text config). Pokec 0.289 (interest text) vs 0.249 majority in pokec_profile_fields.log; 0.28/0.25 context not located.
5. **T10.11 link friction (E09)** and **T10.2/T10.5/T7.0** exist only as text logs (no JSON); numbers must be parsed from logs.
6. **Pokec content-aware blocking**: no `pokec_slice_content_aware.json`; only content-blind for Pokec.
7. **H2 link-level variant** (`h2/edges.json`): no verdict written in h2 README.
8. **H1 per-seed raw data** only summarised in h1_results.json (CI as [gain, ci]); no per-seed arrays. H4a/H4b/H4d have `README` verdicts written by main session from tables (agents hit rate limits); H4b code "not audited line by line".
9. **Not done / open**: T2.6 retrain (Pokec + full GNNs; all `.pt` stale), T3.5-3.7, T4-T6 (superseded), T7.1-7.4 (superseded by T10), T8 active learning, T10.10, T10.12 (edge-level regime map), T11.3 (structure-content alignment NMI/modularity), full-Pokec run of the T10 pipeline (planned by 2026-11-30).
10. **Reddit GNN** (`training/out/reddit/`) has modularity only, no labeled metrics; Gab/VoterFraud never run.
11. **Single-draw caveat**: E20/E21 CIs exclude rewiring-draw variance; E19 settings share users (not independent).
