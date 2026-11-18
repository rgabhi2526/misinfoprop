# Build report

Run: `cd final_results && /opt/anaconda3/bin/python build_results.py` (deterministic, ~10 s).

Rows: results_long 14028; results_summary 22; metrics_glossary 270.
Per experiment (long): E01 24, E02 100, E03 52, E04 33, E05 234, E06 30, E07 2616, E08 509, E09 59, E10 1653, E11 1266, E12 595, E13 2262, E14 1187, E15 29, E16 34, E17 65, E18 168, E19 684, E20 1020, E21 816, E22 592.

Gap 1 filled: training/h5_probe.py now writes results/hypotheses/h5/h5_probe.json (rerun on cached bio embeddings; matches TASKS numbers). Also resolved: T10.5 0.62 vs 0.32 lives in results/scores/*.npz. H6 (E22) included per coordinator message (commit fdfb58f, no fix_round marker); detectors are methods, attacks are in `setting`; random and noise attacks are role control.

## Spot-checks (long row -> raw file, all matched by hand)
1. E10 setting [29] 5% budget random gain: -78.434 = h1_results.json[29].budgets.5%.random[0]*100.
2. E07 Pokec ic:1 participation k=100 infected_share 0.013369 = pokec_slice_content_blind.json.
3. E07 TIMME ic:1 misfit_x_pr k=50 removed pct 22.2566 = 100*(1-inf/none).
4. E22 kumari k2 consensus_entropy succ_vs_fail AUC 0.6476 = h6_timme.json.
5. E13 sweep camp,s=0.9 misfit_x_lowdeg AUC 0.57737 = mean of 8 seeds.
6. E16 cross fixed misfit_x_pr cells CI>0 = 48 (also h4d_summary.txt).
7. E07 TIMME ic:4 participation k=50 removed pct 0.8209.
8. E06 misfit AUC 0.682 = timme_party_check.log line 4.
9. E20 r1.0 k5 wc b6 rehome_text vs none 4.72 pp (README 4.7).
10. E17 bio-only retweet hit 0.599 / chance 0.401 = printed probe output 0.60/0.40.

## Decisions
- Shares, AUC, P@10 stay fractions; gains/removed in percentage points (x100 where stored as fractions).
- BGRL/DGI/MVGRL role = ours (the project's tested approach, superseded); spectral64, neighbour-vote = baseline.
- gateway_struct treated as control (ablation). Blocking SE stored as ci = value +- 1 SE (uncertainty se_plus_minus_1).
- H2 sweeps aggregated per (dataset/model/mode/metric, filter rule): n configs, CI>0, CI<0, best config; recomputed counts disagree with the README (see _gaps.md #4).
- E14 grid/extra/extra2 aggregated to counts and medians; E04 npz used as source for accuracies.
- Verdict mapping from README/TASKS wording: NOT SUPPORTED/NO/NOT ROBUST -> not supported; WEAKLY SUPPORTED, conditional, gate-passed-only-in-regime -> partly supported; E02 failed; E01/E03/E04 descriptive. E06 and E09 marked partly supported (my mapping; TASKS has no explicit label).

## Open questions
- Authors' exact definition of the H2 "configs" counts (README claim of no positive cross CIs is not reproducible).
- Should E02 GNN rows be role ours or baseline?
