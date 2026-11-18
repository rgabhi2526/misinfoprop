# Gaps: numbers not traced to a raw results file, and open mismatches

Hand-entered numbers (source_file = doc quoted from):
1. E17 Kumari-extraction retweet share 0.53 (k=4, leaky text) and matched-control own-community share 0.69: quoted from TASKS.md T11.0; h5_probe.py does not compute them. (2 rows, source_file TASKS.md)

Resolved gaps (now traced):
2. T11.0 probe (E17): h5_probe.py now writes results/hypotheses/h5/h5_probe.json; rerun reproduces TASKS numbers (0.66/0.40/own 0.13 leaky; 0.60/0.40 bio-only, k=4; 0.28/0.12 k=15).
3. T10.5 "TIMME 0.62 vs 0.32; Pokec 0.28 vs 0.25": found as npz keys content_acc / content_majority in results/scores/*.npz (0.622/0.318; 0.278/0.249).

Unresolved or caveated:
4. E11 (H2) config counts differ from the README. Recomputed (non-centrality, non-random-control configs, 95 percent t-CI over seeds): TIMME 7 of 2440 raw configs with CI>0; with the README exclusion of rand controls it gives 1840 configs, 7 positive (matches README); Pokec 1/920 (matches). Synthetic: infected-share 0 positive (matches) but the "1550 configs" total could not be reproduced (we count 6900 per mu-resolved metric, 2300 averaged over mu), and cross-community-share positives are 764 (per mu) / 270 (averaged over mu), contradicting the README statement that all cross CIs include 0. Rows record our definition in `setting`. Needs the authors' definition.
5. E11 link-level variant (h2/edges.json) has no README verdict; rows are aggregates only.
6. E14: agreement-gating raw JSON (gating.json) was never saved; values parsed from gating.log. extra/extra2/grid are aggregated (median, CI counts), not row-for-row.
7. E07: no Pokec content-aware blocking file exists; content-blind JSON cells have only means (no SE) for older files, so ci columns are empty there. E07 removed-pct rows are derived from stored means.
8. E09: only text logs (edge_v2, edge_oracle, edges_timme, edge_seeds); budgets (k links) inferred from edges.py calls (10000/40000 and 2000/10000/40000). Synthetic edge model assumed ic:2 (edges.py default).
9. E05, E08: seed-averaged values only, no per-seed spread stored (uncertainty none / paired CI only).
10. E10: per-seed arrays not stored; CIs are those saved in h1_results.json.
11. E02: single run, no CI; Reddit GNN has modularity only and is not included.
12. E20/E21 CIs exclude rewiring-draw variance; E19 settings share users.
13. Not run / out of scope: T2.6, T3.5-3.7, T7.1-7.4, T8, T10.10, T10.12, T11.3, full-Pokec T10 pipeline.
14. Units: the long file reports gains/removed in percentage points (file fractions x100; H4d already points); H2 README quotes +4.6+-3.5 / +0.26+-0.09 while the raw JSON gives 4.6+-3.8 / 0.3+-0.1 (rounding or CI method).
