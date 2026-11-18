# Gaps: numbers not traced to a raw results file, and open mismatches

Hand-entered numbers (source_file = doc quoted from):
1. E17 Kumari-extraction retweet share 0.53 (k=4, leaky text) and matched-control own-community share 0.69: quoted from TASKS.md T11.0; h5_probe.py does not compute them. (2 rows, source_file TASKS.md)

Resolved gaps (now traced):
2. T11.0 probe (E17): h5_probe.py now writes results/hypotheses/h5/h5_probe.json; rerun reproduces TASKS numbers (0.66/0.40/own 0.13 leaky; 0.60/0.40 bio-only, k=4; 0.28/0.12 k=15).
3. T10.5 "TIMME 0.62 vs 0.32; Pokec 0.28 vs 0.25": found as npz keys content_acc / content_majority in results/scores/*.npz (0.622/0.318; 0.278/0.249).

Unresolved or caveated:
4. E11 (H2) RESOLVED 2026-11-18: the README claim that all synthetic cross-community CIs include 0 was wrong and the README is corrected. Recomputed (non-centrality, non-random-control configs, 95 percent t-CI over 8 seeds): synthetic infected share 0 of 6900 positive; cross-community share 764 of 6900 positive (misfit 159/2250, prod 351/2400, band 63/750, rankavg 141/750, mthenc 50/750; beta 10: 481, 6: 282, 3: 1; hub 595, random 169); random-from-shortlist 0/2250, centrality-only 1/600. The README's "0/1550" total is not reproduced and is a narrower count of the infected-share conclusion. TIMME 7/1840 (1 infected + 6 cross, of 920 each) and Pokec 1/920 match. Per-family counts now in results_long (E11).
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
