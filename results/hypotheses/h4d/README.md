# H4d — extended node-level regime sweep (beta 4-20, 12 seeds) — verdict: statistically visible, practically negligible

Written by main session from `h4d_cells.json` / `h4d_summary.txt` / `h4d_surface.txt` (agent died on a rate limit; first run crashed on a full disk, rerun is complete: 96 jobs, 480 records).
Grid: mu {0.3,0.5} x plant {1%,5%} x placement {hub,random} x beta {4,6,8,12,20} x budget {40,80,200 nodes} = 120 cells, 12 seeds, 60 sims/seed, IC at 2x threshold.
Misfit strategies: misfit, misfit_x_pr, gateway, misfit_rerank. Structural: pagerank, betweenness, degree, collective_influence, adaptive_degree. Units: percentage points of infected share.

## Results
- FIXED strategy misfit_x_pr vs best structural, infected share: CI>0 in 0/120 cells, CI<0 in 120/120 (loses everywhere, even at beta=20; the response-surface gap shrinks with beta but stays negative: -0.11 pts at beta 6, -0.05 at beta 20).
- BEST-SELECTED misfit strategy (picked per cell by mean, so optimistic): CI>0 in 25/120 cells (chance ~3), CI<0 in 54, Holm-surviving 10. All wins are plant 5%, hub placement, budget 200, beta >= 6, gateway-type strategy.
  Size: gain at most +0.06 +/- 0.04 points (e.g. 2.5 vs 2.4 points removed from a 9.7% baseline). That is noise-level in practice.
- Cross-community share: selected strategy CI>0 in 58/120 (Holm 36), fixed misfit_x_pr 48/120, but gains <= 0.03 points on a baseline of ~0.3% -- not meaningful.
- Response surface (infected): gain increases with log(beta) (+0.06 per e-fold) but intercept is negative; larger plant fraction, higher mu and bigger budget make it worse. No setting in range predicts a material win.

## Verdict
Extending beta to 12 and 20 does not rescue misfit-guided blocking. The 2/72 positive cells from the first map were real but tiny; with more seeds and a broader baseline set (collective influence, adaptive degree) they shrink to <= 0.06 points.
Blocking should be reported as a tested, negative extension; the real TIMME agreement effect (~0.3, see h4b) is far below even beta=4.
Caveats: synthetic, static single-source IC, selection bias in 'best-selected', single graph family (DC-SBM n=4000).
