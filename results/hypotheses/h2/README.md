# H2: misfit as a filter over PageRank/betweenness shortlists — verdict: NOT SUPPORTED on infected share; synthetic cross-community share improves at strong content gating (beta>=6); real data: noise

Written by the main session from raw `synth.json`, `real_timme.json`, `real_pokec_knm.json`
(the H2 subagent hit a rate limit before writing its own summary; its compute had finished, `ALLDONE`).
Gain = reduction in infected share vs best of PageRank/betweenness top-k, points; CI = t-interval over seeds.

Synthetic (beta 0-10, hub/random placement, mu 0.1-0.5, k 0.1-5%, shortlist 2k..20%, filter by misfit / product / band):
- 0 of 1550 misfit-filter configurations have a 95% CI above 0 on infected share.
- Best config per cell is almost always negative (e.g. beta=6 hub k=1%: -0.9±0.2; beta=10: -0.9±0.1; beta=3: -1.2±0.2).
- Gap to the random-from-shortlist control is small only where nothing matters (beta>=6); at beta<=3 random-from-shortlist is far worse, so the shortlist itself is not the problem — misfit adds nothing over plain centrality ordering.
- Cross-community share (corrected 2026-11-18: earlier text generalised from the k=0.1% table and said all CIs include 0; that is wrong). Recomputed from `synth.json` (156000 rows; grouped by mu, place, beta, metric, name; 8 seeds; 95% t-interval): 764 of 6900 misfit-type filter configs have CI>0 (misfit 159/2250, prod 351/2400, band 63/750, rankavg 141/750, mthenc 50/750). By beta: 10 -> 481, 6 -> 282, 3 -> 1. By placement: hub 595, random 169. Median gain of the positives +4.4 points, max +13.3. Controls: random-from-shortlist 0/2250, centrality-only 1/600. About 170 false positives would be expected at a one-sided 2.5% rate, so 764 with a 0/2250 random control is a real effect on this metric, confined to strong content gating (beta>=6).
- Infected share, same recount: 0 of 6900 misfit-type configs have CI>0 (the "0 of 1550" above is the same conclusion on a narrower count).
Real data:
- TIMME: 7 of 1840 configs have CI>0 (best +4.6±3.5 on cross-community share, +2.7±2.1 on infected). At that multiplicity (~0.4% hit rate vs 2.5% expected by chance for one-sided 95%) this is noise.
- Pokec slice: 1 of 920, +0.26±0.09 on cross share only.
Caveat: filter pool/budgets chosen by the agent; no pre-registration, so even the few hits are not evidence.
