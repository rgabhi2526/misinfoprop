# H4b — real TIMME interaction data: are misfit nodes special bridges? (verdict: NO)

Written by the main session from the agent's logs (`summary.txt`, `gating.log`, `extra*.log`, `*.json`);
the agent hit a rate limit before writing its own README, and the code (`training/h4b_*.py`) was not audited line by line.
Setup: communities from FOLLOW edges only; outcomes from retweet/mention/reply/like edges (+ pooled "spread"); matched
controls by degree decile (and degree x participation); permutation tests; 12 outcome/definition combos -> read p-values with that in mind.

## Findings
1. Misfit is NOT a degree/centrality proxy: Spearman with PageRank -0.02, betweenness 0.005, degree -0.02. (With disagreement 0.92; with topology participation 0.24.)
2. Cross-community interaction COUNT is not higher for top-misfit nodes than degree-matched controls:
   pooled diff -1.6 [-17.5, +12.3]; retweet +11.7 [+2.4, +20.2] (small); mention -6.8 [-15.3, +2.1].
   Poisson regression with log-degree, PageRank, betweenness: misfit coefficient -0.015 [-0.047, +0.017] (pooled), -0.003 (retweet) -> nothing.
   Stratified by community x degree quintile the effect is NEGATIVE (log1p cross_n -0.40 [-0.54, -0.27] pooled).
3. Cross-community interactions RECEIVED are lower for misfit nodes (pooled -33 [-45, -19]; p2 ~ 0). They are not hubs of incoming cross-community attention.
4. Cross-community SHARE is higher (+0.29 [+0.27, +0.30]); still +0.14 after degree/PR/betweenness controls. Caveat: a share is a ratio, misfit partly encodes topology ambiguity
   (corr 0.24 with participation), and participation-based controls shrink but do not remove it. Read as "a larger fraction of their (not larger number of) interactions crosses communities".
5. Centrality benchmarks do far better on volume: betweenness-top nodes +446 cross interactions vs controls; participation-top +204.
6. REAL agreement-gating effect (does a cross-community interaction prefer target communities whose text profile agrees?): log-odds coefficient
   retweet 0.29-0.33, mention 0.23-0.27, reply 0.18-0.21, like 0.28-0.32 (CIs about +/-0.03, stable across resolutions/seeds).
   The blocking simulations use the same exp(beta*agree) scale and needed beta >= 6 (H4d is re-checking up to 20). Real effect is ~20x too small.
   Follow edges show NEGATIVE coefficients (-0.2 to -0.9) -- unexplained; likely artefact of communities being built from follow edges. Not interpreted.

## Verdict
On real data misfit nodes are not measurably better bridges in volume, receive less cross-community attention, and the real agreement effect (~0.3) is far below the regime where
misfit-guided blocking beat centrality (beta >= 6). This closes the real-data loop: consistent with the blocking hypotheses failing (H1, H2, H3 realistic-beta cases).
The one positive: a higher *fraction* of misfit nodes' interactions cross community lines -- a descriptive property, not an intervention advantage.
