# H3: where could misfit-based targeting help? (synthetic scenario tests)

Code: `training/h3_scenarios.py` (runs), `training/h3_report.py` (table). Data: `h3_results.json`,
`h3_run.log`, `h3_table.md`. Command (from `training/`, CPU only, ~25 min; set
`OMP_NUM_THREADS=1`, otherwise a loaded machine thrashes):
`python h3_scenarios.py --seeds 8 --sims 60`. No GPU needed; `gpu_needed.md` not touched.

Setup: planted DC-SBM, n=4000, 8 communities, mu=0.3, text noise as in `regime.py`; scores from the
real pipeline on observed text; spread uses TRUE content (acceptance weight exp(beta*[content==source])).
IC at 2x threshold, 60 sims per seed, **8 paired seeds**, budgets k=20 and 40 (0.5%, 1% of nodes).
Gain = removed share (points) of best misfit-based strategy (misfit, misfit_x_pr, gateway,
misfit_rerank) minus best structural (pagerank, betweenness, degree, CI, adaptive degree); both picked
by mean over seeds, so the gain is optimistically biased toward whichever has more candidates.
The column "fixed misfit_x_pr" removes that selection. 95% CI = t-interval over seeds.

## Ranking (headline k=20; full table in h3_table.md)
| scenario | gain, points (95% CI) | verdict |
|---|---|---|
| (e) detect infiltrators, precision@20, vs best structural | +76 (±6) random, +36 (±14) hub-placed, +78 (±8) regular | WORKS (but detection, not blocking; vs best non-structural score content_outlier: +73/+38/+61) |
| (f) weak degree heterogeneity, beta=6 | +5.6 (±1.4); k=40 +9.8 (±2.2) | WORKS, only if gating strong; absolute removal still only 7.5% (k=20) |
| (a) topic-gated, hub-placed mismatches, beta=10 | +5.4 (±1.2); k=40 +6.6 (±3.3) | WORKS (same regime as the known win) |
| (g) 3 simultaneous sources, hub-placed, beta=6 | +4.0 (±0.8) | WORKS, same as single source |
| (d) protected top-2% hubs, hub-placed, beta=6 | +4.7 (±1.3) | works, but control without protection is +4.5 (±1.1): protection adds nothing |
| (a) gated, random placement, beta=10 | +4.1 (±4.8); fixed misfit_x_pr +0.5 (±1.7) | MARGINAL (CI includes 0) |
| (c) niche-community harm, beta=10 | +0.1 (±0.6) random, +0.2 (±0.4) hub | USELESS (both ~4% removed) |
| (d) protected hubs, beta=1.4 (realistic) | -0.9 (±0.4) | USELESS/loses |
| (g) 3 sources, random, beta=3 | -1.1 (±0.7) | loses |
| (f) regular graph, beta=1.4 | -0.1 (±0.1) | tie (both remove ~2%) |
| control: hub-placed, beta=1.4 (TIMME-like) | -2.1 (±0.4) | loses (reproduces prior finding) |
| (b) delivery to hard-to-reach community (seed k nodes, maximise reach) | -0.1 to -0.7 (CI excludes 0 at low transmissibility) | USELESS; "edges into target community" (structural) wins; supercritical settings saturate at ~92% reach |

## Findings
- Misfit's blocking advantage needs BOTH strong content gating (beta >= 6) and either hubs among the
  mismatched or no hubs for centrality to exploit. At beta ~1.4 (TIMME) it loses or ties in every
  structural variant tried (hub, protected, regular, multi-source).
- Protecting hubs does not create a misfit niche: structural methods fall back to CI/betweenness on
  the next tier and stay equal or ahead at realistic beta.
- Regular-degree result is real but small in absolute terms (7-14% of cross-community spread removed
  at 1% budget versus 2-4% for structural). Both are weak interventions.
- Delivery: a structural score that needs only the community labels (number of edges into the
  target community) beats every misfit variant; misfit only beats a pure text baseline by ~+1 point.
- Detection is the one clear win: planted-mismatch AUC 0.92-0.94, precision@20 of 73-83% versus
  3-5% for degree/PageRank/betweenness on random placement; on hub-placed mismatches the structural
  baselines reach 37% simply because degree correlates with being planted.

## Caveats
Synthetic only (planted mismatch is exactly what misfit measures; real mismatches are murkier). n=4000,
single mu/plant per scenario, 8 seeds. Scenario (c) uses a single fixed source community of equal size,
not a truly small niche; not tested with unequal community sizes. "Harm" outcome = infected nodes whose
true content equals the source. Adaptive degree is empty when all top-k hubs are protected.
Real-data validation of (e) exists separately (party-misfit AUC in `results/logs/timme_party_check.log`).
