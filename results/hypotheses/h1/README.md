# H1: radicalisation-funnel synthetic test

Verdict: NOT SUPPORTED. Misfit never finds the right nodes to block, even where blocking the true recruiters would work.

Code: `training/h1_funnel.py` (imports views, ambiguity, blocking; nothing existing modified).
Run: `cd training; python h1_funnel.py --seeds 8 --sims 30 --workers 5`. Outputs: `h1_results.json`, `h1_run.log` (tables), `h1_progress.log`.

## Setup
n=4000. 6 mainstream communities, sympathiser fringe (10%), radical core (5%). Recruiters (1%) are labelled
mainstream, send 40% of edges to the core, text shows 30% core. Hub variant: degree ~10x. Non-hub variant: ordinary degree.
Decoys (1% or 3%) are hub news/celebrity accounts with edges across mainstream and pure mainstream text.
Spread: live-edge IC (ic:2), 10 core seeds, edge u->v scaled by K*exp(beta*a_v)/(e^beta+K-1), K=8 (regime.py's beta),
a_v = receiver affinity to the core message (mainstream ~Beta, mean 0.15; fringe 0.5; core 1; recruiters 1).
Harm = share of mainstream nodes reached. Scores come from noisy observed text via the real pipeline; spread uses true affinity.
Budgets 0.5/1/2/5% (20/40/80/200 nodes). 8 seeds (fresh graph each), 30 cascades/seed, paired common cascades.
Gain = share of harm removed minus best-of {degree, pagerank, betweenness} (chosen by mean per setting/budget). CI = t-interval over seeds.
30 settings x 4 budgets.

## Result
Cells with 95% CI above 0 (of 120): misfit 0, misfit_x_pr 1, gateway 7 (all <= +0.057, mostly +0.02, no pattern; expect ~6 by chance). CI strictly below 0: misfit 120/120, misfit_x_pr 65, gateway 46.

Gain over best structural (mean, ±95% CI), budget 1% (40 nodes) unless noted:

| setting | best struct removed | misfit | misfit_x_pr | gateway | oracle (block true recruiters) | AUC misfit->recruiter / degree->recruiter |
|---|---|---|---|---|---|---|
| hub, beta=3 (base) | 0.53 | -0.50±0.02 | -0.02±0.02 | +0.01±0.03 | +0.21±0.02 | 0.65 / 0.95 |
| hub, beta=6 | 0.27 | -0.25±0.07 | +0.02±0.03 | +0.02±0.06 | +0.67±0.06 | 0.65 / 0.95 |
| non-hub, beta=3 | 0.68 | -0.63±0.08 | -0.04±0.04 | -0.05±0.03 | -0.40±0.09 | 0.59 / 0.22 |
| non-hub, beta=6 | 0.34 | -0.31±0.10 | -0.06±0.12 | -0.12±0.07 | +0.30±0.15 | 0.59 / 0.22 |
| decoys 1%, beta=6 | 0.29 | -0.25±0.08 | +0.01±0.02 | -0.03±0.04 | +0.66±0.07 | 0.63 / 0.94 |
| hub+3% decoys, beta=6 | 0.26 | -0.23±0.05 | -0.05±0.05 | -0.16±0.05 | +0.69±0.05 | 0.64 / 0.93 |
| hub, beta=0 | 0.46 | -0.44±0.02 | -0.05±0.01 | -0.02±0.02 | -0.31±0.04 | 0.65 / 0.95 |
| text fully core-like (rec_txt=1), beta=3 | 0.53 | -0.50±0.02 | +0.01±0.01 | 0.00±0.03 | +0.21±0.02 | 0.84 / 0.95 |

## Reading
- The enabling conditions are real: when beta>=3 and recruiters are hubs, removing the TRUE recruiters beats the best structural
  strategy by +0.2 to +0.7 (oracle column; at 5% budget the margin shrinks to ~0 or negative as structural catches up).
  With non-hub recruiters or beta<=1.5 the oracle loses to structural.
- But misfit cannot find them. Recruiter-detection AUC is 0.59-0.65 (0.84 even if the text is fully core-like) against 0.95 for plain degree
  in the hub case. Misfit alone picks low-degree text outliers and loses 25-90 points to structural.
- misfit_x_pr is PageRank re-ordered: statistically indistinguishable from structural (best case +0.09±0.06 at 0.5%, beta=6, decoys).
- Decoys do not change the picture (misfit already fails without them); they hurt gateway/misfit_x_pr slightly at beta=6.
- The non-hub variant shows where content could help in principle (no degree signature), but detection is also weakest there.

## Caveats
Synthetic by construction: recruiter edges, text mixing, acceptance and "agreement gating" are assumptions I chose; the funnel is a
two-step path with a single message source. Real recruiters may have richer, detectable text signatures; this generator gives them
only a mixed embedding with Gaussian noise plus nuisance dimensions (accuracy of text->community 0.59, same as regime.py). Stronger
detectors (supervised, behavioural) are not tested. Static IC, no adaptation by recruiters. Best-of-structural is picked per
setting by mean so it is slightly optimistic for the baseline. The 0.5% budget is 20 nodes: coarse.
Run note: ~20 min wall time with 5 workers on 8 cores, CPU only.
