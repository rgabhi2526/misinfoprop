# H5 / T11 — misfit re-homing as link recommendation (TIMME, 2026-11-11)

Scripts: `training/h5_probe.py` (bio-only encode + community probe), `training/h5_rec.py` (this test). Raw: `h5_rec.json`.
Setup: communities = Leiden on FOLLOW edges (h4b_scores r1.0 s0/s1, r2.0 s0 -> k = 4/5/15). p_C = cross-fitted
classifier on **bio-only** MiniLM (leak control). Held-out = retweet/like/mention ties u->v with u NOT following v.
Misfits = top-10% misfit, has bio, >=5 held-out ties, argmax p_C != t (~1.5k). Controls = low-misfit nodes
matched 1:1 on degree decile x participation decile. 95% bootstrap CIs over users.

## Claim B — misfits are under-served by their follow graph: SUPPORTED (all 3 settings)
| k | misfit held-out ties in text target / own community | controls (text target / own) |
|---|---|---|
| 4 | 0.43 / 0.18 | 0.47 / 0.47 |
| 5 | 0.38 / 0.18 | 0.44 / 0.45 |
| 15 | 0.20 / 0.07 | 0.26 / 0.26 |
Structure-only link prediction is weakest exactly on misfits: Adamic-Adar P@10 0.090 vs 0.138 for matched controls (k=4).

## Community level — which community will u engage with most: TEXT WINS for misfits (all 3)
| k | text argmax p_C | structure argmax p_T | Kumari extraction | own follow community |
|---|---|---|---|---|
| 4 | **0.54** | 0.14 | 0.41 | 0.14 |
| 5 | **0.48** | 0.19 | 0.36 | 0.19 |
| 15 | **0.27** | 0.08 | 0.15 | 0.08 |
For controls text ~= structure (0.56 vs 0.55): the text signal is useful specifically for misfits.

## Claim A (account level) — text beats structure-only recommenders: MIXED, resolution-dependent
Misfit P@10 (best text variant vs AA vs AA re-weighted by structural community):
- k=15: AA x p_C[t_v] **0.122** vs AA 0.076 vs AA x p_T 0.062 -> clear win (+60% over AA).
- k=4: AA + bio similarity **0.106** vs AA 0.090 (+18%); but the same hybrid helps controls more (0.182 vs 0.138) -> not misfit-specific.
- k=5: AA x p_T 0.135 beats AA x p_C 0.096 -> text loses.
Community x PageRank alone ("ours", "kumari", "struct_comm") is a weak account ranker (P@10 ~0.03) — community choice
needs a structural ranker inside it.

## Read with care
3 Leiden settings, 1 seed each for the bio classifier; ~11 scorers x 3 settings -> multiple comparisons.
Cross-layer held-out (follow -> engagement), no timestamps: claim is "follow graph misses ties visible in engagement".


## T11.1b robustness
Pre-registered: primary scorer aa_x_text = AA * p_C(u)[t_v] vs `aa` and `aa_x_struct` = AA * p_T(u)[t_v]; metric P@10 (R@100 secondary).
27 settings = 9 communities (r0.5/1.0/2.0 x s0..s2) x bio-classifier seeds 0,1,2. Misfits/controls built as in `h5_rec.run`.
Paired per-user difference, 95% bootstrap CI over users (2000 draws). Misfit-specific gain = mean diff on misfits - mean diff on controls (groups bootstrapped independently).
Counts are the number of settings (of 27) whose CI is > 0 / < 0 / spans 0. The 27 settings share the same users (and overlapping communities), so they are not independent replicates. Per-setting means and CIs: `h5_robust.json`. Run: `python h5_rec.py --robust`.

| paired difference (aa_x_text minus ...) | group | CI>0 | CI<0 | spans 0 |
|---|---|---|---|---|
| text-aa P@10 | misfit | 12 | 9 | 6 |
| text-aa P@10 | control | 15 | 0 | 12 |
| text-aa P@10 | gain | 3 | 11 | 13 |
| text-aa_x_struct P@10 | misfit | 15 | 9 | 3 |
| text-aa_x_struct P@10 | control | 8 | 5 | 14 |
| text-aa_x_struct P@10 | gain | 16 | 9 | 2 |
| text-aa R@100 | misfit | 24 | 2 | 1 |
| text-aa R@100 | control | 17 | 5 | 5 |
| text-aa R@100 | gain | 13 | 3 | 11 |
| text-aa_x_struct R@100 | misfit | 24 | 0 | 3 |
| text-aa_x_struct R@100 | control | 12 | 0 | 15 |
| text-aa_x_struct R@100 | gain | 24 | 0 | 3 |

Mean P@10 by resolution (averaged over 9 settings per row): misfits r0.5 AA 0.090 / text 0.080 / struct 0.115; r1.0 0.096 / 0.092 / 0.094; r2.0 0.077 / 0.123 / 0.070. Controls r0.5 0.158 / 0.157 / 0.157; r1.0 0.146 / 0.154 / 0.160; r2.0 0.141 / 0.178 / 0.166.

**Verdict: NOT ROBUST.** The account-level text advantage is resolution-dependent. On P@10, AA x p_C beats AA with a positive misfit CI in only 12/27 settings (negative in 9/27), and the misfit-specific gain over AA is positive in 3/27 and negative in 11/27 (so it is, if anything, not misfit-specific); versus AA x p_T the misfit gain is positive in 16/27 and negative in 9/27. The gain is concentrated at fine resolution (r2.0, k~14-15: misfit P@10 0.123 vs AA 0.077 vs AA x p_T 0.070); at r0.5/r1.0 it is null or negative (r0.5: 0.080 vs 0.090 vs 0.115). R@100 is more favourable (misfit CI>0 vs AA in 24/27, vs AA x p_T in 24/27, misfit-specific gain vs AA x p_T positive in 24/27, never negative), so the secondary metric supports text > structural community weighting while the primary P@10 does not.

## T11.2 rewiring simulation (TIMME)

Code: `training/h5_rewire.py` (`python h5_rewire.py timme`; `--selftest`). Raw: `h5_rewire_timme.json` (per cell: mean, and paired bootstrap CIs `vs_none`, `vs_random`, `vs_text`). ~16 min, Pool(4), 100 sims.

**Design.** Treated = the 1541 (r1.0_s0) / 1464 (r2.0_s0) misfits of T11.1. Each treated node drops up to k own-community follow edges with no engagement (retweet/like/mention/reply, either direction) and adds the same number to non-neighbours in a target community, ranked by **plain Adamic-Adar inside that community** (not aa_x_text; within one community p_C is constant, so rehome_text and rehome_struct differ ONLY in the target community; ties random). Treated nodes' own degree is preserved exactly (edges never touch other treated nodes); neighbours' degrees change. Interventions: `rehome_text` (target argmax p_C, bio-only), `rehome_struct` (Kumari extraction community), `random_rewire` (same per-node drop counts, own-community random drops, random adds anywhere), `rehome_text_random_nodes` (same procedure on random non-misfit nodes with argmax p_C != t; it changes ~10% more edges, 13564 vs 12266 at k=5 r1.0, which is conservative against rehome_text), `none`.
**Spread.** Live-edge IC on the undirected follow graph; both directions of an edge are tested against their own gated probability min(1, p_uv K aff[v,c0]) using ONE shared uniform per (min,max,sim) hash (common random numbers); seeds and sources identical across interventions. aff = softmax(beta * onehot(tc_v)). Models wc and ic:2. Outcomes: infected share; share of misfits reached; correction reach (random misfit u, 10 seeds uniform in {t == tc_u}, share of misfits with tc == tc_u reached). CIs: bootstrap over 100 sims of paired per-sim differences. Reduced grid: k in {1,5}, beta in {0.3,1,3,6}; r2.0_s0 only k=5 with beta {0.3,6}.

Correction reach in percentage points (95% CI). `rand-none` = generic effect of rewiring; `text-rand` = targeting effect (the attributable one); `text-rnodes` = rehome_text minus rehome_text_random_nodes; `text-struct` = rehome_text minus rehome_struct (both paired per sim).

| setting | none (%) | rand-none | text-none | text-rand | text-struct | text-rnodes |
|---|---|---|---|---|---|---|
| r1.0 k5 wc b6 | 34.5 | +3.5 [1.6, 5.9] | +4.7 [3.1, 6.9] | +1.3 [-0.3, 2.2] | +1.7 [1.4, 2.1] | +5.3 [3.2, 8.1] |
| r1.0 k5 wc b3 | 31.7 | +2.4 [1.1, 4.4] | +3.4 [2.2, 5.0] | +1.0 [-0.6, 1.9] | +1.3 [1.0, 1.7] | +4.0 [2.3, 6.4] |
| r1.0 k5 ic:2 b6 | 7.1 | -0.07 [-0.23, 0.15] | +0.46 [0.28, 0.70] | +0.53 [0.41, 0.66] | +0.20 [0.13, 0.27] | +0.31 [0.22, 0.41] |
| r1.0 k5 ic:2 b3 | 6.1 | +0.01 [-0.20, 0.32] | +0.28 [0.19, 0.39] | +0.27 [-0.04, 0.51] | +0.14 [0.06, 0.21] | +0.29 [0.04, 0.55] |
| r1.0 k1 wc b6 | 34.5 | +0.55 [0.41, 0.69] | +1.42 [1.13, 1.72] | +0.88 [0.67, 1.11] | +0.89 [0.69, 1.10] | +1.43 [1.12, 1.72] |
| r2.0 k5 wc b6 | 40.9 | +1.2 [0.6, 1.9] | +3.4 [2.4, 4.5] | +2.2 [1.5, 3.1] | +2.4 [1.7, 3.2] | +2.5 [0.1, 4.2] |
| r2.0 k5 ic:2 b6 | 12.0 | -0.34 [-0.47, -0.21] | +1.07 [0.78, 1.38] | +1.40 [1.08, 1.73] | +0.99 [0.74, 1.24] | +1.0 [0.7, 1.3] |
| r1.0 k5 wc b0.3 (real TIMME) | 2.9 | -0.3 [-0.9, 0.1] | +0.1 [-0.4, 0.6] | +0.4 [-0.2, 1.2] | -0.03 [-0.4, 0.3] | +0.2 [-0.3, 0.7] |

(text-struct and text-rnodes columns are the sign-flipped `vs_text` contrasts of rehome_struct / rehome_text_random_nodes from the JSON.) Other outcomes: total infected share barely moves (+0.02 to +0.07 pp for rehome_text at k=5, beta 6); misfits-reached rises with rehome_text (r1.0 k5 wc b6: +0.98 pp [0.71, 1.25] vs none, +0.64 [0.45, 0.82] vs random). At the empirical beta=0.3 nothing is distinguishable from zero.

**Verdict: WEAKLY SUPPORTED, small and conditional.** (1) Much of the headline gain under `wc` is generic rewiring at r1.0 (r1.0 k5 wc b6: 3.5 of the 4.7 pp, 38.0% vs 34.5%) but not at r2.0 (1.2 of 3.4 pp); random_rewire alone raises correction reach (model wc, beta>=3) by +0.4 to +3.5 pp (most at r1.0; r2.0 k5 b6 +1.2); under ic:2 it is about zero (-0.34 [-0.47, -0.21] to +0.06 pp at beta>=3, so the generic-rewiring caveat applies to wc only), so "+4.7 pp over none" overstates what text-targeting buys at r1.0. The attributable effect is the residual over random_rewire: CI > 0 in 4 of the 7 beta>=3 cells shown (r1.0 k5 ic:2 b6 +0.53 [0.41, 0.66], r2.0 k5 wc b6 +2.2 [1.5, 3.1], r2.0 k5 ic:2 b6 +1.4 [1.1, 1.7], r1.0 k1 wc b6 +0.88 [0.67, 1.11]) and CI includes 0 in the other 3 (r1.0 k5 wc b6, wc b3, ic:2 b3). (2) Which node is treated matters, directionally: rehome_text beats rehome_text_random_nodes by a paired contrast with CI excluding 0 in every beta>=3 cell (even though the random-nodes arm moves ~10% more edges), and beats rehome_struct (text target beats structural target, CI > 0 in all beta>=3 cells). (3) Total spread is essentially unchanged, and at the real TIMME beta (0.3) all contrasts are null. The effect exists only under strong selective exposure, which H4b measured as weak on TIMME.

Main limitation: ONE rewiring draw per (community, k) and 100 sims; the sim CIs capture cascade noise but not rewiring-draw variance, which is the principal reason the text-vs-random comparison (wide, sign-unstable under wc, where random adds create hubs) is uncertain. Also: AA from the original graph; follows treated as bidirectional transmission; r2.0 only k=5.

## T11.2 rewiring simulation (Pokec)

Code: `training/h5_rewire.py` (`python h5_rewire.py pokec`). Raw: `h5_rewire_pokec.json` (same structure as the TIMME file). ~35 s, Pool(4), 100 sims.

**Design.** Same interventions, spread model, outcomes and common random numbers as the TIMME section; slice `pokec_knm` (5599 nodes, 45043 friendship edges, 11 communities, graph loaded via `views.load_pokec` and asserted aligned with `out/pokec_knm_views/ambiguity.npz`). Content signal = the stored `p_C` (Pokec profile-interest text; no retweet leak, so the TIMME bio-only constraint does not apply), `tc = argmax p_C`. Treated = 554 misfits: top-10% `misfit` among `t >= 0` with `argmax p_C != t` (no engagement filter, no matched controls). **Drop rule differs from TIMME:** Pokec has no engagement layer, so "dormant" is undefined and each treated node drops up to k *random own-community* edges (not dormant-only); adds are plain Adamic-Adar inside the target community. `rehome_struct` target = neighbour-majority community other than own. Edges changed: k=1 1002 (random-nodes arm 1076), k=5 3660 (4094). Grid: models wc, ic:2; k in {1,5}; beta in {0.3,1,3,6}. **Pokec has no measured selective-exposure beta**, so no beta is "real"; beta is a swept assumption and the low-beta rows are not an empirical anchor.

Correction reach in percentage points (95% CI, paired over 100 sims; contrast definitions as in the TIMME table).

| setting | none (%) | rand-none | text-none | text-rand | text-struct | text-rnodes |
|---|---|---|---|---|---|---|
| k5 wc b6 | 50.3 | +2.9 [2.4, 3.5] | +10.9 [9.8, 12.1] | +8.0 [6.9, 9.1] | +8.5 [7.6, 9.4] | +10.3 [9.1, 11.5] |
| k5 wc b3 | 50.7 | +2.5 [1.6, 3.4] | +10.0 [8.8, 11.2] | +7.5 [6.3, 8.8] | +7.3 [6.4, 8.3] | +9.5 [8.0, 10.8] |
| k5 wc b1 | 32.0 | +1.7 [0.5, 2.9] | +7.0 [5.7, 8.3] | +5.3 [4.0, 6.6] | +5.8 [4.5, 7.1] | +7.2 [5.7, 8.7] |
| k5 ic:2 b6 | 53.6 | +1.9 [1.4, 2.4] | +10.6 [9.0, 12.5] | +8.7 [7.0, 10.7] | +9.7 [8.1, 11.8] | +10.3 [8.6, 12.2] |
| k5 ic:2 b3 | 59.0 | +0.9 [-0.05, 1.8] | +7.4 [6.0, 9.2] | +6.5 [4.8, 8.4] | +7.3 [5.3, 9.7] | +6.8 [5.5, 8.2] |
| k5 wc b0.3 | 4.3 | +1.1 [0.3, 1.9] | +1.5 [0.5, 2.5] | +0.4 [-0.9, 1.7] | +0.1 [-2.5, 1.9] | +2.3 [1.3, 3.3] |
| k1 wc b6 | 50.3 | +0.8 [0.5, 1.1] | +6.5 [5.8, 7.1] | +5.7 [5.0, 6.3] | +6.4 [5.6, 7.2] | +6.1 [5.5, 6.7] |
| k1 ic:2 b6 | 53.6 | +0.7 [0.4, 1.1] | +4.8 [4.2, 5.5] | +4.1 [3.5, 4.8] | +4.6 [3.8, 5.3] | +4.8 [3.9, 5.6] |
| k1 ic:2 b3 | 59.0 | +0.7 [0.3, 1.1] | +4.7 [3.0, 7.1] | +4.0 [2.3, 6.4] | +3.7 [1.9, 6.1] | +4.4 [2.4, 7.0] |

(All 16 cells are in the JSON.) Other outcomes at k5 b6: total infected share is essentially unchanged (wc +0.11 pp [0.07, 0.16], ic:2 +0.01 [-0.03, 0.05] for rehome_text vs none); misfits-reached rises (wc +1.67 pp [1.40, 1.95], ic:2 +1.37 [1.10, 1.66]; random_rewire +0.46 / +0.33).

**Verdict: SUPPORTED on Pokec at beta >= 3 (and at beta = 1 for wc only; selective exposure assumed), null at low beta.** (1) The targeting effect is large and not a generic-rewiring artefact: rehome_text beats random_rewire (CI > 0) in every one of the 8 cells with beta >= 3 (+4.0 to +8.7 pp for k1/k5 across wc and ic:2) and in both wc beta=1 cells (k1, k5), while random_rewire alone adds only +0.1 to +2.9 pp at beta >= 3 (it can be negative at beta < 3, e.g. k5 ic:2 b1 -1.3 pp). Unlike TIMME, where the residual over random was small and sign-unstable, here it is the bulk of the gain. (2) Which target and which node matter: rehome_text beats rehome_struct (CI > 0 in every beta >= 3 cell) and rehome_text_random_nodes (CI > 0 in every beta >= 3 cell, despite the latter moving ~7-12% more edges). (3) At low beta the effect vanishes or is ambiguous: k5 wc b0.3 text-rand +0.4 [-0.9, 1.7]; k1 ic:2 b0.3 and k1 ic:2 b1 CIs include 0 for all contrasts vs none/random (k1 wc b0.3 is positive but wide, text-rand +1.9 [0.5, 4.1]); k5 ic:2 b0.3 is positive vs random (+2.2 [1.0, 3.3]) mainly because random rewiring slightly hurts there. Since Pokec has no measured beta, the headline holds only conditional on strong content-selective exposure.

Caveats: ONE rewiring draw per k and 100 sims (CIs do not cover rewiring-draw variance); Pokec drops are random own-community edges (no dormancy), so the drop side is cruder than TIMME's; correction-reach baselines are high (50-59% at beta >= 3) because the slice is small and dense (mean degree ~16), which leaves headroom-limited but still sizeable gains; p_C accuracy on Pokec text is imperfect, so "misfit" is partly classifier error, and the sim takes tc at face value; AA from the original graph; friendships treated as bidirectional transmission.
