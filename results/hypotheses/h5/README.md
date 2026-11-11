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
Counts are the number of settings (of 27) whose CI is > 0 / < 0 / spans 0. Per-setting means and CIs: `h5_robust.json`. Run: `python h5_rec.py --robust`.

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

Code: `training/h5_rewire.py` (`python h5_rewire.py timme`; `--selftest`). Raw: `h5_rewire_timme.json`. Runtime ~16 min (Pool(4), 100 sims).

**Design.** Treated = the 1541 (r1.0_s0) / 1464 (r2.0_s0) misfits of T11.1. Each treated node drops up to k own-community follow edges with no engagement (retweet/like/mention/reply, either direction) and adds the same number to top-Adamic-Adar non-neighbours in a target community (ties random). Treated nodes' own degree is preserved exactly (edges never touch other treated nodes); their neighbours' degrees change. Interventions: `rehome_text` (target argmax p_C bio-only), `rehome_struct` (Kumari extraction community), `random_rewire` (same per-node drop counts, own-community random drops, random adds anywhere), `rehome_text_random_nodes` (same procedure on random non-misfit nodes with argmax p_C != t), `none`. Each rewiring changes ~6k edges of 892k.
**Spread.** Live-edge IC on the undirected follow graph; both directions of an edge are tested against their own gated probability min(1, p_uv K aff[v,c0]) using ONE shared uniform per (min,max,sim) hash, so shared edges get identical draws in every graph (CRN); seeds and sources identical across interventions. aff = softmax(beta * onehot(tc_v)). Models wc and ic:2. Outcomes: infected share; share of misfits reached; correction reach (random misfit u, 10 seeds uniform in {t == tc_u}, share of misfits with tc == tc_u reached). CIs: bootstrap over 100 sims of paired differences. Reduced grid: k in {1,5}, beta in {0.3,1,3,6}, r2.0_s0 only k=5 with beta {0.3,6}.

Correction reach, paired difference in percentage points (baseline `none` in brackets; 95% CI):

| setting | none | rehome_text - none | rehome_text - random_rewire | rehome_struct - none | text_random_nodes - none |
|---|---|---|---|---|---|
| r1.0 k5 wc b6 | 34.5 | +4.7 [3.0, 6.9] | +1.3 [-0.3, 2.2] | +3.0 [1.5, 5.1] | -0.6 [-2.1, 0.2] |
| r1.0 k5 wc b3 | 31.7 | +3.4 [2.2, 5.0] | +1.0 [-0.6, 1.9] | +2.1 [1.1, 3.6] | -0.6 [-2.2, 0.2] |
| r1.0 k5 ic:2 b6 | 7.1 | +0.46 [0.29, 0.70] | +0.53 [0.41, 0.66] | +0.27 [0.10, 0.49] | +0.15 [0.03, 0.36] |
| r1.0 k5 ic:2 b3 | 6.1 | +0.29 [0.19, 0.39] | +0.27 [-0.04, 0.51] | +0.15 [0.06, 0.23] | 0.00 [-0.25, 0.22] |
| r1.0 k1 wc b6 | 34.5 | +1.4 [1.1, 1.7] | +0.9 [0.7, 1.1] | +0.5 [0.4, 0.7] | 0.0 [-0.06, 0.05] |
| r2.0 k5 wc b6 | 40.9 | +3.4 [2.4, 4.5] | +2.2 [1.5, 3.1] | +1.0 [0.6, 1.5] | +0.9 [-0.08, 2.8] |
| r2.0 k5 ic:2 b6 | 12.0 | +1.07 [0.78, 1.38] | +1.40 [1.08, 1.74] | +0.08 [-0.06, 0.22] | +0.06 [0.03, 0.09] |
| r1.0 k5 wc b0.3 (real TIMME) | 2.9 | +0.1 [-0.4, 0.6] | +0.4 [-0.2, 1.2] | +0.1 [-0.1, 0.5] | -0.1 [-0.6, 0.4] |

Other outcomes (k=5, beta 6): share of misfits reached rises with rehome_text (r1.0 wc +0.98 pp [0.71, 1.25] vs none, vs random +0.64 [0.45, 0.82]; r2.0 wc +0.47 [0.30, 0.65], vs random -0.01); total infected share barely moves (+0.02 to +0.07 pp for rehome_text; CI excludes 0 but effect is negligible), as expected since edges moved are ~0.7% of the graph. In the real-TIMME beta (0.3) regime nothing is distinguishable from zero in any outcome.

**Verdict: PARTLY SUPPORTED, gated by content affinity.** When receivers are selective (beta >= 3), re-homing misfits toward their text community raises correction reach (the share of misfits reachable by a message seeded in their text community) by 0.3-4.7 pp over no rewiring, CI > 0 in every k=5 cell with beta >= 3 (both models, both resolutions), and it beats `rehome_text_random_nodes` (choosing misfits matters) and, point-estimate-wise, `rehome_struct` (text target beats structural target). It does NOT reliably beat `random_rewire` (CI includes 0 in 3 of the 7 beta>=3 cells shown above; random is also very noisy under wc because random adds create hubs), it does nothing for total spread, and at the empirical beta=0.3 it is null. Honest reading: the effect exists but is conditional on strong selective exposure, which H4b measured as weak on TIMME.

Caveats: 100 sims per cell (random_rewire variance under wc is large); one rewiring draw per (community, k); AA is computed on the original graph; undirected edges (a follow is treated as bidirectional transmission); r2.0 only k=5.
