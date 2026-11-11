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
