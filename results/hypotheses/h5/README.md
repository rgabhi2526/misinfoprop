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
