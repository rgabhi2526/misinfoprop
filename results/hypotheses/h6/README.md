# H6 — can misfit detection catch structural community deception?

**Question.** An attacker hides target nodes from community detection by rewiring only their edges (Kumari et al. 2021; Fionda & Pirro safeness). Bio text is untouched. Does the misfit score (1 - p_C[own community], text-only, cross-fitted) flag the hidden nodes better than structure-only scores?

**Verdict (no, not as an all-target detector; yes, partially, for successful kumari-type moves).** Over all targets misfit is near chance (AUC 0.48 to 0.57) while participation and persistence_low reach 0.58 to 0.81 under `safeness` and persistence_low is best under `kumari` (Pokec 0.66/0.59 vs misfit 0.50/0.55; TIMME 0.54/0.56 vs 0.49/0.53). Restricted to successfully hidden targets, misfit (0.60 to 0.66) is the best score in all 4 `kumari` cells (marginally at Pokec k=2: 0.603 vs participation 0.600) but is beaten by participation/persistence_low in all 4 `safeness` cells (misfit 0.66 to 0.69 vs 0.71 to 0.85). Reason: when the attack succeeds, the node sits in a community its text does not match, so misfit fires; unsuccessful rewires leave misfit unchanged but still change structural mixing. Limited to this simplified attacker, one resolution, 3 seeds.

## Setup
- Code: `training/h6_deception.py` (`python h6_deception.py timme|pokec`, `--selftest`). Runtime: TIMME 4 min, Pokec 2 min (Pool(4)); all 3 seeds kept.
- Data: TIMME undirected follow graph (`h4b_inter.npz`, LCC) with bio-only MiniLM (`timme_bio_lcc.npy`, has_bio flag); original communities t0 = `h4b_scores_r1.0_s0.npz['t']` (Leiden, resolution 1.0, 4 communities >= 1% size). Pokec: `pokec_knm` bundle, 384-d interest-text vectors, `has` flag, t0 from `pokec_knm_views/ambiguity.npz`.
- Targets per seed (0,1,2): 200 random nodes with t0 >= 0, degree >= 3, has text, below the top-10% misfit threshold on the original graph. t0 is the same for all seeds; seeds vary targets, attack RNG and Leiden seeds.
- Attacks, budget k in {2,5} rewires per target (remove one edge + add one edge per step; no self-loops or duplicates; step skipped if no valid removal/addition):
  - `kumari`: destination = neighbouring community with most of u's edges (random other community if u has none); remove edge to the own-community neighbour with highest internal degree, add edge to the highest-internal-degree non-neighbour in the destination. Internal degrees are from the original graph and fixed, so targets pile onto the same hubs.
  - `safeness`: remove a random own-community edge, add an edge to a random node of a random other community.
  - `random` (control): remove a random edge of u, add a random non-edge anywhere (not incident to u, literal reading of "anywhere"). It is a control for global noise, so it leaves u's neighbourhood mostly alone and shifts other nodes' edges.
- After the attack: 10 Leiden runs (resolution 1.0), t1 = highest-modularity run with communities < 1% set to -1. Deception success = t1[u] differs from the majority t1 label of u's non-target original community-mates.
- Scores w.r.t. t1 on the attacked graph: misfit (`ambiguity.content_probs` on bio/interest text, 5-fold cross-fit), participation, consensus_entropy (10 runs, aligned to the best run), persistence_low = 1 - zeta (zeta = 0.5 (I / (max(Ex,1) K) + in-community clustering coefficient); ponytail: Ex floored at 1 to avoid division by zero), degree.
- Detection: ROC-AUC of targets vs all non-target nodes with text. Uncertainty: mean +- sd over the 3 seeds (not a bootstrap). Unattacked AUC uses the original graph and t0 with the same targets.

## Results
Deception success (mean over seeds) and rewires actually applied (budget 400/1000 per seed; shortfalls come only from nodes with no valid removable edge):
- TIMME: kumari 0.23 (k=2), 0.44 (k=5); safeness 0.19, 0.33; random control 0.06, 0.14.
- Pokec: kumari 0.37, 0.67; safeness 0.28, 0.49; random control 0.12, 0.29.

The random-control success rate is the Leiden instability floor (communities are re-found differently after any perturbation, 4 or 11 communities). Success rates of the real attacks exceed that floor, but at k=2 on TIMME most targets are still not hidden.

### TIMME (n=20811, 4 communities)

Cells: mean +- sd of per-seed AUC over 3 seeds. `all` = all targets vs non-targets; `succ` = successful targets only; `unatt` = same targets on the unattacked graph.

| attack | k | success rate | score | AUC all | AUC succ | AUC unatt |
|---|---|---|---|---|---|---|
| kumari | 2 | 0.232 +- 0.033 | misfit | 0.487 +- 0.011 | 0.640 +- 0.014 | 0.444 +- 0.009 |
|  |  |  | participation | 0.541 +- 0.018 | 0.575 +- 0.028 | 0.497 +- 0.026 |
|  |  |  | consensus_entropy | 0.505 +- 0.001 | 0.614 +- 0.014 | 0.466 +- 0.006 |
|  |  |  | persistence_low | 0.542 +- 0.009 | 0.446 +- 0.022 | 0.492 +- 0.010 |
|  |  |  | degree | 0.534 +- 0.024 | 0.330 +- 0.022 | 0.534 +- 0.024 |
| kumari | 5 | 0.437 +- 0.035 | misfit | 0.528 +- 0.003 | 0.655 +- 0.010 | 0.444 +- 0.009 |
|  |  |  | participation | 0.562 +- 0.020 | 0.577 +- 0.024 | 0.497 +- 0.026 |
|  |  |  | consensus_entropy | 0.511 +- 0.011 | 0.530 +- 0.033 | 0.466 +- 0.006 |
|  |  |  | persistence_low | 0.564 +- 0.026 | 0.482 +- 0.050 | 0.492 +- 0.010 |
|  |  |  | degree | 0.534 +- 0.024 | 0.391 +- 0.033 | 0.534 +- 0.024 |
| safeness | 2 | 0.187 +- 0.008 | misfit | 0.484 +- 0.022 | 0.663 +- 0.047 | 0.444 +- 0.009 |
|  |  |  | participation | 0.588 +- 0.017 | 0.713 +- 0.043 | 0.497 +- 0.026 |
|  |  |  | consensus_entropy | 0.515 +- 0.009 | 0.733 +- 0.043 | 0.466 +- 0.006 |
|  |  |  | persistence_low | 0.579 +- 0.006 | 0.737 +- 0.039 | 0.492 +- 0.010 |
|  |  |  | degree | 0.534 +- 0.024 | 0.296 +- 0.040 | 0.534 +- 0.024 |
| safeness | 5 | 0.333 +- 0.031 | misfit | 0.522 +- 0.005 | 0.691 +- 0.012 | 0.444 +- 0.009 |
|  |  |  | participation | 0.665 +- 0.015 | 0.767 +- 0.016 | 0.497 +- 0.026 |
|  |  |  | consensus_entropy | 0.534 +- 0.009 | 0.670 +- 0.025 | 0.466 +- 0.006 |
|  |  |  | persistence_low | 0.647 +- 0.018 | 0.775 +- 0.020 | 0.492 +- 0.010 |
|  |  |  | degree | 0.534 +- 0.024 | 0.363 +- 0.031 | 0.534 +- 0.024 |
| random | 2 | 0.060 +- 0.023 | misfit | 0.456 +- 0.004 | 0.646 +- 0.030 | 0.444 +- 0.009 |
|  |  |  | participation | 0.483 +- 0.031 | 0.543 +- 0.132 | 0.497 +- 0.026 |
|  |  |  | consensus_entropy | 0.475 +- 0.007 | 0.694 +- 0.106 | 0.466 +- 0.006 |
|  |  |  | persistence_low | 0.497 +- 0.014 | 0.624 +- 0.114 | 0.492 +- 0.010 |
|  |  |  | degree | 0.500 +- 0.025 | 0.288 +- 0.070 | 0.534 +- 0.024 |
| random | 5 | 0.135 +- 0.014 | misfit | 0.435 +- 0.000 | 0.433 +- 0.034 | 0.444 +- 0.009 |
|  |  |  | participation | 0.546 +- 0.016 | 0.864 +- 0.054 | 0.497 +- 0.026 |
|  |  |  | consensus_entropy | 0.459 +- 0.003 | 0.518 +- 0.039 | 0.466 +- 0.006 |
|  |  |  | persistence_low | 0.546 +- 0.009 | 0.871 +- 0.058 | 0.492 +- 0.010 |
|  |  |  | degree | 0.459 +- 0.022 | 0.135 +- 0.039 | 0.534 +- 0.024 |

### Pokec slice (n=5599, 11 communities)

Cells: mean +- sd of per-seed AUC over 3 seeds. `all` = all targets vs non-targets; `succ` = successful targets only; `unatt` = same targets on the unattacked graph.

| attack | k | success rate | score | AUC all | AUC succ | AUC unatt |
|---|---|---|---|---|---|---|
| kumari | 2 | 0.373 +- 0.022 | misfit | 0.500 +- 0.023 | 0.603 +- 0.024 | 0.454 +- 0.021 |
|  |  |  | participation | 0.598 +- 0.013 | 0.600 +- 0.017 | 0.519 +- 0.011 |
|  |  |  | consensus_entropy | 0.478 +- 0.022 | 0.521 +- 0.049 | 0.482 +- 0.020 |
|  |  |  | persistence_low | 0.662 +- 0.011 | 0.517 +- 0.013 | 0.518 +- 0.013 |
|  |  |  | degree | 0.561 +- 0.015 | 0.389 +- 0.021 | 0.561 +- 0.014 |
| kumari | 5 | 0.670 +- 0.015 | misfit | 0.554 +- 0.002 | 0.598 +- 0.005 | 0.454 +- 0.021 |
|  |  |  | participation | 0.563 +- 0.014 | 0.510 +- 0.018 | 0.519 +- 0.011 |
|  |  |  | consensus_entropy | 0.434 +- 0.023 | 0.427 +- 0.033 | 0.482 +- 0.020 |
|  |  |  | persistence_low | 0.588 +- 0.010 | 0.446 +- 0.006 | 0.518 +- 0.013 |
|  |  |  | degree | 0.562 +- 0.015 | 0.452 +- 0.015 | 0.561 +- 0.014 |
| safeness | 2 | 0.277 +- 0.023 | misfit | 0.510 +- 0.004 | 0.658 +- 0.028 | 0.454 +- 0.021 |
|  |  |  | participation | 0.702 +- 0.007 | 0.819 +- 0.018 | 0.519 +- 0.011 |
|  |  |  | consensus_entropy | 0.529 +- 0.009 | 0.695 +- 0.038 | 0.482 +- 0.020 |
|  |  |  | persistence_low | 0.650 +- 0.009 | 0.703 +- 0.010 | 0.518 +- 0.013 |
|  |  |  | degree | 0.562 +- 0.014 | 0.364 +- 0.027 | 0.561 +- 0.014 |
| safeness | 5 | 0.488 +- 0.010 | misfit | 0.565 +- 0.011 | 0.683 +- 0.019 | 0.454 +- 0.021 |
|  |  |  | participation | 0.814 +- 0.005 | 0.850 +- 0.011 | 0.519 +- 0.011 |
|  |  |  | consensus_entropy | 0.571 +- 0.014 | 0.680 +- 0.023 | 0.482 +- 0.020 |
|  |  |  | persistence_low | 0.731 +- 0.003 | 0.737 +- 0.008 | 0.518 +- 0.013 |
|  |  |  | degree | 0.561 +- 0.015 | 0.406 +- 0.023 | 0.561 +- 0.014 |
| random | 2 | 0.120 +- 0.015 | misfit | 0.467 +- 0.015 | 0.643 +- 0.051 | 0.454 +- 0.021 |
|  |  |  | participation | 0.496 +- 0.015 | 0.679 +- 0.038 | 0.519 +- 0.011 |
|  |  |  | consensus_entropy | 0.481 +- 0.011 | 0.735 +- 0.009 | 0.482 +- 0.020 |
|  |  |  | persistence_low | 0.505 +- 0.020 | 0.597 +- 0.049 | 0.518 +- 0.013 |
|  |  |  | degree | 0.509 +- 0.016 | 0.365 +- 0.043 | 0.561 +- 0.014 |
| random | 5 | 0.287 +- 0.013 | misfit | 0.452 +- 0.009 | 0.471 +- 0.022 | 0.454 +- 0.021 |
|  |  |  | participation | 0.540 +- 0.010 | 0.690 +- 0.016 | 0.519 +- 0.011 |
|  |  |  | consensus_entropy | 0.472 +- 0.003 | 0.541 +- 0.005 | 0.482 +- 0.020 |
|  |  |  | persistence_low | 0.558 +- 0.016 | 0.663 +- 0.008 | 0.518 +- 0.013 |
|  |  |  | degree | 0.432 +- 0.020 | 0.161 +- 0.018 | 0.561 +- 0.014 |

## Reading the numbers
- Unattacked AUC for misfit is 0.44 to 0.45, below 0.5, by construction: targets are drawn from below the top-10% misfit, while negatives include the existing high-misfit nodes. Structural scores are ~0.5 there, as expected. So the attack-versus-unattacked gap for misfit is +0.04 to +0.12 at best (safeness/kumari k=5), small compared with participation (+0.17 safeness k=5 TIMME, +0.30 Pokec).
- `degree` is a sanity baseline: 0.5 to 0.56 over all targets (about the same as unattacked) and below 0.5 for successful targets, so the attack is not leaking through degree.
- Random control: misfit stays below 0.5 (0.43 to 0.47), so misfit is not just responding to noise; structural scores sit at 0.46 to 0.56 (random k=5 raises participation/persistence mildly).
- The restricted-to-successful AUCs are noisy for small n_succ (random control k=2 has ~12 to 24 successes per seed); read those as indicative.

## Limitations
- The attacker does not fake content; a content-aware attacker would also edit the bio.
- Simplified versions of the published attacks (static internal degrees, no persistence-driven stopping, no budget optimisation, no knowledge of the detector).
- One resolution (1.0), one t0 for all seeds, 3 seeds, sd over seeds only.
- Few communities on TIMME (4) make "move to another community" a coarse event; Pokec (11) is the finer test.
- Negatives include nodes that Leiden moves for non-attack reasons and pre-existing misfits.
