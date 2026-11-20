# H6 — can misfit detection catch structural community deception?

**Question.** An attacker hides target nodes from community detection by rewiring only their edges (Kumari et al. 2021; Fionda & Pirro safeness). Bio text is untouched. Does the misfit score (1 - p_C[own community], text-only, cross-fitted) flag the hidden nodes better than structure-only scores?

**Verdict (revised after review): no. Misfit is at best a weak, mostly non-specific signal; structure-only scores (participation, persistence_low) detect rewired targets better, and misfit's apparent success on "successful" targets is circular.** (1) Over all targets misfit is 0.48 to 0.57 against a depressed unattacked baseline of 0.44 to 0.45 (targets are drawn below the top-10% misfit); on the like-for-like population (negatives also below the original top-10% misfit) misfit gains +0.04 to +0.10 over its unattacked 0.49 to 0.51 under kumari/safeness (e.g. Pokec kumari k=5 0.599 vs 0.506) but only +0.01 to +0.03 under the matched random rewire, while participation/persistence_low reach 0.56 to 0.82 in the same setting, and also respond strongly to the matched random rewire (Pokec k=5: participation 0.744). (2) Success is defined as a label flip and misfit is how little the text supports the new label, so misfit must rise for successful targets by construction. The matched random control confirms there is no attack-specific signal: misfit on successful targets is 0.66/0.68 (TIMME k=2/5) and 0.62/0.66 (Pokec) for the matched random rewire vs 0.64/0.66 and 0.60/0.60 for kumari. (3) The real test, successful vs unsuccessful targets under the same attack (s-vs-f), gives misfit 0.64 to 0.74 for kumari and 0.74 to 0.76 for safeness, but the matched random control gives the same (0.73 to 0.75 TIMME, 0.69 to 0.73 Pokec). So the misfit rise is explained by "the node changed label", not by the attack being a deception attack. For kumari at Pokec k=5 misfit's s-vs-f (0.64) is below participation under safeness and above participation/persistence under kumari, but that is a statement about which structural scores are blind after a successful move (persistence_low s-vs-f 0.10 means successful kumari targets look MORE persistent than failed ones), not about misfit being a detector. Limited to this simplified attacker, one resolution, 3 seeds sharing one t0.

## Setup
- Code: `training/h6_deception.py` (`python h6_deception.py timme|pokec`, `--selftest`). Runtime after the fix: TIMME 5.5 min, Pokec 2.3 min (Pool(4)); 3 seeds.
- Data: TIMME undirected follow graph (`h4b_inter.npz`, LCC) with bio-only MiniLM (`timme_bio_lcc.npy`, has_bio flag); t0 = `h4b_scores_r1.0_s0.npz['t']` (Leiden, resolution 1.0, 4 communities >= 1% size). Pokec: `pokec_knm` bundle, 384-d interest-text vectors (not bio MiniLM), `has` flag, t0 from `pokec_knm_views/ambiguity.npz`.
- Targets per seed (0,1,2): 200 random nodes, t0 >= 0, degree >= 3, has text, below the original top-10% misfit threshold. t0 is shared by all seeds; seeds vary targets, attack RNG and Leiden seeds.
- Attacks, budget k in {2,5} rewires per target (each step removes one edge and adds one; additions are never self-loops, existing neighbours or duplicates; a step is skipped if no valid removal/addition):
  - `kumari`: destination = neighbouring community with most of u's edges (random other community if none); remove edge to the own-community neighbour with highest internal degree, add edge to the highest-internal-degree non-neighbour of the destination. Internal degrees come from the original graph, fixed, so targets pile onto the same hubs.
  - `safeness`: remove a random own-community edge, add an edge to a random node of a random other community.
  - `random` (matched control, new): remove a random edge of u, add an edge from u to a uniformly random non-neighbour (any community). Same k and targets.
  - `noise` (kept, trivial): remove a random edge of u, add a random non-edge anywhere not incident to u. This is the Leiden-instability floor, not a rewire control.
- After the attack: 10 Leiden runs (resolution 1.0), t1 = highest-modularity run with communities < 1% set to -1. Success = t1[u] differs from the majority t1 label of u's non-target original community-mates.
- Scores w.r.t. t1: misfit (`ambiguity.content_probs`, 5-fold cross-fit), participation, consensus_entropy (10 runs aligned to the best run), persistence_low = 1 - zeta (zeta = 0.5 (I / (max(Ex,1) K) + in-community clustering coeff); ponytail: Ex floored at 1 to avoid /0), degree.
- Detection: ROC-AUC. Uncertainty: mean +- sd over the 3 seeds (np.std, ddof=0; seeds share t0 and sometimes targets, so sd understates uncertainty; no bootstrap). Effective rewires differ by attack: kumari/safeness need an own-community neighbour to drop (Pokec k=5 about 910 to 920 of 1000), random/noise about 970 to 975, so success rates compare attacks at unequal effective budgets.
- New in this round: `succ` baseline on the unattacked graph for the same successful targets; `s-vs-f`; `all_low`; matched random control.

## Results
Deception success (mean over seeds): TIMME kumari 0.23/0.44, safeness 0.19/0.33, matched random 0.09/0.15, noise 0.06/0.14. Pokec kumari 0.37/0.67, safeness 0.28/0.49, matched random 0.16/0.28, noise 0.14/0.29. The matched-random and noise rates are the Leiden instability floor plus whatever random rewiring achieves, so a quarter of the kumari effect at Pokec k=5 (0.28 of 0.67) is not attack-specific.

### TIMME (n=20811, 4 communities)

Cells: mean over 3 seeds (sd, ddof=0, in the JSON). `all` = all targets vs all non-target nodes with text; `all_low` (unatt) = negatives restricted to nodes below the original top-10% misfit threshold, with the same quantity on the unattacked graph in brackets; `succ` (unatt) = successful targets vs non-targets, with the same successful targets on the unattacked graph in brackets; `s-vs-f` = successful vs unsuccessful targets under the same attack. `unatt` (all targets, all negatives) is in the JSON.

| attack | k | success (n_succ/seed) | rewires done/budget | score | all | all_low (unatt) | succ (unatt) | s-vs-f |
|---|---|---|---|---|---|---|---|---|
| kumari | 2 | 0.232 (45/39/55) | 398/400/398/400 | misfit | 0.487 | 0.532 (0.494) | 0.640 (0.465) | 0.713 |
|  |  |  |  | participation | 0.541 | 0.557 (0.514) | 0.575 (0.601) | 0.560 |
|  |  |  |  | consensus_entropy | 0.505 | 0.517 (0.487) | 0.614 (0.603) | 0.648 |
|  |  |  |  | persistence_low | 0.542 | 0.561 (0.512) | 0.446 (0.485) | 0.378 |
|  |  |  |  | degree | 0.534 | 0.533 (0.533) | 0.330 (0.330) | 0.217 |
| kumari | 5 | 0.437 (84/81/97) | 954/969/943/1000 | misfit | 0.528 | 0.575 (0.494) | 0.655 (0.472) | 0.738 |
|  |  |  |  | participation | 0.562 | 0.580 (0.514) | 0.577 (0.621) | 0.571 |
|  |  |  |  | consensus_entropy | 0.511 | 0.521 (0.487) | 0.530 (0.540) | 0.534 |
|  |  |  |  | persistence_low | 0.564 | 0.584 (0.512) | 0.482 (0.529) | 0.363 |
|  |  |  |  | degree | 0.534 | 0.533 (0.533) | 0.391 (0.391) | 0.230 |
| safeness | 2 | 0.187 (38/35/39) | 398/400/398/400 | misfit | 0.484 | 0.529 (0.494) | 0.663 (0.477) | 0.735 |
|  |  |  |  | participation | 0.588 | 0.606 (0.514) | 0.713 (0.603) | 0.675 |
|  |  |  |  | consensus_entropy | 0.515 | 0.528 (0.487) | 0.733 (0.608) | 0.770 |
|  |  |  |  | persistence_low | 0.579 | 0.600 (0.512) | 0.737 (0.443) | 0.727 |
|  |  |  |  | degree | 0.534 | 0.534 (0.533) | 0.296 (0.295) | 0.188 |
| safeness | 5 | 0.333 (67/59/74) | 954/969/943/1000 | misfit | 0.522 | 0.569 (0.494) | 0.691 (0.452) | 0.760 |
|  |  |  |  | participation | 0.665 | 0.682 (0.514) | 0.767 (0.634) | 0.709 |
|  |  |  |  | consensus_entropy | 0.534 | 0.544 (0.487) | 0.670 (0.546) | 0.709 |
|  |  |  |  | persistence_low | 0.647 | 0.668 (0.512) | 0.775 (0.501) | 0.750 |
|  |  |  |  | degree | 0.534 | 0.533 (0.533) | 0.363 (0.363) | 0.228 |
| random | 2 | 0.087 (17/16/19) | 400/400/400/400 | misfit | 0.459 | 0.502 (0.494) | 0.663 (0.486) | 0.736 |
|  |  |  |  | participation | 0.545 | 0.561 (0.514) | 0.694 (0.604) | 0.678 |
|  |  |  |  | consensus_entropy | 0.478 | 0.496 (0.487) | 0.791 (0.663) | 0.841 |
|  |  |  |  | persistence_low | 0.567 | 0.587 (0.512) | 0.747 (0.381) | 0.747 |
|  |  |  |  | degree | 0.534 | 0.533 (0.533) | 0.275 (0.275) | 0.199 |
| random | 5 | 0.145 (24/29/34) | 974/984/970/1000 | misfit | 0.469 | 0.514 (0.494) | 0.679 (0.480) | 0.753 |
|  |  |  |  | participation | 0.599 | 0.617 (0.514) | 0.716 (0.543) | 0.670 |
|  |  |  |  | consensus_entropy | 0.487 | 0.502 (0.487) | 0.695 (0.606) | 0.747 |
|  |  |  |  | persistence_low | 0.657 | 0.679 (0.512) | 0.819 (0.395) | 0.770 |
|  |  |  |  | degree | 0.534 | 0.533 (0.533) | 0.268 (0.269) | 0.168 |
| noise | 2 | 0.057 (11/12/11) | 400/400/400/400 | misfit | 0.450 | 0.493 (0.494) | 0.587 (0.562) | 0.658 |
|  |  |  |  | participation | 0.479 | 0.493 (0.514) | 0.464 (0.644) | 0.479 |
|  |  |  |  | consensus_entropy | 0.465 | 0.483 (0.487) | 0.648 (0.643) | 0.693 |
|  |  |  |  | persistence_low | 0.496 | 0.514 (0.512) | 0.565 (0.485) | 0.583 |
|  |  |  |  | degree | 0.500 | 0.499 (0.533) | 0.224 (0.279) | 0.210 |
| noise | 5 | 0.138 (32/23/28) | 974/984/970/1000 | misfit | 0.443 | 0.487 (0.494) | 0.452 (0.444) | 0.504 |
|  |  |  |  | participation | 0.543 | 0.557 (0.514) | 0.870 (0.496) | 0.878 |
|  |  |  |  | consensus_entropy | 0.462 | 0.472 (0.487) | 0.506 (0.564) | 0.548 |
|  |  |  |  | persistence_low | 0.546 | 0.562 (0.512) | 0.882 (0.329) | 0.887 |
|  |  |  |  | degree | 0.459 | 0.458 (0.533) | 0.119 (0.215) | 0.108 |

### Pokec slice (n=5599, 11 communities)

Cells: mean over 3 seeds (sd, ddof=0, in the JSON). `all` = all targets vs all non-target nodes with text; `all_low` (unatt) = negatives restricted to nodes below the original top-10% misfit threshold, with the same quantity on the unattacked graph in brackets; `succ` (unatt) = successful targets vs non-targets, with the same successful targets on the unattacked graph in brackets; `s-vs-f` = successful vs unsuccessful targets under the same attack. `unatt` (all targets, all negatives) is in the JSON.

| attack | k | success (n_succ/seed) | rewires done/budget | score | all | all_low (unatt) | succ (unatt) | s-vs-f |
|---|---|---|---|---|---|---|---|---|
| kumari | 2 | 0.373 (81/71/72) | 399/399/397/400 | misfit | 0.500 | 0.541 (0.506) | 0.603 (0.458) | 0.672 |
|  |  |  |  | participation | 0.598 | 0.603 (0.525) | 0.600 (0.617) | 0.530 |
|  |  |  |  | consensus_entropy | 0.478 | 0.480 (0.482) | 0.521 (0.597) | 0.575 |
|  |  |  |  | persistence_low | 0.662 | 0.662 (0.520) | 0.517 (0.498) | 0.253 |
|  |  |  |  | degree | 0.561 | 0.558 (0.558) | 0.389 (0.388) | 0.184 |
| kumari | 5 | 0.670 (137/135/130) | 914/912/919/1000 | misfit | 0.554 | 0.599 (0.506) | 0.598 (0.447) | 0.638 |
|  |  |  |  | participation | 0.563 | 0.568 (0.525) | 0.510 (0.552) | 0.352 |
|  |  |  |  | consensus_entropy | 0.434 | 0.436 (0.482) | 0.427 (0.529) | 0.475 |
|  |  |  |  | persistence_low | 0.588 | 0.589 (0.520) | 0.446 (0.487) | 0.098 |
|  |  |  |  | degree | 0.562 | 0.558 (0.558) | 0.452 (0.452) | 0.117 |
| safeness | 2 | 0.277 (60/57/49) | 399/399/397/400 | misfit | 0.510 | 0.553 (0.506) | 0.658 (0.463) | 0.713 |
|  |  |  |  | participation | 0.702 | 0.709 (0.525) | 0.819 (0.620) | 0.732 |
|  |  |  |  | consensus_entropy | 0.529 | 0.533 (0.482) | 0.695 (0.615) | 0.728 |
|  |  |  |  | persistence_low | 0.650 | 0.651 (0.520) | 0.703 (0.492) | 0.597 |
|  |  |  |  | degree | 0.562 | 0.559 (0.558) | 0.364 (0.363) | 0.189 |
| safeness | 5 | 0.488 (100/98/95) | 914/909/919/1000 | misfit | 0.565 | 0.609 (0.506) | 0.683 (0.450) | 0.739 |
|  |  |  |  | participation | 0.814 | 0.821 (0.525) | 0.850 (0.586) | 0.665 |
|  |  |  |  | consensus_entropy | 0.571 | 0.574 (0.482) | 0.680 (0.556) | 0.715 |
|  |  |  |  | persistence_low | 0.731 | 0.732 (0.520) | 0.737 (0.478) | 0.537 |
|  |  |  |  | degree | 0.561 | 0.558 (0.558) | 0.406 (0.406) | 0.152 |
| random | 2 | 0.158 (29/27/39) | 400/400/400/400 | misfit | 0.469 | 0.511 (0.506) | 0.623 (0.508) | 0.686 |
|  |  |  |  | participation | 0.641 | 0.647 (0.525) | 0.776 (0.613) | 0.711 |
|  |  |  |  | consensus_entropy | 0.499 | 0.500 (0.482) | 0.680 (0.581) | 0.714 |
|  |  |  |  | persistence_low | 0.614 | 0.615 (0.520) | 0.668 (0.469) | 0.570 |
|  |  |  |  | degree | 0.561 | 0.558 (0.558) | 0.322 (0.323) | 0.176 |
| random | 5 | 0.278 (66/55/46) | 975/970/967/1000 | misfit | 0.497 | 0.537 (0.506) | 0.660 (0.487) | 0.733 |
|  |  |  |  | participation | 0.744 | 0.751 (0.525) | 0.794 (0.539) | 0.623 |
|  |  |  |  | consensus_entropy | 0.529 | 0.531 (0.482) | 0.697 (0.587) | 0.735 |
|  |  |  |  | persistence_low | 0.706 | 0.706 (0.520) | 0.706 (0.418) | 0.507 |
|  |  |  |  | degree | 0.562 | 0.558 (0.558) | 0.330 (0.329) | 0.136 |
| noise | 2 | 0.143 (24/37/25) | 400/400/400/400 | misfit | 0.466 | 0.506 (0.506) | 0.566 (0.505) | 0.621 |
|  |  |  |  | participation | 0.498 | 0.504 (0.525) | 0.617 (0.638) | 0.639 |
|  |  |  |  | consensus_entropy | 0.487 | 0.491 (0.482) | 0.739 (0.701) | 0.792 |
|  |  |  |  | persistence_low | 0.517 | 0.518 (0.520) | 0.561 (0.520) | 0.550 |
|  |  |  |  | degree | 0.508 | 0.505 (0.558) | 0.297 (0.371) | 0.252 |
| noise | 5 | 0.287 (58/54/60) | 974/967/967/1000 | misfit | 0.449 | 0.487 (0.506) | 0.462 (0.467) | 0.517 |
|  |  |  |  | participation | 0.537 | 0.541 (0.525) | 0.699 (0.549) | 0.713 |
|  |  |  |  | consensus_entropy | 0.473 | 0.474 (0.482) | 0.516 (0.591) | 0.562 |
|  |  |  |  | persistence_low | 0.573 | 0.574 (0.520) | 0.701 (0.425) | 0.684 |
|  |  |  |  | degree | 0.431 | 0.428 (0.558) | 0.155 (0.328) | 0.130 |

## Reading the numbers
- Unattacked `all` misfit AUC is 0.44 to 0.45 by construction (targets below the top-10% misfit, negatives include the existing high-misfit nodes). `all_low` removes that: unattacked 0.49 to 0.51, so "near chance" there is accurate, and attacked values of 0.53 to 0.61 (kumari/safeness) are modest gains.
- `succ` has its own baseline now (brackets): misfit on the same successful targets before the attack is 0.44 to 0.49, so the succ rise to 0.60 to 0.69 is real but equally present for matched random (0.62 to 0.68).
- Degree: 0.50 to 0.56 over all targets for most attacks, but 0.43 to 0.46 for noise/random k=5 (u loses edges) and 0.12 to 0.45 for successful targets in every attack (successful targets lost degree or are low-degree).
- Successful counts per seed are small for the controls (TIMME matched random k=2: 17/16/19; noise k=2: 11/12/11; Pokec matched random k=5: 66/55/46). s-vs-f and succ AUCs for those cells are noisy.

## Limitations
- The attacker does not fake content; a content-aware attacker would also edit the bio.
- Simplified published attacks (static internal degrees, no persistence-driven stopping, no budget optimisation, no knowledge of the detector).
- One resolution (1.0), one t0 for all seeds, 3 seeds, sd over seeds with ddof=0.
- 4 communities on TIMME make "move" a coarse event; Pokec (11) is the finer test.
- Success is defined from Leiden labels and misfit is defined from the same labels, hence the circularity noted above; s-vs-f does not remove it, the matched random control is what shows it.
- Negatives include pre-existing misfits and nodes Leiden moves for non-attack reasons.
