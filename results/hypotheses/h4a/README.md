# H4a — can misfit DETECT infiltrators / campaigns / recruiters with realistic text? (sweep + main run done)

Synthetic DC-SBM (n=4000, 8 communities), `training/h4a_detect.py`, tables `tables_sweep.md` (8 seeds). Text accuracy calibrated to classifier accuracy (0.21-0.90).
The 'main' run (24 configs x 10 seeds, `tables_main.md`) was rerun after a disk-full crash and agrees with the sweep (text acc 0.61: misfit AUC 0.77 vs best baseline 0.51 for infiltrators; hub/peripheral placement and recruiters lose to pagerank/low-degree/topology ambiguity at 0.98-1.00; text acc 0.30: 0.62 vs 0.51). Written by main session from the tables.

## Results (AUC; gain = misfit minus best baseline, paired CI over seeds; * = CI > 0)
- Infiltrators with RANDOM placement: baselines are near chance (best = betweenness, AUC 0.51) since nothing structural marks them.
  Misfit AUC at text-acc 0.56: 0.57 (mix 0.2), 0.66 (0.4), 0.75 (0.6), 0.83 (0.8), 0.89 (1.0) -> gains +0.05 to +0.37 (all CI>0).
  At text-acc 0.46 and mix >= 0.4: +0.11 to +0.31. At weak text (acc 0.27-0.33) needs mix >= 0.6 (+0.08 to +0.21). Below acc ~0.25 or mix <= 0.2: no reliable gain.
- Coordinated campaigns (groups sharing a text vector across communities): gain +0.06 (acc 0.21) to +0.35 (acc 0.89); +0.21 at acc 0.56. Precision@k is low (0.07-0.29) -- ranking helps, top-k is not clean.
- Recruiters (hub, 40% cross-community edges), HUB placement and PERIPHERAL placement: misfit LOSES clearly to structural baselines
  (pagerank AUC 1.00, topology ambiguity 0.98, low-degree 0.98): gains -0.06 to -0.44 even at text-acc 0.88. Same as H1.

## Verdict
Infiltrator/sockpuppet detection IS a defensible *conditional* headline: misfit works when the bad accounts have no structural signature and carry a moderately strong,
mostly-mismatched text signal (needs text->community accuracy >= ~0.45 with >= 40% off-community text, or >= ~0.3 with >= 60%). It is the wrong tool for
recruiters/hubs or peripheral accounts, where degree/PageRank/low-degree already give AUC ~0.98-1.0.
Caveats: synthetic and partly circular (planted text mismatch is what misfit measures); the 'random placement' baseline weakness is by construction;
real TIMME text accuracy is 0.62 (Pokec 0.28), so TIMME-like data sits in the working region, Pokec-like data does not.
Real-data validation is still the 11-node TIMME party check (AUC 0.68).
