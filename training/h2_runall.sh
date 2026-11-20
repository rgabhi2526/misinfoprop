#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
P=/opt/anaconda3/bin/python
$P h2_filter.py --seeds 8 --sims 60 --procs 6 > ../results/hypotheses/h2/synth.log 2>&1
$P h2_real.py pokec_knm --reps 8 --sims 40 --procs 6 > ../results/hypotheses/h2/real_pokec.log 2>&1
$P h2_real.py timme --reps 6 --sims 25 --procs 6 --piv 100 > ../results/hypotheses/h2/real_timme.log 2>&1
$P h2_edges.py > ../results/hypotheses/h2/edges.log 2>&1
touch ../results/hypotheses/h2/ALLDONE
