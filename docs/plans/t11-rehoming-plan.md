# T11 — misfit re-homing: robustness + rewiring simulation

Spec: `TASKS.md` T11 and `SUGGESTIONS.md` S-25 (decisions agreed with the user 2026-11-11).
Existing code: `training/h5_probe.py`, `training/h5_rec.py`, results in `results/hypotheses/h5/`.

## Global Constraints
- Python: `/opt/anaconda3/bin/python`. Run scripts from `training/` (`cd "<repo>/training"`). The repo path has a trailing space: `.../Major project /` — always quote it.
- **Never start model training** (`train_*.py`, `gclib.run_all`, notebooks, GPU). Everything here is CPU analysis/simulation and is allowed.
- Text for content signals = **bio-only** MiniLM embeddings `training/out/timme_bio_lcc.npy` (20811 x 385, last column = has_bio flag), never the leaky `text`/GloVe in `h4b_scores_*.npz`. p_C = `ambiguity.content_probs(xb, t, seed=...)`.
- Communities: `training/out/h4b_scores_r{0.5,1.0,2.0}_s{0,1,2}.npz` (keys t, p_T, participation, pagerank, ...). Interactions: `training/out/h4b_inter.npz` (keys `<type>_s/_d/_c` for follow, like, mention, reply, retweet; LCC index).
- Misfit set and matched controls exactly as `h5_rec.run` defines them (top-10% bio misfit, has bio, >=5 held-out ties, argmax p_C != t; controls matched 1:1 on degree decile x participation decile, low misfit).
- Style: match the repo — short module docstring stating design + run command, numpy/scipy, `from metrics import log` for progress, no new dependencies, no classes unless needed. Each new script has ONE runnable self-check (`--selftest` or assert in `__main__`) on a tiny synthetic case.
- Results go to `results/hypotheses/h5/` (JSON + a section in `README.md`). Report numbers honestly, including negatives; 95% bootstrap CIs over users / simulations.
- Commits: author is the user only. NEVER add any `Co-Authored-By` or Claude attribution line. Commit only files you created/changed for the task (`git add <paths>`), never `git add -A` (the tree has unrelated uncommitted work).

## Task 1: Account-level robustness (T11.1b)
File: modify `training/h5_rec.py` (add a `robust` mode; keep existing behaviour as default).
Pre-registered comparison (fixed before results): primary scorer `aa_x_text` = AA * p_C(u)[t_v]; compare against `aa` and `aa_x_struct` = AA * p_T(u)[t_v]. Metric P@10 (and R@100 secondary).
- Run over all 9 community files (r0.5, r1.0, r2.0 x s0..s2) and bio-classifier seeds 0, 1, 2 (`content_probs(..., seed=)`) = 27 settings; for each, misfits and matched controls.
- For each setting report: mean P@10 per scorer, and the PAIRED per-user differences (aa_x_text − aa) and (aa_x_text − aa_x_struct) with 95% bootstrap CI.
- Summary: count of settings where each paired difference CI > 0, < 0, or spans 0, separately for misfits and controls; also "misfit-specific gain" = (diff on misfits) − (diff on controls) with bootstrap CI (independent groups).
- Write `results/hypotheses/h5/h5_robust.json` and append a "## T11.1b robustness" section to `results/hypotheses/h5/README.md` with the summary table and an explicit verdict line.
- Command: `python h5_rec.py --robust`. Self-check: existing `prec_rec` assert plus an assert that the paired-diff function returns 0 CI for identical inputs.

## Task 2: Rewiring simulation on TIMME (T11.2, M2)
File: create `training/h5_rewire.py`.
Graph = undirected follow graph F (from `h4b_inter.npz`), communities t from `h4b_scores_r1.0_s0.npz` (main) — also run r2.0_s0. Content community tc = argmax p_C (bio-only, seed 0).
Interventions, each with the SAME edge budget B = k edges per treated node, k in {1, 3, 5}:
- `rehome_text` (ours): treated = misfits; for each misfit u drop k follow edges to its own community t_u with NO engagement in either direction (retweet/like/mention/reply, from h4b_inter) — if fewer dormant edges exist, drop only those and add the same number; add the same number of edges to the top-ranked non-neighbours in community tc_u by `aa_x_text` (ties broken randomly). Degree of u preserved.
- `rehome_struct`: same treated nodes and drop rule, add to top AA nodes inside the structural extraction community (Kumari target: neighbouring community with most follow edges, excluding own).
- `random_rewire`: same treated nodes, drop random own-community edges, add random non-neighbour edges anywhere.
- `rehome_text_random_nodes`: same procedure as rehome_text but treated = random nodes with argmax p_C != t (same count) — tests whether choosing misfits matters.
- `none`: original graph.
Spread: live-edge Independent Cascade reusing `blocking.edge_probs` (models `wc` and `ic:1`, `ic:2`) and `blocking.reach`; content gating as in `h4d_sweep.py`: aff = exp(beta * onehot(tc)) row-normalised, edge u->v transmits with min(1, p_uv * K * aff[v, tc_u]) where K = number of communities; beta in {0.3, 1, 3, 6} (0.3 = real TIMME value from H4b). Common random numbers across interventions (same seeds and edge uniforms per sim; note rewired graphs have different edge sets — draw a uniform per (u,v) pair via a hashed RNG or a dense-per-pair approach so that shared edges see the same draw).
Outcomes per sim (sims = 200, n_seeds = 10 random seeds): (a) total infected share; (b) share of misfits reached; (c) **correction reach** = share of misfits reached when seeds are drawn from each misfit's content community tc (seeded uniformly among nodes with t == tc_u for a randomly chosen misfit u's tc).
Report per (k, beta, model): mean outcome per intervention and the paired difference vs `none` and vs `random_rewire` with 95% CI over sims. Write `results/hypotheses/h5/h5_rewire_timme.json` and a README section "## T11.2 rewiring simulation (TIMME)" with a verdict.
Command: `python h5_rewire.py timme`. Self-check `python h5_rewire.py --selftest`: on a 2-block planted graph, rewire preserves every treated node's degree, drops only own-community dormant edges, adds only edges into the target community, and never creates self-loops or duplicate edges.

## Task 3: Rewiring simulation on Pokec (T11.2 second dataset)
File: modify `training/h5_rewire.py` (add `pokec` mode). Pokec has no engagement layer: "dormant" is undefined, so drop rule = random own-community edges (state this in the README). Data: `training/out/pokec_knm_views/ambiguity.npz` (t, p_C, misfit, ...) and the graph from `views` for `pokec_knm` (see how `blocking.py`/`views.py` load it). Misfits = top-10% misfit with argmax p_C != t; no controls needed. Same interventions (except dormant rule), same spread/outcome grid. Write `results/hypotheses/h5/h5_rewire_pokec.json` and README section "## T11.2 rewiring simulation (Pokec)". Command: `python h5_rewire.py pokec`.

## Task 4: Docs
Update `TASKS.md` (T11.1b, T11.2 status lines with headline numbers), `DEVELOPED.md` (bump "Last updated", add a dated change-log line, add h5 files to evidence section if one lists hypothesis results), `SUGGESTIONS.md` S-25 (one line: outcome). Numbers must be copied from the JSON/README produced by Tasks 1-3, not invented. Do not touch unrelated lines (these files have other uncommitted edits — commit them anyway since they are docs the user maintains, but only after reading `git diff` to confirm the other edits are documentation).
