# Major project — misinformation / clustering disagreement

**Read `TASKS.md` first.** It is the single source of truth for scope and status.
`SUGGESTIONS.md` holds ideas and decisions, keyed by task ID.

## Doc map
- `TASKS.md` — what's done / next (T0…T9). Update status when work lands.
- `DEVELOPED.md` — what has been built, how it works, and why (pipeline, components,
  evidence, environment, change log).
- `SUGGESTIONS.md` — options, trade-offs, decisions (S-xx → T-x.y).
- `docs/MODELS.md` — why BGRL/DGI/MVGRL. `training/README.md` — how to run.
- `data_final/DATASETS.md` — dataset provenance. `docs/ideas/project-idea-1.md` — long-term idea.
- `DISCUSSION.md` — holds the disgreement between various sesions working on this project .
  This needs to be resolved , treat it as the point of common messaging between session so that a single source of truth prevails.
- `docs/papers/active_learning.pdf` — the AAS paper the disagreement step follows.

## Working rules
- Be blunt; criticise the plan when evidence says so. Discuss before large builds.
- Run Python with `/opt/anaconda3/bin/python` (conda base). `conda run` drops stdin.
- The project path has a trailing space: `.../Major project /`. Quote it.
- Commits: user is sole author, never add an AI `Co-Authored-By` line.
- **Never start model training yourself.** Training (`train_*.py`, notebooks,
  `gclib.run_all`, any GPU job) is run by the user or only after explicit approval
  for that specific run. Prepare the exact command and hand it over instead.
- After finishing a task, update `TASKS.md` / `SUGGESTIONS.md`.
- **Update `DEVELOPED.md` whenever something major changes** — a stage is built or
  reworked, a dataset/feature/model decision changes, a result supersedes an old one,
  or the environment changes. Edit the affected section, bump "Last updated", and add
  a dated line to its change log. A change isn't done until `DEVELOPED.md` reflects it.
