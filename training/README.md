# Embedding training — BGRL / DGI / MVGRL

One script + one notebook per dataset. All three models share the same GCN
encoder and training harness (`gclib.py`), so a run only varies the *objective
family* — the comparison stays fair (docs/MODELS.md §5).

```
gclib.py            GCN encoder, BGRL/DGI/MVGRL, train loop, diffusion, eval
common.py           id remap, symmetrize, feature standardize, CLI args
train_<ds>.py       per-dataset loader (load_data) + CLI main
train_<ds>.ipynb    same, as a notebook
requirements.txt
```

## Run

```bash
pip install -r requirements.txt          # torch + PyG must match your CUDA
python train_timme.py                     # all 3 models, full data
python train_reddit.py     --limit 100000 # quick sanity pass on a slice
python train_pokec.py --region "zilinsky kraj" # small slice; omit for full 1.63M
python train_gab.py        --limit 200000 # 48GB dump; parse+encode cached once
python eval_embeddings.py timme           # full metrics report on saved embeddings
python plot_umap.py timme                 # UMAP grid (timme | reddit)
```

Common flags: `--dim 256 --layers 2 --epochs 500 --clusters K --models bgrl dgi mvgrl --device cuda --limit N`.
Outputs go to `out/<ds>/`: `<ds>_<model>_emb.pt` (node embeddings) and
`<ds>_metrics.json` (best-epoch metrics, wall-clock, and the full `metrics.report`:
health / labeled / unlabeled / task sections — see `metrics.py`).

## Per-dataset notes

Final datasets (T0, 2026-09-27): **timme** and **pokec**. gab and reddit are reserve.
All text goes through `common.encode_texts` (`paraphrase-multilingual-MiniLM-L12-v2`,
384-d + `has_metadata` flag); nodes without metadata are kept.

| Dataset | Graph | Node features | Labels (NMI) |
|---|---|---|---|
| **timme** | `P_all` (21k users): pooled follow/mention/favorite/reply/retweet | bio + latest tweet text → multilingual encoder | R/D: `dict.csv` + `additional_labels/` (1,206 in LCC); independents unlabeled |
| **reddit** | subreddit→subreddit hyperlinks (body+title) | shipped 300-d subreddit embeddings | none |
| **pokec** | 30.6M directed friendships (1.63M users) | interest free text (Slovak) → multilingual encoder + gender, age, region one-hot | none |
| **gab** | repost: actuser→author; reply/quote: author→parent author | sentence-transformer over each author's own posts | none |

## Design decisions (from docs/MODELS.md)

- **Largest connected component only** — `make_data` drops isolated nodes and
  small components (40% of reddit was degree-0); `data.orig_idx` maps kept
  nodes back, and loaders return labels already indexed by it.
- **Best epoch selected by modularity, never NMI** — selecting on the labels you
  report against is leakage. Treat best-epoch modularity as selection-biased.
- **Symmetrized for embedding only** — `common.make_data` undirects a copy;
  raw directed edges stay in `data_final/` for downstream diffusion/centrality.
- **Same feature matrix into all 3 models** per dataset → output differences are
  attributable to the model, not the input.
- **MVGRL diffusion** uses PyG `GDC` (sparse/approx PPR), never a dense N² matrix.
  It is CPU-bound preprocessing — computed once in `run_all`; cache
  `data.diff_edge_index/diff_edge_weight` to disk to avoid recomputing per run.
- **Scale (VoterFraud, 1.9M nodes)** — full-batch on an A100 with sparse ops.
  The trainers are batch-agnostic, so wrap them in a `NeighborLoader` if VRAM is
  tight; use `--limit` for a quick pass first.

## Not built

- Gab feature cap (`MAX_CHARS=4000`) and encoder (`all-MiniLM-L6-v2`) are
  defaults — raise/swap in `train_gab.py` if the ideology signal looks thin.
- Cluster count `K` for the label-less datasets (reddit, gab) is a guess (10).
  Set `--clusters` once you have a target community count.
