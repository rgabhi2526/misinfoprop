# Embedding training — BGRL / DGI / MVGRL

One script + one notebook per dataset. All three models share the same GCN
encoder and training harness (`gclib.py`), so a run only varies the *objective
family* — the comparison stays fair (MODELS.md §5).

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
python train_voterfraud.py --models bgrl  # one model only
python train_gab.py        --limit 200000 # gab is 6GB; start small
```

Common flags: `--dim 256 --layers 2 --epochs 500 --clusters K --models bgrl dgi mvgrl --device cuda --limit N`.
Outputs go to `out/<ds>/`: `<ds>_<model>_emb.pt` (node embeddings) and
`<ds>_metrics.json` (modularity, NMI where labels exist, wall-clock).

## Per-dataset notes

| Dataset | Graph | Node features | Labels (NMI) |
|---|---|---|---|
| **timme** | pooled follow/mention/favorite/reply/retweet | GloVe description+status (`features.npz`) | R/D party (`dict.csv`) |
| **reddit** | subreddit→subreddit hyperlinks (body+title) | shipped 300-d subreddit embeddings | none |
| **voterfraud** | retweet edges (all months) | user community centralities + one-hot community | `user_community` |
| **gab** | repost/reply/quote: actor→author | sentence-transformer over user's posts | none |

## Design decisions (from MODELS.md)

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
