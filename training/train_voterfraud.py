"""VoterFraud2020 (Group A, the 1.9M-node scale case): retweet graph.

Edges: retweeted_id -> user_id from every retweets-*.csv (union). Nodes: all
ids that appear. Features: numeric per-user columns from users.csv (community
centralities, retweet/quote counts) + one-hot community; ids not present in
users.csv (retweeted-only accounts) get zero features. Labels: user_community
is the ground-truth partition used for NMI.

Scale note (MODELS.md §1): full-batch on an A100 with sparse ops. If VRAM is
tight, wrap the trainers in a NeighborLoader — the models here are batch-agnostic.
Use --limit for a quick sanity pass on a slice of the edges.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch

import common
import gclib

DEFAULT_DIR = os.path.join(os.path.dirname(__file__),
                           "../data_final/voterfraud2020")


def load_data(data_dir=DEFAULT_DIR, limit=None):
    # ---- edges ----
    src, dst = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "retweets-*.csv"))):
        df = pd.read_csv(f, usecols=["retweeted_id", "user_id"],
                         dtype=str, nrows=limit)
        src.extend(df["retweeted_id"].tolist())
        dst.extend(df["user_id"].tolist())
    edge_index, id2idx, _ = common.remap_ids(src, dst)
    n = len(id2idx)

    # ---- features + labels from users.csv ----
    users = pd.read_csv(os.path.join(data_dir, "users.csv"), dtype={"user_id": str})
    users = users[users["user_id"].isin(id2idx)]
    num_cols = [c for c in users.columns
                if c not in ("user_id", "user_community")
                and pd.api.types.is_numeric_dtype(users[c])]

    n_comm = int(users["user_community"].max()) + 1 if "user_community" in users else 0
    feat_dim = len(num_cols) + n_comm
    x = np.zeros((n, feat_dim), dtype=np.float32)
    labels = np.full(n, -1, dtype=np.int64)

    rows = users["user_id"].map(id2idx).to_numpy()
    x[rows, :len(num_cols)] = np.nan_to_num(users[num_cols].to_numpy(dtype=np.float32))
    if n_comm:
        comm = users["user_community"].fillna(-1).astype(int).to_numpy()
        valid = comm >= 0
        x[rows[valid], len(num_cols) + comm[valid]] = 1.0
        labels[rows[valid]] = comm[valid]

    data = common.make_data(edge_index, torch.from_numpy(x), num_nodes=n)
    return data, labels


def main():
    a = common.base_args("voterfraud", DEFAULT_DIR, clusters=2, epochs=200).parse_args()
    data, labels = load_data(a.data_dir, a.limit)
    k = int(labels.max()) + 1 if (labels >= 0).any() else a.clusters
    cfg = common.cfg_from_args(a)
    cfg.clusters = k  # match ground-truth community count for a fair NMI
    print(f"voterfraud: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
          f"{data.num_features} feat dims, {int((labels >= 0).sum())} labeled, k={k}")
    results = gclib.run_all(data, cfg, labels=labels, models=a.models)
    gclib.save(results, a.out_dir, "voterfraud")


if __name__ == "__main__":
    main()
