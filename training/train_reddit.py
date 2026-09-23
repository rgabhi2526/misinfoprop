"""Reddit hyperlinks (Group B): subreddit -> subreddit hyperlink graph.

No ideology derivation needed here — the dataset ships 300-dim subreddit
embeddings (web-redditEmbeddings-subreddits.csv), which we use directly as node
features. Edges are the union of body + title hyperlink TSVs, kept only when
both endpoints have an embedding. No ground-truth labels (NMI is skipped).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

import common
import gclib

DEFAULT_DIR = os.path.join(os.path.dirname(__file__),
                           "../data_final/reddit-hyperlinks")
EDGE_FILES = ["soc-redditHyperlinks-body.tsv", "soc-redditHyperlinks-title.tsv"]


def load_data(data_dir=DEFAULT_DIR, limit=None):
    emb = pd.read_csv(os.path.join(data_dir, "web-redditEmbeddings-subreddits.csv"),
                      header=None)
    names = emb.iloc[:, 0].astype(str).tolist()
    id2idx = {v: i for i, v in enumerate(names)}
    x = torch.from_numpy(emb.iloc[:, 1:].to_numpy(dtype=np.float32))
    n = len(names)

    src, dst = [], []
    for fn in EDGE_FILES:
        df = pd.read_csv(os.path.join(data_dir, fn), sep="\t",
                         usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"],
                         dtype=str, nrows=limit)
        for a, b in zip(df["SOURCE_SUBREDDIT"], df["TARGET_SUBREDDIT"]):
            if a in id2idx and b in id2idx:
                src.append(id2idx[a]); dst.append(id2idx[b])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = common.make_data(edge_index, x, num_nodes=n)
    return data, None


def main():
    a = common.base_args("reddit", DEFAULT_DIR, clusters=10).parse_args()
    data, labels = load_data(a.data_dir, a.limit)
    print(f"reddit: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
          f"{data.num_features} feat dims")
    cfg = common.cfg_from_args(a)
    results = gclib.run_all(data, cfg, labels=labels, models=a.models)
    gclib.save(results, a.out_dir, "reddit")


if __name__ == "__main__":
    main()
