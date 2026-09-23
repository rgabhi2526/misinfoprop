"""TIMME (Group A): follow/mention/favorite/reply/retweet relations + GloVe
description/status features + R/D party labels.

Nodes are the accounts in all_twitter_ids.csv (that ordering also indexes the
feature matrix). Relations are the *_list.csv files (`from<TAB>to<TAB>count`);
we pool every relation type into one graph for the embedding stage.
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
                           "../data_final/timme/repo/data/PureP")


def load_data(data_dir=DEFAULT_DIR, limit=None):
    # node order = feature-matrix row order
    ids = pd.read_csv(os.path.join(data_dir, "all_twitter_ids.csv"),
                      header=None).iloc[:, 0].astype(str)
    # first row may be a header token; drop if non-numeric
    if not ids.iloc[0].strip().lstrip("-").isdigit():
        ids = ids.iloc[1:]
    id_list = ids.tolist()
    id2idx = {v: i for i, v in enumerate(id_list)}
    n = len(id_list)

    feats = np.load(os.path.join(data_dir, "features.npz"))
    x = np.concatenate([feats["description"], feats["status"]], axis=1)
    assert x.shape[0] == n, f"features rows {x.shape[0]} != ids {n}"
    x = torch.from_numpy(x)

    # pool all relation files into one edge list
    src, dst = [], []
    for f in glob.glob(os.path.join(data_dir, "*_list.csv")):
        df = pd.read_csv(f, sep="\t", header=None, usecols=[0, 1],
                         names=["from", "to"], dtype=str, nrows=limit)
        for a, b in zip(df["from"], df["to"]):
            if a in id2idx and b in id2idx:
                src.append(id2idx[a]); dst.append(id2idx[b])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = common.make_data(edge_index, x, num_nodes=n)

    # party labels: dict.csv (name, twitter_id, twitter_name, party)
    labels = np.full(n, -1, dtype=np.int64)
    dic = pd.read_csv(os.path.join(data_dir, "dict.csv"), sep="\t", dtype=str)
    party_col = [c for c in dic.columns if c.lower() == "party"][0]
    id_col = [c for c in dic.columns if "id" in c.lower()][0]
    for tid, party in zip(dic[id_col].astype(str), dic[party_col]):
        if tid in id2idx and isinstance(party, str):
            labels[id2idx[tid]] = 1 if party.strip().upper().startswith("R") else 0
    return data, labels


def main():
    a = common.base_args("timme", DEFAULT_DIR, clusters=2).parse_args()
    data, labels = load_data(a.data_dir, a.limit)
    print(f"timme: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
          f"{data.num_features} feat dims, {int((labels >= 0).sum())} labeled")
    cfg = common.cfg_from_args(a)
    results = gclib.run_all(data, cfg, labels=labels, models=a.models)
    gclib.save(results, a.out_dir, "timme")


if __name__ == "__main__":
    main()
