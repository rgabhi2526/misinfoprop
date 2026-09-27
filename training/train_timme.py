"""TIMME (Group A): follow/mention/favorite/reply/retweet relations + text
features from each user's bio and latest tweet + R/D party labels.

Default subset is P_all (21k users, 6.5M relation rows); PureP (583 politicians,
all labeled) is too small/easy to separate models — use it only as a smoke test
via --data-dir .../PureP. Nodes are the accounts in all_twitter_ids.csv (that
ordering also indexes the feature matrices). Relations are the *_list.csv files
(`from<TAB>to<TAB>count`), pooled into one graph for the embedding stage.

Features: raw `description` (bio) + `status` (latest tweet) from
../formatted_location/simplified_user_info.json (covers every P_all user; 84% /
93% non-empty), encoded with common.ENCODER + has_metadata flag. Replaces the
shipped GloVe averages (decision T0, 2026-09-27). Cached as timme_text.npy in
data_dir. Location text is not used.

Labels: dict.csv (politicians) + ../additional_labels/new_dict_cleaned.csv
(manually labeled users). R=1, D=0, anything else (independents) = -1.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

import common
import gclib

DEFAULT_DIR = os.path.join(os.path.dirname(__file__),
                           "../data_final/timme/repo/data/P_all")
PARTY = {"R": 1, "D": 0}


def load_data(data_dir=DEFAULT_DIR, limit=None):
    # node order = feature-matrix row order
    ids = pd.read_csv(os.path.join(data_dir, "all_twitter_ids.csv"),
                      header=None, dtype=str).iloc[:, 0].str.strip()
    # first row may be a header token; drop if non-numeric
    if not ids.iloc[0].lstrip("-").isdigit():
        ids = ids.iloc[1:]
    id2idx = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    n = len(ids)

    info = json.load(open(os.path.join(data_dir, "../formatted_location/simplified_user_info.json")))
    texts = [" ".join(filter(None, ((info.get(i) or {}).get("description"),
                                    (info.get(i) or {}).get("status")))) for i in ids]
    x = common.encode_texts(texts, cache=os.path.join(data_dir, "timme_text.npy"))
    assert x.shape[0] == n, f"features rows {x.shape[0]} != ids {n}"

    # pool all relation files into one edge list
    src, dst = [], []
    for fn in sorted(glob.glob(os.path.join(data_dir, "*_list.csv"))):
        df = pd.read_csv(fn, sep="\t", header=None, usecols=[0, 1],
                         dtype=str, nrows=limit)
        a, b = df[0].str.strip().map(id2idx), df[1].str.strip().map(id2idx)
        ok = a.notna() & b.notna()
        src.append(a[ok].to_numpy(np.int64)); dst.append(b[ok].to_numpy(np.int64))
    edge_index = torch.from_numpy(np.stack([np.concatenate(src), np.concatenate(dst)]))

    labels = np.full(n, -1, dtype=np.int64)
    for fn in (os.path.join(data_dir, "dict.csv"),
               os.path.join(data_dir, "../additional_labels/new_dict_cleaned.csv")):
        if not os.path.exists(fn):
            continue
        d = pd.read_csv(fn, sep="\t", dtype=str)
        idx = d["twitter_id"].str.strip().map(id2idx)
        y = d["party"].str.strip().str.upper().map(PARTY)
        ok = idx.notna() & y.notna()
        labels[idx[ok].to_numpy(np.int64)] = y[ok].to_numpy(np.int64)

    data = common.make_data(edge_index, x, num_nodes=n)
    return data, labels[data.orig_idx.numpy()]


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
