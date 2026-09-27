"""Score saved embeddings with metrics.report, no retraining.

    python eval_embeddings.py timme            # all models in out/timme/
    python eval_embeddings.py reddit --clusters 10

Writes out/<ds>/<ds>_report.json.
"""
import argparse
import glob
import importlib
import json
import os

import torch

import metrics

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("dataset")
p.add_argument("--clusters", type=int, default=None, help="default: #label classes, else 10")
p.add_argument("--limit", type=int, default=None)
a = p.parse_args()

data, labels = importlib.import_module(f"train_{a.dataset}").load_data(limit=a.limit)
k = a.clusters or (int(labels.max()) + 1 if labels is not None and (labels >= 0).any() else 10)
out_dir = os.path.join(HERE, "out", a.dataset)
ei = data.edge_index.numpy()
graph = metrics.graph_stats(ei, data.num_nodes) if data.num_nodes <= metrics.GRAPH_MAX else None
rep = {}
for f in sorted(glob.glob(os.path.join(out_dir, f"{a.dataset}_*_emb.pt"))):
    name = os.path.basename(f)[len(a.dataset) + 1:-len("_emb.pt")]
    emb = torch.load(f, map_location="cpu", weights_only=True)
    assert emb.shape[0] == data.num_nodes, (f"{name}: {emb.shape[0]} rows != {data.num_nodes} nodes — "
                                           "embedding predates the current loader, retrain")
    rep[name] = metrics.report(emb.numpy(), ei, k, labels, graph=graph)
    print(name, json.dumps(rep[name], indent=1))
with open(os.path.join(out_dir, f"{a.dataset}_report.json"), "w") as fh:
    json.dump(rep, fh, indent=2)
