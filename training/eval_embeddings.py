"""Score saved embeddings with metrics.report, no retraining.

    python eval_embeddings.py timme            # all models in out/timme/
    python eval_embeddings.py pokec --region "zilinsky kraj, kysucke nove mesto" --tag pokec_knm

Writes out/<tag>/<tag>_report.json (tag defaults to the dataset).
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
p.add_argument("--region", default=None, help="pokec only: region slice")
p.add_argument("--tag", default=None, help="out/<tag>/ (default: dataset)")
a = p.parse_args()
tag = a.tag or a.dataset

kw = dict(region=a.region) if a.region else {}
data, labels = importlib.import_module(f"train_{a.dataset}").load_data(limit=a.limit, **kw)
k = a.clusters or (int(labels.max()) + 1 if labels is not None and (labels >= 0).any() else 10)
out_dir = os.path.join(HERE, "out", tag)
ei = data.edge_index.numpy()
graph = metrics.graph_stats(ei, data.num_nodes) if data.num_nodes <= metrics.GRAPH_MAX else None
rep = {}
for f in sorted(glob.glob(os.path.join(out_dir, f"{tag}_*_emb.pt"))):
    name = os.path.basename(f)[len(tag) + 1:-len("_emb.pt")]
    emb = torch.load(f, map_location="cpu", weights_only=True)
    assert emb.shape[0] == data.num_nodes, (f"{name}: {emb.shape[0]} rows != {data.num_nodes} nodes — "
                                           "embedding predates the current loader, retrain")
    rep[name] = metrics.report(emb.numpy(), ei, k, labels, graph=graph)
    print(name, json.dumps(rep[name], indent=1))
with open(os.path.join(out_dir, f"{tag}_report.json"), "w") as fh:
    json.dump(rep, fh, indent=2)
