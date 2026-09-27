"""How modular are our graphs? (T7.0, S-22) Bridge-targeting beats hub-targeting only
under strong community structure (Salathé & Jones 2010), so measure it before T7.

Louvain (igraph, C) on the exact graph the models see (symmetrized, LCC), against a
degree-preserving rewired null: sparse random graphs also get Q ~0.2-0.3 from Louvain,
so only the gap above the null means real communities. No training, no text encoding.

    python check_modularity.py timme
    python check_modularity.py pokec --region "zilinsky kraj, kysucke nove mesto"
    python check_modularity.py pokec            # full 1.63M, ~1-2 GB RAM
"""
import argparse
import os
import random

import igraph as ig
import numpy as np
import pandas as pd
import torch

import common


def pokec_full_edges(data_dir):
    e = pd.read_csv(os.path.join(data_dir, "soc-pokec-relationships.txt"), sep="\t",
                    header=None, dtype=np.int64).to_numpy().T - 1        # ids are 1..N
    n = int(e.max()) + 1
    return common.make_data(torch.from_numpy(e), torch.zeros(n, 1), num_nodes=n).edge_index


def louvain(g, seed):
    ig.set_random_number_generator(random.Random(seed))
    c = g.community_multilevel()
    m = np.array(c.membership)
    e = np.array(g.get_edgelist()) if g.ecount() < 5_000_000 else \
        np.array(g.get_edgelist()[::10])            # ponytail: 10% edge sample on huge graphs
    sizes = np.bincount(m)
    return c.modularity, len(sizes), sizes.max() / g.vcount(), (m[e[:, 0]] != m[e[:, 1]]).mean()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["timme", "pokec"])
    p.add_argument("--region", default=None)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()

    if a.dataset == "pokec" and not a.region:
        ei = pokec_full_edges(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data"))
    else:
        import importlib
        mod = importlib.import_module(f"train_{a.dataset}")
        ei = (mod.load_data(region=a.region) if a.region else mod.load_data())[0].edge_index
    s, d = ei.numpy()
    keep = s < d
    g = ig.Graph(n=int(ei.max()) + 1, edges=np.stack([s[keep], d[keep]], 1))
    print(f"{a.dataset}[{a.region or 'all'}]: {g.vcount()} nodes, {g.ecount()} edges, "
          f"mean degree {2 * g.ecount() / g.vcount():.1f}")

    for name, graph in [("real", g), ("rewired null", None)]:
        if graph is None:
            graph = g.copy()
            graph.rewire(n=2 * g.ecount())          # degree-preserving edge swaps
        r = np.array([louvain(graph, sd) for sd in range(a.seeds)])
        print(f"  {name:13s} Q = {r[:, 0].mean():.3f} ± {r[:, 0].std():.3f}   "
              f"communities {r[:, 1].mean():.0f}   largest {r[:, 2].mean():.1%}   "
              f"cross-community edges {r[:, 3].mean():.1%}")
