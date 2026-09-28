"""UMAP of saved embeddings: rows = models, columns = colourings.

    python plot_umap.py timme     # party | k-means k=2 | boundary margin
    python plot_umap.py pokec --region "zilinsky kraj, kysucke nove mesto" --tag pokec_knm
                                  # k-means k=10 | log degree | boundary margin

Writes out/<tag>/<tag>_umap.png (tag defaults to the dataset).
"""
import argparse
import importlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.preprocessing import normalize

import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("bgrl", "dgi", "mvgrl")
# dataviz reference palette: categorical slots in fixed order; blue/orange ramps
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER, INK, MUTED = "#b8b6b0", "#1f1f1c", "#6b6a66"
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("orange", ["#fbe0d3", "#f4a37f", "#eb6834", "#b8461b", "#7a2a0c"])
PT = dict(s=3, linewidths=0, rasterized=True)


def margin(emb, k):
    _, centers = metrics.cluster(emb, k)
    sims = np.sort(normalize(emb) @ centers.T, axis=1)
    return sims[:, -1] - sims[:, -2]


def categorical(ax, xy, lab, names, colors=CAT):
    # ponytail: >8 categories fold into grey "other" (largest 8 keep a hue)
    order = np.argsort(-np.bincount(lab))[:len(colors)]
    ax.scatter(*xy[~np.isin(lab, order)].T, c=OTHER, label="other", **PT)
    for slot, c in enumerate(order):
        ax.scatter(*xy[lab == c].T, c=colors[slot], label=names[c], **PT)
    ax.legend(markerscale=4, fontsize=7, frameon=False, loc="best", labelcolor=MUTED)


def continuous(ax, xy, v, cmap, label, fig):
    o = np.argsort(v)                   # draw high values last
    sc = ax.scatter(*xy[o].T, c=v[o], cmap=cmap, **PT)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01).set_label(label, color=MUTED, fontsize=8)


def main(ds, region=None, tag=None):
    tag = tag or ds
    loader = importlib.import_module(f"train_{ds}")
    data, labels = loader.load_data(region=region) if region else loader.load_data()
    n = data.num_nodes
    deg = np.bincount(data.edge_index[0].numpy(), minlength=n)
    k = 2 if ds == "timme" else 10
    cols = (["party (R/D)", "k-means k=2", "boundary margin (low = boundary)"] if ds == "timme"
            else ["k-means k=10", "log(1+degree), degree 0 in black", "boundary margin (low = boundary)"])

    fig, axes = plt.subplots(3, 3, figsize=(16, 15), facecolor="white")
    for r, m in enumerate(MODELS):
        emb = torch.load(os.path.join(HERE, "out", tag, f"{tag}_{m}_emb.pt"),
                         map_location="cpu", weights_only=True).numpy()
        assert len(emb) == n, f"{m}: {len(emb)} rows != {n} nodes — retrain with current loader"
        metrics.log(f"umap: {m} ({n} nodes)")
        xy = umap.UMAP(metric="cosine", n_neighbors=15, min_dist=0.1,
                       random_state=0).fit_transform(normalize(emb))
        pred, _ = metrics.cluster(emb, k)
        mg = margin(emb, k)
        a, b, c = axes[r]
        if ds == "timme":
            # party convention: D blue, R red (slot 1 / slot 8); R is the larger class
            u = labels < 0                  # unlabeled: grey underlay
            a.scatter(*xy[u].T, c=OTHER, label=f"unlabeled (n={u.sum()})", **PT)
            categorical(a, xy[~u], labels[~u], {0: "D", 1: "R"}, colors=[CAT[7], CAT[0]])
            categorical(b, xy, pred, {i: f"c{i}" for i in range(k)})
        else:
            categorical(a, xy, pred, {i: f"c{i}" for i in range(k)})
            z = deg == 0
            continuous(b, xy[~z], np.log1p(deg[~z]).astype(float), ORANGE, "log(1+degree)", fig)
            b.scatter(*xy[z].T, c=INK, label=f"degree 0 (n={z.sum()})", **PT)
            b.legend(markerscale=4, fontsize=7, frameon=False, labelcolor=MUTED)
        continuous(c, xy, mg, BLUE.reversed(), "nearest − 2nd-nearest centroid cos", fig)
        for j, ax in enumerate(axes[r]):
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_color("#e3e2de")
            ax.set_title(f"{m.upper()} · {cols[j]}", fontsize=10, color=INK, loc="left")
    fig.suptitle(f"{tag}: UMAP (cosine, n_neighbors=15, min_dist=0.1) of L2-normalized embeddings",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = os.path.join(HERE, "out", tag, f"{tag}_umap.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--region", default=None, help="pokec only: region slice")
    p.add_argument("--tag", default=None, help="out/<tag>/ (default: dataset)")
    a = p.parse_args()
    main(a.dataset, a.region, a.tag)
