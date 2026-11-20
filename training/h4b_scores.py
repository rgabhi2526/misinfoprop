"""Scores from the FOLLOW graph only (T = Leiden on follow edges; misfit/disagreement from follow structure + text)."""
import os, random, numpy as np, igraph as ig
from multiprocessing import Pool
from sklearn.preprocessing import normalize
import ambiguity, views, h4b_data
OUTC = os.path.join(os.path.dirname(__file__), "out")
CONFIGS = [(r, s) for r in (0.5, 1.0, 2.0) for s in (0, 1, 2)]

def follow_bundle(b):
    s, d, _ = h4b_data.load_inter()["follow"]
    return views._bundle(np.stack([s, d]), b["text"], b["has"], b["y"])

def job(cfg):
    res, seed = cfg
    fn = os.path.join(OUTC, f"h4b_scores_r{res}_s{seed}.npz")
    if os.path.exists(fn): return fn
    b = views.load_timme(); bf = follow_bundle(b)
    text = np.hstack([b["text"], normalize(b["glove"])])
    runs = views.leiden_runs(bf["g"], 1, resolution=res, seed=seed)
    o = ambiguity.compute(bf, runs, text)
    o["deg_follow"] = bf["deg"]
    np.savez_compressed(fn, **o); print("done", cfg, "k", o["t"].max() + 1, flush=True)
    return fn

if __name__ == "__main__":
    with Pool(4) as P: P.map(job, CONFIGS)
    b = views.load_timme(); bf = follow_bundle(b)
    cp = os.path.join(OUTC, "h4b_btw_follow.npy")
    piv = np.random.default_rng(0).choice(b["n"], 300, replace=False).tolist()
    np.save(cp, np.asarray(bf["g"].betweenness(sources=piv), float))
    print("btw done")
