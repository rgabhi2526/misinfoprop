"""H2 on real graphs. 'Seeds' = independent replicate batches of cascades (scores fixed).
    python h2_real.py timme|pokec_knm [--reps 8 --sims 30]  -> ../results/hypotheses/h2/real_<ds>.json
"""
import argparse, json, os, sys
import numpy as np
from multiprocessing import Pool
import ambiguity, views
import h2_filter as H

D = None
def init(ds, rep_args):
    global D
    D = rep_args + (ds,)

def run(args):
    rep, model, mode, sims = args
    b, s, cents, comm, aff, ds_name = D
    mis = s["misfit"]
    rng = np.random.default_rng(1000)
    sets = H.build_sets(b["n"], cents, mis, rng, True)
    a = aff if mode == "aware" else None
    res, base, npool = H.cascades(b, sets, comm, a, sims, model, 77 + rep)
    rows = []
    for col, cn in ((0, "inf"), (1, "cross")):
        ps = {nm: res[nm][:, col] for nm in res}
        for nm in ps:
            pct = float(nm.split("|")[2])
            bst = max(("pr", "btw"), key=lambda f: ps[f"{f}|-|{pct}|cent"].mean())
            rows.append({"ds": ds_name, "model": model, "mode": mode, "seed": rep, "metric": cn, "name": nm,
                         "removed": float(ps[nm].mean()),
                         "gain": float((ps[nm] - ps[f"{bst}|-|{pct}|cent"]).mean()),
                         "best_struct": bst, "base": float(base[col]), "pool": npool})
    print(ds_name, model, mode, rep, flush=True)
    return rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--sims", type=int, default=30); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--models", default="wc,ic:2"); ap.add_argument("--piv", type=int, default=200)
    a = ap.parse_args(); ds_name = a.ds
    od = H.OUTD
    if a.ds == "timme":
        b = views.load_timme()
        s = dict(np.load(os.path.join(od, "..", "..", "scores", "timme_scores_minilm_only.npz")))
        modes = ["blind", "aware"]
    else:
        b = views.load_pokec(a.ds)
        s = dict(np.load(os.path.join(od, "..", "..", "scores", "pokec_slice_scores.npz")))
        modes = ["blind"]
    n = b["n"]; assert len(s["misfit"]) == n
    comm = np.where(s["t"] < 0, s["t"].max() + 1, s["t"])
    cp = os.path.join(od, f"btw_{a.ds}.npy")
    if os.path.exists(cp): btw = np.load(cp)
    else:
        piv = np.random.default_rng(0).choice(n, min(n, a.piv), replace=False).tolist()
        btw = np.asarray(b["g"].betweenness(sources=piv), float); np.save(cp, btw)
    pr = s["pagerank"]; deg = b["deg"].astype(float)
    un = -np.minimum(H.rk(pr), H.rk(btw))
    cents = {"pr": pr, "btw": btw, "deg": deg, "union": un.astype(float)}
    aff = ambiguity.content_probs(b["glove"], s["t"]) if a.ds == "timme" else None
    print("n", n, "misfit-pr corr", np.corrcoef(s["misfit"], pr)[0, 1], flush=True)
    jobs = [(r, m, md, a.sims) for m in a.models.split(",") for md in modes for r in range(a.reps)]
    with Pool(a.procs, initializer=init, initargs=(a.ds, (b, s, cents, comm, aff))) as P:
        rows = [r for rr in P.imap_unordered(run, jobs) for r in rr]
    json.dump(rows, open(os.path.join(od, f"real_{a.ds}.json"), "w"))
