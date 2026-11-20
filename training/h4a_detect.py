"""H4a: does misfit detect infiltrators / campaign accounts / recruiters with realistic noisy text?
DC-SBM, K communities, text = s*mean_vec + N(0,I) (s calibrated -> classifier accuracy). Existing modules imported unchanged."""
from __future__ import annotations
import argparse, json, os, sys, time, itertools
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v,'1')
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.stats as st
from scipy.stats import rankdata
import ambiguity, views

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'hypotheses', 'h4a')
BASE = dict(n=4000, K=8, dim=32, mu=0.2, s=1.6, kind='infil', prev=0.02, m=0.6, place='random', grp=10)
METHODS = ['misfit','disagreement','degree','pagerank','betweenness','text_iso','topo_participation',
           'topo_ambT','low_degree','misfit_nonhub','misfit_x_lowdeg','knn_dup']

def gen(P, seed):
    rng = np.random.default_rng(seed)
    n, K, dim = P['n'], P['K'], P['dim']
    comm = rng.integers(0, K, n)
    deg = np.minimum((rng.pareto(1.8, n) + 1) * 5, 250)
    npos = max(2, int(round(P['prev'] * n)))
    pos = rng.choice(n, npos, replace=False)
    kind = P['kind']
    if kind == 'rec':
        deg[pos] = np.minimum(deg[pos] * 6, 500)
    elif P['place'] == 'hub':
        deg[pos] = np.minimum(np.maximum(deg[pos], 20) * 5, 500)
    elif P['place'] == 'peripheral':
        deg[pos] = rng.integers(2, 4, npos)
    deg = np.maximum(deg.astype(int), 1)
    theta = np.where(np.isin(np.arange(n), pos) & (kind == 'rec'), 1.0, deg).astype(float)
    byc = [np.flatnonzero(comm == c) for c in range(K)]
    cdf = [np.cumsum(theta[p]) / theta[p].sum() for p in byc]
    src = np.repeat(np.arange(n), deg)
    # target community per stub
    tc = comm[src].copy()
    out = rng.random(len(src)) < P['mu']
    tc[out] = rng.integers(0, K, out.sum())
    if kind == 'rec':      # recruiters send 40% of stubs to ONE other community
        isr = np.isin(src, pos)
        other = (comm[src] + 1 + (src % (K - 1))) % K
        sw = isr & (rng.random(len(src)) < 0.4)
        tc[sw] = other[sw]
    dst = np.empty(len(src), int)
    for c in range(K):
        idx = np.flatnonzero(tc == c)
        dst[idx] = byc[c][np.searchsorted(cdf[c], rng.random(len(idx)))]
    k = src != dst
    ei = np.stack([np.r_[src[k], dst[k]], np.r_[dst[k], src[k]]])
    # text
    cen = rng.normal(size=(K, dim)); cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    mean = cen[comm].copy()
    truth = np.zeros(n, bool); truth[pos] = True
    if kind == 'infil':
        tgt = (comm[pos] + 1 + rng.integers(0, K - 1, npos)) % K
        mean[pos] = (1 - P['m']) * cen[comm[pos]] + P['m'] * cen[tgt]
    elif kind == 'camp':
        g = P['grp']; gid = np.arange(npos) // g
        z = rng.normal(size=(gid.max() + 1, dim)); z /= np.linalg.norm(z, axis=1, keepdims=True)
        mean[pos] = (1 - P['m']) * cen[comm[pos]] + P['m'] * z[gid]
    elif kind == 'rec':
        # recruiters' text hedged toward the community they recruit into
        tgt = (comm[pos] + 1 + (pos % (K - 1))) % K
        mean[pos] = (1 - P['m']) * cen[comm[pos]] + P['m'] * cen[tgt]
    text = P['s'] * mean + rng.normal(size=(n, dim))
    return ei, text, comm, truth, deg

def auc(sc, y):
    r = rankdata(sc); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def pk(sc, y):
    k = int(y.sum()); return y[np.argsort(-sc, kind='stable')[:k]].mean()

def scores(P, seed):
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    ei, text, comm, truth, deg = gen(P, seed)
    n = P['n']
    b = views._bundle(ei, text, np.ones(n, bool), comm)
    runs = views.leiden_runs(b['g'], 10, seed=seed)
    s = ambiguity.compute(b, runs)
    d = b['deg']
    bt = np.array(b['g'].betweenness(cutoff=4), dtype=float)
    iso = -IsolationForest(n_estimators=100, random_state=seed).fit(text).score_samples(text)
    nn = NearestNeighbors(n_neighbors=6).fit(text / np.linalg.norm(text, axis=1, keepdims=True))
    dist, _ = nn.kneighbors(text / np.linalg.norm(text, axis=1, keepdims=True))
    knn = -dist[:, 1:].mean(1)
    mis = s['misfit']
    sc = dict(misfit=mis, disagreement=s['disagreement'], degree=d, pagerank=s['pagerank'], betweenness=bt,
              text_iso=iso, topo_participation=s['participation'], topo_ambT=s['ambiguity_T'],
              low_degree=-d, knn_dup=knn)
    hub = d >= np.quantile(d, 0.9)
    sc['misfit_nonhub'] = np.where(hub, -1.0, mis)
    sc['misfit_x_lowdeg'] = rankdata(mis) * rankdata(-d)
    acc = float(s['content_acc'])
    return sc, truth, acc, float(d[truth].mean() / d.mean()), float((s['t'] >= 0).mean())

def one(args):
    P, seed = args
    t0 = time.time()
    sc, y, acc, dr, _ = scores(P, seed)
    res = {m: (auc(v, y), pk(v, y)) for m, v in sc.items()}
    return dict(P=P, seed=seed, acc=acc, degratio=dr, res=res, sec=time.time() - t0)

def calib():
    for s in (0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5):
        P = dict(BASE, s=s, prev=0.0025)
        a = [scores(P, sd)[2] for sd in range(2)]
        print('s', s, 'content_acc', np.mean(a), flush=True)

def ci(x):
    x = np.asarray(x, float); m = x.mean()
    if len(x) < 2: return m, np.nan
    return m, st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

def configs():
    C = {}
    def add(name, **kw): C[name] = dict(BASE, **kw)
    add('base')
    for k in ('camp', 'rec'): add(f'kind={k}', kind=k)
    for p in (0.005, 0.01, 0.05): add(f'prev={p}', prev=p)
    for m in (0.2, 0.4, 0.8, 1.0): add(f'm={m}', m=m)
    for pl in ('hub', 'peripheral'): add(f'place={pl}', place=pl)
    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--calib', action='store_true'); ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--tag', default='main')
    a = ap.parse_args()
    if a.calib: return calib()
    C = {}
    if a.sweep:   # text strength x mixing sweep, random infiltrators, plus weak-text variants of kinds
        for s, m in itertools.product(SW_S, (0.2, 0.4, 0.6, 0.8, 1.0)):
            C[f's={s},m={m}'] = dict(BASE, s=s, m=m)
        for k in ('camp', 'rec'):
            for s in SW_S: C[f'{k},s={s}'] = dict(BASE, s=s, kind=k)
        for pl in ('hub', 'peripheral'):
            for s in SW_S: C[f'{pl},s={s}'] = dict(BASE, s=s, place=pl)
    else:
        for sname, s in (('acc0.6', S60), ('acc0.3', S30)):
            for nm, P in configs().items(): C[f'{sname}|{nm}'] = dict(P, s=s)
    jobs = [(P, sd) for P in C.values() for sd in range(a.seeds)]
    names = [nm for nm in C for _ in range(a.seeds)]
    t0 = time.time(); out = {nm: [] for nm in C}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (nm, r) in enumerate(zip(names, ex.map(one, jobs))):
            out[nm].append(r)
            print(f'[{time.time()-t0:6.0f}s] {i+1}/{len(jobs)} {nm} seed{r["seed"]} acc={r["acc"]:.2f}', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f'h4a_{a.tag}.json'), 'w'))

S60, S30 = 1.75, 0.8
SW_S = (0.5, 0.7, 0.9, 1.3, 1.6, 2.2, 3.0)
if __name__ == '__main__':
    main()
