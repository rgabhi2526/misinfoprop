import os, sys, json, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, '.')
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize
import common, ambiguity, views
J = 'out'
D = '../data_final/timme/repo/data/P_all'
ids = pd.read_csv(f'{D}/all_twitter_ids.csv', header=None, dtype=str).iloc[:, 0].str.strip()
if not ids.iloc[0].lstrip('-').isdigit(): ids = ids.iloc[1:]
ids = ids.to_list(); n = len(ids)
info = json.load(open(f'{D}/../formatted_location/simplified_user_info.json'))
bios = [((info.get(i) or {}).get('description') or '') for i in ids]
xb = common.encode_texts(bios, cache=f'{J}/timme_bio.npy').numpy()
# LCC index exactly as h4b_data (verified there to equal views)
it = np.load('out/h4b_inter.npz')
id2 = {v: k for k, v in enumerate(ids)}
src, dst = [], []
for fn in ['friend_list.csv','favorite_list.csv','mention_list.csv','reply_list.csv','retweet_list.csv']:
    df = pd.read_csv(f'{D}/{fn}', sep='\t', dtype=str)
    a = df.iloc[:,0].str.strip().map(id2); b = df.iloc[:,1].str.strip().map(id2); ok = (a.notna()&b.notna()).to_numpy()
    src.append(a[ok].to_numpy(np.int64)); dst.append(b[ok].to_numpy(np.int64))
s = np.concatenate(src); d = np.concatenate(dst); m = s != d
A = sp.coo_matrix((np.ones(m.sum()), (s[m], d[m])), shape=(n, n)).tocsr(); A = A + A.T
_, comp = connected_components(A, directed=False); keep = np.flatnonzero(comp == np.bincount(comp).argmax())
full = np.load(f'{D}/timme_text.npy'); bv = views.load_timme()
print('LCC align check (max abs diff vs views text):', np.abs(full[keep, :-1] - bv['text']).max(), len(keep), bv['n'])
xb = xb[keep]; np.save(f'{J}/timme_bio_lcc.npy', xb); has_bio = xb[:, -1] > 0; print('bio coverage', has_bio.mean())

# ---- graph-only target predictors (follow layer only; engagement layers never used) ----
def _onehot(t, k):
    ok = t >= 0
    return sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), t[ok])), shape=(len(t), k))

def _mask_own(S, t_u):
    S = np.asarray(S, float).copy(); S[np.arange(len(t_u)), t_u] = -np.inf
    return S

def most_linked(F, t, k, us):
    """non-own community with most follow edges of u (argmax; ties -> lowest index)."""
    C = (F[us] @ _onehot(t, k)).toarray()
    return _mask_own(C, t[us]).argmax(1), (C.sum(1) == 0)

def label_prop(F, t, k, us):
    """one step of label spreading: sum_{v in N(u)} onehot(t_v)/deg(v)."""
    deg = np.asarray(F.sum(1)).ravel(); w = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)
    C = (F[us] @ sp.diags(w) @ _onehot(t, k)).toarray()
    return _mask_own(C, t[us]).argmax(1)

def adamic_adar(F, t, k, us):
    """per community c: sum of AA(u,v) over non-neighbours v != u with t_v==c; AA=sum_z 1/log deg(z)."""
    deg = np.asarray(F.sum(1)).ravel(); w = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    Au = F[us]; M = (Au @ sp.diags(w) @ F).tocsr()
    M = M - M.multiply(Au)                                  # drop existing neighbours
    M = M.tolil(); 
    for i, u in enumerate(us): M[i, u] = 0                  # drop self
    S = (M.tocsr() @ _onehot(t, k)).toarray()
    return _mask_own(S, t[us]).argmax(1), (S.max(1) <= 0)

def _selfcheck():
    # path 0-1-2-3 plus 1-4; communities: 0,4->0 ; 2,3->1 ; 5 isolated-ish tied to 4 ->2
    e = [(0,1),(1,2),(2,3),(1,4),(4,5)]; r = [a for a,b in e]+[b for a,b in e]; c = [b for a,b in e]+[a for a,b in e]
    G = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(6, 6)); tt = np.array([0, 0, 1, 1, 0, 2]); us = np.array([1, 4])
    ml, _ = most_linked(G, tt, 3, us); assert list(ml) == [1, 2], ml     # u1: nbrs 0(c0),2(c1),4(c0) -> c1 ; u4: nbrs 1(c0),5(c2) -> c2
    aa, _ = adamic_adar(G, tt, 3, us)
    # u1 non-nbrs: 3 (via 2, deg2: 1/ln2) in c1 -> AA=1/ln2; 5 (via 4, deg2) in c2 -> 1/ln2: tie -> lowest idx = 1
    assert list(aa[:1]) == [1], aa
    # u4 non-nbrs: 0,2 (via 1, deg3: 1/ln3 each) c0 own,c1 -> c1 ; 
    assert aa[1] == 1, aa
    lp = label_prop(G, tt, 3, us); assert list(lp) == [1, 2], lp
    # shortest-path sanity: AA must not count neighbours or self
    assert adamic_adar(G, tt, 3, np.array([2]))[0][0] in (0, 2)
_selfcheck()

OUT = {}
rng_seed = 0
FOLLOW = sp.coo_matrix((np.ones(len(it['follow_s'])), (it['follow_s'], it['follow_d'])), shape=(len(keep), len(keep))).tocsr()
FOLLOW = FOLLOW + FOLLOW.T; FOLLOW.setdiag(0); FOLLOW.eliminate_zeros(); FOLLOW.data[:] = 1.0
for f in ['r1.0_s0', 'r1.0_s1', 'r2.0_s0']:
    z = np.load(f'out/h4b_scores_{f}.npz'); t = z['t']; ok = t >= 0
    pc = ambiguity.content_probs(np.hstack([xb[:, :-1], xb[:, -1:]]), t)
    acc = (pc[ok].argmax(1) == t[ok]).mean()
    for name, P, mis_s in [('leaky (bio+status+GloVe)', z['p_C'], z['misfit']), ('bio only', pc, 1 - pc[np.arange(len(t)), np.where(ok, t, 0)])]:
        tgt = P.argmax(1); k = P.shape[1]
        sel = ok & has_bio  # same population for both
        q = np.quantile(mis_s[sel], 0.9); mis = sel & (mis_s >= q) & (tgt != t)
        sz = np.bincount(t[ok], minlength=k) / ok.sum()
        us = np.flatnonzero(mis); tk = P.shape[1]
        pT_nonown = _mask_own(z['p_T'][us], t[us]).argmax(1)
        pC_nonown = _mask_own(P[us], t[us]).argmax(1)
        ml, ml_zero = most_linked(FOLLOW, t, tk, us); lp = label_prop(FOLLOW, t, tk, us); aa, aa_zero = adamic_adar(FOLLOW, t, tk, us)
        PRED = {'argmax_pC': tgt[us], 'argmax_pC_nonown': pC_nonown, 'most_linked': ml, 'label_prop': lp, 'p_T': pT_nonown, 'adamic_adar': aa}
        loc = np.full(len(t), -1); loc[us] = np.arange(len(us))
        row = []; rec = {'n_misfits': int(mis.sum()), 'k': int(k), 'acc': float(acc) if name == 'bio only' else None}
        for L in ['retweet', 'like', 'mention']:
            s2, d2 = it[L+'_s'], it[L+'_d']; mm = mis[s2] & (t[d2] >= 0) & (t[d2] != t[s2])
            hit = (t[d2[mm]] == tgt[s2[mm]]).mean()
            ch = np.mean(sz[tgt[s2[mm]]] / (1 - sz[t[s2[mm]]]))
            a2 = mis[s2] & (t[d2] >= 0); own = (t[d2[a2]] == t[s2[a2]]).mean()
            row.append(f'{L} {hit:.2f}/{ch:.2f} own {own:.2f}')
            lu = loc[s2[mm]]; nU = len(us); nu = np.bincount(lu, minlength=nU).astype(float)
            hv = {nm: np.bincount(lu, weights=(t[d2[mm]] == pr[lu]).astype(float), minlength=nU) for nm, pr in PRED.items()}
            share = {nm: float(h.sum() / nu.sum()) for nm, h in hv.items()}
            assert abs(share['argmax_pC'] - hit) < 1e-12
            act = np.flatnonzero(nu > 0); rg = np.random.default_rng(rng_seed); B = rg.integers(0, len(act), (1000, len(act)))
            den = nu[act][B].sum(1); tb = hv['argmax_pC'][act][B].sum(1) / den
            st = {'hit_share': {nm: v for nm, v in share.items()}, 'chance': float(ch), 'n_users_with_ties': int(len(act)),
                  'n_misfits_zero_follow_comm_links': int(ml_zero.sum()), 'n_misfits_zero_AA': int(aa_zero.sum()), 'text_minus': {}}
            for nm in ['argmax_pC_nonown', 'most_linked', 'label_prop', 'p_T', 'adamic_adar']:
                dd = tb - hv[nm][act][B].sum(1) / den
                st['text_minus'][nm] = {'diff': share['argmax_pC'] - share[nm], 'ci95': [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
            row[-1] += ' | ' + ' '.join(f'{nm[:6]}={share[nm]:.3f}' for nm in ['most_linked','label_prop','p_T','adamic_adar'])
            rec[L] = {'hit_argmax_pC': float(hit), 'chance': float(ch), 'own_community_share': float(own), 'n_cross': int(mm.sum()), 'structural': st}
        print(f'{f} {name:26s} acc={"%.2f"%acc if name=="bio only" else "-   "} misfits={mis.sum():4d} | ' + ' | '.join(row))
        OUT.setdefault(f, {})[name] = rec
os.makedirs('../results/hypotheses/h5', exist_ok=True)
json.dump(OUT, open('../results/hypotheses/h5/h5_probe.json', 'w'), indent=1)
