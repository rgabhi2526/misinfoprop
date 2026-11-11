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
for f in ['r1.0_s0', 'r1.0_s1', 'r2.0_s0']:
    z = np.load(f'out/h4b_scores_{f}.npz'); t = z['t']; ok = t >= 0
    pc = ambiguity.content_probs(np.hstack([xb[:, :-1], xb[:, -1:]]), t)
    acc = (pc[ok].argmax(1) == t[ok]).mean()
    for name, P, mis_s in [('leaky (bio+status+GloVe)', z['p_C'], z['misfit']), ('bio only', pc, 1 - pc[np.arange(len(t)), np.where(ok, t, 0)])]:
        tgt = P.argmax(1); k = P.shape[1]
        sel = ok & has_bio  # same population for both
        q = np.quantile(mis_s[sel], 0.9); mis = sel & (mis_s >= q) & (tgt != t)
        sz = np.bincount(t[ok], minlength=k) / ok.sum()
        row = []
        for L in ['retweet', 'like', 'mention']:
            s2, d2 = it[L+'_s'], it[L+'_d']; mm = mis[s2] & (t[d2] >= 0) & (t[d2] != t[s2])
            hit = (t[d2[mm]] == tgt[s2[mm]]).mean()
            ch = np.mean(sz[tgt[s2[mm]]] / (1 - sz[t[s2[mm]]]))
            a2 = mis[s2] & (t[d2] >= 0); own = (t[d2[a2]] == t[s2[a2]]).mean()
            row.append(f'{L} {hit:.2f}/{ch:.2f} own {own:.2f}')
        print(f'{f} {name:26s} acc={"%.2f"%acc if name=="bio only" else "-   "} misfits={mis.sum():4d} | ' + ' | '.join(row))
