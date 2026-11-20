"""Summarise h4a_<tag>.json -> markdown tables (AUC, P@k, paired gain over best baseline)."""
import json, sys, os, numpy as np, scipy.stats as st
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'hypotheses', 'h4a')
BASEL = ['degree','pagerank','betweenness','low_degree','text_iso','topo_participation','topo_ambT','knn_dup']
OURS = ['misfit','disagreement','misfit_nonhub','misfit_x_lowdeg']
def ci(x):
    x = np.asarray(x, float)
    return x.mean(), st.t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))
def table(tag, which):   # which 0=AUC 1=P@k
    d = json.load(open(os.path.join(OUT, f'h4a_{tag}.json')))
    L = [f'| config | text acc | '+' | '.join(OURS)+' | best baseline (name) | '+' | '.join('gain '+o for o in OURS)+' |', '|'+'---|'*(4+2*len(OURS))]
    n_pos = {o: 0 for o in OURS}; n_neg = {o: 0 for o in OURS}; cells = 0
    for nm, rs in d.items():
        M = {m: np.array([r['res'][m][which] for r in rs]) for m in rs[0]['res']}
        best = max(BASEL, key=lambda m: M[m].mean())
        acc = np.mean([r['acc'] for r in rs])
        cols = []; gains = []
        for o in OURS:
            mu, h = ci(M[o]); cols.append(f'{mu:.2f}±{h:.2f}')
            g, gh = ci(M[o] - M[best]); gains.append(f'{g:+.2f}±{gh:.2f}' + ('*' if g - gh > 0 else ''))
            n_pos[o] += g - gh > 0; n_neg[o] += g + gh < 0
        cells += 1
        mb, hb = ci(M[best])
        L.append(f'| {nm} | {acc:.2f} | ' + ' | '.join(cols) + f' | {mb:.2f}±{hb:.2f} ({best}) | ' + ' | '.join(gains) + ' |')
    L.append(f'\ncells={cells}; CI>0 (marked *): ' + ', '.join(f'{o} {n_pos[o]}' for o in OURS) + '; CI<0: ' + ', '.join(f'{o} {n_neg[o]}' for o in OURS))
    return '\n'.join(L)
def allmethods(tag, which=0):
    d = json.load(open(os.path.join(OUT, f'h4a_{tag}.json')))
    ms = list(next(iter(d.values()))[0]['res']); L = ['| config | '+' | '.join(ms)+' |', '|'+'---|'*(1+len(ms))]
    for nm, rs in d.items():
        L.append(f'| {nm} | ' + ' | '.join(f'{np.mean([r["res"][m][which] for r in rs]):.2f}' for m in ms) + ' |')
    return '\n'.join(L)
if __name__ == '__main__':
    tag = sys.argv[1]
    s = f'## {tag}: AUC (gain = misfit-family minus best baseline, paired over seeds, mean±95% CI)\n\n' + table(tag, 0)
    s += f'\n\n## {tag}: precision@k (k = #positives)\n\n' + table(tag, 1)
    s += f'\n\n## {tag}: mean AUC, all methods\n\n' + allmethods(tag, 0)
    s += f'\n\n## {tag}: mean precision@k, all methods\n\n' + allmethods(tag, 1)
    open(os.path.join(OUT, f'tables_{tag}.md'), 'w').write(s); print(s)
