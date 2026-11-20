#!/usr/bin/env python
"""Build the ground-truth results tables from RAW result files.

Run:  cd final_results && /opt/anaconda3/bin/python build_results.py
Writes results_long.csv (one row per number), results_summary.csv (one row per
experiment), metrics_glossary.csv. Every number is read from a raw JSON / npz / text log
under the repo root; the few hand-entered ones are marked `ponytail:` and listed in _gaps.md.

Unit conventions (documented in the report): shares, AUC, P@10, R@100, success rates stay as
stored (fractions 0-1). Anything called gain / removed / *_pts is in PERCENTAGE POINTS
(fractions x100 where the file stores a fraction; H4d already stores points).
"""
import csv, json, re, collections, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.dirname(os.path.abspath(__file__))
COLS = ['experiment_id', 'phase', 'use_case', 'dataset', 'data_type', 'setting', 'method', 'method_role',
        'metric', 'value', 'ci_low', 'ci_high', 'uncertainty', 'n', 'source_file', 'source_key']
ROWS = []
# ponytail: two-sided 95% t critical values, df 1..30, hand-typed instead of importing scipy
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

# ---------------------------------------------------------------- roles
OURS = {'misfit', 'misfit_x_pr', 'disagreement', 'disagreement_x_pr', 'gateway', 'misfit_rerank', 'misfit_nonhub',
        'misfit_x_lowdeg', 'rehome_text', 'ours', 'ours+aa', 'aa_x_text', 'aa+text_sim', 'bridge_struct_x_recv',
        'bridge_content', 'misfit_alone', 'misfit_features', 'text+structure+misfit', 'misfit_only', 'BGRL', 'DGI', 'MVGRL',
        'T Leiden (best of runs)', 'prod_misfit', 'recv', 'mthenc', 'rankavg', 'band', 'prod'}
CONTROL = {'random', 'none', 'random_rewire', 'rehome_text_random_nodes', 'rand', 'gateway_struct', 'real_misfit_shuffled',
           'misfit_SHUFFLED_TEXT', 'disagreement_SHUFFLED_TEXT', 'rewired null', 'random_edges', 'random_attack', 'noise attack', 'random attack'}
REFERENCE = {'oracle_recruiters', 'bridge_oracle', 'bridge_struct_x_recv_ORACLE', 'real graph'}


def role(m):
    if m in CONTROL:
        return 'control'
    if m in REFERENCE:
        return 'reference'
    return 'ours' if m in OURS else 'baseline'


def rel(p):
    return os.path.relpath(p, ROOT)


def P(*parts):
    return os.path.join(ROOT, *parts)


def A(exp, phase, use, ds, dt, setting, method, metric, value, src, key, lo='', hi='', unc='none', n='', role_=None):
    """Append one long row. value must be a real number."""
    assert value is not None and value == value, (exp, setting, method, metric)
    r = lambda x: '' if x == '' or x is None else round(float(x), 6)
    ROWS.append(dict(experiment_id=exp, phase=phase, use_case=use, dataset=ds, data_type=dt, setting=setting,
                     method=method, method_role=role_ or role(method), metric=metric, value=r(value),
                     ci_low=r(lo), ci_high=r(hi), uncertainty=unc, n=n, source_file=src, source_key=key))


def tci(vals):
    v = np.asarray(vals, float)
    n = len(v)
    if n < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), T975[min(n - 1, 30)] * float(v.std(ddof=1)) / np.sqrt(n)


def jl(p):
    return json.load(open(P(p)))


def lines(p):
    return [(i + 1, l.rstrip('\n')) for i, l in enumerate(open(P(p), errors='replace'))]


class X:  # per-experiment constructor so each row call stays short
    def __init__(self, exp, phase, use, ds, dt):
        self.a = (exp, phase, use, ds, dt)

    def add(self, setting, method, metric, value, src, key, **kw):
        A(*self.a[:4], self.a[4], setting, method, metric, value, src, key, **kw)

    def addds(self, ds, dt, setting, method, metric, value, src, key, **kw):
        A(self.a[0], self.a[1], self.a[2], ds, dt, setting, method, metric, value, src, key, **kw)


# ================================================================= E01
def e01_modularity():
    x = X('E01', 'T7.0', 'dataset characterisation', 'TIMME', 'real')
    files = [('TIMME', 'results/logs/modularity_timme.log'), ('Pokec slice', 'results/logs/modularity_pokec_slice.log'),
             ('Pokec full', 'results/logs/modularity_pokec_full.log')]
    rx = re.compile(r'^\s+(real|rewired null)\s+Q = ([\d.]+) ± ([\d.]+)\s+communities (\d+)\s+largest ([\d.]+)%\s+cross-community edges ([\d.]+)%')
    for ds, f in files:
        for ln, l in lines(f):
            m = rx.match(l)
            if not m:
                continue
            meth = 'real graph' if m.group(1) == 'real' else 'rewired null'
            q, sd = float(m.group(2)), float(m.group(3))
            x.addds(ds, 'real', 'Louvain, graph-only', meth, 'modularity_Q', q, f, f'log line {ln}', lo=q - sd, hi=q + sd, unc='sd_over_runs')
            x.addds(ds, 'real', 'Louvain, graph-only', meth, 'n_communities', m.group(4), f, f'log line {ln}')
            x.addds(ds, 'real', 'Louvain, graph-only', meth, 'largest_community_share_pct', m.group(5), f, f'log line {ln}')
            x.addds(ds, 'real', 'Louvain, graph-only', meth, 'cross_community_edge_share_pct', m.group(6), f, f'log line {ln}')


# ================================================================= E02
def e02_gnn():
    x = X('E02', 'T2/T3', 'embedding quality', 'TIMME', 'real')
    f = 'results/gnn_embeddings/timme_gnn_metrics_2026-09-28.json'
    d = jl(f)
    for m, nm in [('bgrl', 'BGRL'), ('dgi', 'DGI'), ('mvgrl', 'MVGRL')]:
        r = d[m]
        s = 'k-means k=2 on GCN embedding, single run (superseded)'
        for k in ['modularity', 'nmi', 'emb_std', 'best_epoch']:
            x.add(s, nm, k if k != 'modularity' else 'modularity_Q', r[k], f, f'{m}.{k}')
        for sec, ks in [('health', ['effective_rank', 'corr_norm_logdeg', 'dim_std_mean']),
                        ('unlabeled', ['conductance_mean', 'edge_reconstruction_auc', 'seed_stability_ari']),
                        ('labeled', ['probe_acc', 'probe_f1_macro', 'nmi', 'ari', 'knn10_label_agreement', 'silhouette_true_labels'])]:
            for k in ks:
                x.add(s + ' (old biased probe)' if k.startswith('probe') else s, nm, 'nmi_party' if (sec == 'labeled' and k == 'nmi') else k,
                      r['report'][sec][k], f, f'{m}.report.{sec}.{k}')
    # degree diagnostics log
    f = 'results/logs/gnn_degree_diagnostics.log'
    cur = None
    for ln, l in lines(f):
        m = re.match(r'^(bgrl|dgi|mvgrl): probe ([\d.]+) \| probe\[emb\+text\] ([\d.]+)', l)
        if m:
            cur = m.group(1).upper()
            x.add('probe on raw embedding (not standardised)', cur, 'probe_acc', m.group(2), f, f'log line {ln}')
            x.add('probe on raw embedding + text', cur, 'probe_acc', m.group(3), f, f'log line {ln}')
        m = re.match(r'^\s+k=2 split vs logdeg \|r\|=([\d.]+) vs has_text \|r\|=([\d.]+) vs party NMI=([\d.]+)', l)
        if m and cur:
            x.add('k=2 split', cur, 'abs_corr_split_vs_logdegree', m.group(1), f, f'log line {ln}')
            x.add('k=2 split', cur, 'nmi_party', m.group(3), f, f'log line {ln}')
        m = re.match(r'^\s+k=6: sizes .* Q=([-\d.]+) party NMI=([\d.]+)', l)
        if m and cur:
            x.add('k=6', cur, 'modularity_Q', m.group(1), f, f'log line {ln}')
            x.add('k=6', cur, 'nmi_party', m.group(2), f, f'log line {ln}')
        m = re.match(r'^probe (text-only|spectral\(graph\)|logdeg only)\s+([\d.]+)', l)
        if m:
            nm = {'text-only': 'text-only probe', 'spectral(graph)': 'spectral probe', 'logdeg only': 'log-degree-only probe'}[m.group(1)]
            x.add('linear probe party accuracy, unstandardised', nm, 'probe_acc', m.group(2), f, f'log line {ln}')
        m = re.match(r'^labeled 1206\s+R share ([\d.]+)\s+\(majority baseline ([\d.]+)\)', l)
        if m:
            x.add('1206 labeled nodes', 'majority class', 'probe_acc', m.group(2), f, f'log line {ln}')
    # fair probe log
    f = 'results/logs/gnn_fair_probe_and_baselines.log'
    for ln, l in lines(f):
        m = re.match(r'^neighbour-vote \(graph, out-of-fold\) ([\d.]+)', l)
        if m:
            x.add('standardised probe, graph-only baseline', 'neighbour-vote', 'probe_acc', m.group(1), f, f'log line {ln}')
        m = re.match(r'^text std-probe ([\d.]+) \| spectral64 std-probe ([\d.]+)', l)
        if m:
            x.add('standardised probe', 'text-only probe', 'probe_acc', m.group(1), f, f'log line {ln}')
            x.add('standardised probe', 'spectral64 probe', 'probe_acc', m.group(2), f, f'log line {ln}')
        m = re.match(r'^(bgrl|dgi|mvgrl): raw C=1 nostd ([\d.]+) \| L2 C=1 nostd \(=report\) ([\d.]+) \| std ([\d.]+) \| std emb\+text ([\d.]+)', l)
        if m:
            nm = m.group(1).upper()
            for g, st in [(2, 'raw, not standardised'), (4, 'standardised (fair probe)'), (5, 'standardised, embedding + text')]:
                x.add(st, nm, 'probe_acc', m.group(g), f, f'log line {ln}')
    f = 'results/logs/gnn_cluster_after_removing_degree.log'
    for ln, l in lines(f):
        m = re.match(r'^(bgrl|dgi|mvgrl) k=2 raw: Q=([-\d.]+) NMI=([-\d.]+) \| k=2 -deg: Q=([-\d.]+) NMI=([-\d.]+) \| k=6 raw: Q=([-\d.]+) NMI=([-\d.]+) \| k=6 -deg: Q=([-\d.]+) NMI=([-\d.]+)', l)
        if m:
            nm = m.group(1).upper()
            for i, (st) in enumerate(['k=2 raw', 'k=2 degree regressed out', 'k=6 raw', 'k=6 degree regressed out']):
                x.add(st, nm, 'modularity_Q', m.group(2 + 2 * i), f, f'log line {ln}')
                x.add(st, nm, 'nmi_party', m.group(3 + 2 * i), f, f'log line {ln}')


# ================================================================= E03
def e03_views():
    x = X('E03', 'T10.1/T10.2', 'view construction', 'TIMME', 'real')
    rx = re.compile(r'^([TC])\s+(.+?)\s+k=\s*(\d+) \(>=1%:\s*(\d+)\) largest=([\d.]+) Q=([+-][\d.]+)(?: NMI=([-\d.]+) ARI=([-\d.]+))? eta2=([\d.]+) stab=([\d.]+)')
    for ds, f in [('TIMME', 'results/logs/views_timme.log'), ('Pokec slice', 'results/logs/views_pokec.log')]:
        for ln, l in lines(f):
            m = rx.match(l)
            if m:
                view = 'topology view' if m.group(1) == 'T' else 'text view'
                meth = ('T ' if m.group(1) == 'T' else 'C ') + re.sub(r'\s+', ' ', m.group(2))
                if meth.startswith('C text kNN Leiden') or meth.startswith('T Leiden'):
                    r = 'ours'
                else:
                    r = 'baseline'
                s = view + ' (modularity on SOCIAL graph)'
                mets = [('modularity_Q', 6), ('n_clusters', 3), ('largest_cluster_share', 5), ('degree_eta2', 9), ('seed_stability_ari', 10)]
                if m.group(7) is not None:
                    mets += [('nmi_party', 7), ('ari_party', 8)]
                for nm, g in mets:
                    x.addds(ds, 'real', s, meth, nm, m.group(g), f, f'log line {ln}', role_=r)
            m = re.match(r'^T vs C: NMI=([\d.]+) ARI=([-\d.]+)', l)
            if m:
                x.addds(ds, 'real', 'topology vs text partition', 'T Leiden vs C text kNN Leiden', 'nmi_T_vs_C', m.group(1), f, f'log line {ln}', role_='reference')
                x.addds(ds, 'real', 'topology vs text partition', 'T Leiden vs C text kNN Leiden', 'ari_T_vs_C', m.group(2), f, f'log line {ln}', role_='reference')


# ================================================================= E04
def e04_content():
    x = X('E04', 'T10.3/T10.5', 'content informativeness', 'TIMME', 'real')
    for ds, f, nm in [('TIMME', 'results/scores/timme_scores_minilm_glove.npz', 'p_C text classifier (MiniLM bio+tweet, GloVe tweets)'),
                      ('TIMME', 'results/scores/timme_scores_minilm_only.npz', 'p_C text classifier (MiniLM only)'),
                      ('Pokec slice', 'results/scores/pokec_slice_scores.npz', 'p_C text classifier (interest text)')]:
        z = np.load(P(f))
        x.addds(ds, 'real', 'cross-fitted, predicts graph community', nm, 'text_to_community_acc', float(z['content_acc']), f, 'npz key content_acc', role_='ours')
        x.addds(ds, 'real', 'cross-fitted, predicts graph community', 'majority class', 'text_to_community_acc', float(z['content_majority']), f, 'npz key content_majority', role_='control')
    f = 'results/logs/amb_minilm.log'
    x.addds('TIMME', 'real', 'MiniLM only, log of the run', 'p_C text classifier (MiniLM only)', 'text_to_community_acc', 0.496, f, 'log line 2', role_='ours')
    ROWS.pop()  # duplicate of npz row: kept out; the npz is the source
    f = 'results/logs/block_gateway.log'
    x.addds('TIMME', 'real', 'GloVe-only affinity classifier', 'p_C text classifier (GloVe only)', 'text_to_community_acc', 0.637, f, 'log line 1', role_='ours')
    f = 'results/logs/timme_party_check.log'
    x.addds('TIMME', 'real', 'party-check run, different text configuration', 'p_C text classifier (party-check config)', 'text_to_community_acc', 0.275, f, 'log line 3', role_='ours')
    x.addds('TIMME', 'real', 'party-check run, different text configuration', 'majority class', 'text_to_community_acc', 0.318, f, 'log line 3', role_='control')
    f = 'results/logs/text_only_clustering_timme.log'
    rx = re.compile(r'^(MiniLM bio\+latest tweet|GloVe avg of tweets|both concatenated) \| probe ([\d.]+)\s+k=\s*(\d+) \(>=1%:\s*(\d+)\) largest=([\d.]+) Q=([+-][\d.]+) NMI=([\d.]+) ARI=([-\d.]+) eta2=([\d.]+) stab=([\d.]+)')
    for ln, l in lines(f):
        m = rx.match(l)
        if m:
            for nm, g in [('party_probe_acc', 2), ('modularity_Q_social_graph', 6), ('nmi_party', 7), ('ari_party', 8), ('degree_eta2', 9), ('seed_stability_ari', 10)]:
                x.addds('TIMME', 'real', 'text-only k-means clustering of raw text features', 'text k-means: ' + m.group(1), nm, m.group(g), f, f'log line {ln}', role_='baseline')
    f = 'results/logs/pokec_profile_fields.log'
    for ln, l in lines(f):
        m = re.match(r'^(age\+gender\+completion|interest text|text\+profile)\s+acc ([\d.]+)\s+\(majority ([\d.]+)\)', l)
        if m:
            x.addds('Pokec slice', 'real', 'predict graph community from profile fields', m.group(1), 'text_to_community_acc', m.group(2), f, f'log line {ln}', role_='ours' if m.group(1) != 'age+gender+completion' else 'baseline')
            x.addds('Pokec slice', 'real', 'predict graph community from profile fields', 'majority class', 'text_to_community_acc', m.group(3), f, f'log line {ln}', role_='control')


# ================================================================= E05
def e05_bench():
    x = X('E05', 'T10.4', 'detection (planted mismatch)', 'planted DC-SBM n=4000 K=8', 'synthetic')
    f = 'results/detection/synthetic_benchmark.json'
    d = jl(f)
    for i, r in enumerate(d):
        s = f"nuisance={r['nuisance']} mu={r['mu']} sigma={r['sigma']} (mean of 3 seeds)"
        x.add(s, 'topology partition T', 'T_ari_vs_planted', r['T_ari'], f, f'[{i}].T_ari', n=3)
        for sc, (a, b) in r['auc'].items():
            x.add(s, sc, 'auc_overall', a, f, f'[{i}].auc.{sc}[0]', n=3)
            x.add(s, sc, 'auc_within_degree_terciles', b, f, f'[{i}].auc.{sc}[1]', n=3)


# ================================================================= E06
def e06_party():
    x = X('E06', 'T10.6', 'detection (real party validation)', 'TIMME', 'real')
    f = 'results/logs/timme_party_check.log'
    for ln, l in lines(f):
        m = re.match(r'^\s+(\S+)\s+AUC ([\d.]+)\s+within-degree ([\d.]+)\s+one-sided p ([\d.]+)', l)
        if m:
            s = '11 party misfits among 595 labeled party-core nodes'
            for nm, g in [('auc', 2), ('auc_within_degree', 3), ('p_one_sided', 4)]:
                x.add(s, m.group(1), nm, m.group(g), f, f'log line {ln}', n=11)


# ================================================================= E07
def e07_blocking():
    x = X('E07', 'T10.7', 'blocking (gate)', 'TIMME', 'simulation-on-real')
    for ds, f, mode, sims in [('TIMME', 'results/blocking/timme_content_blind.json', 'content-blind', 100),
                              ('TIMME', 'results/blocking/timme_content_aware.json', 'content-aware', 100),
                              ('Pokec slice', 'results/blocking/pokec_slice_content_blind.json', 'content-blind', 200)]:
        d = jl(f)
        for model, strat in d.items():
            nb = strat['none']['0']
            for sname, cells in strat.items():
                for k, c in cells.items():
                    if sname == 'none':
                        k = 'no blocking'
                    s = f'{mode} model={model} k={k} sims={sims}'
                    se = (c[2], c[3]) if len(c) == 4 else (None, None)
                    key = f'{model}.{sname}.{"0" if sname == "none" else k}'
                    for i, (nm, nb_i) in enumerate([('infected_share', nb[0]), ('cross_community_infected_share', nb[1])]):
                        lo = hi = ''
                        unc = 'none'
                        if se[i] is not None:
                            lo, hi, unc = c[i] - se[i], c[i] + se[i], 'se_plus_minus_1'
                        x.addds(ds, 'simulation-on-real', s, sname, nm, c[i], f, f'{key}[{i}]', lo=lo, hi=hi, unc=unc, n=sims)
                        if sname != 'none':
                            x.addds(ds, 'simulation-on-real', s, sname, nm.replace('_share', '_removed_pct'), 100 * (1 - c[i] / nb_i), f,
                                    f'derived: 100*(1-{key}[{i}]/{model}.none.0[{i}])', n=sims)


# ================================================================= E08
def e08_regime():
    x = X('E08', 'T10.8/T10.9', 'blocking (regime boundary)', 'planted DC-SBM n=4000 K=8', 'synthetic')
    f = 'results/regime/node_regime_map.json'
    for i, r in enumerate(jl(f)):
        s = f"mu={r['mu']} beta={r['beta']} plant={r['plant']} place={r['place']} (2 seeds)"
        x.add(s, 'misfit scores', 'detect_auc', r['detect_auc'], f, f'[{i}].detect_auc', n=2)
        x.add(s, 'best of PageRank/betweenness', 'cross_spread_removed_pts', 100 * r['best_struct_removed'], f, f'[{i}].best_struct_removed x100', n=2, role_='baseline')
        for m in ['misfit', 'misfit_x_pr', 'gateway', 'misfit_rerank', 'gateway_struct']:
            g, c = r[m]
            x.add(s, m, 'gain_vs_best_structural_pts', 100 * g, f, f'[{i}].{m}[0] x100', lo=100 * (g - c), hi=100 * (g + c), unc='ci95_paired', n=2)
    f = 'results/logs/timme_position_on_regime_map.log'
    x2 = X('E08', 'T10.8/T10.9', 'blocking (regime boundary)', 'TIMME', 'real')
    for ln, l in lines(f):
        m = re.match(r'^TIMME acceptance of own community: median ([\d.]+) -> effective agreement effect beta ~ ([\d.]+)', l)
        if m:
            x2.add('GloVe acceptance of own community', 'TIMME placement', 'median_acceptance_own_community', m.group(1), f, f'log line {ln}', role_='reference')
            x2.add('GloVe acceptance of own community', 'TIMME placement', 'effective_agreement_beta', m.group(2), f, f'log line {ln}', role_='reference')
        m = re.match(r"^top-5% misfit nodes: median degree percentile ([\d.]+)", l)
        if m:
            x2.add('top-5% misfit nodes', 'TIMME placement', 'median_degree_percentile', m.group(1), f, f'log line {ln}', role_='reference')
        m = re.match(r'^synthetic hub placement: planted median degree percentile ([\d.]+); random placement ~([\d.]+)', l)
        if m:
            x.add('hub placement', 'synthetic placement', 'median_degree_percentile', m.group(1), f, f'log line {ln}', role_='reference')
            x.add('random placement', 'synthetic placement', 'median_degree_percentile', m.group(2), f, f'log line {ln}', role_='reference')


# ================================================================= E09
def e09_edges():
    x = X('E09', 'T10.11', 'blocking (edge-level link friction)', 'planted DC-SBM + TIMME', 'synthetic')
    f = 'results/logs/edge_v2.log'
    for ln, l in lines(f):
        m = re.match(r'^synthetic beta=([\d.]+) (\S+): ([+-][\d.]+) \+- ([\d.]+) vs bridge_struct', l)
        if m:
            v, c = float(m.group(3)), float(m.group(4))
            x.add(f'synthetic mu=0.3 5% hub mismatches beta={m.group(1)} 4 seeds pooled ic:2', m.group(2), 'diff_vs_bridge_struct_pts', v, f, f'log line {ln}', lo=v - c, hi=v + c, unc='ci95_paired', n=4)
    f = 'results/logs/edge_seeds.log'
    for ln, l in lines(f):
        m = re.match(r'^beta=([\d.]+) seed=(\d): content - struct = ([+-][\d.]+) \+- ([\d.]+) points', l)
        if m:
            v, c = float(m.group(3)), float(m.group(4))
            x.add(f'synthetic beta={m.group(1)} seed={m.group(2)}', 'bridge_content', 'diff_vs_bridge_struct_pts', v, f, f'log line {ln}', lo=v - c, hi=v + c, unc='ci95_paired', n=1)

    def table(f, budgets, keep=None):
        for ln, l in lines(f):
            m = re.match(r'^no cut: cross-community infected ([\d.]+);', l)
            if m:
                x2.add('no cut baseline, content-aware ic:2', 'no blocking', 'cross_community_infected_share', m.group(1), f, f'log line {ln}', role_='control')
            m = re.match(r'^  (\S+)\s+(.*)$', l)
            if m and '%' in l and m.group(1) not in ('',):
                cells = re.findall(r'([\d.]+)% \(\s*([+-][\d.]+)±\s*([\d.]+)\)', m.group(2))
                if keep and m.group(1) not in keep:
                    continue
                for k, (rem, d, c) in zip(budgets, cells):
                    s = f'TIMME content-aware ic:2 cut {k} links'
                    x2.add(s, m.group(1), 'cross_infection_removed_pct', rem, f, f'log line {ln}', n=60)
                    x2.add(s, m.group(1), 'diff_vs_bridge_struct_pts', d, f, f'log line {ln}', lo=float(d) - float(c), hi=float(d) + float(c), unc='ci95_paired', n=60)
    x2 = X('E09', 'T10.11', 'blocking (edge-level link friction)', 'TIMME', 'simulation-on-real')
    table('results/logs/edges_timme.log', [2000, 10000, 40000])
    table('results/logs/edge_v2.log', [10000, 40000], keep={'bridge_struct_x_recv', 'bridge_struct_x_recv_ORACLE'})
    table('results/logs/edge_oracle.log', [10000, 40000], keep={'bridge_oracle'})
    f = 'results/logs/edge_oracle.log'
    for ln, l in lines(f):
        m = re.match(r'^node-level spearman\(p_C minilm, affinity glove\).*: ([\d.]+)', l)
        if m:
            x2.add('p_C (MiniLM) vs GloVe affinity, all node-community pairs', 'receptivity proxy', 'spearman', m.group(1), f, f'log line {ln}', role_='reference')
        m = re.match(r'^argmax agreement: ([\d.]+)', l)
        if m:
            x2.add('p_C argmax vs affinity argmax', 'receptivity proxy', 'argmax_agreement', m.group(1), f, f'log line {ln}', role_='reference')


# ================================================================= E10
def e10_h1():
    x = X('E10', 'H1', 'blocking (radicalisation funnel)', 'synthetic funnel n=4000', 'synthetic')
    f = 'results/hypotheses/h1/h1_results.json'
    d = jl(f)
    ncell = pos = neg = 0
    for i, r in enumerate(d):
        p = r['params']
        base = f"name={r['name']} beta={p['beta']} rec_hub={p['rec_hub']} decoy={p['decoy']} model={p['model']}"
        x.add(base, 'misfit', 'recruiter_detection_auc', r['det']['auc_misfit_recruiter'], f, f'[{i}].det.auc_misfit_recruiter', n=r['n_seeds'])
        x.add(base, 'degree', 'recruiter_detection_auc', r['det']['auc_degree_recruiter'], f, f'[{i}].det.auc_degree_recruiter', n=r['n_seeds'])
        x.add(base, 'no blocking', 'base_harm_share', r['base_harm'], f, f'[{i}].base_harm', n=r['n_seeds'], role_='control')
        for b, c in r['budgets'].items():
            s = base + f' budget={b}'
            for m, v in c['removed'].items():
                x.add(s, m, 'harm_removed_pts', 100 * v, f, f"[{i}].budgets.{b}.removed.{m} x100", n=r['n_seeds'])
            for m in ['misfit', 'misfit_x_pr', 'gateway', 'random', 'oracle_recruiters']:
                g, ci = c[m]
                x.add(s, m, 'gain_vs_best_structural_pts', 100 * g, f, f'[{i}].budgets.{b}.{m}[0] x100', lo=100 * (g - ci), hi=100 * (g + ci), unc='ci95_t_over_seeds', n=r['n_seeds'])
            ncell += 1
            g, ci = c['misfit']
            pos += g - ci > 0
            neg += g + ci < 0
    x.add('all 30 settings x 4 budgets', 'misfit', 'n_cells', ncell, f, 'derived: count of cells', role_='ours')
    x.add('all 30 settings x 4 budgets', 'misfit', 'n_cells_ci_above_0', pos, f, 'derived: count of cells with gain-ci>0')
    x.add('all 30 settings x 4 budgets', 'misfit', 'n_cells_ci_below_0', neg, f, 'derived: count of cells with gain+ci<0')


# ================================================================= E11
def rule_of(name):
    return name.split('|')[-1] if '|' in name else name


def e11_h2():
    x = X('E11', 'H2', 'blocking (misfit as filter)', '', '')
    def agg(f, ds, dt, keyf, setf):
        d = jl(f)
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in d:
            nm = r['name']
            if nm.endswith('|cent'):
                continue  # structural baselines: gain 0 by construction
            g[keyf(r)][r['seed']].append(r['gain'])
        # g[(group, config...)] -> per seed list (mu collapsed per config by key)
        groups = collections.defaultdict(list)
        for key, sd in g.items():
            seeds = sorted(sd)
            vals = [np.mean(sd[s]) for s in seeds]
            m, c = tci(vals)
            groups[setf(key)].append((m, c, key, len(seeds)))
        for (grp, rule), lst in sorted(groups.items(), key=lambda kv: str(kv[0])):
            ok = rule != 'rand'
            role_ = 'control' if rule == 'rand' else 'ours'
            s = f'{grp}; filter rule={rule}; aggregated over configs (pool x budget' + ('x mu)' if ds.startswith('planted') else ')')
            mth = f'misfit filter ({rule})' if ok else 'random-from-shortlist'
            pos = sum(1 for m, c, k, n in lst if m - c > 0)
            neg = sum(1 for m, c, k, n in lst if m + c < 0)
            best = max(lst, key=lambda t: t[0])
            x.addds(ds, dt, s, mth, 'n_configs', len(lst), f, 'derived: count of configs', role_=role_)
            x.addds(ds, dt, s, mth, 'n_configs_ci_above_0', pos, f, 'derived: configs with mean-ci>0 (t over seeds)', role_=role_)
            x.addds(ds, dt, s, mth, 'n_configs_ci_below_0', neg, f, 'derived: configs with mean+ci<0', role_=role_)
            x.addds(ds, dt, s + f'; best config={best[2][-2:]}', mth, 'best_config_gain_pts', 100 * best[0], f, 'derived: max over configs of mean gain x100',
                    lo=100 * (best[0] - best[1]), hi=100 * (best[0] + best[1]), unc='ci95_t_over_seeds', n=best[3], role_=role_)
    agg('results/hypotheses/h2/real_timme.json', 'TIMME', 'simulation-on-real',
        lambda r: (r['model'], r['mode'], r['metric'], r['name']), lambda k: (f'TIMME model={k[0]} mode={k[1]} metric={k[2]}', rule_of(k[3]) if k[3].startswith(('pr|', 'btw|', 'deg|', 'union|')) else ('rand' if k[3].startswith('random') else 'prod' if '_x_' in k[3] else 'misfit')))
    agg('results/hypotheses/h2/real_pokec_knm.json', 'Pokec slice', 'simulation-on-real',
        lambda r: (r['model'], r['mode'], r['metric'], r['name']), lambda k: (f'Pokec slice model={k[0]} mode={k[1]} metric={k[2]}', rule_of(k[3]) if k[3].startswith(('pr|', 'btw|', 'deg|', 'union|')) else ('rand' if k[3].startswith('random') else 'prod' if '_x_' in k[3] else 'misfit')))
    agg('results/hypotheses/h2/synth.json', 'planted DC-SBM n=4000', 'synthetic',
        lambda r: (r['place'], r['beta'], r['metric'], r['mu'], r['name']), lambda k: (f'synthetic place={k[0]} beta={k[1]} metric={k[2]}', rule_of(k[4]) if k[4].startswith(('pr|', 'btw|', 'deg|', 'union|')) else ('rand' if k[4].startswith('random') else 'prod' if '_x_' in k[4] else 'misfit')))
    # totals (non-rand)
    for f, ds, dt in [('results/hypotheses/h2/real_timme.json', 'TIMME', 'simulation-on-real'), ('results/hypotheses/h2/real_pokec_knm.json', 'Pokec slice', 'simulation-on-real'),
                      ('results/hypotheses/h2/synth.json', 'planted DC-SBM n=4000', 'synthetic')]:
        d = jl(f)
        for met in ('inf', 'cross'):
            g = collections.defaultdict(lambda: collections.defaultdict(list))
            for r in d:
                nm = r['name']
                if nm.endswith('|cent') or rule_of(nm) == 'rand' or nm.startswith('random'):
                    continue
                if r['metric'] != met:
                    continue
                key = tuple(r.get(k) for k in ('ds', 'model', 'mode', 'place', 'beta', 'mu')) + (nm,)
                g[key][r['seed']].append(r['gain'])
            cnt = pos = neg = 0
            for key, sd in g.items():
                m, c = tci([np.mean(v) for v in sd.values()])
                cnt += 1; pos += m - c > 0; neg += m + c < 0
            if ds.startswith('planted'):
                # per-family counts (added 2026-11-18 after the H2 README correction)
                fam = collections.defaultdict(lambda: [0, 0])
                posg, bybeta, byplace = [], collections.Counter(), collections.Counter()
                for key, sd in g.items():
                    m, c = tci([np.mean(v) for v in sd.values()])
                    fm = rule_of(key[-1])
                    fam[fm][0] += 1
                    if m - c > 0:
                        fam[fm][1] += 1; posg.append(100 * m); bybeta[key[4]] += 1; byplace[key[3]] += 1
                sfam = f'synthetic family counts, metric={"infected" if met == "inf" else "cross-community"} share'
                for fm, (nc, npos) in sorted(fam.items()):
                    x.addds(ds, dt, sfam + f'; family={fm}', f'misfit filter ({fm})', 'n_configs', nc, f, 'derived: configs per filter family (mu x place x beta x pool x budget)', role_='ours')
                    x.addds(ds, dt, sfam + f'; family={fm}', f'misfit filter ({fm})', 'n_configs_ci_above_0', npos, f, 'derived: configs with mean-ci>0 (t over seeds)', role_='ours')
                if met == 'cross':
                    for b, nb in sorted(bybeta.items()):
                        x.addds(ds, dt, sfam + f'; positives by beta={b}', 'misfit filters (all rules)', 'n_configs_ci_above_0', nb, f, 'derived: positives with this beta', role_='ours')
                    for pl, npl in sorted(byplace.items()):
                        x.addds(ds, dt, sfam + f'; positives by place={pl}', 'misfit filters (all rules)', 'n_configs_ci_above_0', npl, f, 'derived: positives with this placement', role_='ours')
                    x.addds(ds, dt, sfam + '; positives', 'misfit filters (all rules)', 'median_positive_gain_pts', float(np.median(posg)), f, 'derived: median over positive configs of mean gain x100', role_='ours')
                    x.addds(ds, dt, sfam + '; positives', 'misfit filters (all rules)', 'max_positive_gain_pts', float(np.max(posg)), f, 'derived: max over positive configs of mean gain x100', role_='ours')
                # random-from-shortlist and centrality-only controls
                gc = collections.defaultdict(lambda: collections.defaultdict(list))
                for r in d:
                    nm = r['name']
                    if r['metric'] != met:
                        continue
                    if rule_of(nm) == 'rand' or nm.startswith('random'):
                        lab = 'random-from-shortlist'
                    elif nm.endswith('|cent'):
                        lab = 'centrality-only'
                    else:
                        continue
                    gc[(lab,) + tuple(r.get(k) for k in ('place', 'beta', 'mu')) + (nm,)][r['seed']].append(r['gain'])
                cc = collections.defaultdict(lambda: [0, 0])
                for key, sd in gc.items():
                    m, c = tci([np.mean(v) for v in sd.values()])
                    cc[key[0]][0] += 1; cc[key[0]][1] += m - c > 0
                for lab, (nc, npos) in sorted(cc.items()):
                    sc = sfam + f'; control={lab}'
                    x.addds(ds, dt, sc, lab, 'n_configs', nc, f, 'derived: control configs', role_='control')
                    x.addds(ds, dt, sc, lab, 'n_configs_ci_above_0', int(npos), f, 'derived: control configs with mean-ci>0', role_='control')
            s = f'ALL {ds} configs, metric={"infected" if met == "inf" else "cross-community"} share (rand controls and plain centrality excluded)'
            x.addds(ds, dt, s, 'misfit filters (all rules)', 'n_configs', cnt, f, 'derived: count of configs', role_='ours')
            x.addds(ds, dt, s, 'misfit filters (all rules)', 'n_configs_ci_above_0', pos, f, 'derived: configs with mean-ci>0 (t over seeds)', role_='ours')
            x.addds(ds, dt, s, 'misfit filters (all rules)', 'n_configs_ci_below_0', neg, f, 'derived: configs with mean+ci<0', role_='ours')
    # link-level variant
    f = 'results/hypotheses/h2/edges.json'
    d = jl(f)
    g = collections.defaultdict(list)
    for r in d:
        g[(r['beta'], r['k'], r['name'])].append((r['gain'], r['removed']))
    for (beta, k, nm), v in sorted(g.items()):
        m, c = tci([a for a, b in v])
        meth = nm if nm == 'struct' else 'link ' + nm
        x.addds('planted DC-SBM (link-level)', 'synthetic', f'link-level beta={beta} k={k} links', meth, 'gain_vs_struct_pts', 100 * m, f,
                'derived: mean over seeds of gain x100', lo=100 * (m - c), hi=100 * (m + c), unc='ci95_t_over_seeds', n=len(v),
                role_='baseline' if nm == 'struct' else ('control' if nm.endswith('rand') else 'ours'))
        x.addds('planted DC-SBM (link-level)', 'synthetic', f'link-level beta={beta} k={k} links', meth, 'removed_pts', 100 * np.mean([b for a, b in v]), f,
                'derived: mean over seeds of removed x100', n=len(v), role_='baseline' if nm == 'struct' else ('control' if nm.endswith('rand') else 'ours'))


# ================================================================= E12
def e12_h3():
    x = X('E12', 'H3', 'blocking + detection + delivery scenarios', 'planted DC-SBM n=4000 K=8', 'synthetic')
    f = 'results/hypotheses/h3/h3_results.json'
    d = jl(f)
    for sc, v in d.items():
        cfg = ','.join(f'{k}={vv}' for k, vv in v['cfg'].items())
        for kk, res in v['res'].items():
            if kk.startswith('k='):
                s = f'scenario={sc} ({cfg}) {kk}'
                for m, (mv, c) in res['all'].items():
                    x.add(s, m, 'removed_pts', 100 * mv, f, f'{sc}.res.{kk}.all.{m} x100', lo=100 * (mv - c), hi=100 * (mv + c), unc='ci95_t_over_seeds', n=res['n_seeds'])
                for key, val in res.items():
                    if key.startswith('gain'):
                        g, c = val
                        nm = {'gain': 'best misfit-based (picked per cell)', 'gain_fixed_misfit_x_pr': 'misfit_x_pr'}.get(key, key)
                        x.add(s, nm, 'gain_vs_best_structural_pts', 100 * g, f, f'{sc}.res.{kk}.{key}[0] x100', lo=100 * (g - c), hi=100 * (g + c), unc='ci95_t_over_seeds', n=res['n_seeds'],
                              role_='ours')
                x.add(s, 'best misfit-based (picked per cell)', 'wins_out_of_seeds', res['wins'], f, f'{sc}.res.{kk}.wins', n=res['n_seeds'], role_='ours')
            elif kk == 'auc':
                for m, (mv, sd) in res.items():
                    x.add(f'scenario={sc} ({cfg}) detection', m, 'auc', mv, f, f'{sc}.res.auc.{m}[0]', lo=mv - sd, hi=mv + sd, unc='sd_over_seeds', n=8)
            elif kk.startswith('prec@'):
                s = f'scenario={sc} ({cfg}) detection {kk}'
                for m, (mv, c) in res['all'].items():
                    x.add(s, m, 'precision_at_k', mv, f, f'{sc}.res.{kk}.all.{m}[0]', lo=mv - c, hi=mv + c, unc='ci95_t_over_seeds', n=res['n_seeds'])
                for key, val in res.items():
                    if key.startswith('gain'):
                        g, c = val
                        nm = {'gain': 'best misfit-based (picked per cell)', 'gain_fixed_misfit_x_pr': 'misfit_x_pr', 'gain_vs_best_other_nonstruct': 'best misfit-based minus best other non-structural'}.get(key, key)
                        x.add(s, nm, 'gain_vs_best_structural_pts', 100 * g, f, f'{sc}.res.{kk}.{key}[0] x100', lo=100 * (g - c), hi=100 * (g + c), unc='ci95_t_over_seeds', n=res['n_seeds'], role_='ours')


# ================================================================= E13
BASELINES_H4A = ['degree', 'pagerank', 'betweenness', 'text_iso', 'topo_participation', 'topo_ambT', 'low_degree', 'knn_dup']


def e13_h4a():
    x = X('E13', 'H4a', 'detection (infiltrators/campaigns/recruiters)', 'planted DC-SBM n=4000 K=8', 'synthetic')
    for f, label in [('results/hypotheses/h4a/h4a_main.json', 'main'), ('results/hypotheses/h4a/h4a_sweep.json', 'sweep')]:
        d = jl(f)
        for cfg, runs in d.items():
            n = len(runs)
            P0 = runs[0]['P']
            s = f"{label} config={cfg} kind={P0['kind']} place={P0['place']} mu={P0['mu']} (text acc {np.mean([r['acc'] for r in runs]):.2f})"
            for sc in runs[0]['res']:
                for j, nm in enumerate(['auc', 'precision_at_k']):
                    v = [r['res'][sc][j] for r in runs]
                    m = float(np.mean(v)); sd = float(np.std(v, ddof=1)) if n > 1 else 0
                    x.add(s, sc, nm, m, f, f'{cfg}[*].res.{sc}[{j}] mean over seeds', lo=m - sd, hi=m + sd, unc='sd_over_seeds', n=n)
            # paired gain misfit - best baseline (best by mean AUC)
            means = {b: np.mean([r['res'][b][0] for r in runs]) for b in BASELINES_H4A}
            bb = max(means, key=means.get)
            diffs = [r['res']['misfit'][0] - r['res'][bb][0] for r in runs]
            m, c = tci(diffs)
            x.add(s, 'misfit', 'auc_gain_vs_best_baseline', m, f, f'derived: misfit AUC - {bb} AUC, paired over seeds', lo=m - c, hi=m + c, unc='ci95_t_over_seeds', n=n)
            x.add(s, bb, 'is_best_baseline', 1, f, 'derived: baseline with highest mean AUC', role_='baseline')


# ================================================================= E14
def e14_h4b():
    x = X('E14', 'H4b', 'bridge validation (real interactions)', 'TIMME', 'real')
    f = 'results/hypotheses/h4b/primary.json'
    d = jl(f)
    for k, v in d['spearman'].items():
        x.add('follow-only communities res=1.0 seed=0 k=4', k, 'spearman', v, f, f'spearman.{k}', role_='reference')
    for grp, sc in d['q1q2q3'].items():
        for score, outs in sc.items():
            if score in ('regression', 'activity'):
                continue
            for out, v in outs.items():
                if not isinstance(v, dict) or 'diff' not in v:
                    continue
                s = f'primary res=1.0 seed=0 interaction={grp} outcome={out} vs matched controls (degree decile)'
                key = f'q1q2q3.{grp}.{score}.{out}'
                x.add(s, score, 'diff_vs_matched_controls', v['diff'], f, key + '.diff', lo=v['ci'][0], hi=v['ci'][1], unc='permutation_ci95', n=v['n_high'])
                x.add(s, score, 'control_mean', v['ctrl_mean'], f, key + '.ctrl_mean', n=v['n_ctrl'], role_='control')
                x.add(s, score, 'p_one_sided', v['p_one'], f, key + '.p_one', n=v['n_high'])
    for grp, outs in d['circular_pooled'].items():
        for out, v in outs.items():
            x.add(f'circular pooled interaction={grp} outcome={out}', 'misfit', 'diff_vs_matched_controls', v['diff'], f, f'circular_pooled.{grp}.{out}.diff', lo=v['ci'][0], hi=v['ci'][1], unc='permutation_ci95', n=v['n_high'])
    # gating
    f = 'results/hypotheses/h4b/gating.log'
    rx = re.compile(r'^\(([\d.]+), (\d)\) (\d) (\w+) cross agree beta (-?[\d.]+) \[(-?[\d.]+),(-?[\d.]+)\] same-T (-?[\d.]+)')
    for ln, l in lines(f):
        m = rx.match(l)
        if m:
            s = f'res={m.group(1)} seed={m.group(2)} classifier={m.group(3)} interaction={m.group(4)}'
            x.add(s, 'case-control logit', 'agreement_beta', m.group(5), f, f'log line {ln}', lo=m.group(6), hi=m.group(7), unc='ci95', role_='reference')
            x.add(s, 'case-control logit', 'agreement_beta_same_T', m.group(8), f, f'log line {ln}', role_='reference')
    # robustness grids, aggregated
    for f, label, keyf in [('results/hypotheses/h4b/grid.json', 'grid res x seed x k', lambda r: (r['group'], r['score'], r['outcome'], r['k'])),
                           ('results/hypotheses/h4b/extra.json', 'extra (log outcome, alt matching)', lambda r: (r['group'], r['score'], r['outcome'], r['matching'])),
                           ('results/hypotheses/h4b/extra2.json', 'extra2 (community x degree matching)', lambda r: (r['group'], r['score'], r['outcome'], r['matching']))]:
        g = collections.defaultdict(list)
        for r in jl(f):
            if r['score'] in ('misfit', 'disagreement'):
                g[keyf(r)].append(r)
        for (grp, score, out, extra), rs in sorted(g.items(), key=str):
            s = f'{label}: interaction={grp} outcome={out} {"k" if label.startswith("grid") else "matching"}={extra}; aggregated over {len(rs)} (res,seed) cells'
            x.add(s, score, 'median_diff_vs_matched_controls', float(np.median([r['diff'] for r in rs])), f, 'derived: median of diff over cells', n=len(rs))
            x.add(s, score, 'n_cells_ci_above_0', sum(r['ci'][0] > 0 for r in rs), f, 'derived: cells with ci_low>0', n=len(rs))
            x.add(s, score, 'n_cells_ci_below_0', sum(r['ci'][1] < 0 for r in rs), f, 'derived: cells with ci_high<0', n=len(rs))


# ================================================================= E15
def e15_h4c():
    x = X('E15', 'H4c', 'misinformation detection (rumour threads)', 'PHEME', 'real')
    f = 'results/hypotheses/h4c/h4c_pheme_results.json'
    d = jl(f)
    names = {'text': 'text only', 'struct': 'text + structure', 'struct+misfit': 'text + structure + misfit', 'misfit': 'misfit alone'}
    for cv in ['stratified5', 'leave_event_out']:
        for k, nm in names.items():
            r = d[f'{cv}/{k}']
            x.add(cv, nm, 'auc', r['auc'], f, f'{cv}/{k}.auc', lo=r['auc_ci'][0], hi=r['auc_ci'][1], unc='bootstrap95', n=6425, role_='ours' if 'misfit' in k else 'baseline')
            x.add(cv, nm, 'f1', r['f1'], f, f'{cv}/{k}.f1', n=6425, role_='ours' if 'misfit' in k else 'baseline')
        for k, nm in [('delta text->struct', 'adding structure to text'), ('delta struct->struct+misfit', 'adding misfit to text+structure')]:
            r = d[f'{cv}/{k}']
            x.add(cv, nm, 'delta_auc_paired', r['mean'], f, f'{cv}/{k}.mean', lo=r['ci'][0], hi=r['ci'][1], unc='bootstrap95_paired', n=6425, role_='ours' if 'misfit' in k else 'baseline')
    for k, v in d['univariate_auc'].items():
        x.add('univariate AUC of thread aggregate', k, 'auc', v, f, f'univariate_auc.{k}', role_='ours' if k.startswith('mis') else 'baseline')
    x.add('threads with defined misfit vs log size', 'n_defined', 'correlation', d['corr_ndefined_logsize'], f, 'corr_ndefined_logsize', role_='reference')


# ================================================================= E16
def e16_h4d():
    x = X('E16', 'H4d', 'blocking (extended regime sweep)', 'planted DC-SBM n=4000 K=8', 'synthetic')
    f = 'results/hypotheses/h4d/h4d_cells.json'
    d = jl(f)
    for met in ('infected', 'cross'):
        c = [r for r in d if r['metric'] == met]
        s = f'all 120 cells, metric={met} share, 12 seeds'
        x.add(s, 'misfit_x_pr (fixed)', 'n_cells_ci_above_0', sum(r['fix_gain'] - r['fix_ci'] > 0 for r in c), f, 'derived: fix_gain-fix_ci>0', n=len(c))
        x.add(s, 'misfit_x_pr (fixed)', 'n_cells_ci_below_0', sum(r['fix_gain'] + r['fix_ci'] < 0 for r in c), f, 'derived: fix_gain+fix_ci<0', n=len(c))
        x.add(s, 'best misfit-based (selected per cell)', 'n_cells_ci_above_0', sum(r['gain'] - r['gain_ci'] > 0 for r in c), f, 'derived: gain-gain_ci>0', n=len(c))
        x.add(s, 'best misfit-based (selected per cell)', 'n_cells_ci_below_0', sum(r['gain'] + r['gain_ci'] < 0 for r in c), f, 'derived: gain+gain_ci<0', n=len(c))
        b = max(c, key=lambda r: r['gain'])
        x.add(s + f" (best cell mu={b['mu']} plant={b['plant']} place={b['place']} beta={b['beta']} budget={b['budget']})", 'best misfit-based (selected per cell)', 'max_cell_gain_pts', b['gain'], f, 'derived: max gain over cells', lo=b['gain'] - b['gain_ci'], hi=b['gain'] + b['gain_ci'], unc='ci95_t_over_seeds', n=12)
        for beta in sorted({r['beta'] for r in c}):
            cb = [r for r in c if r['beta'] == beta]
            s2 = f'metric={met} beta={beta}, mean over {len(cb)} cells'
            x.add(s2, 'misfit_x_pr (fixed)', 'mean_gain_vs_best_structural_pts', float(np.mean([r['fix_gain'] for r in cb])), f, 'derived: mean fix_gain', n=len(cb))
            x.add(s2, 'best misfit-based (selected per cell)', 'mean_gain_vs_best_structural_pts', float(np.mean([r['gain'] for r in cb])), f, 'derived: mean gain', n=len(cb))
    f = 'results/hypotheses/h4d/h4d_summary.txt'
    rx = re.compile(r'^\[(infected|cross)\] (best misfit-based \(selected\)|fixed misfit_x_pr) vs best structural: cells=(\d+); CI>0: (\d+).*CI<0: (\d+); Holm-surviving wins \(alpha \.05\): (\d+)')
    for ln, l in lines(f):
        m = rx.match(l)
        if m:
            meth = 'misfit_x_pr (fixed)' if m.group(2).startswith('fixed') else 'best misfit-based (selected per cell)'
            x.add(f'all 120 cells, metric={m.group(1)} share, Holm correction alpha .05', meth, 'n_cells_holm_surviving_wins', m.group(6), f, f'summary line {ln}', n=m.group(3))


# ================================================================= E17
def e17_probe():
    x = X('E17', 'T11.0', 're-homing motivation probe', 'TIMME', 'real')
    f = 'results/hypotheses/h5/h5_probe.json'
    d = jl(f)
    for st, v in d.items():
        for var, r in v.items():
            text = 'bio only (clean text, no leak)' if var == 'bio only' else 'bio + latest tweet + GloVe (leak control, not a valid result)'
            leaky = var != 'bio only'
            base = f'follow communities {st} (k={r["k"]})'
            if r['acc'] is not None:
                x.add(base + '; ' + text, 'text classifier p_C', 'text_to_community_acc', r['acc'], f, f'{st}.{var}.acc', role_='ours')
            x.add(base + '; ' + text, 'top-10% misfits', 'n_misfits', r['n_misfits'], f, f'{st}.{var}.n_misfits', role_='ours')
            for L in ['retweet', 'like', 'mention']:
                s = f'{base}; {text}; held-out {L}'
                x.add(s, 'argmax p_C (text target)', 'cross_ties_in_target_share', r[L]['hit_argmax_pC'], f, f'{st}.{var}.{L}.hit_argmax_pC', n=r[L]['n_cross'], role_='reference' if leaky else 'ours')
                x.add(s, 'chance (size-weighted)', 'cross_ties_in_target_share', r[L]['chance'], f, f'{st}.{var}.{L}.chance', n=r[L]['n_cross'], role_='control')
                x.add(s, 'own follow community', 'ties_staying_in_own_community_share', r[L]['own_community_share'], f, f'{st}.{var}.{L}.own_community_share', role_='reference')
                if 'structural' in r[L]:
                    sg = r[L]['structural']
                    for pn, pv in sg['hit_share'].items():
                        if pn == 'argmax_pC':
                            continue
                        lab = {'argmax_pC_nonown': 'argmax p_C restricted to non-own communities', 'most_linked': 'most-linked non-own community (follow graph)', 'label_prop': 'label propagation, 1 step (follow graph)',
                               'p_T': 'argmax p_T excluding own (follow graph)', 'adamic_adar': 'Adamic-Adar community score (follow graph)'}[pn]
                        x.add(s, lab, 'cross_ties_in_target_share', pv, f, f'{st}.{var}.{L}.structural.hit_share.{pn}', n=r[L]['n_cross'], role_='reference' if leaky else ('ours' if pn == 'argmax_pC_nonown' else 'baseline'))
                        d_ = sg['text_minus'][pn]
                        x.add(s, 'argmax p_C minus ' + lab, 'text_minus_predictor_hit_share', d_['diff'], f, f'{st}.{var}.{L}.structural.text_minus.{pn}.diff', lo=d_['ci95'][0], hi=d_['ci95'][1], unc='bootstrap95_users', role_='reference' if leaky else 'baseline')
    f = 'TASKS.md'
    # ponytail: the Kumari-extraction retweet share (0.53) and control own-community share (0.69) are not printed by
    # h5_probe.py; hand-entered from TASKS.md T11.0 (listed in _gaps.md)
    x.add('follow communities r1.0_s0 (k=4); bio + latest tweet + GloVe (leak control, not a valid result); held-out retweet', 'Kumari extraction target', 'cross_ties_in_target_share', 0.53, f, 'T11.0 text (hand-entered)', role_='reference')
    x.add('follow communities r1.0_s0 (k=4); matched controls; held-out retweet', 'matched controls', 'ties_staying_in_own_community_share', 0.69, f, 'T11.0 text (hand-entered)', role_='control')


# ================================================================= E18
def e18_rec():
    x = X('E18', 'T11.1', 'link recommendation (re-homing)', 'TIMME', 'real')
    f = 'results/hypotheses/h5/h5_rec.json'
    d = jl(f)
    for st, grps in d.items():
        for g, r in grps.items():
            gname = g
            for m, v in r.items():
                if m in ('comm_acc',):
                    for t, val in v.items():
                        x.add(f'{st} {gname}', {'text argmax p_C': 'text target (argmax p_C)', 'struct argmax p_T': 'structural community (argmax p_T)', 'kumari extraction': 'kumari', 'own follow community': 'own follow community'}[t],
                              'community_top1_hit', val, f, f'{st}.{g}.comm_acc.{t}', role_='ours' if t.startswith('text') else 'baseline')
                elif m in ('share_in_text_target', 'share_in_own'):
                    x.add(f'{st} {gname}', m.replace('share_in_', 'held-out ties in '), 'share_of_heldout_ties', v[0], f, f'{st}.{g}.{m}[0]', lo=v[1], hi=v[2], unc='bootstrap95_users', role_='reference')
                else:
                    for k in ('P@10', 'R@100'):
                        x.add(f'{st} {gname}', m, k, v[k][0], f, f'{st}.{g}.{m}.{k}[0]', lo=v[k][1], hi=v[k][2], unc='bootstrap95_users')


# ================================================================= E19
def e19_robust():
    x = X('E19', 'T11.1b', 'link recommendation robustness', 'TIMME', 'real')
    f = 'results/hypotheses/h5/h5_robust.json'
    d = jl(f)
    for st, r in d['settings'].items():
        for g in ('misfit', 'control'):
            for m, mv in r[g]['mean'].items():
                for k, v in mv.items():
                    x.add(f'{st} {g}', m, k, v, f, f'settings.{st}.{g}.mean.{m}.{k}', n=r['n_' + g])
            for k, (mn, lo, hi) in r[g]['diff'].items():
                x.add(f'{st} {g}', k.replace('text-', 'aa_x_text minus '), 'paired_diff', mn, f, f'settings.{st}.{g}.diff.{k}[0]', lo=lo, hi=hi, unc='bootstrap95_paired', n=r['n_' + g], role_='ours')
        for k, (mn, lo, hi) in r['gain'].items():
            x.add(f'{st} misfit-specific gain', k.replace('text-', 'aa_x_text minus '), 'misfit_minus_control_diff', mn, f, f'settings.{st}.gain.{k}[0]', lo=lo, hi=hi, unc='bootstrap95_paired', role_='ours')
    for k, v in d['summary'].items():
        for g, c in v.items():
            for outcome in ('pos', 'neg', 'span'):
                x.add(f'27 settings, {g} group', k.replace('text-', 'aa_x_text minus '), f'n_settings_ci_{ {"pos": "above_0", "neg": "below_0", "span": "spanning_0"}[outcome] }', c[outcome], f, f'summary.{k}.{g}.{outcome}', n=27, role_='ours')


# ================================================================= E20/E21
def rewire(exp, ds, f, label):
    x = X(exp, 'T11.2', 'rewiring (correction reach)', ds, 'simulation-on-real')
    d = jl(f)
    sims = d['sims']
    for key, r in d['results'].items():
        com, k, model, b = key.split('|')
        s = f'{com} {k} model={model} beta={b[1:]} sims={sims}'
        for met, ints in r.items():
            for iv, v in ints.items():
                unit = 100 if met != 'infected' else 1
                x.add(s, iv, met if met == 'infected' else met + '_pts', 100 * v['mean'] if met != 'infected' else v['mean'], f, f'results.{key}.{met}.{iv}.mean' + (' x100' if met != 'infected' else ''), n=sims)
                for ctr in ('vs_none', 'vs_random', 'vs_text'):
                    if iv in ('none',) and ctr == 'vs_none':
                        continue
                    if (iv == 'random_rewire' and ctr == 'vs_random') or (iv == 'rehome_text' and ctr == 'vs_text'):
                        continue
                    dd, lo, hi = v[ctr]
                    nm = {'vs_none': 'contrast_vs_none_pp', 'vs_random': 'contrast_vs_random_rewire_pp', 'vs_text': 'contrast_vs_rehome_text_pp'}[ctr]
                    x.add(s, iv, met + '_' + nm, 100 * dd, f, f'results.{key}.{met}.{iv}.{ctr}[0] x100', lo=100 * lo, hi=100 * hi, unc='bootstrap95_over_sims', n=sims)


def e20_rewire_timme():
    rewire('E20', 'TIMME', 'results/hypotheses/h5/h5_rewire_timme.json', 'timme')


def e21_rewire_pokec():
    rewire('E21', 'Pokec slice', 'results/hypotheses/h5/h5_rewire_pokec.json', 'pokec')


# ================================================================= E22 (DISABLED until the re-run lands)
def e22_h6():
    """H6 community-deception detection (re-run committed fdfb58f; no marker key needed any more)."""
    for fn in ('results/hypotheses/h6/h6_timme.json', 'results/hypotheses/h6/h6_pokec.json'):
        d = jl(fn)
        ds = 'TIMME' if d['ds'] == 'timme' else 'Pokec slice'
        x = X('E22', 'H6', 'deception detection', ds, 'simulation-on-real')
        ns = len(d['seeds'])
        for tk, t in d['table'].items():
            atk, k = tk.split('|')
            s = f'attack={atk} {k} targets={d["n_targets"]}/seed seeds={ns}'
            ar = 'control' if atk in ('random', 'noise') else 'reference'
            m, sd = t['success_rate']
            x.add(s, atk + ' attack', 'deception_success_rate', m, fn, f'table.{tk}.success_rate[0]', lo=m - sd, hi=m + sd, unc='sd_over_seeds', n=ns, role_=ar)
            x.add(s, atk + ' attack', 'n_successful_targets_per_seed', float(np.mean(t['n_succ'])), fn, f'table.{tk}.n_succ mean', n=ns, role_=ar)
            for det, aucs in t['auc'].items():
                for sub, (m, sd) in aucs.items():
                    x.add(s + f' negatives={sub}', det, 'auc', m, fn, f'table.{tk}.auc.{det}.{sub}[0]', lo=m - sd, hi=m + sd, unc='sd_over_seeds', n=ns)


EXPERIMENTS = [e01_modularity, e02_gnn, e03_views, e04_content, e05_bench, e06_party, e07_blocking, e08_regime, e09_edges,
               e10_h1, e11_h2, e12_h3, e13_h4a, e14_h4b, e15_h4c, e16_h4d, e17_probe, e18_rec, e19_robust,
               e20_rewire_timme, e21_rewire_pokec, e22_h6]


# ================================================================= summary
def L(exp, **c):
    """Look up exactly one long row of experiment `exp`. Condition value ('in', s) means substring."""
    hits = []
    for r in ROWS:
        if r['experiment_id'] != exp:
            continue
        ok = True
        for k, v in c.items():
            if isinstance(v, tuple):
                ok = ok and v[1] in str(r[k])
            else:
                ok = ok and str(r[k]) == str(v)
        if ok:
            hits.append(r)
    assert len(hits) == 1, (exp, c, len(hits))
    return hits[0]


def fmt(r, label, nd=3):
    v = float(r['value'])
    s = f'{v:.{nd}g}' if abs(v) < 1000 else str(v)
    return f'{label} {s}'


SUMMARY = []
USED = []  # (exp, 'our'/'comp', row) for the assertion


def S(exp, phase, use, q, ds, dt, our_m, ours, best_c, comps, headline, verdict, cav):
    """ours/comps: list of (label, lookup-dict) -> string built from long rows."""
    def build(items, tag):
        out = []
        for lab, c in items:
            r = L(exp, **c)
            USED.append((exp, tag, r))
            out.append(fmt(r, lab))
        return '; '.join(out)
    SUMMARY.append(dict(experiment_id=exp, phase=phase, use_case=use, question=q, dataset=ds, data_type=dt, our_method=our_m,
                        our_result=build(ours, 'our'), best_competitor=best_c, competitor_result=build(comps, 'comp'),
                        headline_metric=headline, verdict=verdict, caveat=cav))


def build_summary():
    E = lambda **c: c
    S('E01', 'T7.0', 'dataset characterisation', 'Do the graphs have real community structure (Q far above a degree-preserving rewired null)?',
      'TIMME; Pokec slice; Pokec full', 'real', 'Louvain modularity on the real graph',
      [('TIMME Q', E(dataset='TIMME', method='real graph', metric='modularity_Q')), ('Pokec slice Q', E(dataset='Pokec slice', method='real graph', metric='modularity_Q')), ('Pokec full Q', E(dataset='Pokec full', method='real graph', metric='modularity_Q'))],
      'rewired null (degree-preserving)',
      [('TIMME null Q', E(dataset='TIMME', method='rewired null', metric='modularity_Q')), ('Pokec slice null Q', E(dataset='Pokec slice', method='rewired null', metric='modularity_Q')), ('Pokec full null Q', E(dataset='Pokec full', method='rewired null', metric='modularity_Q'))],
      'modularity Q', 'descriptive', 'Real communities in all three graphs. TIMME cross-community edges 49.3 percent vs 83.1 percent in the null.')
    S('E02', 'T2/T3', 'embedding quality', 'Do self-supervised GNN embeddings (BGRL/DGI/MVGRL) give meaningful clusters?', 'TIMME', 'real',
      'BGRL / DGI / MVGRL (k-means k=2)',
      [('BGRL Q', E(method='BGRL', metric='modularity_Q', setting=('in', 'single run'))), ('DGI Q', E(method='DGI', metric='modularity_Q', setting=('in', 'single run'))), ('MVGRL Q', E(method='MVGRL', metric='modularity_Q', setting=('in', 'single run')))],
      'spectral64 probe / neighbour-vote (graph-only)',
      [('spectral64 fair-probe acc', E(method='spectral64 probe', metric='probe_acc')), ('neighbour-vote acc', E(method='neighbour-vote', metric='probe_acc'))],
      'modularity Q of k-means clusters (and fair-probe accuracy)', 'failed', 'Clusters sort by degree (Q <= 0, NMI about 0); fair probe not better than graph-only baselines. Abandoned phase-1 method, superseded by T10 (S-23). Single run, no CI; old probe was biased.')
    S('E03', 'T10.1/T10.2', 'view construction', 'Can graph-only (T) and text-only (C) partitions be built cleanly, and do they differ?', 'TIMME; Pokec slice', 'real',
      'T view: Leiden x100 on the social graph',
      [('TIMME T-Leiden Q', E(dataset='TIMME', method='T Leiden (best of runs)', metric='modularity_Q')), ('Pokec T-Leiden Q', E(dataset='Pokec slice', method='T Leiden (best of runs)', metric='modularity_Q'))],
      'C view: text kNN-graph Leiden (rejected, Q about 0)',
      [('TIMME text-Leiden Q', E(dataset='TIMME', method='C text kNN Leiden (best of runs)', metric='modularity_Q')), ('TIMME T-vs-C NMI', E(dataset='TIMME', metric='nmi_T_vs_C')), ('Pokec T-vs-C NMI', E(dataset='Pokec slice', metric='nmi_T_vs_C'))],
      'modularity Q on social graph; T-vs-C NMI', 'descriptive', 'Views differ (low NMI). Raw text clusters have no graph structure (Q about 0.01).')
    S('E04', 'T10.3/T10.5', 'content informativeness', 'How well does text predict the graph community?', 'TIMME; Pokec slice', 'real',
      'cross-fitted text classifier p_C',
      [('TIMME acc (MiniLM+GloVe)', E(dataset='TIMME', setting=('in', 'cross-fitted'), method=('in', 'MiniLM bio+tweet'), metric='text_to_community_acc')), ('Pokec acc', E(dataset='Pokec slice', method=('in', 'interest text'), metric='text_to_community_acc', source_file=('in', 'scores')))],
      'majority-class baseline',
      [('TIMME majority', E(dataset='TIMME', setting=('in', 'cross-fitted'), method='majority class', metric='text_to_community_acc', source_key=('in', 'content_majority'), source_file=('in', 'minilm_glove'))), ('Pokec majority', E(dataset='Pokec slice', method='majority class', metric='text_to_community_acc', source_file=('in', 'pokec_slice_scores')))],
      'accuracy of text -> graph community', 'descriptive', 'Informative on TIMME (0.62 vs 0.32), barely above chance on Pokec (0.28 vs 0.25; profile+text 0.356).')
    S('E05', 'T10.4', 'detection (planted mismatch)', 'Can a score recover nodes whose text says A while edges say B?', 'planted DC-SBM', 'synthetic',
      'misfit', [('misfit AUC (nu=1.5 mu=0.5 sigma=2)', E(method='misfit', metric='auc_overall', setting=('in', 'nuisance=1.5 mu=0.5 sigma=2.0')))],
      'content_outlier', [('content_outlier AUC (same setting)', E(method='content_outlier', metric='auc_overall', setting=('in', 'nuisance=1.5 mu=0.5 sigma=2.0'))), ('degree AUC', E(method='degree', metric='auc_overall', setting=('in', 'nuisance=1.5 mu=0.5 sigma=2.0')))],
      'AUC (overall)', 'supported', 'Synthetic and partly circular (planted text mismatch is what misfit measures). Values are means over 3 seeds without stored spread.')
    S('E06', 'T10.6', 'detection (real party validation)', 'Do high-misfit nodes sit in the wrong community for their party?', 'TIMME', 'real',
      'misfit', [('misfit AUC', E(method='misfit', metric='auc'))],
      'misfit with shuffled text (null)', [('shuffled-text AUC', E(method='misfit_SHUFFLED_TEXT', metric='auc')), ('degree AUC', E(method='degree', metric='auc'))],
      'AUC for party misfits', 'partly supported', 'Only 11 positive nodes; one-sided p 0.019. Rows that beat misfit on within-degree AUC exist (participation 0.769).')
    S('E07', 'T10.7', 'blocking (gate)', 'Does blocking high-misfit nodes stop spread better than structural targeting?', 'TIMME; Pokec slice', 'simulation-on-real',
      'gateway (content-weighted), content-aware wc, k=50',
      [('gateway infected removed pct', E(dataset='TIMME', method='gateway', metric='infected_removed_pct', setting='content-aware model=wc k=50 sims=100'))],
      'betweenness',
      [('betweenness infected removed pct', E(dataset='TIMME', method='betweenness', metric='infected_removed_pct', setting='content-aware model=wc k=50 sims=100'))],
      'percent of infected removed', 'not supported', 'Gate failed: betweenness/PageRank win on TIMME and Pokec slice. Content-blind rows where disagreement_x_pr tops TIMME (41.52) are a content-blind structural result; they do not test misfit and are not a positive for our method. Gateway within noise (SE about 15 points).')
    S('E08', 'T10.8/T10.9', 'blocking (regime boundary)', 'When does mismatch-aware node targeting beat the best of PageRank/betweenness?', 'planted DC-SBM; TIMME placement', 'synthetic',
      'misfit_x_pr (cell mu=0.5 beta=6 plant=10 hub)', [('gain pts', E(method='misfit_x_pr', metric='gain_vs_best_structural_pts', setting=('in', 'mu=0.5 beta=6.0 plant=0.1 place=hub')))],
      'best of PageRank/betweenness', [('best structural removed pts', E(method='best of PageRank/betweenness', metric='cross_spread_removed_pts', setting=('in', 'mu=0.5 beta=6.0 plant=0.1 place=hub'))), ('TIMME effective beta', E(metric='effective_agreement_beta'))],
      'paired gain (points) in cross-community spread removed', 'partly supported', 'Content targeting wins only at strong agreement effect (beta 6) with hub-placed mismatches; TIMME (beta about 1.4) sits in the losing region. 2 seeds.')
    S('E09', 'T10.11', 'blocking (edge-level)', 'Does link friction (bridge score x receptivity) beat structural edge scores?', 'planted DC-SBM; TIMME', 'synthetic + simulation-on-real',
      'bridge_struct_x_recv (synthetic beta=4)', [('synthetic beta=4 diff pts', E(method='bridge_struct_x_recv', metric='diff_vs_bridge_struct_pts', setting=('in', 'beta=4.0')))],
      'bridge_struct_x_recv on TIMME (tie with oracle receptivity); synthetic control beta=0',
      [('synthetic beta=0 control diff pts', E(method='bridge_struct_x_recv', metric='diff_vs_bridge_struct_pts', setting=('in', 'beta=0.0'))), ('TIMME 40000 links diff pts', E(dataset='TIMME', method='bridge_struct_x_recv', metric='diff_vs_bridge_struct_pts', setting=('in', 'cut 40000')))],
      'paired difference vs bridge_struct (points)', 'partly supported', 'Wins only on synthetic data with strong agreement; ties on real TIMME even with oracle receptivity; control (beta=0) is negative.')
    S('E10', 'H1', 'blocking (radicalisation funnel)', 'Can misfit find recruiters to block in a radicalisation funnel?', 'synthetic funnel', 'synthetic',
      'misfit', [('cells with CI>0', E(method='misfit', metric='n_cells_ci_above_0')), ('cells with CI<0', E(method='misfit', metric='n_cells_ci_below_0'))],
      'degree / PageRank / betweenness (best structural)', [('cells total', E(method='misfit', metric='n_cells'))],
      'cells (setting x budget) where misfit beats best structural', 'not supported', 'Recruiter-detection AUC of misfit 0.59-0.65 vs degree 0.95 in the hub case (rows in long file).')
    S('E11', 'H2', 'blocking (misfit as filter)', 'Does misfit improve a PageRank/betweenness shortlist?', 'synthetic; TIMME; Pokec slice', 'synthetic + simulation-on-real',
      'misfit filter rules over centrality shortlists',
      [('TIMME configs CI>0 (infected)', E(dataset='TIMME', setting=('in', 'ALL TIMME configs, metric=infected'), metric='n_configs_ci_above_0')), ('TIMME configs CI>0 (cross)', E(dataset='TIMME', setting=('in', 'ALL TIMME configs, metric=cross'), metric='n_configs_ci_above_0')), ('Pokec configs CI>0 (cross)', E(dataset='Pokec slice', setting=('in', 'metric=cross'), metric='n_configs_ci_above_0', method='misfit filters (all rules)')), ('synthetic configs CI>0 (infected)', E(dataset='planted DC-SBM n=4000', setting=('in', 'ALL planted DC-SBM n=4000 configs, metric=infected'), metric='n_configs_ci_above_0')), ('synthetic configs CI>0 (cross)', E(dataset='planted DC-SBM n=4000', setting=('in', 'ALL planted DC-SBM n=4000 configs, metric=cross'), metric='n_configs_ci_above_0'))],
      'best of PageRank/betweenness top-k', [('TIMME configs total (infected)', E(dataset='TIMME', setting=('in', 'ALL TIMME configs, metric=infected'), metric='n_configs')), ('synthetic configs total (infected)', E(dataset='planted DC-SBM n=4000', setting=('in', 'ALL planted DC-SBM n=4000 configs, metric=infected'), metric='n_configs')), ('synthetic configs total (cross)', E(dataset='planted DC-SBM n=4000', setting=('in', 'ALL planted DC-SBM n=4000 configs, metric=cross'), metric='n_configs'))],
      'configs with 95% CI above 0', 'mixed', 'Infected share: no support (0 of 6900 synthetic, 7/1840 TIMME, 1/920 Pokec = noise). Synthetic cross-community share: positives at strong content gating (beta>=6), mostly hub placement; random-from-shortlist control has 0 positives, so not noise. Real data: noise. Counts recomputed from raw files (README corrected 2026-11-18; its 0/1550 is a narrower count of the same conclusion). No pre-registration.')
    S('E12', 'H3', 'blocking + detection scenarios', 'In which scenarios could misfit targeting help?', 'planted DC-SBM', 'synthetic',
      'misfit (detection, random placement) precision@20', [('misfit precision@20', E(method='misfit', metric='precision_at_k', setting=('in', 'scenario=e_detect_random'), source_key=('in', 'prec@20')))],
      'PageRank (precision@20)', [('pagerank precision@20', E(method='pagerank', metric='precision_at_k', setting=('in', 'scenario=e_detect_random'), source_key=('in', 'prec@20')))],
      'precision@20 (detection)', 'partly supported', 'Only detection clearly works; blocking needs beta>=6 and hub/no-hub placement; at realistic beta 1.4 loses. 8 seeds.')
    S('E13', 'H4a', 'detection (infiltrators/campaigns/recruiters)', 'When does misfit detect bad accounts given realistic noisy text?', 'planted DC-SBM', 'synthetic',
      'misfit (infiltrators, random placement, text acc 0.6)', [('misfit AUC', E(method='misfit', metric='auc', setting=('in', 'main config=acc0.6|base ')))],
      'best baseline (betweenness)', [('betweenness AUC', E(method='betweenness', metric='auc', setting=('in', 'main config=acc0.6|base ')))],
      'AUC', 'partly supported', 'Conditional: works for non-structural infiltrators/campaigns when text accuracy >= about 0.45; recruiters/hub/peripheral lose to pagerank/low-degree. Synthetic, partly circular.')
    S('E14', 'H4b', 'bridge validation (real interactions)', 'Are misfit nodes special bridges on real TIMME interactions; does real gating match blocking assumptions?', 'TIMME', 'real',
      'misfit (cross-community count vs matched controls)',
      [('cross count diff', E(method='misfit', metric='diff_vs_matched_controls', setting=('in', 'primary res=1.0 seed=0 interaction=spread outcome=cross_n'))), ('retweet agreement beta (res1 seed0 clf0)', E(method='case-control logit', metric='agreement_beta', setting='res=1.0 seed=0 classifier=0 interaction=retweet'))],
      'PageRank / betweenness / participation (centrality controls)', [('misfit~pagerank Spearman', E(method='misfit~pagerank', metric='spearman'))],
      'difference in cross-community count vs matched controls; agreement beta', 'not supported', 'Verdict NO in README: misfit not a centrality proxy, real agreement beta 0.18-0.33 vs >=6 needed. Cross SHARE is higher (descriptive). Code not audited line by line.')
    S('E15', 'H4c', 'misinformation detection', 'Does a per-account misfit feature help rumour detection on PHEME?', 'PHEME', 'real',
      'text + structure + misfit (stratified 5-fold)', [('delta AUC', E(setting='stratified5', method='adding misfit to text+structure', metric='delta_auc_paired'))],
      'text + structure (baseline)', [('baseline AUC', E(setting='stratified5', method='text + structure', metric='auc'))],
      'paired delta AUC', 'not supported', 'Test is weak: no follow graph, misfit near constant; leave-event-out delta also negative.')
    S('E16', 'H4d', 'blocking (extended regime sweep)', 'Does extending beta to 4-20 rescue misfit-guided blocking?', 'planted DC-SBM', 'synthetic',
      'misfit_x_pr (fixed)', [('cells CI>0 (infected)', E(method='misfit_x_pr (fixed)', metric='n_cells_ci_above_0', setting=('in', 'metric=infected'))), ('cells CI<0 (infected)', E(method='misfit_x_pr (fixed)', metric='n_cells_ci_below_0', setting=('in', 'metric=infected')))],
      'best of PageRank/betweenness/degree/CI/adaptive degree', [('best-selected misfit cells CI>0', E(method='best misfit-based (selected per cell)', metric='n_cells_ci_above_0', setting=('in', 'metric=infected')))],
      'cells (of 120) with gain CI above/below 0', 'not supported', 'Statistically visible but practically negligible (best-selected gain <= +0.06 points); selection bias in best-selected.')
    S('E17', 'T11.0', 're-homing motivation', 'Do misfits follow one community but engage with their text-predicted one?', 'TIMME', 'real',
      'argmax p_C (bio-only text, k=4, retweets)', [('hit share', E(setting=('in', 'r1.0_s0 (k=4); bio only'), method='argmax p_C (text target)', metric='cross_ties_in_target_share', source_key=('in', 'retweet')))],
      'chance (size-weighted)', [('chance', E(setting=('in', 'r1.0_s0 (k=4); bio only'), method='chance (size-weighted)', metric='cross_ties_in_target_share', source_key=('in', 'retweet')))],
      'share of cross-community retweets landing in argmax p_C', 'supported', 'Survives bio-only text (clean, no leak); leaky-text rows are kept as role=reference, labelled leak control, not a valid result. Probe rerun 2026-11-19 gives 0.601 (JSON of 2026-11-18 had 0.599; unmodified script now also gives 0.601, tiny run drift); graph-only predictors added; Kumari 0.53 and controls 0.69 are quoted from TASKS.md.')
    S('E18', 'T11.1', 'link recommendation', 'Are misfits under-served by the follow graph, and does a text-aware recommender beat structure-only ones?', 'TIMME', 'real',
      'aa_x_text (misfits, k=15)', [('P@10', E(setting='r2.0_s0 misfit', method='aa_x_text', metric='P@10'))],
      'Adamic-Adar (misfits, k=15)', [('P@10', E(setting='r2.0_s0 misfit', method='aa', metric='P@10'))],
      'P@10', 'mixed', 'Claim B (under-served) supported; community-level text target wins; account-level MIXED (k=15 +60 percent, k=4 not misfit-specific, k=5 loses).')
    S('E19', 'T11.1b', 'link recommendation robustness', 'Is the account-level text advantage robust across resolutions/seeds?', 'TIMME', 'real',
      'aa_x_text vs aa (P@10, misfits)', [('settings with CI>0', E(setting='27 settings, misfit group', method='aa_x_text minus aa P@10', metric='n_settings_ci_above_0'))],
      'aa (P@10)', [('settings with CI<0', E(setting='27 settings, misfit group', method='aa_x_text minus aa P@10', metric='n_settings_ci_below_0'))],
      'settings (of 27) with paired P@10 CI above/below 0', 'not supported', 'NOT ROBUST on P@10 (README). 27 settings share users (not independent); gain concentrated at r2.0; R@100 favourable.')
    S('E20', 'T11.2', 'rewiring (correction reach)', 'Does degree-preserving re-homing of misfits raise correction reach (TIMME)?', 'TIMME', 'simulation-on-real',
      'rehome_text (r1.0 k5 wc beta 6)', [('correction reach vs none pp', E(setting='r1.0_s0 k5 model=wc beta=6 sims=100', method='rehome_text', metric='correction_reach_contrast_vs_none_pp'))],
      'random_rewire (same cell)', [('random_rewire vs none pp', E(setting='r1.0_s0 k5 model=wc beta=6 sims=100', method='random_rewire', metric='correction_reach_contrast_vs_none_pp'))],
      'correction reach (pp) vs no rewiring', 'partly supported', 'Headline is gain vs no rewiring (4.72 pp, CI excludes 0). In this cell the text-vs-random_rewire residual is +1.26 [-0.33, 2.22]: not distinguishable from generic rewiring (README: CI>0 in only 4/7 beta>=3 cells shown); null at real beta 0.3. CIs exclude rewiring-draw variance.')
    S('E21', 'T11.2', 'rewiring (correction reach)', 'Does degree-preserving re-homing of misfits raise correction reach (Pokec slice)?', 'Pokec slice', 'simulation-on-real',
      'rehome_text (k5 wc beta 6)', [('correction reach vs none pp', E(setting='pokec_knm k5 model=wc beta=6 sims=100', method='rehome_text', metric='correction_reach_contrast_vs_none_pp'))],
      'random_rewire (same cell)', [('random_rewire vs none pp', E(setting='pokec_knm k5 model=wc beta=6 sims=100', method='random_rewire', metric='correction_reach_contrast_vs_none_pp'))],
      'correction reach (pp) vs no rewiring', 'supported', 'SUPPORTED at beta>=3 (swept, not measured; selective exposure assumed); null at beta 0.3. Total infected share essentially unchanged.')
    S('E22', 'H6', 'deception detection', 'Can misfit detect nodes that hide from community detection by rewiring their own edges?', 'TIMME; Pokec slice', 'simulation-on-real',
      'misfit (kumari k=2, TIMME, all targets)', [('misfit AUC', E(dataset='TIMME', method='misfit', metric='auc', setting='attack=kumari k2 targets=200/seed seeds=3 negatives=all'))],
      'participation (same cell)', [('participation AUC', E(dataset='TIMME', method='participation', metric='auc', setting='attack=kumari k2 targets=200/seed seeds=3 negatives=all'))],
      'ROC-AUC targets vs non-targets', 'not supported', 'README verdict: no; misfit is a weak, mostly non-specific signal; the succ-only result is circular because the matched random control gives the same AUC. Mean+-sd over 3 seeds only.')


# ================================================================= glossary
GLOSS = {
 # metrics
 'modularity_Q': ('metric', 'Fraction of edges inside communities minus that expected at random; higher means stronger community structure.'),
 'n_communities': ('metric', 'Number of communities found.'), 'n_clusters': ('metric', 'Number of clusters found.'),
 'largest_community_share_pct': ('metric', 'Percent of nodes in the biggest community.'), 'largest_cluster_share': ('metric', 'Share of nodes in the biggest cluster.'),
 'cross_community_edge_share_pct': ('metric', 'Percent of edges whose endpoints sit in different communities.'),
 'nmi': ('metric', 'Normalised mutual information between a clustering and reference labels.'), 'nmi_party': ('metric', 'NMI between clusters and party (R/D) labels.'),
 'ari': ('metric', 'Adjusted Rand index between a clustering and reference labels, corrected for chance.'), 'ari_party': ('metric', 'ARI between clusters and party labels.'),
 'nmi_T_vs_C': ('metric', 'NMI between the topology partition and the text partition; low means the views differ.'), 'ari_T_vs_C': ('metric', 'ARI between the topology and text partitions.'),
 'degree_eta2': ('metric', 'Share of log-degree variance explained by cluster membership; high means clusters just sort by degree.'),
 'seed_stability_ari': ('metric', 'Agreement (ARI) of clusterings across random seeds.'),
 'emb_std': ('metric', 'Mean per-dimension standard deviation of an embedding (collapse check).'), 'best_epoch': ('metric', 'Training epoch selected by modularity.'),
 'effective_rank': ('metric', 'Effective number of independent embedding dimensions.'), 'corr_norm_logdeg': ('metric', 'Correlation between embedding norm and log degree.'),
 'dim_std_mean': ('metric', 'Mean per-dimension standard deviation of the embedding.'), 'conductance_mean': ('metric', 'Mean fraction of cluster edges leaving the cluster (1 = no structure).'),
 'edge_reconstruction_auc': ('metric', 'How well embedding dot products separate real from fake edges.'), 'probe_acc': ('metric', 'Accuracy of a linear probe predicting party (or a label) from the representation.'),
 'probe_f1_macro': ('metric', 'Macro-averaged F1 of the linear probe.'), 'knn10_label_agreement': ('metric', 'Label purity of each node 10 nearest neighbours.'),
 'silhouette_true_labels': ('metric', 'Cluster separation measured under the true labels.'), 'abs_corr_split_vs_logdegree': ('metric', 'Absolute correlation of the k=2 split with log degree.'),
 'text_to_community_acc': ('metric', 'Accuracy of a text-only classifier at predicting a node graph community (vs majority-class baseline).'),
 'party_probe_acc': ('metric', 'Probe accuracy predicting party from raw text features.'), 'modularity_Q_social_graph': ('metric', 'Modularity of a text-derived partition measured on the social graph.'),
 'T_ari_vs_planted': ('metric', 'ARI between the recovered topology partition and the planted one.'),
 'auc': ('metric', 'Probability a random positive node outranks a random negative; 0.5 is chance.'), 'auc_overall': ('metric', 'AUC over all nodes.'),
 'auc_within_degree_terciles': ('metric', 'AUC inside degree terciles so degree in disguise cannot win.'), 'auc_within_degree': ('metric', 'Within-degree AUC.'),
 'p_one_sided': ('metric', 'One-sided p-value (permutation or test) for the effect being positive.'),
 'infected_share': ('metric', 'Fraction of nodes reached by a simulated cascade.'), 'cross_community_infected_share': ('metric', 'Infected nodes outside every seed community, as a fraction of all nodes.'),
 'infected_removed_pct': ('metric', 'Percent reduction of infected share vs no blocking (derived from stored means).'), 'cross_community_infected_removed_pct': ('metric', 'Percent reduction of cross-community infected share vs no blocking (derived).'),
 'detect_auc': ('metric', 'AUC of misfit at detecting planted mismatches in the regime map.'), 'cross_spread_removed_pts': ('metric', 'Points of cross-community spread removed by the best structural baseline.'),
 'gain_vs_best_structural_pts': ('metric', 'Percentage points removed by a method minus the best structural baseline, paired over common random numbers.'),
 'median_acceptance_own_community': ('metric', 'Median GloVe acceptance of own community across TIMME nodes.'), 'effective_agreement_beta': ('metric', 'Agreement-gating strength implied by TIMME acceptance.'),
 'median_degree_percentile': ('metric', 'Median degree percentile of a node set (0.5 = typical).'),
 'diff_vs_bridge_struct_pts': ('metric', 'Percentage-point difference in cross-community infection removed vs the structural bridge score.'),
 'cross_infection_removed_pct': ('metric', 'Percent of cross-community infection removed by cutting links.'), 'cross_community_infected_share ': ('metric', ''),
 'spearman': ('metric', 'Rank correlation.'), 'argmax_agreement': ('metric', 'Fraction of nodes where two classifiers pick the same top community.'),
 'recruiter_detection_auc': ('metric', 'AUC at ranking planted recruiters above other accounts.'), 'base_harm_share': ('metric', 'Harm share (mainstream nodes reached) with no blocking.'),
 'harm_removed_pts': ('metric', 'Percentage points of harm removed by blocking.'),
 'n_cells': ('metric', 'Number of (setting, budget) cells counted.'), 'n_cells_ci_above_0': ('metric', 'Cells whose 95 percent CI lies entirely above 0 (method beats baseline).'),
 'n_cells_ci_below_0': ('metric', 'Cells whose 95 percent CI lies entirely below 0 (method loses).'),
 'n_configs': ('metric', 'Number of filter configurations counted.'), 'n_configs_ci_above_0': ('metric', 'Configurations with 95 percent CI above 0.'), 'n_configs_ci_below_0': ('metric', 'Configurations with 95 percent CI below 0.'),
 'best_config_gain_pts': ('metric', 'Gain of the best configuration (selection-biased).'), 'gain_vs_struct_pts': ('metric', 'Link-level gain vs the structural bridge score, in points.'),
 'removed_pts': ('metric', 'Percentage points of spread removed.'), 'wins_out_of_seeds': ('metric', 'Number of seeds where the best misfit-based strategy beat the best structural one.'),
 'precision_at_k': ('metric', 'Share of the top-k ranked items that are true positives.'), 'auc_gain_vs_best_baseline': ('metric', 'Misfit AUC minus the best baseline AUC, paired over seeds.'),
 'is_best_baseline': ('metric', 'Marker: this baseline had the highest mean AUC in the config.'),
 'diff_vs_matched_controls': ('metric', 'Difference vs degree-matched controls (with permutation CI).'), 'control_mean': ('metric', 'Mean outcome of matched controls.'),
 'agreement_beta': ('metric', 'Log-odds coefficient of text agreement in a case-control logit of observed interactions.'), 'agreement_beta_same_T': ('metric', 'Same coefficient restricted to same-topology-community pairs.'),
 'median_diff_vs_matched_controls': ('metric', 'Median across robustness cells of the difference vs matched controls.'),
 'f1': ('metric', 'F1 score.'), 'delta_auc_paired': ('metric', 'Change in AUC from adding a feature, paired bootstrap.'), 'correlation': ('metric', 'Pearson correlation.'),
 'max_cell_gain_pts': ('metric', 'Largest per-cell gain among the best-selected strategies, in points.'), 'mean_gain_vs_best_structural_pts': ('metric', 'Mean gain over cells at a given beta, in points.'),
 'n_cells_holm_surviving_wins': ('metric', 'Cells still significant after Holm multiple-comparison correction.'),
 'cross_ties_in_target_share': ('metric', 'Share of held-out cross-community ties landing in the target community.'), 'ties_staying_in_own_community_share': ('metric', 'Share of held-out ties staying in the node own follow community.'),
 'n_misfits': ('metric', 'Number of top-10 percent misfit nodes analysed.'), 'community_top1_hit': ('metric', 'Share of held-out ties whose community equals the predicted target community.'),
 'share_of_heldout_ties': ('metric', 'Share of held-out ties in a given community.'), 'P@10': ('metric', 'Share of top-10 recommended accounts that are held-out true ties.'),
 'R@100': ('metric', 'Recall of held-out ties within the top-100 recommendations.'), 'paired_diff': ('metric', 'Paired per-user difference between two recommenders.'),
 'misfit_minus_control_diff': ('metric', 'Difference between the misfit-group diff and the control-group diff (misfit-specific gain).'),
 'n_settings_ci_above_0': ('metric', 'Settings (of 27) with paired CI above 0.'), 'n_settings_ci_below_0': ('metric', 'Settings with paired CI below 0.'), 'n_settings_ci_spanning_0': ('metric', 'Settings whose CI spans 0.'),
 'infected': ('metric', 'Infected share in a rewiring simulation.'), 'misfits_reached': ('metric', 'Share of treated misfits reached by the cascade.'), 'correction_reach': ('metric', 'Share of misfits sharing a random misfit text-community that a cascade seeded there reaches.'),
 'n_successful_targets_per_seed': ('metric', 'Mean number of successfully hidden targets per seed.'),
 'kumari attack': ('term', 'Move target toward neighbouring community by rewiring (simplified Kumari 2021).'), 'safeness attack': ('term', 'Drop random own-community edge, add edge to a random other-community node.'), 'random attack': ('control', 'Matched random rewiring control.'), 'noise attack': ('control', 'Random edge swap, the Leiden-instability floor.'),
 'persistence_low': ('baseline', 'One minus Kumari community persistence; high means easy to move.'),
 'deception_success_rate': ('metric', 'Fraction of attacked targets whose community label changed.'),
 # terms
 'misfit': ('method', 'One minus the text-predicted probability of the node own graph community; high means the text disagrees with the placement (ours).'),
 'disagreement': ('method', 'Jensen-Shannon divergence between structural and text community membership.'),
 'misfit_x_pr': ('method', 'Misfit multiplied by PageRank (influential and ambiguous).'), 'disagreement_x_pr': ('method', 'Disagreement multiplied by PageRank.'),
 'gateway': ('method', 'Node score of cross-community links weighted by the receiver text membership.'), 'gateway_struct': ('control', 'Gateway without text: isolates what text adds.'),
 'misfit_rerank': ('method', 'PageRank shortlist reordered by misfit.'), 'misfit_nonhub': ('method', 'Misfit restricted to non-hub accounts.'), 'misfit_x_lowdeg': ('method', 'Misfit multiplied by an inverse-degree term.'),
 'consensus_entropy': ('baseline', 'Instability of a node community across Leiden runs.'), 'consensus_entropy_x_pr': ('baseline', 'Consensus entropy multiplied by PageRank.'),
 'participation': ('baseline', 'How spread a node edges are across communities.'), 'participation_x_pr': ('baseline', 'Participation multiplied by PageRank.'), 'topo_participation': ('baseline', 'Participation coefficient used as a detector.'),
 'content_outlier': ('baseline', 'Text far from the own-community text centroid.'), 'content_outlier_x_pr': ('baseline', 'Content outlier multiplied by PageRank.'),
 'ambiguity_T': ('baseline', 'Normalised entropy of the structural membership p_T.'), 'ambiguity_T_x_pr': ('baseline', 'Structural ambiguity multiplied by PageRank.'), 'topo_ambT': ('baseline', 'Structural ambiguity used as a detector.'),
 'degree': ('baseline', 'Node degree.'), 'low_degree': ('baseline', 'Inverse of degree.'), 'pagerank': ('baseline', 'PageRank centrality.'), 'betweenness': ('baseline', 'Betweenness centrality.'),
 'collective_influence': ('baseline', 'Collective influence heuristic summing neighbour degrees in a ball.'), 'adaptive_degree': ('baseline', 'Iteratively remove the highest-degree node and recompute.'),
 'random': ('control', 'Uniformly random nodes or edges.'), 'none': ('control', 'No intervention.'), 'no blocking': ('control', 'No blocking (unblocked cascade).'),
 'text_iso': ('baseline', 'Text isolation score.'), 'knn_dup': ('baseline', 'Near-duplicate text among kNN neighbours.'), 'oracle_recruiters': ('baseline', 'Blocks the true planted recruiters (upper bound).'),
 'bridge_struct': ('baseline', 'Cross-community link score PR_u times PR_v (structure only).'), 'bridge_content': ('method', 'Cross-community link score using content-based receptivity.'),
 'bridge_struct_x_recv': ('method', 'Structural bridge score multiplied by receptivity.'), 'bridge_struct_x_recv_ORACLE': ('baseline', 'Same with oracle (true) receptivity.'), 'bridge_oracle': ('baseline', 'Bridge score with oracle receptivity.'),
 'degree_product': ('baseline', 'Link score: product of endpoint degrees.'), 'pr_product': ('baseline', 'Link score: product of endpoint PageRanks.'),
 'link prod': ('method', 'H2 link rule: product of centrality and receptivity.'), 'link prod_misfit': ('method', 'H2 link rule: product with misfit.'), 'link rankavg': ('method', 'H2 link rule: rank average.'), 'link recv': ('method', 'H2 link rule: receptivity only.'), 'link misfit': ('method', 'H2 link rule: misfit only.'), 'link rand': ('control', 'Random links from the shortlist.'), 'struct': ('baseline', 'Structural link score baseline.'),
 'random-from-shortlist': ('control', 'Random pick from the centrality shortlist.'),
 'centrality-only': ('control', 'Plain PageRank/betweenness/degree top-k with no misfit filter (gain 0 by construction, up to ties).'),
 'median_positive_gain_pts': ('metric', 'Median gain (points) over configs whose CI lies above 0.'), 'max_positive_gain_pts': ('metric', 'Largest gain (points) among configs whose CI lies above 0.'),
 'case-control logit': ('method', 'Logit of observed vs random interactions on text agreement, giving the agreement beta.'),
 'BGRL': ('method', 'Self-supervised GNN embedding with bootstrapped latents, no negatives.'), 'DGI': ('method', 'Self-supervised GNN embedding by local-global infomax.'), 'MVGRL': ('method', 'Self-supervised GNN embedding contrasting adjacency and PPR-diffusion views.'),
 'spectral probe': ('baseline', 'Linear probe on spectral embedding of the graph.'), 'spectral64 probe': ('baseline', 'Standardised probe on a 64-d spectral embedding.'), 'neighbour-vote': ('baseline', 'Predict label by out-of-fold vote of graph neighbours.'),
 'text-only probe': ('baseline', 'Probe on text features only.'), 'log-degree-only probe': ('baseline', 'Probe on log degree only.'), 'majority class': ('control', 'Always predict the most common class.'),
 'real graph': ('term', 'The observed graph.'), 'rewired null': ('control', 'Degree-preserving rewired graph.'),
 'T Leiden (best of runs)': ('method', 'Topology partition: best of 100 Leiden runs on the social graph.'), 'C text kNN Leiden (best of runs)': ('method', 'Text partition: Leiden on a text kNN graph.'),
 'T reg-spectral + k-means': ('baseline', 'Regularised spectral embedding plus k-means on the graph.'), 'C text k-means': ('baseline', 'k-means on text embeddings.'),
 'topology partition T': ('term', 'Recovered topology community partition.'), 'T Leiden vs C text kNN Leiden': ('term', 'Comparison of the two partitions.'),
 'misfit_SHUFFLED_TEXT': ('control', 'Misfit computed after shuffling text across nodes.'), 'disagreement_SHUFFLED_TEXT': ('control', 'Disagreement computed after shuffling text.'),
 'misfit alone': ('method', 'PHEME classifier using only misfit features.'), 'text only': ('baseline', 'PHEME classifier on source text only.'), 'text + structure': ('baseline', 'PHEME classifier on text plus structural features.'), 'text + structure + misfit': ('method', 'PHEME classifier adding misfit features.'),
 'pop': ('baseline', 'Recommend by PageRank.'), 'cn': ('baseline', 'Recommend by common neighbours.'), 'aa': ('baseline', 'Adamic-Adar link score.'), 'struct_comm': ('baseline', 'p_T(u)[t_v] weighting times PageRank.'),
 'kumari': ('baseline', 'Structural extraction community target (Kumari et al. 2021) times PageRank.'), 'text_sim': ('baseline', 'Cosine similarity of bio embeddings.'), 'ours': ('method', 'p_C(u)[t_v] times PageRank recommender.'), 'ours+aa': ('method', 'Rank average of ours and Adamic-Adar.'),
 'aa_x_text': ('method', 'Adamic-Adar multiplied by text community membership p_C.'), 'aa_x_struct': ('baseline', 'Adamic-Adar multiplied by structural membership p_T.'), 'aa+text_sim': ('method', 'Adamic-Adar plus bio similarity.'),
 'rehome_text': ('method', 'Rewire dormant own-community follows toward the text-predicted community (degree-preserving).'), 'rehome_struct': ('baseline', 'Same rewiring toward the structural (Kumari) community.'),
 'random_rewire': ('control', 'Same drop counts, random adds anywhere.'), 'rehome_text_random_nodes': ('control', 'Same procedure on random non-misfit nodes.'),
 'argmax p_C (text target)': ('method', 'Community with highest text-predicted probability.'), 'text target (argmax p_C)': ('method', 'Community with highest text-predicted probability.'), 'structural community (argmax p_T)': ('baseline', 'Community with highest structural membership.'),
 'argmax p_C restricted to non-own communities': ('method', 'Text target with the own community excluded (identical to argmax p_C for misfits).'), 'most-linked non-own community (follow graph)': ('baseline', 'Non-own community holding most of the node follow edges (Kumari-style structural target).'), 'label propagation, 1 step (follow graph)': ('baseline', 'Non-own community with highest sum over follow neighbours of onehot(community)/degree.'), 'argmax p_T excluding own (follow graph)': ('baseline', 'Highest structural-membership probability among non-own communities.'), 'Adamic-Adar community score (follow graph)': ('baseline', 'Non-own community with largest summed Adamic-Adar score to non-neighbours.'), 'argmax p_C minus argmax p_C restricted to non-own communities': ('term', 'Paired contrast, text target minus this predictor.'), 'argmax p_C minus most-linked non-own community (follow graph)': ('term', 'Paired contrast, text target minus this predictor.'), 'argmax p_C minus label propagation, 1 step (follow graph)': ('term', 'Paired contrast, text target minus this predictor.'), 'argmax p_C minus argmax p_T excluding own (follow graph)': ('term', 'Paired contrast, text target minus this predictor.'), 'argmax p_C minus Adamic-Adar community score (follow graph)': ('term', 'Paired contrast, text target minus this predictor.'), 'text_minus_predictor_hit_share': ('metric', 'Hit share of argmax p_C minus that of a graph-only predictor, paired bootstrap over misfit users.'),
 'own follow community': ('baseline', 'Node own follow-graph community.'), 'chance (size-weighted)': ('control', 'Hit rate expected by chance given community sizes.'), 'Kumari extraction target': ('baseline', 'Neighbouring community with most of the node edges.'),
 'matched controls': ('control', 'Low-misfit nodes matched on degree and participation decile.'), 'top-10% misfits': ('method', 'Nodes in the top decile of misfit with text target different from own community.'),
 'text classifier p_C': ('method', 'Cross-fitted text-only classifier giving community membership.'), 'held-out ties in text_target': ('term', ''), 'held-out ties in own': ('term', ''),
 'best misfit-based (picked per cell)': ('method', 'Best-performing misfit-based strategy chosen per cell (optimistic).'), 'best misfit-based (selected per cell)': ('method', 'Best misfit-based strategy selected per cell (optimistic).'), 'misfit_x_pr (fixed)': ('method', 'misfit_x_pr applied unconditionally (no selection).'),
 'best of PageRank/betweenness': ('baseline', 'Better of PageRank and betweenness.'), 'best misfit-based minus best other non-structural': ('method', 'Gain over the best non-structural alternative.'),
 'misfit filters (all rules)': ('method', 'All misfit-based filter rules pooled.'), 'gain_vs_text_toB': ('term', ''),
 'text k-means: MiniLM bio+latest tweet': ('baseline', 'k-means on MiniLM text.'), 'text k-means: GloVe avg of tweets': ('baseline', 'k-means on GloVe text.'), 'text k-means: both concatenated': ('baseline', 'k-means on concatenated text features.'),
 'TIMME placement': ('term', 'Position of TIMME on the regime map.'), 'synthetic placement': ('term', 'Planted-node placement in synthetic graphs.'),
 'age+gender+completion': ('baseline', 'Pokec profile fields as predictors.'), 'interest text': ('method', 'Pokec interest text as predictor.'), 'text+profile': ('method', 'Text and profile fields combined.'),
 'receptivity proxy': ('term', 'p_C as a proxy for GloVe affinity.'), 'no blocking ': ('control', ''),
 'GloVe-only': ('term', ''),
 'kumari ': ('baseline', ''),
}
# kumari attack / deception names (E22) added by the glossary builder only if present


def pattern_gloss(nm):
    # ponytail: pattern-based one-liners for families of generated names
    if nm.startswith(('C text k-means', 'T reg-spectral')):
        return ('baseline', 'Baseline partition: ' + ('k-means on text embeddings' if nm.startswith('C') else 'regularised spectral embedding plus k-means on the graph') + ' (k as named).')
    if nm.startswith('aa_x_text minus'):
        return ('metric', 'Paired difference between aa_x_text and the named recommender on the named measure.')
    if nm.startswith('adding '):
        return ('method', 'Feature-set change whose paired AUC delta is reported.')
    if nm == 'btw':
        return ('baseline', 'Betweenness centrality.')
    if nm.endswith(('_pp', '_pts')) and nm.split('_')[0] in ('correction', 'infected', 'misfits'):
        return ('metric', 'Rewiring-simulation outcome or contrast (percentage points) for the named intervention comparison.')
    if nm.startswith('held-out ties in'):
        return ('metric', 'Share of held-out ties in the text-target or own community.')
    if nm.startswith('link '):
        return ('method', 'H2 link-level rule: ' + nm[5:] + ' (shortlist size|rule; bridge_content = content-based bridge score).')
    if nm.startswith('misfit filter ('):
        return ('method', 'H2 node filter: shortlist by centrality then choose by the named rule.')
    if nm.startswith('p_C text classifier'):
        return ('method', 'Cross-fitted text-only classifier predicting graph community (text source as named).')
    if nm in ('log_size_baseline', 'mis_frac', 'mis_max', 'mis_mean', 'out_frac', 'out_max', 'out_mean', 'n_defined'):
        return ('method', 'PHEME univariate thread feature (misfit or outlier aggregate, thread-size baseline or defined-count).')
    if nm.startswith('misfit~'):
        return ('metric', 'Spearman partner for misfit: ' + nm[7:] + '.')
    if nm in ('edges_into_B', 'misfit_toB', 'misfit_toB_x_edges', 'text_toB', 'text_toB_x_edges'):
        return ('baseline', 'H3 delivery-scenario strategy targeting community B (edges into B, or misfit/text toward B).')
    if nm in ('gain_vs_best_text', 'gain_vs_text_toB'):
        return ('metric', 'Delivery-scenario gain vs the named text-based competitor (points).')
    if nm == 'misfit scores':
        return ('method', 'Misfit score used for detection AUC in the regime map.')
    return None


def build_glossary():
    names = {}
    for r in ROWS:
        names[r['method']] = 1
        names[r['metric']] = 1
    out = []
    missing = []
    for nm in sorted(names):
        if nm in GLOSS and GLOSS[nm][1]:
            out.append(dict(name=nm, kind=GLOSS[nm][0], one_line=GLOSS[nm][1]))
        elif pattern_gloss(nm):
            k, t = pattern_gloss(nm); out.append(dict(name=nm, kind=k, one_line=t))
        else:
            missing.append(nm)
    return out, missing


# ================================================================= main
def main():
    for fn in EXPERIMENTS:
        fn()
    seen = set()
    for r in ROWS:
        assert r['value'] != '' and r['value'] is not None, r
        k = tuple(r[c] for c in COLS)
        assert k not in seen, ('duplicate row', k)
        seen.add(k)
        # same (exp, setting, method, metric, dataset) must be unique too
    keys = collections.Counter((r['experiment_id'], r['dataset'], r['setting'], r['method'], r['metric'], r['source_file'], r['source_key']) for r in ROWS)
    dups = [k for k, c in keys.items() if c > 1]
    assert not dups, dups[:5]
    with open(os.path.join(OUT, 'results_long.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS)
        w.writeheader()
        w.writerows(ROWS)
    build_summary()
    # check: every number in our_result / competitor_result is a value present for that experiment
    for exp, tag, r in USED:
        assert any(q is r for q in ROWS)
    SC = ['experiment_id', 'phase', 'use_case', 'question', 'dataset', 'data_type', 'our_method', 'our_result', 'best_competitor',
          'competitor_result', 'headline_metric', 'verdict', 'caveat']
    allowed = {'supported', 'partly supported', 'mixed', 'not supported', 'failed', 'descriptive'}
    for s in SUMMARY:
        assert s['verdict'] in allowed, s
    with open(os.path.join(OUT, 'results_summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, SC)
        w.writeheader()
        w.writerows(SUMMARY)
    gl, missing = build_glossary()
    with open(os.path.join(OUT, 'metrics_glossary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['name', 'kind', 'one_line'])
        w.writeheader()
        w.writerows(gl)
    cnt = collections.Counter(r['experiment_id'] for r in ROWS)
    print('long rows', len(ROWS), dict(sorted(cnt.items())))
    print('summary rows', len(SUMMARY), 'glossary rows', len(gl))
    if missing:
        print('GLOSSARY MISSING:', missing)


if __name__ == '__main__':
    main()
