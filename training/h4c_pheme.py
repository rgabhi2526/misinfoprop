"""H4c: does a per-account text-vs-neighbourhood misfit score help rumour detection on PHEME?

Task: rumour vs non-rumour thread (PHEME all-rnr-annotated-threads, 5-9 events).
Users = authors of source tweets and replies. Graph = reply edges replier -> parent author
(+ co-thread edges are NOT used; they would be a thread-id proxy). Text = profile
description + the user's tweets seen in the data (TF-IDF + SVD).
Misfit(u) = 1 - p_C[t_u], t_u = Leiden community of u in the reply graph, p_C = cross-fitted
text-only classifier (same recipe as training/ambiguity.py, reimplemented, CPU only).

Honest protocol: leave-event-out CV is the main result; stratified 5-fold also reported.
Misfit uses no rumour labels (unsupervised communities + cross-fitted text classifier).

Run: OMP_NUM_THREADS=1 /opt/anaconda3/bin/python -I h4c_pheme.py DATA_DIR OUT_DIR
"""
import glob, json, os, sys, re, time
from datetime import datetime
import numpy as np
import igraph as ig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

def ts(s): return datetime.strptime(s, "%a %b %d %H:%M:%S +0000 %Y").timestamp()

# ---------------------------------------------------------------- load
threads = []  # dicts
users = {}    # uid -> dict(desc, texts[], followers, verified, ...)
def add_user(u, text):
    d = users.setdefault(u["id"], dict(desc=u.get("description") or "", texts=[],
        fol=u.get("followers_count", 0), fr=u.get("friends_count", 0), ver=int(bool(u.get("verified"))),
        st=u.get("statuses_count", 0), created=ts(u["created_at"])))
    d["texts"].append(text)

def depth(struct):
    best = 0
    def rec(node, d):
        nonlocal best
        best = max(best, d)
        if isinstance(node, dict):
            for k, v in node.items(): rec(v, d + 1)
    rec(struct, 0); return best

for ev in sorted(glob.glob(os.path.join(DATA, "*-all-rnr-threads"))):
    evname = os.path.basename(ev).replace("-all-rnr-threads", "")
    for lab, sub in ((1, "rumours"), (0, "non-rumours")):
        for td in glob.glob(os.path.join(ev, sub, "*/")):
            try:
                src = json.load(open(glob.glob(td + "source-tweets/*.json")[0]))
                st = json.load(open(td + "structure.json"))
            except Exception:
                continue
            tid = src["id_str"]
            add_user(src["user"], src["text"])
            t_src = ts(src["created_at"])
            rx = []
            for f in glob.glob(td + "reactions/*.json"):
                try: r = json.load(open(f))
                except Exception: continue
                add_user(r["user"], r["text"])
                rx.append((r["user"]["id"], r.get("in_reply_to_user_id"), ts(r["created_at"]) - t_src, r["text"]))
            threads.append(dict(ev=evname, y=lab, id=tid, src=src["text"], su=src["user"]["id"],
                                sfol=src["user"]["followers_count"], sver=int(bool(src["user"]["verified"])),
                                sage=(t_src - ts(src["user"]["created_at"])) / 86400.0,
                                sstat=src["user"]["statuses_count"], rx=rx, depth=depth(st), n_media=len(src.get("entities", {}).get("urls", []))))
log("threads", len(threads), "users", len(users), "rumour frac", np.mean([t["y"] for t in threads]))
for e in sorted({t["ev"] for t in threads}):
    s = [t["y"] for t in threads if t["ev"] == e]; log("  event", e, len(s), "rumour frac %.2f" % np.mean(s))

# ---------------------------------------------------------------- user graph
uids = sorted(users); uix = {u: i for i, u in enumerate(uids)}
edges = set()
for t in threads:
    for (u, pu, dt, _) in t["rx"]:
        if pu and pu in uix and u in uix and pu != u:
            edges.add((uix[u], uix[pu]))
# replies to non-source tweets have parent author = in_reply_to_user_id, mostly in data
g = ig.Graph(n=len(uids), edges=list(edges)); g.simplify()
comp = g.connected_components(); log("user graph", g.vcount(), g.ecount(), "components", len(comp),
    "largest", max(comp.sizes()), "users with deg>0", int(np.sum(np.array(g.degree()) > 0)))
# Leiden communities, resolution chosen so communities are not ~ threads (threads are components anyway)
best = None
labels = np.zeros(g.vcount(), int)
lcc = np.array(comp.membership) == int(np.argmax(comp.sizes()))
# communities on the whole graph: components are small stars, so Leiden = component-ish.
part = g.community_leiden(objective_function="modularity", resolution=1.0, n_iterations=5)
labels = np.array(part.membership); log("leiden communities", labels.max() + 1, "sizes>=5:", int(np.sum(np.bincount(labels) >= 5)))

# ---------------------------------------------------------------- user text -> SVD
utext = [(users[u]["desc"] + " " + " ".join(users[u]["texts"][:20])).lower() for u in uids]
utext = [re.sub(r"http\S+|@\w+", " ", s) for s in utext]
tf = TfidfVectorizer(min_df=3, max_features=50000, sublinear_tf=True, stop_words="english")
X = tf.fit_transform(utext)
Z = TruncatedSVD(64, random_state=0).fit_transform(X)
Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
log("user text SVD", Z.shape)

# ---------------------------------------------------------------- misfit (cross-fitted by event)
# a user's event = event of the first thread they appear in
uev = {}
for t in threads:
    for u in [t["su"]] + [r[0] for r in t["rx"]]:
        uev.setdefault(u, t["ev"])
uev_arr = np.array([uev[u] for u in uids])
# restrict classifier to communities with >= 8 members, others get class -1 (misfit undefined -> NaN)
cnt = np.bincount(labels)
ok = cnt[labels] >= 8
cls = np.unique(labels[ok]); cmap = {c: i for i, c in enumerate(cls)}
y_u = np.array([cmap[c] if c in cmap else -1 for c in labels])
log("classes", len(cls), "users in classes", int((y_u >= 0).sum()))
pC = np.full((len(uids), len(cls)), np.nan)
from sklearn.model_selection import KFold
# communities are within-event (reply graph components never cross events), so cross-fit by
# random user folds; rumour labels are never used here, so no label leakage.
for tr, te in KFold(5, shuffle=True, random_state=0).split(Z):
    tr = tr[y_u[tr] >= 0]
    # fast cross-fitted classifier: cosine-to-train-centroid softmax (T=0.1); LR with 179 classes was too slow on CPU
    C_ = np.zeros((len(cls), Z.shape[1])); np.add.at(C_, y_u[tr], Z[tr])
    C_ /= np.linalg.norm(C_, axis=1, keepdims=True) + 1e-9
    S = Z[te] @ C_.T / 0.1; S -= S.max(1, keepdims=True); P = np.exp(S); P /= P.sum(1, keepdims=True)
    pC[te] = P
misfit = np.full(len(uids), np.nan)
m = y_u >= 0
misfit[m] = 1 - pC[np.where(m)[0], y_u[m]]
# fallback / second score: text outlier = 1 - cosine to the centroid of own community
cent = np.zeros((len(cls), Z.shape[1]))
for c in range(len(cls)): cent[c] = Z[y_u == c].mean(0)
cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
outl = np.full(len(uids), np.nan); outl[m] = 1 - np.sum(Z[m] * cent[y_u[m]], 1)
log("misfit defined for", int(m.sum()), "users; mean %.3f" % np.nanmean(misfit))

# ---------------------------------------------------------------- cascade features
def agg(v):
    v = np.array([x for x in v if not np.isnan(x)])
    if len(v) == 0: return [0.5, 0.5, 0.0, 0.0]
    return [v.mean(), v.max(), (v > 0.8).mean(), float(len(v))]

rows, yv, grp, srctxt = [], [], [], []
mis_rows = []
for t in threads:
    rx = t["rx"]; n = len(rx)
    dts = np.array([r[2] for r in rx]) / 60.0 if n else np.array([0.0])
    fol = [np.log1p(users[r[0]]["fol"]) for r in rx] or [0.0]
    ages = [users[r[0]]["fol"] for r in rx]
    base = [np.log1p(n), t["depth"], np.log1p(np.median(dts)), np.log1p(dts.max()),
            np.log1p(t["sfol"]), t["sver"], np.log1p(t["sage"]), np.log1p(t["sstat"]), np.mean(fol),
            len({r[0] for r in rx}) / max(n, 1), t["n_media"], len(t["src"])]
    rows.append(base)
    sh = [t["su"]] + [r[0] for r in rx]
    mis_rows.append(agg([misfit[uix[u]] for u in sh]) + agg([outl[uix[u]] for u in sh])[:3])
    yv.append(t["y"]); grp.append(t["ev"]); srctxt.append(t["src"])
B = np.array(rows, float); M = np.array(mis_rows, float); y = np.array(yv); grp = np.array(grp)
log("cascade feature matrices", B.shape, M.shape)

# ---------------------------------------------------------------- evaluation
def model():
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_depth=3, random_state=0)

def cv_pred(fn_feats, splits):
    p = np.zeros(len(y))
    for tr, te in splits:
        Xtr, Xte = fn_feats(tr, te)
        p[te] = model().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    return p

_cache = {}
def src_svd(tr, te):  # source-text TF-IDF fit on train fold only (cached per fold)
    key = (len(tr), int(tr[:5].sum()), int(te[:5].sum()), len(te))
    if key in _cache: return _cache[key]
    _cache[key] = _src_svd(tr, te); return _cache[key]
def _src_svd(tr, te):
    v = TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2), max_features=30000)
    A = v.fit_transform([srctxt[i] for i in tr]); s = TruncatedSVD(40, random_state=0).fit(A)
    return s.transform(A), s.transform(v.transform([srctxt[i] for i in te]))

def feats(kind):
    def f(tr, te):
        a, b = src_svd(tr, te)
        parts_tr, parts_te = [a], [b]
        if kind in ("struct", "struct+misfit"):
            parts_tr.append(B[tr]); parts_te.append(B[te])
        if kind in ("struct+misfit", "text+misfit", "misfit"):
            parts_tr.append(M[tr]); parts_te.append(M[te])
        if kind == "misfit":
            return M[tr], M[te]
        return np.hstack(parts_tr), np.hstack(parts_te)
    return f

kinds = ["text", "struct", "struct+misfit", "misfit"]
sp_strat = list(StratifiedKFold(5, shuffle=True, random_state=0).split(B, y))
sp_event = list(GroupKFold(len(set(grp))).split(B, y, grp))
res = {}
B_idx = np.arange(len(y))
def boot(p_a, p_b, idx_list=None, n=1000):
    r = np.random.default_rng(1); d = []; da = []; fa = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(set(y[i])) < 2: continue
        a, b = roc_auc_score(y[i], p_a[i]), roc_auc_score(y[i], p_b[i]); d.append(b - a)
    return float(np.mean(d)), [float(x) for x in np.percentile(d, [2.5, 97.5])]
def ci(p):
    r = np.random.default_rng(2); a = []
    for _ in range(1000):
        i = r.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: a.append(roc_auc_score(y[i], p[i]))
    return float(roc_auc_score(y, p)), [float(x) for x in np.percentile(a, [2.5, 97.5])]
preds = {}
for sname, sp in (("stratified5", sp_strat), ("leave_event_out", sp_event)):
    for k in kinds:
        p = cv_pred(feats(k), sp); preds[(sname, k)] = p
        auc, c = ci(p); f1 = f1_score(y, p > 0.5)
        res[f"{sname}/{k}"] = dict(auc=auc, auc_ci=c, f1=float(f1))
        log(sname, k, "AUC %.3f [%.3f,%.3f] F1 %.3f" % (auc, c[0], c[1], f1))
    for a, b in (("text", "struct"), ("struct", "struct+misfit"), ("text", "text+misfit")):
        if b == "text+misfit": continue
        d, c = boot(preds[(sname, a)], preds[(sname, b)])
        res[f"{sname}/delta {a}->{b}"] = dict(mean=d, ci=c)
        log(sname, f"delta AUC {a}->{b}: {d:+.4f} [{c[0]:+.4f},{c[1]:+.4f}]")

# univariate: does misfit alone separate classes, and is it just size?
uni = {}
names = ["mis_mean", "mis_max", "mis_frac", "n_defined", "out_mean", "out_max", "out_frac"]
for j, nme in enumerate(names):
    uni[nme] = float(roc_auc_score(y, M[:, j]))
uni["log_size_baseline"] = float(roc_auc_score(y, B[:, 0]))
corr = float(np.corrcoef(M[:, 3], B[:, 0])[0, 1])
log("univariate AUC", uni, "corr(n_defined, log size) %.2f" % corr)
res["univariate_auc"] = uni; res["corr_ndefined_logsize"] = corr
res["meta"] = dict(threads=len(threads), users=len(users), graph_edges=g.ecount(), communities=int(len(cls)),
                   users_with_misfit=int(m.sum()), events=sorted(set(grp.tolist())))
json.dump(res, open(os.path.join(OUT, "h4c_pheme_results.json"), "w"), indent=1)
log("done")
