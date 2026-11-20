import sys, warnings, os; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import cross_val_score, StratifiedKFold
import views, train_timme
while not os.path.exists(os.path.join(train_timme.DEFAULT_DIR, "timme_views.npz")): pass
b = views.load_timme()
import torch  # orig_idx for the GloVe rows: rebuild from LCC order cached in views? recompute cheaply
data, _ = train_timme.load_data()
glove = np.load(os.path.join(train_timme.DEFAULT_DIR, "tweet_features.npz"))["tweets_average"][data.orig_idx.numpy()]
m = b["y"] >= 0; cv = StratifiedKFold(5, shuffle=True, random_state=0)
probe = lambda z: cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), z[m], b["y"][m], cv=cv).mean()
zero = (np.abs(glove).sum(1) == 0)
print(f"GloVe all-zero rows: {zero.mean():.3f}; MiniLM empty: {(~b['has']).mean():.3f}")
for name, z in [("MiniLM bio+latest tweet", b["text"]), ("GloVe avg of tweets", glove), ("both concatenated", np.hstack([b["text"], normalize(glove)]))]:
    zz = normalize(z)
    runs = views.leiden_runs(views.knn_graph(zz), 5)
    p = views.best_run(runs, views.knn_graph(zz))
    print(views.fmt(f"{name} | probe {probe(z):.3f}", views.score(p, b), views.stability(runs)), flush=True)
