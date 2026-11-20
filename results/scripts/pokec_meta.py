import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, views, ambiguity as am, train_pokec
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
b = views.load_pokec("pokec_knm")
s = np.load("out/pokec_knm_views/ambiguity.npz"); t = s["t"]
# profile fields for the slice, same row order as the loader (region filter, LCC via orig_idx)
data, _ = train_pokec.load_data(region=views.POKEC_SLICES["pokec_knm"])
p = pd.read_csv("../data/soc-pokec-profiles.txt", sep="\t", header=None, names=train_pokec.COLS + ["_t"],
                usecols=["user_id", "gender", "region", "AGE", "completion_percentage", "public"], dtype=str, na_values=["null"], quoting=3)
p = p[p["region"].str.startswith(views.POKEC_SLICES["pokec_knm"], na=False)].reset_index(drop=True).iloc[data.orig_idx.numpy()]
age = pd.to_numeric(p["AGE"], errors="coerce").fillna(0).to_numpy()
X = np.c_[age, age > 0, pd.to_numeric(p["gender"], errors="coerce").fillna(.5), pd.to_numeric(p["completion_percentage"], errors="coerce").fillna(0)]
ok = t >= 0; cv = StratifiedKFold(5, shuffle=True, random_state=0)
for name, Z in [("age+gender+completion", X), ("interest text", b["text"]), ("text+profile", np.c_[b["text"], X])]:
    pred = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), Z[ok], t[ok], cv=cv)
    print(f"{name:24s} acc {np.mean(pred == t[ok]):.3f}  (majority {np.bincount(t[ok]).max()/ok.sum():.3f})")
print("age known:", (age > 0).mean().round(3), " median age by community:", [int(np.median(age[(t==c)&(age>0)])) for c in range(t.max()+1)])
