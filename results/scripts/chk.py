import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from train_pokec import COLS, INTEREST
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
p = pd.read_csv("../data/soc-pokec-profiles.txt", sep="\t", header=None, names=COLS+["_trailing"],
                usecols=INTEREST, dtype=str, na_values=["null"], quoting=3, nrows=300000)
texts = ["; ".join(f"{c}: {v}" for c, v in zip(INTEREST, r) if isinstance(v, str)) for r in p[INTEREST].itertuples(index=False)]
long = [t for t in texts if len(t) > 1000]
print("sample", len(texts), "len>1000:", len(long))
diff = 0; minlen = 10**9
for t in long:
    a = tok(t, truncation=True, max_length=128)["input_ids"]
    b = tok(t[:1000], truncation=True, max_length=128)["input_ids"]
    diff += a != b
    ids = tok(t[:2000], truncation=True, max_length=128, return_offsets_mapping=True)["offset_mapping"]
    minlen = min(minlen, ids[-2][1])
print("changed after cap:", diff, " max chars needed for 128 tokens:", max(tok(t[:3000], truncation=True, max_length=128, return_offsets_mapping=True)["offset_mapping"][-2][1] for t in long))
