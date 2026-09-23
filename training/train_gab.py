"""Gab (Group B): user interaction graph + text-embedding ideology features.

The raw dump is 6GB JSONL (one post per line), no ready graph or features. We
derive both in a single streaming pass:

  * Edge: on a repost / reply / quote, actuser -> post.user (the original
    author). Original posts add no edge.
  * Feature: concatenate each user's own post bodies, embed with a
    sentence-transformer (all-MiniLM-L6-v2, 384-dim) as the ideology signal.

No ground-truth labels (NMI skipped). --limit caps lines read for a quick pass.

ponytail: bodies are truncated per user (MAX_CHARS) so the text encoder sees a
bounded prompt; raise it if the signal looks thin.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import common
import gclib

DEFAULT_DIR = os.path.join(os.path.dirname(__file__),
                           "../data_final/gab/extracted")
JSON_FILE = "gab_posts_jan_2018.json"
MAX_CHARS = 4000          # per-user text cap fed to the encoder
EMB_MODEL = "all-MiniLM-L6-v2"


def _uid(u):
    return str(u.get("id")) if isinstance(u, dict) and u.get("id") is not None else None


def load_data(data_dir=DEFAULT_DIR, limit=None):
    path = os.path.join(data_dir, JSON_FILE)
    src, dst = [], []
    texts = {}                      # uid -> concatenated body text

    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            actor = _uid(rec.get("actuser"))
            post = rec.get("post") or {}
            if actor is None:
                continue
            if post.get("repost") or post.get("is_reply") or post.get("is_quote"):
                tgt = _uid(post.get("user"))
                if tgt and tgt != actor:
                    src.append(actor); dst.append(tgt)
            body = post.get("body")
            if body and len(texts.get(actor, "")) < MAX_CHARS:
                texts[actor] = (texts.get(actor, "") + " " + body)[:MAX_CHARS]

    edge_index, id2idx, idx2id = common.remap_ids(src, dst)
    n = len(id2idx)

    # ---- text-embedding features ----
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device=gclib.Cfg().device)
    corpus = [texts.get(str(u), "") for u in idx2id]
    emb = model.encode(corpus, batch_size=256, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    x = torch.from_numpy(emb.astype(np.float32))

    data = common.make_data(edge_index, x, num_nodes=n)
    return data, None


def main():
    a = common.base_args("gab", DEFAULT_DIR, clusters=10).parse_args()
    data, labels = load_data(a.data_dir, a.limit)
    print(f"gab: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
          f"{data.num_features} feat dims")
    cfg = common.cfg_from_args(a)
    results = gclib.run_all(data, cfg, labels=labels, models=a.models)
    gclib.save(results, a.out_dir, "gab")


if __name__ == "__main__":
    main()
