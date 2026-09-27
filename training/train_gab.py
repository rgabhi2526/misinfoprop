"""Gab (Group B): user interaction graph + text-embedding ideology features.

The raw dump is a 48GB JSON array, one record per line, no ready graph or
features. Record shape (checked against the file): top-level `type` is "post"
or "repost"; `actuser` is who acted; `post.user` is the post's author;
replies/quotes carry the replied-to/quoted post in `post.parent`. Note
`post.repost` is always False — reposts are marked only by `type`.

  * Edge (actor -> author, direction of attention):
      repost          actuser   -> post.user
      reply / quote   post.user -> post.parent.user
  * Feature: each author's own post bodies (type == "post" only, so reposted
    text is not credited to the reposter), encoded with common.ENCODER +
    has_metadata flag (users who only repost/reply get the flag = 0).

The parse + encode runs once and is cached next to the data
(gab_cache_<limit|all>.pt); delete the cache to rebuild.

No ground-truth labels (NMI skipped).
ponytail: --limit reads the first N lines, and the dump is grouped by user
timeline (first 1337 lines = 35 users), so a limited run is a few users' full
histories, not a random subgraph. Fine for smoke tests only.
ponytail: bodies are truncated per user (MAX_CHARS) so the text encoder sees a
bounded prompt; raise it if the signal looks thin.
"""
from __future__ import annotations

import json
import os

import torch

import common
import gclib

DEFAULT_DIR = os.path.join(os.path.dirname(__file__),
                           "../data_final/gab/extracted")
JSON_FILE = "gab_posts_jan_2018.json"
MAX_CHARS = 4000          # per-user text cap fed to the encoder


def _uid(u):
    return str(u.get("id")) if isinstance(u, dict) and u.get("id") is not None else None


def parse(path, limit=None):
    """One streaming pass -> (src uids, dst uids, {uid: text})."""
    src, dst, texts = [], [], {}
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
            post = rec.get("post") or {}
            author = _uid(post.get("user"))
            if author is None:
                continue
            if rec.get("type") == "repost":
                a, b = _uid(rec.get("actuser")), author
            else:
                body = post.get("body")
                if body and len(texts.get(author, "")) < MAX_CHARS:
                    texts[author] = (texts.get(author, "") + " " + body)[:MAX_CHARS]
                a, b = (author, _uid((post.get("parent") or {}).get("user"))) \
                    if post.get("is_reply") or post.get("is_quote") else (None, None)
            if a and b and a != b:
                src.append(a); dst.append(b)
    return src, dst, texts


def load_data(data_dir=DEFAULT_DIR, limit=None):
    cache = os.path.join(data_dir, f"gab_cache_{limit or 'all'}.pt")
    if os.path.exists(cache):
        c = torch.load(cache)
        edge_index, x = c["edge_index"], c["x"]
    else:
        src, dst, texts = parse(os.path.join(data_dir, JSON_FILE), limit)
        edge_index, _, idx2id = common.remap_ids(src, dst)
        x = common.encode_texts([texts.get(str(u), "") for u in idx2id])
        torch.save({"edge_index": edge_index, "x": x, "uids": list(idx2id)}, cache)

    data = common.make_data(edge_index, x, num_nodes=x.size(0))
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
