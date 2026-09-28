"""Pokec (SNAP soc-Pokec): Slovak social network, 1.63M users, 30.6M directed
friendships (54% reciprocated).

Features (decision T0, 2026-09-27; metadata used as features only):
  * interest free text (hobbies, music, movies, books, sports, ...; Slovak),
    concatenated "field: value; ..." and encoded with common.ENCODER +
    has_metadata flag (61% of users fill at least one interest field),
  * gender, age (0 = unknown -> age_known flag), region one-hot (187 regions).

No labels (R6). Physical / lifestyle fields (eyes, smoking, zodiac, ...) are not used.
--region restricts to users whose region starts with the given text (e.g.
"zilinsky kraj") and their induced friendships — a small slice for fast iteration.
Give a slice its own --tag so downstream scripts find it:
    python train_pokec.py --region "zilinsky kraj, kysucke nove mesto" --tag pokec_knm
    -> out/pokec_knm/pokec_knm_<model>_emb.pt
Text encodings are cached as pokec_text_<region|all>.npy next to the data.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

import common
import gclib

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "../data")
COLS = ("user_id public completion_percentage gender region last_login registration "
        "AGE body I_am_working_in_field spoken_languages hobbies I_most_enjoy_good_food "
        "pets body_type my_eyesight eye_color hair_color hair_type "
        "completed_level_of_education favourite_color relation_to_smoking "
        "relation_to_alcohol sign_in_zodiac on_pokec_i_am_looking_for love_is_for_me "
        "relation_to_casual_sex my_partner_should_be marital_status children "
        "relation_to_children I_like_movies I_like_watching_movie I_like_music "
        "I_mostly_like_listening_to_music the_idea_of_good_evening "
        "I_like_specialties_from_kitchen fun I_am_going_to_concerts my_active_sports "
        "my_passive_sports profession I_like_books life_style music cars politics "
        "relationships art_culture hobbies_interests science_technology "
        "computers_internet education sport movies travelling health companies_brands "
        "more").split()
# most-filled first: the encoder truncates at 128 tokens
INTEREST = ["hobbies", "I_like_music", "I_like_movies", "I_like_books", "my_active_sports",
            "my_passive_sports", "I_mostly_like_listening_to_music", "I_like_watching_movie",
            "the_idea_of_good_evening", "I_am_going_to_concerts", "life_style", "fun",
            "music", "hobbies_interests", "sport", "movies", "politics", "art_culture",
            "science_technology", "computers_internet", "cars", "travelling", "health"]


def load_data(data_dir=DEFAULT_DIR, limit=None, region=None):
    common.log(f"pokec: reading profiles ({region or 'all regions'})")
    p = pd.read_csv(os.path.join(data_dir, "soc-pokec-profiles.txt"), sep="\t",
                    header=None, names=COLS + ["_trailing"], usecols=COLS[:8] + INTEREST,
                    dtype=str, na_values=["null"], quoting=3)
    if region:
        p = p[p["region"].str.startswith(region, na=False)]
    p = p.reset_index(drop=True)
    uid2idx = pd.Series(np.arange(len(p)), index=p["user_id"].astype(np.int64).to_numpy())
    n = len(p)
    common.log(f"pokec: {n} users; reading friendships (30.6M rows for full Pokec)")

    e = pd.read_csv(os.path.join(data_dir, "soc-pokec-relationships.txt"), sep="\t",
                    header=None, dtype=np.int64, nrows=limit)
    a, b = e[0].map(uid2idx), e[1].map(uid2idx)
    ok = a.notna() & b.notna()                     # induced subgraph when --region is set
    edge_index = torch.from_numpy(np.stack([a[ok].to_numpy(np.int64), b[ok].to_numpy(np.int64)]))
    common.log(f"pokec: {int(ok.sum())} friendships kept; building interest texts")

    texts = ["; ".join(f"{c}: {v}" for c, v in zip(INTEREST, row) if isinstance(v, str))
             for row in p[INTEREST].itertuples(index=False)]
    tag = (region or "all").replace(" ", "_").replace(",", "")
    x_text = common.encode_texts(texts, cache=os.path.join(data_dir, f"pokec_text_{tag}.npy"))

    age = pd.to_numeric(p["AGE"], errors="coerce").fillna(0).to_numpy()
    regions = pd.get_dummies(p["region"].fillna("unknown")).to_numpy(np.float32)
    x = torch.cat([x_text, torch.from_numpy(np.column_stack([
        pd.to_numeric(p["gender"], errors="coerce").fillna(0.5).to_numpy(),
        age / 100.0, (age > 0).astype(float),        # age 0 = unknown (30% of users)
        regions]).astype(np.float32))], dim=1)

    data = common.make_data(edge_index, x, num_nodes=n)
    return data, None


def main():
    p = common.base_args("pokec", DEFAULT_DIR, clusters=10)
    p.add_argument("--region", default=None, help='e.g. "zilinsky kraj" (default: all)')
    p.add_argument("--tag", default="pokec", help="out/<tag>/<tag>_<model>_emb.pt, e.g. pokec_knm")
    p.set_defaults(out_dir=None)                   # default: out/<tag>/ next to this file
    a = p.parse_args()
    data, labels = load_data(a.data_dir, a.limit, a.region)
    print(f"pokec[{a.region or 'all'}]: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, "
          f"{data.num_features} feat dims")
    cfg = common.cfg_from_args(a)
    results = gclib.run_all(data, cfg, labels=labels, models=a.models)
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", a.tag)
    gclib.save(results, out_dir, a.tag)


if __name__ == "__main__":
    main()
