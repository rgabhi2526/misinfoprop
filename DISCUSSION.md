# Discussion — open disagreements between sessions

Shared channel between sessions working on this project. Anything two sessions (or a
session and a doc) disagree on goes here until resolved, so one source of truth wins.

**Convention**
- One `#` topic per disagreement: date, status (`open` / `resolved`), evidence, how to verify.
- Resolving: apply the fix to the doc it concerns (`TASKS.md`, `SUGGESTIONS.md`,
  `DATASETS.md`, code), then mark the topic `resolved` with date + what changed.
  Keep resolved topics — they record why.
- Read this file at session start; claims here outrank the docs they dispute until resolved.

---

# Topic 1 — `DATASETS.md` vs the actual files · 2026-09-27 · resolved 2026-09-27

Checked against the files in `data_final/`. `data_final/DATASETS.md` has **not** been
corrected yet; where it disagrees with this topic, trust this topic and re-run the
checks below.

## Dataset criteria (as understood — confirm or correct)

Must have:
1. Real-world social network. Nodes = people, or communities (Reddit subreddits accepted).
2. Communities exist — given, or found by clustering. Labels not required.
3. Per-node metadata that (a) feeds the embedding with topology, so it shapes clusters,
   and (b)(optional ,better if possible) explains a cluster in plain words ("supporters of X", "Delhi community","liberal").
   Ideology/party flags **not** required — location, topics, followed accounts,
   interests all qualify.
4. Scale ≥ ~10k nodes.
5. Directed preferred (bridges/diffusion later), unweighted or binarizable;
   symmetrized for the embedding step.

Pipeline needs (`TASKS.md`), not per-dataset:
6. At least one dataset with ground truth, for T6 validation. Today only TIMME.
7. Timestamped cascades are a bonus for T8. Only Gab has them at user level.

## Discrepancies found

✅ matches · ❌ wrong · ~ minor · ➕ missing from the doc

### VoterFraud2020
| Claim in `DATASETS.md` | Reality | |
|---|---|---|
| Authors: Zannettou et al. | Abilov et al. (Cornell Tech), per Figshare API metadata | ❌ |
| ~1.89M nodes, 16.7M edges | 25,566,698 retweet rows; 1,790,745 unique retweeters; 1,214,846 unique retweeted tweets; `users.csv` 2,559,018 users (1,697,944 with a community label) | ❌ |
| Directed user→user retweet graph | `retweeted_id` is a **tweet ID** (99.6% found in `tweets-*.csv`, 0% in `users.csv`). Tweets files have no author column → user→user graph cannot be built. `training/train_voterfraud.py` therefore builds a tweet↔user bipartite graph | ❌ |
| Per-cluster news-domain stats, ~90/10 split | `urls.csv` has per-community aggregates (true); the 90/10 figure is unverified | ~ |
| "Primary, no derivation needed" | Per-user metadata = community ID + active/suspended/deleted status only. No text, bio, location or hashtags → fails criterion 3 | ❌ |

### TIMME (`P_all`)
| Claim | Reality | |
|---|---|---|
| 21,015 nodes | `all_twitter_ids.csv` has 21,014 IDs; TIMME readme and `TASKS.md` say 20,811 after filtering | ~ |
| 586 politicians + ~2,976 labeled users | `dict.csv` = 585 (313 R / 270 D / 2 I). `additional_labels/new_dict_cleaned.csv` = 2,975 rows, but only **633** are in P_all → ~1.2k labels total (matches `TASKS.md`'s 1,206). Doc overstates ~3× | ❌ |
| `P_all/features.npz` broken; tweet features OK | Correct. `tweet_features.npz` = 21,014 × 300 (averaged GloVe) | ✅ |
| (not mentioned) | `formatted_location/formatted_location.csv`: state known for 50.6% of P_all users, city for 37.3%. DC-skewed (DC 2,470, CA 1,048, NY 987, TX 648). Unused by the pipeline — candidate cluster-explanation attribute | ➕ |

### Reddit hyperlinks
| Claim | Reality | |
|---|---|---|
| 55,863 nodes | Hyperlink graph has **67,180** unique subreddits | ❌ |
| 858,490 edges | 286,561 body + 571,927 title = 858,488 | ✅ |
| 86-dim LIWC per edge | 86 properties per edge (LIWC + basic text stats); 9.6% links negative | ✅ |
| (not mentioned) | `web-redditEmbeddings-subreddits.csv`: 51,278 × 300, covers only **46.5%** of graph nodes → usable graph with features ≈ 31k subreddits (why T1.4 dropped 20.5k) | ➕ |

### Gab
| Claim | Reality | |
|---|---|---|
| ~341K users, 22.1M posts | Zenodo: 336,752 users, 22,112,812 posts | ~ |
| Features include follower/following edges | **False** — same entry says no follow graph. Records are `post`/`repost` only; graph built from repost/reply/quote | ❌ |
| Hashtags field | No hashtag field; hashtags only inside `post.body`. Present: `topic`, `category`, `language`, `like_count`, `score`, `is_reply`, `is_quote`, `parent`; user flags `verified`, `is_donor`, `is_pro`, `is_investor` | ~ |
| Source: Zannettou / Mathew et al., ICWSM 2018 | Zannettou et al., **WWW'18 companion** (CySoc workshop). Mathew et al. is a separate crawl | ❌ |

### Doc-level
| Claim | Reality | |
|---|---|---|
| Groups A/B framed around ideology labels | Out of date — criteria no longer require ideology (see criterion 3) | ❌ |

## Verdict vs criteria
- **TIMME** — passes all; only dataset with validation labels. Add location as an explanation attribute.
- **Gab** — passes; clusters explainable via topics/text/reposted accounts; only user-level cascades. No labels.
- **Reddit** — passes (community-level nodes accepted); effectively ~31k nodes with features.
- **VoterFraud2020** — fails criterion 3, and the documented graph cannot be built from the files. Options (S-12): drop, keep as topology-only ablation (co-retweet user graph), or replace. Candidate replacement already on disk: `data/soc-pokec-*` — 1.63M users, 30.6M directed friendship edges, profiles with region, interests, age, languages (Slovak text; not political; no timestamps).

## Open decisions
- [ ] Confirm criteria 1–8 above
- [ ] VoterFraud2020: drop / ablation / replace with Pokec (→ T1.8, S-12)
- [ ] Rewrite `data_final/DATASETS.md` with the corrections above
- [ ] Use TIMME location as an explanation (and/or feature) attribute?

## How to re-verify
Run from the project root (≈1–2 min, VoterFraud load dominates):

```bash
cd "data_final" && /opt/anaconda3/bin/python - <<'EOF'
import pandas as pd, glob, numpy as np, json, ast, itertools
# VoterFraud: is retweeted_id a tweet or a user?
rt = pd.concat(pd.read_csv(f, dtype=str) for f in sorted(glob.glob('voterfraud2020/retweets-*.csv')))
tw = set(pd.concat(pd.read_csv(f, dtype=str, usecols=['tweet_id']) for f in glob.glob('voterfraud2020/tweets-*.csv')).tweet_id)
u = pd.read_csv('voterfraud2020/users.csv', dtype=str)
print('VF rows', len(rt), 'retweeters', rt.user_id.nunique(), 'retweeted', rt.retweeted_id.nunique())
print('VF retweeted_id in tweets', rt.retweeted_id.isin(tw).mean(), 'in users', rt.retweeted_id.isin(set(u.user_id)).mean())
print('VF users', len(u), u.user_community.value_counts(dropna=False).to_dict())
# TIMME labels in P_all
d = 'timme/repo/data/'
ids = set(open(d+'P_all/all_twitter_ids.csv').read().split())
print('TIMME ids', len(ids)-1, 'dict', pd.read_csv(d+'P_all/dict.csv', sep='\t').party.value_counts().to_dict())
a = pd.read_csv(d+'additional_labels/new_dict_cleaned.csv', sep=None, engine='python', dtype=str)
print('TIMME extra labels', len(a), 'in P_all', a.twitter_id.isin(ids).sum())
loc = pd.read_csv(d+'formatted_location/formatted_location.csv', header=None, names=['i','id','City','State','Country'], dtype=str).iloc[1:]
loc = loc[loc.id.isin(ids)]
print('TIMME state known', (loc.State != 'Unknown').mean(), 'city known', (loc.City != 'Unknown').mean())
# Reddit node count and embedding coverage
e = pd.concat(pd.read_csv(f'reddit-hyperlinks/soc-redditHyperlinks-{s}.tsv', sep='\t') for s in ['body', 'title'])
n = set(e.SOURCE_SUBREDDIT) | set(e.TARGET_SUBREDDIT)
emb = set(pd.read_csv('reddit-hyperlinks/web-redditEmbeddings-subreddits.csv', header=None, usecols=[0])[0])
print('Reddit edges', len(e), 'nodes', len(n), 'emb coverage', len(n & emb) / len(n))
# Gab record fields (first 2000 lines)
keys = set()
for line in itertools.islice(open('gab/extracted/gab_posts_jan_2018.json'), 2000):
    line = line.strip().rstrip(',')
    if line.startswith('{'):
        p = json.loads(line)['post']; keys |= set((ast.literal_eval(p) if isinstance(p, str) else p).keys())
print('Gab post keys', sorted(keys))
EOF
```

Expected: VF retweeted_id ≈0.996 in tweets / 0.0 in users; TIMME 21,014 ids, 633 extra labels in P_all,
state ≈0.506; Reddit 858,488 edges, 67,180 nodes, coverage ≈0.465; Gab keys include `topic`, no `hashtags`.

---

# Topic 2 — Dataset finalization: inventory + scoring · 2026-09-27 · resolved 2026-09-27

Requirements R1–R10 agreed with the user: `TASKS.md` T0.1. Key ones: metadata must
reflect thoughts/interests and is used **as features only** (R3); nodes without
metadata are **kept** with empty features + `has_metadata` flag (R4, revised same day —
dropping them would delete edges/bridges); labels and timestamps not required (R6, R7) — success is
measured by spread blocked in the IBM simulation.

Topic 1 (VoterFraud `retweeted_id` = tweet id, not user id) **independently
verified** by this session: 99.1% in tweets, 0% in users (500k-row sample). The
current `train_voterfraud.py` builds a mixed tweet/user graph and must not be trained.

## Inventory (verified from files)

| Dataset | Node | Nodes (≈; all kept per revised R4) | Edges | Metadata (feature) | Coverage | Labels (bonus) | Timestamps |
|---|---|---|---|---|---|---|---|
| TIMME `P_all` | Twitter user (politician-centred crawl) | ~20.8k (LCC) | 5 relation types, directed, 6.5M rows | GloVe avg of user's tweets (300-d) | 93.3% non-zero | 1,206 R/D | no |
| SNAP Twitter-ego | Twitter user | ~81.3k | 2.42M directed follows | hashtags + @mentions used (binary, per-ego vocab) | 89.5% non-empty | circles for 28% of nodes | no |
| Gab | Gab user | ≤337k | repost/reply/quote | own post text → MiniLM | unmeasured (needs full parse) | none | **yes** (cascades) |
| Pokec | Slovak SN user | 1.63M | 30.6M directed friendships | free-text hobbies/music/movies/books/sports (Slovak) | 60.7% ≥1 interest field | region (187), age, gender | no |
| Reddit | subreddit | ~31k | 858k hyperlinks, directed | 300-d subreddit embedding (SNAP: from user co-activity — closer to topology than "thoughts"; unverified) | 46.5% | none | yes (per link) |
| VoterFraud | user / tweet (mixed) | — | user→tweet only; no user→user | community id + status | — | graph-derived | month |
| Orkut | user | 3.07M | 117M undirected | only interest-group membership (= its ground-truth communities) | — | groups | no |
| SNAP Facebook-ego | user | ~4k | 88k undirected | anonymized education/work/location ids | 99.9% | circles | no |

## Scoring vs requirements
- **Pass:** TIMME, Twitter-ego, Gab, Pokec.
- **Fail:** VoterFraud (R3 no metadata; user graph not buildable), Orkut (R3: only
  metadata is the group labels themselves → circular; R9 117M edges),
  Facebook-ego (R9/R1: 4k nodes, 10 egos; R3: demographics not thoughts).
- **Weak:** Reddit (R3: embedding is co-activity, not content; R2 debatable).

## Caveats on the passers
- TIMME and Twitter-ego are **crawled around seed accounts** (politicians / 973 egos):
  some "bridges" may be sampling artifacts (the seeds themselves).
- Pokec is huge; a **region-induced subgraph** (e.g. one kraj) gives a small start (R9).
  Needs a multilingual text encoder (Slovak).
- Gab needs the one-off 48 GB parse (T1.7) before its size/coverage are known.

## Open decisions
- [ ] Pick 2 of {TIMME, Twitter-ego, Gab, Pokec}

## Detail pass: TIMME + Pokec (2026-09-27)
- **TIMME has raw text**: `timme/repo/data/formatted_location/simplified_user_info.json`
  covers all 21,014 P_all users — description 84.4%, location 76.6%, latest tweet
  (`status`) 93.4% non-empty. Better feature source than the GloVe averages; unused so far.
- TIMME relation files carry an interaction `count` (median 1, mean ~4, max ~3.2k);
  friend_list count is always 1. P_all `features.npz` (768-d description/status) is a
  split zip — recoverable by unzipping `features.npz.zip`.
- `yupeng_dataset/` is a separate, older congress crawl (487 members, 3 relations) — not P_all.
- Pokec: 30,622,564 directed edges, 54.3% reciprocated; all 1,632,803 profiles present;
  AGE=0 (unknown) for 30.3%; median profile completion 41%.

## Resolution (2026-09-27)
User decision: **TIMME `P_all` + Pokec**. TIMME features = bio + latest tweet text;
Pokec = interest text + gender/age/region; one multilingual encoder. VoterFraud dropped
(loader deleted), Gab/Reddit reserve. `data_final/DATASETS.md` rewritten (resolves Topic 1).
Still open → `TASKS.md` T0.4a: full Pokec vs regional slice first.
