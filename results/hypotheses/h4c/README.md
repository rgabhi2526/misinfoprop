# H4c: does a per-account text-vs-neighbourhood misfit score help rumour detection?

**Verdict: NO. Feasibility check passed only partially (data exist, but the graph is too sparse for a meaningful misfit), and the score adds nothing: delta AUC -0.0006 [-0.0013,+0.0001] (stratified 5-fold), -0.0010 [-0.0022,+0.0002] (leave-event-out). Not useful as an extra feature on PHEME.**

## Scoping (step 1)
| Source | Misinfo labels | Cascades | Sharer text | Sharer graph | Usable? |
|---|---|---|---|---|---|
| data_final/voterfraud2020 | none (only author-community 0-4 and active status; urls.csv gives domain counts, not credibility) | retweets-*.csv = (retweeted tweet id, retweeter user_id), so cascades exist | none (no tweet text anywhere; tweets-*.csv has no author/user column) | no user->user edges (retweeted_id is a tweet id; can only build retweeter co-retweet graph) | No: no labels, no text |
| data_final/gab | none | repost chains | yes | no follow graph | Not parsed (48GB); no veracity labels |
| data_final/reddit-hyperlinks | none | none | LIWC only | subreddit graph | No |
| Twitter15/16 (Ma et al. 2017, downloaded from Dropbox link in Rumor_RvNN README) | yes (4 classes, 1490+818 trees) | yes, with uid + delay | source tweet only (README: rest not released) | uids only | Structure features only; no sharer text -> misfit undefined. Downloaded to data_final/rumor_ext/tw1516, not used. |
| **PHEME (figshare 6392078, all-rnr-annotated-threads)** | yes: rumour vs non-rumour, 6425 threads, 9 events | yes, reply trees | **yes: tweet text + full user profile (description, followers...) for all repliers** | no follow graph; only reply edges | **Used** |
| FakeNewsNet | -- | -- | -- | -- | Skipped (needs Twitter API hydration) |

Network was available (curl worked for figshare and Dropbox). Downloads live in data_final/rumor_ext/{pheme,tw1516}; all Python read them with -I.

## Method (step 2) - training/h4c_pheme.py
- Users = authors of source tweets and replies (50,593). Graph = reply edges replier -> parent author (63,820 edges, 934 components, giant component 47.8k). Leiden communities (1,033; 252 with >=5 members; 180 classes with >=8 used).
- User text = profile description + their tweets in the data -> TF-IDF + SVD(64). (No neural embedding: CPU only, TF-IDF allowed.)
- misfit(u) = 1 - p_C[t_u], p_C = cross-fitted (5 random user folds) cosine-centroid softmax of text -> community. Also text-outlier score. Aggregated per cascade over sharers: mean, max, frac>0.8, count (+ outlier mean/max/frac). No rumour labels used to build it.
- Classifier: HistGradientBoosting (CPU, small tables). Baseline = source-text TF-IDF+SVD(40) (fit per train fold) ; "struct" adds size, depth, reply-delay quantiles, source-user followers/verified/age/statuses, URL count etc. ; "struct+misfit" adds the 7 misfit features.
- Eval: stratified 5-fold and leave-one-event-out (harder, honest). 1000x bootstrap CIs over threads; paired bootstrap for deltas.

## Results (AUC [95% CI], F1)
| Features | stratified 5-fold | leave-event-out |
|---|---|---|
| source text only | 0.879 [0.870,0.887] F1 .717 | 0.693 [0.680,0.707] F1 .450 |
| text + structure (baseline) | 0.880 [0.871,0.888] F1 .723 | 0.708 [0.694,0.721] F1 .469 |
| baseline + misfit | 0.879 [0.871,0.888] F1 .722 | 0.706 [0.693,0.720] F1 .465 |
| misfit alone | 0.604 [0.591,0.618] | 0.456 [0.441,0.470] (below chance) |

Delta AUC baseline -> +misfit: -0.0006 [-0.0013,+0.0001] / -0.0010 [-0.0022,+0.0002]. (Structure over text-only: +0.001 [-0.001,+0.003] stratified; +0.014 [+0.012,+0.017] leave-event-out.)
Univariate: all misfit aggregates have AUC 0.43-0.47 (outlier 0.48-0.51); log cascade size alone is 0.427, and the "number of sharers with misfit defined" has corr 0.72 with log size. Misfit alone (0.604 stratified) is just a size proxy, and flips below chance across events.

## Why it fails here / caveats (blunt)
1. Mean misfit is 0.98: with 180 tiny reply-star communities, text cannot predict community, so misfit is ~constant and the signal is mostly cascade size. The "neighbourhood" in PHEME is a reply tree around a single post, i.e. a thread, not a social neighbourhood; text-vs-thread-community mismatch measures topic drift in replies, not account anomaly.
2. No follow graph. Users appear in ~1 thread; 934 components. Misfit as defined for TIMME/Pokec (dense social graph + persistent account text) needs follow/retweet edges among accounts with several posts each.
3. Reply trees contain replies, not shares; rumour-ness shows up in stance (support/deny/query), which text/structure baselines partly capture already.
4. Single dataset, one embedding (TF-IDF+SVD), one community resolution. A MiniLM embedding could change the misfit, but it cannot fix points 1-2; not run (CPU time; and GPU/training not needed, so nothing added to gpu_needed.md).
5. Leave-event-out AUC (~0.70) is the realistic number; the 0.88 stratified figure is event leakage via source-text topic.

## Minimal data needed for a fair test
Cascades with veracity labels AND, for each sharer, (a) >=20 own tweets/bio and (b) follow or mutual-mention edges among the sharers (or among a ~1-hop crawl), e.g. FakeNewsNet / Twitter15-16 / CoAID rehydrated with user timelines and follower lists via the Twitter/X API (not available offline). VoterFraud2020 has the graph-ish retweet data and community ids but no text and no veracity labels; joining it with hydrated tweet text and URL-domain credibility (e.g. NewsGuard/MBFC ratings on urls.csv domains) would make a domain-credibility label feasible.

## Files
- training/h4c_pheme.py (script), results/hypotheses/h4c/h4c_pheme.log, h4c_pheme_results.json
- data_final/rumor_ext/ (pheme, tw1516 downloads)
