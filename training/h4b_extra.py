"""H4b extra: robust outcomes (log1p count, share) and matching on follow-degree decile x follow-participation quintile
(so misfit must beat plain boundary-ness). Spread group."""
import json, os, numpy as np
import ambiguity, views, h4b_data, h4b_scores
from h4b_analyze import *
if __name__ == "__main__":
    rng = np.random.default_rng(11)
    b = views.load_timme(); bf = h4b_scores.follow_bundle(b); inter = h4b_data.load_inter()
    btw = np.load(os.path.join(OUTC, "h4b_btw_follow.npy")); rows = []
    for res, seed in ((1.0, 0), (1.0, 1), (1.0, 2), (2.0, 0), (2.0, 1)):
        s, t, deg, pop = load_cfg(res, seed, b, bf, btw)
        d10 = strata_of(deg, pop); q5 = np.digitize(s["participation"], np.quantile(s["participation"][pop], [.2, .4, .6, .8]))
        st2 = d10 * 5 + q5
        for g in ("spread", "retweet"):
            o = outcomes(inter, t, g)
            for sn in ("misfit", "disagreement"):
                for on, y in (("log1p_cross_n", np.log1p(o["cross_n"])), ("cross_share", o["cross_share"]), ("in_cross_share", o["in_cross_share"])):
                    for mn, stv in (("deg_decile", None), ("deg_decile x participation_quintile", st2)):
                        r = matched_test(s[sn], y, deg, pop, rng, nperm=3000, boot=200, strata=stv)
                        rows.append({"res": res, "seed": seed, "group": g, "score": sn, "outcome": on, "matching": mn, **r})
                        print(res, seed, g, sn, on, mn, round(r["diff"], 4), r["ci"], r["p_two"], flush=True)
    json.dump(rows, open(os.path.join(OUTD, "extra.json"), "w"))
