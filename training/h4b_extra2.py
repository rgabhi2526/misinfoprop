"""H4b extra2: community fixed effects. Strata = follow-community x follow-degree decile (+ x participation tercile variant)."""
import json, os, numpy as np
import views, h4b_data, h4b_scores
from h4b_analyze import *
if __name__ == "__main__":
    rng = np.random.default_rng(13)
    b = views.load_timme(); bf = h4b_scores.follow_bundle(b); inter = h4b_data.load_inter()
    btw = np.load(os.path.join(OUTC, "h4b_btw_follow.npy")); rows = []
    for res, seed in ((1.0, 0), (1.0, 1), (1.0, 2), (2.0, 0)):
        s, t, deg, pop = load_cfg(res, seed, b, bf, btw)
        d5 = strata_of(deg, pop, q=5)
        pt = np.digitize(s["participation"], np.quantile(s["participation"][pop], [1/3, 2/3]))
        for nm, stv in (("community x deg_quintile", t * 5 + d5), ("community x deg_quintile x participation_tercile", (t * 5 + d5) * 3 + pt)):
            for g in ("spread", "retweet"):
                o = outcomes(inter, t, g)
                for sn in ("misfit", "disagreement"):
                    for on, y in (("log1p_cross_n", np.log1p(o["cross_n"])), ("cross_share", o["cross_share"]), ("log1p_cross_in", np.log1p(o["cross_in"]))):
                        r = matched_test(s[sn], y, deg, pop, rng, nperm=2000, boot=200, strata=np.where(pop, stv, -1))
                        rows.append({"res": res, "seed": seed, "group": g, "score": sn, "outcome": on, "matching": nm, **r})
                        print(res, seed, g, sn, on, nm, round(r["diff"], 4), [round(x, 4) for x in r["ci"]], round(r["p_two"], 4), flush=True)
    json.dump(rows, open(os.path.join(OUTD, "extra2.json"), "w"))
