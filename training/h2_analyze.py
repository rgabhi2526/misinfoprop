"""Summaries of H2 results -> ../results/hypotheses/h2/tables_<ds>_<metric>.md
    python h2_analyze.py synth|timme|pokec_knm [inf|cross]
Gain = removed(config) - removed(best of PR/btw top-k), points of unblocked share (x100); mean +- t95 CI over seeds."""
import json, sys, os, collections
import numpy as np
from scipy import stats
D = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h2")
PCT = [0.001, 0.005, 0.01, 0.02, 0.05]

def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2: return float(x.mean()), float("nan")
    return float(x.mean()), float(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
def fmt(m, h): return f"{100*m:+.1f}±{100*h:.1f}"
def load(ds): return json.load(open(os.path.join(D, "synth.json" if ds == "synth" else f"real_{ds}.json")))

def main(ds, metric):
    rows = [r for r in load(ds) if r["metric"] == metric]
    if ds == "synth":
        gk = lambda r: (r["place"], r["beta"]); sk = lambda r: r["seed"]   # seed-level, averaged over mu
        cols = [("hub", 0), ("hub", 1), ("hub", 3), ("hub", 6), ("hub", 10), ("random", 3), ("random", 6), ("random", 10)]
    else:
        gk = lambda r: (r["model"], r["mode"]); sk = lambda r: r["seed"]
        cols = sorted({gk(r) for r in rows})
    L = []; P = lambda s="": L.append(s)
    # (name, group) -> per-seed list of (gain, removed)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: acc[(r["name"], gk(r))][sk(r)].append((r["gain"], r["removed"]))
    def cell(name, g, which=0):
        per = acc.get((name, g))
        if not per: return None
        return ci([np.mean([x[which] for x in v]) for v in per.values()]), [np.mean([x[which] for x in v]) for v in per.values()]
    P(f"# {ds}, metric={metric}\nGain over best(PR,btw) top-k, points of unblocked {metric} share, mean±95%CI over {'seeds (averaged over mu)' if ds=='synth' else 'replicate batches'}.\n")
    P("## Headroom: removal achieved by best-structural top-k (points)")
    P("| k% | " + " | ".join(map(str, cols)) + " |"); P("|---" * (len(cols) + 1) + "|")
    for k in PCT:
        out = []
        for g in cols:
            a, b = cell(f"pr|-|{k}|cent", g, 1), cell(f"btw|-|{k}|cent", g, 1)
            out.append("-" if a is None else f"{100*max(a[0][0], b[0][0]):.0f}")
        P(f"| {100*k:g} | " + " | ".join(out) + " |")
    hows = ["cent", "misfit", "rankavg", "prod", "mthenc", "rand"]
    for k in PCT:
        P(f"\n## k = {100*k:g}% of nodes: gain vs best structural")
        P("| filter | M | how | " + " | ".join(map(str, cols)) + " |"); P("|---" * (len(cols) + 3) + "|")
        names = [n for n in sorted({n for n, _ in acc}) if n.split("|")[2] == str(k)]
        def keyf(n):
            f, M, _, h = n.split("|")
            return (f, ["-", "2k", "3k", "5k", "10k", "20%"].index(M), hows.index(h) if h in hows else 9)
        for n in sorted(names, key=keyf):
            f, M, _, h = n.split("|")
            if f in ("random", "misfit_alone") and False: continue
            cs = [cell(n, g) for g in cols]
            P(f"| {f} | {M} | {h} | " + " | ".join("-" if c is None else fmt(*c[0]) for c in cs) + " |")
    # control and summary
    P("\n## Control: misfit-selected minus random-selected from the SAME shortlist (points of removal)")
    P("| filter | M | k% | " + " | ".join(map(str, cols)) + " |"); P("|---" * (len(cols) + 3) + "|")
    for f in ("pr", "btw", "deg", "union"):
        for M in ("3k", "10k", "20%"):
            for k in PCT[1:]:
                out = []
                for g in cols:
                    a, b = cell(f"{f}|{M}|{k}|misfit", g, 1), cell(f"{f}|{M}|{k}|rand", g, 1)
                    out.append("-" if a is None or b is None else fmt(*ci(np.array(a[1]) - np.array(b[1]))))
                P(f"| {f} | {M} | {100*k:g} | " + " | ".join(out) + " |")
    # best configs (selection-biased)
    P("\n## Best 12 configs by mean gain over all columns (selection-biased: ~300 configs searched)")
    names = [n for n in {n for n, _ in acc} if "|cent" not in n]
    sc = {n: np.mean([cell(n, g)[0][0] for g in cols if cell(n, g)]) for n in names}
    P("| config | " + " | ".join(map(str, cols)) + " | mean |"); P("|---" * (len(cols) + 2) + "|")
    for n in sorted(names, key=lambda n: -sc[n])[:12]:
        P(f"| {n} | " + " | ".join("-" if cell(n, g) is None else fmt(*cell(n, g)[0]) for g in cols) + f" | {100*sc[n]:+.2f} |")
    # cells with CI>0
    if ds == "synth":
        cacc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows: cacc[(r["name"], r["mu"], r["place"], r["beta"])][r["seed"]].append(r["gain"])
        cnt = collections.defaultdict(lambda: [0, 0, 0])
        for (n, *_), per in cacc.items():
            if "|cent" in n: continue
            m, h = ci([np.mean(v) for v in per.values()]); cnt[n][2] += 1
            cnt[n][0] += m - h > 0; cnt[n][1] += m + h < 0
        P("\n## Cells (mu x place x beta, 30 per config) with 95%CI>0 / <0 (top 10 by #CI>0)")
        for n in sorted(cnt, key=lambda n: -cnt[n][0])[:10]:
            P(f"{n}: CI>0 {cnt[n][0]}, CI<0 {cnt[n][1]} of {cnt[n][2]}")
    open(os.path.join(D, f"tables_{ds}_{metric}.md"), "w").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "inf")
