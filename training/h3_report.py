"""Print a markdown table from results/hypotheses/h3/h3_results.json."""
import json, os
p = os.path.join(os.path.dirname(__file__), "..", "results", "hypotheses", "h3", "h3_results.json")
R = json.load(open(p))
f = lambda x: f"{100*x[0]:+.1f} ± {100*x[1]:.1f}"
print("| scenario | metric | best misfit | best struct/baseline | gain (pts) ±95%CI | fixed misfit_x_pr gain | wins |")
print("|---|---|---|---|---|---|---|")
for nm, d in R.items():
    for key, o in d["res"].items():
        if key == "auc": continue
        print(f"| {nm} | {key} | {o['best_misfit']} {100*o['misfit_val'][0]:.1f} | {o['best_struct']} {100*o['struct_val'][0]:.1f} | "
              f"{f(o['gain'])} | {f(o['gain_fixed_misfit_x_pr'])} | {o['wins']}/{o['n_seeds']} |")
for nm, d in R.items():
    if "auc" in d["res"]:
        print(nm, "AUC", {k: round(v[0], 3) for k, v in d["res"]["auc"].items()})
        for key, o in d["res"].items():
            if key.startswith("prec") and "gain_vs_best_other_nonstruct" in o:
                print("  ", key, "vs", o["best_other_nonstruct"], f(o["gain_vs_best_other_nonstruct"]))
    for key, o in d["res"].items():
        if "gain_vs_best_text" in o:
            print(nm, key, "misfit vs best text-only baseline:", f(o["gain_vs_best_text"]))
