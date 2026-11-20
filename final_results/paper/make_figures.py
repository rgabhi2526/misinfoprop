"""Re-runnable: reads ../results_long.csv, writes fig_*.pdf and tab_*.tex (generated tables)."""
import os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "results_long.csv"))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
# Okabe-Ito, fixed per method across all figures
COL = {"misfit": "#0072B2", "disagreement": "#56B4E9", "content_outlier": "#E69F00", "degree": "#009E73",
       "participation": "#CC79A7", "consensus_entropy": "#7F7F7F", "radar": "#D55E00",
       "most_linked": "#CC79A7", "label_prop": "#009E73", "p_T": "#F0E442", "adamic_adar": "#7F7F7F",
       "argmax p_C": "#0072B2", "chance": "#999999",
       "rehome_text": "#0072B2", "rehome_struct": "#D55E00", "random_rewire": "#E69F00"}
MK = {"misfit": "o", "disagreement": "s", "content_outlier": "^", "degree": "D", "participation": "v",
      "consensus_entropy": "x", "radar": "P"}
OUT = os.path.join(HERE)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight"); plt.close(fig)

# ---------------- E05 ----------------
e5 = d[(d.experiment_id == "E05") & (d.metric == "auc_overall")].copy()
e5[["nu", "mu", "sg"]] = e5.setting.str.extract(r"nuisance=([\d.]+) mu=([\d.]+) sigma=([\d.]+)").astype(float)
methods = ["misfit", "disagreement", "content_outlier", "radar", "degree", "participation", "consensus_entropy"]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True)
for ax, nu in zip(axs, [0.0, 1.5]):
    for m in methods:
        s = e5[(e5.nu == nu) & (e5.mu == 0.5) & (e5.method == m)].sort_values("sg")
        ax.plot(s.sg, s.value, marker=MK[m], ms=4, lw=1.4 if m in ("misfit", "disagreement") else 1,
                color=COL[m], label=("Radar" if m == "radar" else m.replace("_", " ")), ls="-" if m in ("misfit", "disagreement") else "--")
    ax.axhline(0.5, color="k", lw=.5, ls=":")
    ax.set_xscale("log", base=2); ax.set_xticks([.5, 1, 2]); ax.set_xticklabels(["0.5", "1", "2"])
    ax.set_xlabel(r"text noise $\sigma$"); ax.set_title(f"nuisance = {nu:g}, $\\mu$ = 0.5")
axs[0].set_ylabel("detection AUC (mean of 3 seeds)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.17), columnspacing=1.0, handlelength=1.6)
save(fig, "fig_e05.pdf")

def tex_e05():
    rows = []
    for nu in (0.0, 1.5):
        for sg in (0.5, 1.0, 2.0):
            r = {m: e5[(e5.nu == nu) & (e5.mu == 0.5) & (e5.sg == sg) & (e5.method == m)].value.iloc[0] for m in methods}
            cells = ["%.3f" % v for v in r.values()]
            rows.append(f"{nu:g} & {sg:g} & " + " & ".join(cells) + r" \\")
        if nu == 0.0: rows.append(r"\midrule")
    return "\n".join(rows)
open(os.path.join(OUT, "tab_e05.tex"), "w").write("\\begin{tabular}{@{}cc ccccccc@{}}\n\\toprule\nNuis. & $\\sigma$ & misfit & disagr. & c.\\,outl. & Radar & degree & partic. & cons.\\,ent.\\\\\n\\midrule\n" + tex_e05() + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------- E17 ----------------
e17 = d[(d.experiment_id == "E17") & d.setting.str.contains("bio only")].copy()
e17["k"] = e17.setting.str.extract(r"\(k=(\d+)\)").astype(int)
e17["tie"] = e17.setting.str.extract(r"held-out (\w+)")
PRED = [("argmax p_C (text target)", "argmax p_C", r"text $\arg\max p_C$"),
        ("most-linked non-own community (follow graph)", "most_linked", "most-linked"),
        ("label propagation, 1 step (follow graph)", "label_prop", "label prop."),
        ("argmax p_T excluding own (follow graph)", "p_T", r"$\arg\max p_T$"),
        ("Adamic-Adar community score (follow graph)", "adamic_adar", "Adamic--Adar"),
        ("chance (size-weighted)", "chance", "chance")]
TIES = ("retweet", "like", "mention")
def hit(k, t, meth, metric="cross_ties_in_target_share"):
    return e17[(e17.k == k) & (e17.tie == t) & (e17.method == meth) & (e17.metric == metric)].iloc[0]
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.2), sharey=False)
for ax, k in zip(axs, [4, 5, 15]):
    n = len(PRED); w = .8 / n
    for j, (full, key, lab) in enumerate(PRED):
        ax.bar([i + (j - (n - 1) / 2) * w for i in range(3)], [hit(k, t, full).value for t in TIES], w,
               color=COL[key], label=lab.replace("--", "-"))
    ax.set_xticks(range(3)); ax.set_xticklabels(TIES); ax.set_title(f"$k$ = {k} communities"); ax.set_ylim(0, .66)
axs[0].set_ylabel("share of cross-community ties\nin predicted community")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.1), columnspacing=1.0, handlelength=1.2)
save(fig, "fig_e17.pdf")

STRUCT = [("most-linked non-own community (follow graph)", "most_linked", "most-linked"),
          ("label propagation, 1 step (follow graph)", "label_prop", "LP"),
          ("argmax p_T excluding own (follow graph)", "p_T", "$p_T$"),
          ("Adamic-Adar community score (follow graph)", "adamic_adar", "AA")]
def tex_e17():
    out = []
    for k in (4, 5, 15):
        for t in TIES:
            v = {key: hit(k, t, full).value for full, key, _ in PRED}
            bf, bk, blab = max(STRUCT, key=lambda x: hit(k, t, x[0]).value)
            r = hit(k, t, "argmax p_C minus " + bf.replace("(follow graph)", "(follow graph)"), "text_minus_predictor_hit_share")
            diff = mm(f"{blab}: {r.value:+.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]")
            out.append(f"{k} & {t} & " + " & ".join("%.3f" % v[key] for _, key, _ in PRED) + f" & {diff} \\\\")
        if k != 15: out.append(r"\midrule")
    return "\n".join(out)
mm = lambda t: t.replace("-", "$-$")
open(os.path.join(OUT, "tab_e17.tex"), "w").write("\\begin{tabularx}{\\linewidth}{@{}cl cccccc >{\\raggedright\\arraybackslash}X@{}}\n\\toprule\n$k$ & Tie & text & most-linked & LP & $p_T$ & AA & chance & text $-$ best graph-only [95\\% CI]\\\\\n\\midrule\n" + tex_e17() + "\n\\bottomrule\n\\end{tabularx}\n")

# ---------------- E20 / E21 ----------------
def rew(exp, prefix):
    x = d[(d.experiment_id == exp) & (d.metric == "correction_reach_contrast_vs_none_pp") & d.setting.str.contains(f"{prefix}k5 model=wc")].copy()
    x["beta"] = x.setting.str.extract(r"beta=([\d.]+)").astype(float)
    return x
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.4), sharey=False)
for ax, (exp, pre, ttl) in zip(axs, [("E20", "r1.0_s0 ", "TIMME"), ("E21", "pokec_knm ", "Pokec slice")]):
    x = rew(exp, pre)
    for j, m in enumerate(["rehome_text", "rehome_struct", "random_rewire"]):
        s = x[x.method == m].sort_values("beta"); xs = range(len(s)); off = (j - 1) * .12
        ax.errorbar([i + off for i in xs], s.value, yerr=[s.value - s.ci_low, s.ci_high - s.value], marker="o", ms=4,
                    lw=1.3, capsize=2, color=COL[m], label=m.replace("_", " "))
    ax.axhline(0, color="k", lw=.5, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels(["0.3", "1", "3", "6"])
    ax.set_xlabel(r"content-gating strength $\beta$"); ax.set_title(ttl)
axs[0].set_ylabel("correction reach vs no rewiring (pp)")
axs[0].legend(frameon=False, loc="upper left")
save(fig, "fig_e21.pdf")

def tex_rew():
    out = []
    for exp, pre, nm in [("E20", "r1.0_s0 ", "TIMME"), ("E21", "pokec_knm ", "Pokec")]:
        base = d[(d.experiment_id == exp) & d.setting.str.contains(f"{pre}k5 model=wc beta=6")]
        reach = lambda m: base[(base.method == m) & (base.metric == "correction_reach_pts")].value.iloc[0]
        def con(m, ref):
            r = base[(base.method == m) & (base.metric == f"correction_reach_contrast_vs_{ref}_pp")]
            if r.empty: return "--"
            r = r.iloc[0]; return mm(f"{r.value:+.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]")
        for m, lab in [("none", "\\meth{none}"), ("rehome_text", "\\meth{rehome\\_text} (ours)"), ("rehome_struct", "\\meth{rehome\\_struct}"),
                       ("random_rewire", "\\meth{random\\_rewire}"), ("rehome_text_random_nodes", "\\meth{rehome\\_text}, rand.\\ nodes")]:
            out.append(f"{nm} & {lab} & {reach(m):.2f} & " + ("--" if m == "none" else con(m, "none")) + " & " +
                       ("--" if m in ("none", "random_rewire") else con(m, "random_rewire")) + r" \\")
        if exp == "E20": out.append(r"\midrule")
    return "\n".join(out)
open(os.path.join(OUT, "tab_rew.tex"), "w").write("\\begin{tabularx}{\\linewidth}{@{}l>{\\raggedright\\arraybackslash}Xccc@{}}\n\\toprule\nData & Method & Reach & vs \\meth{none} & vs \\meth{random\\_rewire}\\\\\n\\midrule\n" + tex_rew() + "\n\\bottomrule\n\\end{tabularx}\n")
print("ok")
