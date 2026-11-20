"""Framework overview figure (schematic). Writes framework.pdf / framework.png next to this file.

Run: /opt/anaconda3/bin/python report/figures/make_framework.py
Layout units: 1 unit = 0.1 inch (figure is W x H units, equal aspect).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.fallback": "stixsans",
})

W, H = 72, 44
INK, MUTED = "#1F2937", "#4B5563"
COMM = ["#3B82F6", "#F97316", "#22C55E"]  # community colours
RED = "#E11D48"
P = {  # panel colour, tint
    1: ("#2563EB", "#EFF5FF"),
    2: ("#059669", "#ECFDF5"),
    3: ("#7C3AED", "#F5F3FF"),
    4: ("#EA580C", "#FFF7ED"),
    5: ("#0E7490", "#ECFEFF"),
    6: ("#BE185D", "#FDF2F8"),
}

fig = plt.figure(figsize=(W / 10, H / 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")


# ---------- primitives ----------
def rbox(x, y, w, h, fc, ec, lw=0.6, r=0.6, ls="-", z=1, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, alpha=alpha))


def t(x, y, s, size=5.6, c=INK, ha="left", va="center", w="normal", style="normal", z=5):
    ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, weight=w, style=style, zorder=z)


def arrow(p0, p1, c=INK, lw=1.0, rad=0.0, ms=6, z=6, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, ls=ls,
                                 shrinkA=0, shrinkB=0))


def panel(n, x, y, w, h, title):
    col, tint = P[n]
    rbox(x, y, w, h, tint, col, lw=0.9, r=0.8, z=1)
    rbox(x, y + h - 2.0, w, 2.0, col, col, lw=0.9, r=0.8, z=2)
    ax.add_patch(Rectangle((x, y + h - 2.0), w, 0.9, fc=col, ec="none", zorder=2))  # square bottom of header
    ax.add_patch(Circle((x + 1.25, y + h - 1.0), 0.7, fc="white", ec="none", zorder=3))
    t(x + 1.25, y + h - 1.0, str(n), 6.4, col, ha="center", w="bold")
    t(x + 2.3, y + h - 1.0, title, 7.0, "white", w="bold")


def node(x, y, c, r=0.42, ring=None, z=4):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=0.5, zorder=z))
    if ring:
        ax.add_patch(Circle((x, y), r + 0.28, fc="none", ec=ring, lw=0.9, zorder=z))


def edge(a, b, c="#9CA3AF", lw=0.45, ls="-", z=3):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def chip(x, y, s, col, size=5.0):
    w = 0.42 * size * len(s) / 5.0 * 0.95 + 1.0
    rbox(x, y - 0.55, w, 1.1, "white", col, lw=0.5, r=0.5, z=4)
    t(x + w / 2, y, s, size, col, ha="center", w="bold")
    return x + w


def community_graph(cx, cy, scale, seed, misfit=True, blocked=None, cut=None):
    """Three toy communities; returns positions/labels. One planted misfit (ring)."""
    rng = np.random.default_rng(seed)
    centres = np.array([[-1.0, 0.55], [1.0, 0.55], [0.0, -0.85]]) * scale
    pos, lab = [], []
    for k, c in enumerate(centres):
        for j in range(7):
            ang = 2 * np.pi * j / 7 + rng.uniform(-0.3, 0.3)
            rad = scale * (0.35 + 0.25 * rng.random()) if j else 0.0
            pos.append(c + rad * np.array([np.cos(ang), np.sin(ang)]))
            lab.append(k)
    pos = np.array(pos) + [cx, cy]
    lab = np.array(lab)
    E = []
    for k in range(3):
        idx = np.where(lab == k)[0]
        for i in idx[1:]:
            E.append((idx[0], i))
        for a, b in zip(idx[1:], np.roll(idx[1:], 1)):
            if rng.random() < 0.7:
                E.append((a, b))
    bridges = [(3, 10), (12, 17), (5, 19)]
    for a, b in E:
        edge(pos[a], pos[b])
    for a, b in bridges:
        dashed = cut is not None and (a, b) == cut
        edge(pos[a], pos[b], RED if dashed else "#6B7280", 0.8 if dashed else 0.6,
             ls=(0, (1.5, 1)) if dashed else "-")
    mis = 9 if misfit else None
    for i, (p, k) in enumerate(zip(pos, lab)):
        node(*p, COMM[k], r=0.5 if i % 7 == 0 else 0.38, ring=RED if i == mis else None)
    if blocked is not None:
        x, y = pos[blocked]
        ax.add_patch(Circle((x, y), 0.62, fc="#111827", ec="white", lw=0.5, zorder=6))
        t(x, y, "×", 6.5, "white", ha="center", w="bold", z=7)
    return pos, lab


# ---------- layout ----------
TOP_Y, TOP_H = 25.0, 18.6
BOT_Y, BOT_H = 3.3, 16.5
COLS = [0.5, 24.5, 48.5]
PW = 23.0

# ===== Panel 1: data =====
x0 = COLS[0]
panel(1, x0, TOP_Y, PW, TOP_H, "Data & Network")
community_graph(x0 + 6.6, TOP_Y + 10.9, 2.6, seed=3)
t(x0 + 6.6, TOP_Y + 6.75, "communities + one planted misfit", 4.6, MUTED, ha="center", style="italic")
dy = TOP_Y + 15.0
for name, sub in [("TIMME", "Twitter · 20.8k users"), ("Pokec", "social net · 1.63M users"),
                  ("Synthetic", "DC-SBM · planted conflicts")]:
    rbox(x0 + 13.4, dy - 1.35, 9.0, 2.7, "white", P[1][0], lw=0.45, r=0.5, z=3)
    t(x0 + 14.0, dy + 0.48, name, 5.8, P[1][0], w="bold")
    t(x0 + 14.0, dy - 0.58, sub, 4.8, MUTED)
    dy -= 3.2
rbox(x0 + 0.8, TOP_Y + 0.7, PW - 1.6, 5.2, "white", P[1][0], lw=0.5, r=0.5, ls=(0, (3, 1.5)), z=3)
t(x0 + 1.5, TOP_Y + 4.95, "Preprocess", 5.6, P[1][0], w="bold")
t(x0 + 1.5, TOP_Y + 3.7, "symmetrise · drop self-loops · largest conn. comp.", 5.0)
t(x0 + 1.5, TOP_Y + 2.6, "keep users without text (+ has_metadata flag)", 5.0)
t(x0 + 1.5, TOP_Y + 1.5, r"text $\rightarrow$ MiniLM-L12 $\rightarrow$ $x_i \in \mathbb{R}^{384}$", 5.0)

# ===== Panel 2: two views =====
x0 = COLS[1]
panel(2, x0, TOP_Y, PW, TOP_H, "Two Separate Views")
for j, (title, lines) in enumerate([
    ("Topology view", [r"edges only", r"Leiden $\times R$ seeds", r"$\rightarrow$ partition $t$ (max $Q$)",
                       r"label spreading:", r"$F \leftarrow \alpha S F + (1-\alpha) Y$",
                       r"$\rightarrow p_T(i) \in \Delta^{K-1}$"]),
    ("Content view", [r"text only, no edges", r"cross-fitted logistic reg.", r"target $t$, own fold held out",
                      r"$p_C(i) = \Pr(t_i \mid x_i)$", r"$\in \Delta^{K-1}$", ""]),
]):
    bx = x0 + 0.8 + j * 11.0
    rbox(bx, TOP_Y + 7.2, 10.4, 9.0, "white", P[2][0], lw=0.5, r=0.5, z=3)
    t(bx + 5.2, TOP_Y + 15.3, title, 5.8, P[2][0], ha="center", w="bold")
    for i, s in enumerate(lines):
        t(bx + 5.2, TOP_Y + 14.0 - i * 1.13, s, 5.0, MUTED if i == 0 else INK, ha="center",
          style="italic" if i == 0 else "normal")
arrow((x0 + 11.25, TOP_Y + 11.5), (x0 + 11.75, TOP_Y + 11.5), P[2][0], lw=0.6, ms=4)
t(x0 + 11.5, TOP_Y + 12.3, "$t$", 5.0, P[2][0], ha="center")
# mini bar charts: same node, the two views disagree
pT, pC = [0.78, 0.15, 0.07], [0.12, 0.80, 0.08]
for j, (lbl, pv) in enumerate([(r"$p_T(i)$", pT), (r"$p_C(i)$", pC)]):
    bx = x0 + 2.0 + j * 8.2
    for k, v in enumerate(pv):
        ax.add_patch(Rectangle((bx + k * 1.3, TOP_Y + 1.6), 1.0, v * 4.6, fc=COMM[k], ec="none", zorder=4))
    ax.plot([bx - 0.3, bx + 3.9], [TOP_Y + 1.6] * 2, color=MUTED, lw=0.5, zorder=4)
    t(bx + 1.8, TOP_Y + 0.95, lbl, 5.0, ha="center")
t(x0 + 6.95, TOP_Y + 4.2, "vs", 5.4, MUTED, ha="center", style="italic")
t(x0 + 18.9, TOP_Y + 4.5, "same user,", 4.7, RED, ha="center", style="italic")
t(x0 + 18.9, TOP_Y + 3.5, "views", 4.7, RED, ha="center", style="italic")
t(x0 + 18.9, TOP_Y + 2.5, "disagree", 4.7, RED, ha="center", style="italic")

# ===== Panel 3: scores =====
x0 = COLS[2]
panel(3, x0, TOP_Y, PW, TOP_H, "Per-node Scores")
rbox(x0 + 0.8, TOP_Y + 9.6, PW - 1.6, 6.7, "white", P[3][0], lw=0.5, r=0.5, z=3)
for i, (name, f) in enumerate([
    ("misfit", r"$1 - p_C(i)_{t_i}$"),
    ("disagreement", r"$\mathrm{JS}_2\,(p_T(i)\,\Vert\,p_C(i))$"),
    ("ambiguity", r"$H(p_T(i))\,/\,\log K$"),
]):
    yy = TOP_Y + 15.0 - i * 2.05
    t(x0 + 1.6, yy, name, 5.6, P[3][0], w="bold")
    t(x0 + 8.4, yy, "=", 5.6, MUTED)
    t(x0 + 9.4, yy, f, 6.0)
t(x0 + 1.0, TOP_Y + 8.55, "Baselines to beat", 5.4, P[3][0], w="bold")
cx = x0 + 1.0
for s in ["consensus entropy", "participation", "content outlier"]:
    cx = chip(cx, TOP_Y + 7.2, s, MUTED, 4.6) + 0.4
cx = x0 + 1.0
for s in ["degree", "PageRank", r"score $\times$ PageRank"]:
    cx = chip(cx, TOP_Y + 5.85, s, MUTED, 4.6) + 0.4
# score distribution with top-k tail
xs = np.linspace(0, 1, 40)
ys = np.exp(-((xs - 0.25) ** 2) / 0.03) + 0.25 * np.exp(-((xs - 0.85) ** 2) / 0.01)
ys = ys / ys.max() * 3.0
gx, gy, gw = x0 + 1.6, TOP_Y + 1.0, 13.0
ax.fill_between(gx + xs * gw, gy, gy + ys, color=P[3][0], alpha=0.18, lw=0, zorder=3)
tail = xs >= 0.72
ax.fill_between(gx + xs[tail] * gw, gy, gy + ys[tail], color=P[3][0], alpha=0.75, lw=0, zorder=4)
ax.plot(gx + xs * gw, gy + ys, color=P[3][0], lw=0.7, zorder=4)
ax.plot([gx, gx + gw], [gy, gy], color=MUTED, lw=0.5, zorder=4)
t(gx + 0.72 * gw + 1.6, gy + 2.3, "top-$k$", 5.0, P[3][0], w="bold")
t(gx + gw + 0.8, gy + 2.0, "rank nodes", 4.8, MUTED)
t(gx + gw + 0.8, gy + 1.0, "(or links)", 4.8, MUTED)

# ===== Band: two questions (both fed by the scores, not sequential) =====
BY = 22.0
rbox(48.9, BY - 1.25, 22.2, 2.5, "#FFFFFF", "#9CA3AF", lw=0.6, r=1.2, ls=(0, (3, 1.5)), z=3)
t(60.0, BY, "every score and baseline, same tests", 5.4, MUTED, ha="center", style="italic")
ey = 24.15
ax.plot([60.0, 60.0, 12.0], [TOP_Y, ey, ey], color=INK, lw=0.9, zorder=5, solid_joinstyle="round")
for (bx, bw, s_, n) in [(1.0, 22.0, r"Q1: does $s_i$ find real conflicts?", 4),
                        (25.0, 22.0, r"Q2: does blocking by $s_i$ cut spread?", 5)]:
    rbox(bx, BY - 1.25, bw, 2.5, P[n][1], P[n][0], lw=0.7, r=1.2, z=3)
    t(bx + bw / 2, BY, s_, 6.0, P[n][0], ha="center", w="bold")
    arrow((bx + bw / 2, ey), (bx + bw / 2, BY + 1.25), INK, lw=0.9, ms=5)
    arrow((bx + bw / 2, BY - 1.25), (bx + bw / 2, BOT_Y + BOT_H + 0.05), P[n][0], lw=0.9, ms=5)
t(60.5, ey + 0.42, r"scores $s_i$", 5.0, P[3][0], w="bold")
# row-1 arrows
for xa in (23.5, 47.5):
    arrow((xa + 0.1, TOP_Y + 11.5), (xa + 0.95, TOP_Y + 11.5), INK, lw=1.0, ms=5)

# ===== Panel 4: detection =====
x0 = COLS[0]
panel(4, x0, BOT_Y, PW, BOT_H, "Detection Check")
yy = BOT_Y + 13.3
for head, lines in [("Synthetic benchmark", [r"DC-SBM, power-law degree, mixing $\mu$",
                                             r"text = centroid + $\sigma$ noise (+ nuisance)",
                                             r"planted: text of another community"]),
                    ("TIMME (real labels)", [r"party $\neq$ community majority",
                                             r"($\geq$20 labelled, $\geq$80% one party)"])]:
    t(x0 + 1.0, yy, head, 5.6, P[4][0], w="bold")
    for s in lines:
        yy -= 1.12
        t(x0 + 1.0, yy, s, 4.8)
    yy -= 1.55
rbox(x0 + 0.8, BOT_Y + 0.6, 12.6, 2.2, "white", P[4][0], lw=0.5, r=0.5, ls=(0, (3, 1.5)), z=3)
t(x0 + 7.1, BOT_Y + 1.7, "null: shuffle text rows, recompute", 4.7, P[4][0], ha="center", w="bold")
# ROC
rx, ry, rs = x0 + 15.0, BOT_Y + 4.2, 7.0
ax.add_patch(Rectangle((rx, ry), rs, rs, fc="white", ec=MUTED, lw=0.5, zorder=3))
u = np.linspace(0, 1, 50)
ax.plot(rx + u * rs, ry + u * rs, color="#9CA3AF", lw=0.6, ls=(0, (2, 1.5)), zorder=4)
ax.plot(rx + u * rs, ry + u ** 0.55 * rs, color="#9CA3AF", lw=0.9, zorder=4)
ax.plot(rx + u * rs, ry + u ** 0.18 * rs, color=P[4][0], lw=1.2, zorder=5)
ax.fill_between(rx + u * rs, ry, ry + u ** 0.18 * rs, color=P[4][0], alpha=0.12, lw=0, zorder=3)
t(rx + rs / 2, ry - 0.8, "FPR", 4.6, MUTED, ha="center")
ax.text(rx - 0.75, ry + rs / 2, "TPR", fontsize=4.6, color=MUTED, rotation=90, ha="center", va="center")
t(rx + 3.6, ry + 4.9, "score", 4.6, P[4][0], w="bold")
t(rx + 4.0, ry + 2.7, "baseline", 4.4, "#6B7280")
t(rx + rs / 2, ry + rs + 0.8, "ROC on planted / misfits", 4.6, MUTED, ha="center", style="italic")

# ===== Panel 5: spread simulation =====
x0 = COLS[1]
panel(5, x0, BOT_Y, PW, BOT_H, "Spread Simulation & Blocking")
community_graph(x0 + 5.8, BOT_Y + 10.0, 2.15, seed=7, misfit=False, blocked=3, cut=(12, 17))
t(x0 + 5.8, BOT_Y + 6.1, "block node / cut link", 4.6, MUTED, ha="center", style="italic")
xx = x0 + 11.6
t(xx, BOT_Y + 13.6, "Independent Cascade", 5.6, P[5][0], w="bold")
t(xx, BOT_Y + 12.45, r"WC: $p_{uv} = 1/\deg(v)$", 4.9)
t(xx, BOT_Y + 11.3, r"uniform: $p = c\,p_c$, $p_c = \langle k\rangle / (\langle k^2\rangle - \langle k\rangle)$", 4.9)
t(xx, BOT_Y + 9.75, "Selective exposure", 5.6, P[5][0], w="bold")
t(xx, BOT_Y + 8.6, r"$\tilde{p}_{uv} =\min(1,\,p_{uv}\,K\,a_v(A))$", 4.9)
t(xx, BOT_Y + 7.45, "seeds never blocked · common RNG", 4.6, MUTED, style="italic")
rbox(x0 + 0.8, BOT_Y + 0.6, PW - 1.6, 4.85, "white", P[5][0], lw=0.5, r=0.5, z=3)
t(x0 + 1.5, BOT_Y + 4.55, "Node blocking", 5.4, P[5][0], w="bold")
t(x0 + 7.6, BOT_Y + 4.55, "degree · PageRank · betweenness · CI · misfit", 4.7)
t(x0 + 1.5, BOT_Y + 3.15, r"gateway  $g_i = \Sigma_A\, h_{iA}\,a_i(A)\,(d_i - h_{iA})$", 5.0)
t(x0 + 1.5, BOT_Y + 1.6, "Link friction", 5.4, P[5][0], w="bold")
t(x0 + 7.6, BOT_Y + 1.6, r"bridge $\mathrm{PR}_u\mathrm{PR}_v$ $\times$ receptivity $p_C$", 4.9)
arrow((47.6, BOT_Y + BOT_H / 2), (48.45, BOT_Y + BOT_H / 2), INK, lw=1.0, ms=5)


# ===== Panel 6: evaluation metrics =====
def icon(cx, cy, kind, col):
    ax.add_patch(Circle((cx, cy), 0.72, fc="white", ec=col, lw=0.7, zorder=4))
    s = 0.42
    if kind == "roc":
        u = np.linspace(0, 1, 20)
        ax.plot(cx - s + u * 2 * s, cy - s + u ** 0.3 * 2 * s, color=col, lw=0.8, zorder=5)
        ax.plot([cx - s, cx + s], [cy - s, cy + s], color=col, lw=0.4, ls=":", zorder=5)
    elif kind == "bars":
        for k, h in enumerate([0.5, 0.8, 0.65]):
            ax.add_patch(Rectangle((cx - s + k * 0.3, cy - s), 0.22, h * 2 * s, fc=col, ec="none", zorder=5))
    elif kind == "shuffle":
        ax.plot([cx - s, cx + s], [cy - 0.25, cy + 0.25], color=col, lw=0.7, zorder=5)
        ax.plot([cx - s, cx + s], [cy + 0.25, cy - 0.25], color=col, lw=0.7, zorder=5)
    elif kind == "burst":
        for a in np.linspace(0, 2 * np.pi, 7)[:-1]:
            ax.plot([cx, cx + s * np.cos(a)], [cy, cy + s * np.sin(a)], color=col, lw=0.5, zorder=5)
        ax.add_patch(Circle((cx, cy), 0.2, fc=col, ec="none", zorder=6))
    elif kind == "cross":
        ax.add_patch(Circle((cx - 0.25, cy), 0.2, fc=COMM[0], ec="none", zorder=5))
        ax.add_patch(Circle((cx + 0.3, cy), 0.2, fc=COMM[1], ec="none", zorder=5))
        ax.plot([cx - 0.05, cx + 0.1], [cy, cy], color=col, lw=0.6, zorder=5)
    elif kind == "down":
        arrow((cx, cy + s), (cx, cy - s), col, lw=0.8, ms=3.5, z=6)
    elif kind == "ci":
        ax.plot([cx, cx], [cy - s, cy + s], color=col, lw=0.6, zorder=5)
        ax.plot([cx - 0.18, cx + 0.18], [cy - s] * 2, color=col, lw=0.6, zorder=5)
        ax.plot([cx - 0.18, cx + 0.18], [cy + s] * 2, color=col, lw=0.6, zorder=5)
        ax.add_patch(Circle((cx, cy + 0.08), 0.15, fc=col, ec="none", zorder=6))
    elif kind == "heat":
        vals = [[0.15, 0.4, 0.9], [0.1, 0.3, 0.6], [0.05, 0.1, 0.25]]
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((cx - s + j * 0.28, cy + s - (i + 1) * 0.28), 0.26, 0.26,
                                       fc=col, alpha=vals[i][j], ec="none", zorder=5))


x0 = COLS[2]
panel(6, x0, BOT_Y, PW, BOT_H, "Evaluation Metrics")
rows = [
    (4, "roc", "ROC AUC", "planted conflicts / party misfits"),
    (4, "bars", "AUC by degree tercile", "score must not just track degree"),
    (4, "shuffle", "Mann–Whitney vs null", "beat own shuffled-text version"),
    (5, "burst", "Infected share", "seeds excluded"),
    (5, "cross", "Cross-community share", "infected outside seed community"),
    (5, "down", "Spread removed (%)", "vs no blocking"),
    (5, "ci", "Paired gain, 95% CI", "vs best of PageRank, betweenness"),
    (5, "heat", "Regime map", r"$\mu \times \beta \times$ mismatch share $\times$ placement"),
]
step = 1.78
for i, (n, kind, name, sub) in enumerate(rows):
    yy = BOT_Y + BOT_H - 2.8 - i * step
    icon(x0 + 1.9, yy, kind, P[n][0])
    t(x0 + 3.1, yy + 0.32, name, 5.5, INK, w="bold")
    t(x0 + 3.1, yy - 0.5, sub, 4.4, MUTED)
for n, i0, i1, lbl in [(4, 0, 2, "detection"), (5, 3, 7, "spread")]:
    ytop = BOT_Y + BOT_H - 2.8 - i0 * step + 0.75
    ybot = BOT_Y + BOT_H - 2.8 - i1 * step - 0.75
    ax.plot([x0 + 21.9, x0 + 21.9], [ybot, ytop], color=P[n][0], lw=1.6, solid_capstyle="round", zorder=4)
    ax.text(x0 + 21.25, (ytop + ybot) / 2, lbl, fontsize=4.4, color=P[n][0], rotation=90,
            ha="center", va="center", weight="bold", zorder=5)

# ===== Legend =====
rbox(0.5, 0.3, 71.0, 2.5, "#F9FAFB", "#D1D5DB", lw=0.5, r=0.6, z=1)
ly = 1.55
t(1.2, ly, "Legend:", 5.6, w="bold")
lx = 5.6
for k in range(3):
    node(lx + k * 1.0, ly, COMM[k], r=0.38)
t(lx + 2.8, ly, "communities $t$", 5.0)
node(lx + 10.0, ly, COMM[1], r=0.38, ring=RED)
t(lx + 11.0, ly, "planted / misfit user", 5.0)
ax.add_patch(Circle((lx + 20.8, ly), 0.5, fc="#111827", ec="white", lw=0.5, zorder=6))
t(lx + 20.8, ly, "×", 5.6, "white", ha="center", w="bold", z=7)
t(lx + 21.7, ly, "blocked node", 5.0)
edge((lx + 27.4, ly), (lx + 29.4, ly), RED, 0.9, ls=(0, (1.5, 1)), z=5)
t(lx + 29.9, ly, "cut link", 5.0)
t(lx + 34.4, ly, r"$p_T, p_C$: structural / content membership   $K$: # communities   "
                 r"$a_v(A)$: content affinity   $\beta$: agreement effect", 5.0, MUTED)

out = Path(__file__).resolve().parent
fig.savefig(out / "framework.pdf")
fig.savefig(out / "framework.png", dpi=400)
print("wrote", out / "framework.pdf", out / "framework.png")
