"""
starting_process_circuit.py
Ultra-compact DQC EJPP circuit — well under one column of a two-column paper.
Run:  python starting_process_circuit.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import numpy as np, os

# ── font configuration (avoid Type 3 fonts for PDF compatibility) ─────────────
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts in PDF
plt.rcParams['ps.fonttype'] = 42   # TrueType fonts in PostScript
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans']

# ── colours ───────────────────────────────────────────────────────────────────
CA = "#1565c0"
CB = "#b71c1c"
CE = "#6a1b9a"
CG = "#1b5e20"
CM = "#e65100"
CC = "#37474f"
CR = "#b71c1c"
CW = "#111111"

# ── wire y-positions (tighter spacing) ───────────────────────────────────────
YD  =  1.50   # data  (A)
YC  =  0.90   # comm  (A)
YL  =  0.30   # comm  (B)
YDB = -0.30   # data  (B)

# ── gate x-positions (shorter circuit) ───────────────────────────────────────
GX = dict(
    ebit = 0.70,
    sc   = 1.25,
    sm   = 1.80,
    xc   = 2.45,
    lg   = 3.05,
    eh   = 3.60,
    em   = 4.15,
    zc   = 4.80,
)
X0, X1 = 0.32, 5.20

# ── helpers ───────────────────────────────────────────────────────────────────
def wire(ax, y, x0=None, x1=None):
    _x0 = x0 if x0 is not None else X0
    _x1 = x1 if x1 is not None else X1
    ax.plot([_x0, _x1], [y, y], color=CW, lw=1.1, zorder=1,
            solid_capstyle="round")

def box(ax, x, y, txt, c, w=0.30, h=0.24, fs=5.0, tc="white", z=4):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.025", lw=0.7, ec=c, fc=c, zorder=z))
    ax.text(x, y, txt, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", zorder=z+1)

def cnot(ax, x, cy, ty, c, z=4):
    ax.plot([x,x],[cy,ty], color=c, lw=1.0, zorder=z)
    ax.plot(x, cy, "o", color=c, ms=4.5, zorder=z+1)
    r = 0.095
    ax.add_patch(plt.Circle((x,ty), r, color=c, fill=False, lw=1.0, zorder=z+1))
    ax.plot([x-r,x+r],[ty,ty], color=c, lw=1.0, zorder=z+2)
    ax.plot([x,x],[ty-r,ty+r], color=c, lw=1.0, zorder=z+2)

def meas_box(ax, x, y, c, z=4):
    w, h = 0.32, 0.26
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.025", lw=0.7, ec=c, fc="#fff8f0", zorder=z))
    th = np.linspace(np.pi, 0, 40)
    r = 0.08
    cy = y - 0.015
    ax.plot(x + r*np.cos(th), cy + r*np.sin(th), color=c, lw=0.8, zorder=z+1)
    tip_x = x + 0.068
    tip_y = cy + 0.065
    ax.plot([x, tip_x], [cy, tip_y], color=c, lw=0.7, zorder=z+2)
    angle = np.arctan2(tip_y - cy, tip_x - x)
    hs = 0.038
    pts = np.array([
        [tip_x + hs*np.cos(angle),     tip_y + hs*np.sin(angle)],
        [tip_x + hs*np.cos(angle+2.4), tip_y + hs*np.sin(angle+2.4)],
        [tip_x + hs*np.cos(angle-2.4), tip_y + hs*np.sin(angle-2.4)],
    ])
    ax.add_patch(plt.Polygon(pts, closed=True, fc=c, ec=c, lw=0, zorder=z+3))

def cl_arrow(ax, x0, x1, y0, y1, lbl, rad=0.0, lbl_dx=0.0, lbl_dy=0.0):
    ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
        arrowprops=dict(arrowstyle="-|>", color=CC, lw=0.8,
                        linestyle="dashed",
                        connectionstyle=f"arc3,rad={rad}"), zorder=6)
    mx = (x0+x1)/2 + lbl_dx
    my = (y0+y1)/2 + lbl_dy
    ax.text(mx, my, lbl, fontsize=4.5, color=CC, fontstyle="italic",
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec=CC,
                      alpha=0.90, lw=0.4), zorder=7)

def zigzag(ax, x, y1, y2, c=CE, n=4, amp=0.055):
    ys = np.linspace(y1, y2, 2*n+1)
    xs = np.full_like(ys, x)
    for i in range(1, len(xs)-1):
        xs[i] += amp * (1 if i % 2 == 1 else -1)
    ax.plot(xs, ys, color=c, lw=1.0, zorder=3,
            solid_capstyle="round", solid_joinstyle="round")
    ax.plot(x, y1, "o", color=c, ms=3.0, zorder=4)
    ax.plot(x, y2, "o", color=c, ms=3.0, zorder=4)

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.0, 1.80))
ax.set_xlim(0.0, 5.35)
ax.set_ylim(-0.68, 1.98)
ax.axis("off")

# ── wires ─────────────────────────────────────────────────────────────────────
wire(ax, YD); wire(ax, YC); wire(ax, YL)
wire(ax, YDB, x0=GX["lg"]-0.22, x1=GX["lg"]+0.22)

# ── wire labels ───────────────────────────────────────────────────────────────
for lbl, y, c in [("data (A)", YD,  CA),
                   ("comm (A)", YC,  CA),
                   ("comm (B)", YL,  CB),
                   ("data (B)", YDB, CB)]:
    ax.text(0.30, y, lbl, ha="right", va="center", fontsize=4.8,
            color=c, fontweight="bold")

# ── ebit zigzag ───────────────────────────────────────────────────────────────
zigzag(ax, GX["ebit"], YC, YL)
ax.text(GX["ebit"]+0.14, (YC+YL)/2, r"$|\Phi^+\rangle$",
        ha="left", va="center", fontsize=4.2, color=CE,
        bbox=dict(boxstyle="round,pad=0.04", fc="white", ec=CE,
                  alpha=0.90, lw=0.4))

# ── starting_process ──────────────────────────────────────────────────────────
cnot(ax, GX["sc"], YD, YC, CA)
meas_box(ax, GX["sm"], YC, CM)
ax.text(GX["sm"], YC-0.20, r"$m_A$", ha="center", va="top",
        fontsize=4.5, color=CM, fontweight="bold")
ax.plot([GX["sm"]+0.18, X1], [YC, YC], color=CW, lw=0.8,
        linestyle="dotted", alpha=0.28, zorder=1)
cl_arrow(ax, GX["sm"]+0.18, GX["xc"]-0.18, YC-0.08, YL+0.08,
         r"$m_A$", rad=0.0, lbl_dx=0.10)
box(ax, GX["xc"], YL, "X?", CR, w=0.28, h=0.22, fs=5.0)
ax.text(GX["xc"], YL-0.20, r"$m_A{=}1$", ha="center", va="top",
        fontsize=3.8, color=CR)

# ── local gate: CU1 controlled gate comm(B)→data(B) ──────────────────────────
box(ax, GX["lg"], YD, "ops", CG, w=0.32, h=0.22, fs=4.8)
cnot(ax, GX["lg"], YL, YDB, CG)
ax.text(GX["lg"]+0.22, (YL+YDB)/2, "CU1", ha="left", va="center",
        fontsize=4.5, color=CG, fontweight="bold")

# ── ending_process ────────────────────────────────────────────────────────────
box(ax, GX["eh"], YL, "H", CB, w=0.24, h=0.22, fs=5.0)
meas_box(ax, GX["em"], YL, CM)
ax.text(GX["em"], YL-0.20, r"$m_B$", ha="center", va="top",
        fontsize=4.5, color=CM, fontweight="bold")
ax.plot([GX["em"]+0.18, X1], [YL, YL], color=CW, lw=0.8,
        linestyle="dotted", alpha=0.28, zorder=1)
cl_arrow(ax, GX["em"]+0.18, GX["zc"]-0.18, YL+0.08, YD-0.08,
         r"$m_B$", rad=-0.35, lbl_dy=0.24)
box(ax, GX["zc"], YD, "Z?", CR, w=0.28, h=0.22, fs=5.0)
ax.text(GX["zc"], YD-0.20, r"$m_B{=}1$", ha="center", va="top",
        fontsize=3.8, color=CR)

# ── phase bracket labels ───────────────────────────────────────────────────────
YTOP = 1.75
for x0b, x1b, lbl, c in [
    (0.44, 0.96, "ebit",             CE),
    (1.00, 2.65, "starting_process", CA),
    (2.70, 3.40, "local",            CG),
    (3.44, 5.10, "ending_process",   CB),
]:
    ax.plot([x0b, x1b], [YTOP, YTOP], color=c, lw=0.8, zorder=2)
    ax.plot([x0b, x0b], [YTOP-0.03, YTOP], color=c, lw=0.8, zorder=2)
    ax.plot([x1b, x1b], [YTOP-0.03, YTOP], color=c, lw=0.8, zorder=2)
    ax.text((x0b+x1b)/2, YTOP+0.02, lbl, ha="center", va="bottom",
            fontsize=4.2, color=c, fontweight="bold")

# ── legend ────────────────────────────────────────────────────────────────────
items = [
    mpatches.Patch(color=CE, label="ebit"),
    mpatches.Patch(color=CA, label="starting_process"),
    mpatches.Patch(color=CB, label="ending_process"),
    mpatches.Patch(color=CG, label="local gate"),
    mpatches.Patch(color=CM, label="measurement"),
    mpatches.Patch(color="#c62828", label="Pauli corr."),
    mpatches.Patch(color=CC, label="classical"),
]
ax.legend(handles=items, loc="lower center",
          bbox_to_anchor=(0.5, -0.04), ncol=4,
          fontsize=4.0, framealpha=0.95, edgecolor="#bbb",
          borderpad=0.25, labelspacing=0.12, handlelength=0.7,
          handletextpad=0.25, columnspacing=0.4)

plt.tight_layout(pad=0.15)
out = os.path.join(os.path.dirname(__file__), "starting_process_circuit.pdf")
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
