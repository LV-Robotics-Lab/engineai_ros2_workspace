from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# 与 fig2-1 / plot_grouped_risk_bars 一致：PNG 写入仓库根目录 OutputFigures/（.gitignore）
_out_dir = Path(__file__).resolve().parent.parent / "OutputFigures"
_out_dir.mkdir(parents=True, exist_ok=True)

def add_box(ax, x, y, w, h, title, subtitle=None, lw=1.5):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=lw))
    ax.text(x+0.01*w, y+h-0.04*h, title, va="top", ha="left", fontsize=11, weight="bold")
    if subtitle:
        ax.text(x+0.02*w, y+h-0.14*h, subtitle, va="top", ha="left", fontsize=9)

def grouped_bar(ax, cats, series, labels, title="", ylabel="Normalized value"):
    x = np.arange(len(cats))
    n = len(series)
    width = 0.75 / n
    offsets = (np.arange(n) - (n-1)/2) * width
    for vals, lab, off in zip(series, labels, offsets):
        ax.bar(x + off, vals, width=width, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="best")

def matrix(ax, data, rows, cols, title=""):
    im = ax.imshow(data, aspect='auto')
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i,j] > np.mean(data) else "black")
    return im

def save_fig(fig, name):
    path = str(_out_dir / name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

# Updated Figure 1
fig = plt.figure(figsize=(13,9))
fig.suptitle("Updated Figure 1 mockup — Problem framing and framework overview", fontsize=16, y=0.98, weight="bold")
# a
axa = fig.add_axes([0.05,0.55,0.28,0.30]); axa.set_axis_off()
add_box(axa,0,0,1,1,"(a) Robot with physical protectors")
for x in [0.20,0.55]:
    axa.add_patch(Circle((x+0.08,0.72),0.06, fill=False))
    axa.add_patch(Rectangle((x+0.02,0.42),0.12,0.22, fill=False))
    axa.add_patch(Rectangle((x+0.04,0.20),0.08,0.20, fill=False))
    # highlight protected zones
    axa.add_patch(Rectangle((x+0.015,0.46),0.13,0.06, fill=False, linewidth=2))
    axa.add_patch(Rectangle((x+0.03,0.25),0.10,0.07, fill=False, linewidth=2))
axa.text(0.5,0.08,"photo / render placeholder", ha="center", fontsize=9)
axa.set_xlim(0,1); axa.set_ylim(0,1)
# b
axb = fig.add_axes([0.38,0.55,0.26,0.30]); axb.set_axis_off()
add_box(axb,0,0,1,1,"(b) Multi-risk hardware consequences of a fall")
# one fall sketch
axb.plot([0.18,0.32,0.46],[0.25,0.55,0.25], lw=1.5)
# risk icons
risks = [("Force",0.18,0.75),("Joint",0.50,0.72),("Back-EMF",0.18,0.12),("Accel",0.55,0.15)]
for lab,x,y in risks:
    axb.add_patch(Rectangle((x,y),0.22,0.12, fill=False))
    axb.text(x+0.11,y+0.06,lab,ha="center",va="center",fontsize=8)
axb.set_xlim(0,1); axb.set_ylim(0,1)
# c
axc = fig.add_axes([0.69,0.55,0.26,0.30]); axc.set_axis_off()
add_box(axc,0,0,1,1,"(c) Active vs passive along fall timeline")
xs = [0.08,0.42,0.74]
labs = ["Pre-impact","Contact","After-contact"]
for x,lab in zip(xs,labs):
    axc.add_patch(Rectangle((x,0.42),0.16,0.16, fill=False))
    axc.text(x+0.08,0.63,lab,ha="center",fontsize=8)
axc.add_patch(FancyArrowPatch((0.24,0.50),(0.42,0.50),arrowstyle="->",mutation_scale=12))
axc.add_patch(FancyArrowPatch((0.58,0.50),(0.74,0.50),arrowstyle="->",mutation_scale=12))
axc.text(0.16,0.24,"learned policy\nshapes impact", ha="center", fontsize=8)
axc.text(0.50,0.24,"protector\nmitigates response", ha="center", fontsize=8)
axc.set_xlim(0,1); axc.set_ylim(0,1)
# d
axd = fig.add_axes([0.18,0.12,0.64,0.24]); axd.set_axis_off()
add_box(axd,0,0,1,1,"(d) Unified framework statement")
for x,lab in [(0.14,"same\nfall event"),(0.45,"same multi-risk\nobjective"),(0.76,"shared design\nlogic")]:
    axd.add_patch(Rectangle((x-0.10,0.38),0.20,0.24, fill=False))
    axd.text(x,0.50,lab,ha="center",va="center",fontsize=10)
axd.add_patch(FancyArrowPatch((0.24,0.50),(0.35,0.50),arrowstyle="->",mutation_scale=14))
axd.add_patch(FancyArrowPatch((0.55,0.50),(0.66,0.50),arrowstyle="->",mutation_scale=14))
axd.set_xlim(0,1); axd.set_ylim(0,1)
p1 = save_fig(fig, "figure1_mockup_updated.png")

# Updated Figure 3
fig = plt.figure(figsize=(13,9))
fig.suptitle("Updated Figure 3 mockup — Passive protection mitigates scenario-conditioned multi-risk responses", fontsize=16, y=0.98, weight="bold")
axa = fig.add_axes([0.05,0.58,0.22,0.28]); axa.set_axis_off()
add_box(axa,0,0,1,1,"(a) Representative fall scenarios")
names = ["Power loss","Trip","Slip","Side fall"]
positions = [(0.10,0.58),(0.56,0.58),(0.10,0.18),(0.56,0.18)]
for (x,y), name in zip(positions, names):
    axa.add_patch(Rectangle((x,y),0.24,0.18, fill=False))
    axa.plot([x+0.05,x+0.12,x+0.18],[y+0.03,y+0.12,y+0.03], lw=1.3)
    axa.text(x+0.12, y-0.03, name, ha="center", va="top", fontsize=8)
axa.set_xlim(0,1); axa.set_ylim(0,1)
axb = fig.add_axes([0.32,0.58,0.63,0.28])
grouped_bar(axb, ["Fwd","Back","Left","Right"],
            [[0.95,0.88,0.91,0.86],[0.63,0.59,0.61,0.57]],
            ["No padding","With padding"],
            title="(b) Peak contact force distributions", ylabel="Normalized peak force")
axc = fig.add_axes([0.05,0.12,0.52,0.34])
data = np.array([[0.68,0.72,0.66,0.64],
                 [0.74,0.69,0.63,0.65],
                 [0.81,0.78,0.70,0.71],
                 [0.76,0.74,0.68,0.67]])
matrix(axc, data, ["Force","Wrench","Back-EMF","Accel"], ["Fwd","Back","Left","Right"],
       title="(c) Multi-risk response matrix\n(with-pad / no-pad ratio)")
axd = fig.add_axes([0.64,0.12,0.28,0.34])
grouped_bar(axd, ["Fwd","Back","Left","Right"],
            [[0.89,0.84,0.80,0.82],[0.60,0.57,0.55,0.56]],
            ["No padding","With padding"],
            title="(d) Integrated overall risk", ylabel="Integrated risk")
p3 = save_fig(fig, "figure3_mockup_updated.png")

# Updated Figure 4
fig = plt.figure(figsize=(13,9))
fig.suptitle("Updated Figure 4 mockup — Human-inspired active falling narrows the preferred protector distribution", fontsize=16, y=0.98, weight="bold")
axa = fig.add_axes([0.05,0.58,0.26,0.28])
grouped_bar(axa, ["Impact vel","Peak force","Contact order"],
            [[0.78,0.80,0.62],[0.52,0.56,0.86]],
            ["Baseline human","Protective human"],
            title="(a) Why mimic human trajectories", ylabel="Normalized")
axb = fig.add_axes([0.37,0.58,0.26,0.28])
grouped_bar(axb, ["Hand","Forearm","Knee","Torso"],
            [[0.40,0.18,0.15,0.27],[0.24,0.32,0.28,0.16]],
            ["Passive / PD","Mimic"],
            title="(b) First-contact statistics", ylabel="Fraction")
axc = fig.add_axes([0.69,0.58,0.26,0.28])
grouped_bar(axc, ["Arm","Leg","Torso","Head"],
            [[0.22,0.18,0.52,0.08],[0.34,0.26,0.34,0.06]],
            ["Passive / PD","Mimic"],
            title="(c) Regional impulse share", ylabel="Share")
axd = fig.add_axes([0.10,0.12,0.80,0.32]); axd.set_axis_off()
add_box(axd,0,0,1,1,"(d) Protector design implication")
cols = [0.18,0.50,0.82]
titles = ["Passive / PD","Mimic","Preferred padding"]
topnotes = ["broader hotspots","narrower hotspots","focused coverage"]
for cx, title, note in zip(cols, titles, topnotes):
    axd.add_patch(Circle((cx,0.68),0.05, fill=False))
    axd.add_patch(Rectangle((cx-0.06,0.42),0.12,0.18, fill=False))
    axd.add_patch(Rectangle((cx-0.03,0.24),0.06,0.16, fill=False))
    if title != "Preferred padding":
        axd.add_patch(Rectangle((cx-0.10,0.36),0.20,0.28, fill=False, linestyle="--"))
    else:
        axd.add_patch(Rectangle((cx-0.07,0.41),0.14,0.14, fill=False, linewidth=2))
        axd.add_patch(Rectangle((cx-0.05,0.26),0.10,0.08, fill=False, linewidth=2))
    axd.text(cx,0.10,title,ha="center",fontsize=9)
    axd.text(cx,0.86,note,ha="center",fontsize=8)
axd.add_patch(FancyArrowPatch((0.28,0.55),(0.40,0.55),arrowstyle="->",mutation_scale=14))
axd.add_patch(FancyArrowPatch((0.60,0.55),(0.72,0.55),arrowstyle="->",mutation_scale=14))
axd.text(0.34,0.63,"mimic makes hotspot\nstatistics more concentrated", ha="center", fontsize=9)
axd.text(0.66,0.63,"design can shrink and\nfocus protector layout", ha="center", fontsize=9)
axd.set_xlim(0,1); axd.set_ylim(0,1)
p4 = save_fig(fig, "figure4_mockup_updated.png")

# Updated Figure 5
fig = plt.figure(figsize=(13,9))
fig.suptitle("Updated Figure 5 mockup — Complementary multi-risk reduction by active + passive protection", fontsize=16, y=0.98, weight="bold")
axa = fig.add_axes([0.05,0.58,0.22,0.28]); axa.set_axis_off()
add_box(axa,0,0,1,1,"(a) Experimental groups")
groups = ["None","Passive","Active","A+P"]
for i,g in enumerate(groups):
    x = 0.10 + (i%2)*0.42
    y = 0.56 if i<2 else 0.18
    axa.add_patch(Rectangle((x,y),0.22,0.16, fill=False))
    axa.text(x+0.11,y+0.08,g,ha="center",va="center",fontsize=9)
axa.set_xlim(0,1); axa.set_ylim(0,1)
axb = fig.add_axes([0.32,0.58,0.63,0.28])
x = np.arange(3); width = 0.18
vals = np.array([
    [1.00,0.72,0.70,0.48],
    [1.00,0.83,0.74,0.58],
    [1.00,0.88,0.69,0.51],
])
labels = ["Collision","Joint/struct.","Motor"]
for k,g in enumerate(groups):
    axb.bar(x + (k-1.5)*width, vals[:,k], width=width, label=g)
axb.set_xticks(x)
axb.set_xticklabels(labels, fontsize=9)
axb.set_ylabel("Normalized risk", fontsize=9)
axb.set_title("(b) Channel-wise risk comparison", fontsize=10)
axb.legend(frameon=False, fontsize=8)
axc = fig.add_axes([0.07,0.12,0.34,0.32])
grouped_bar(axc, groups,
            [[1.0,0.78,0.71,0.49]],
            ["Integrated risk"],
            title="(c) Integrated overall risk", ylabel="Integrated risk")
axd = fig.add_axes([0.48,0.12,0.44,0.32]); axd.set_axis_off()
add_box(axd,0,0,1,1,"(d) Representative case study")
xs = [0.08,0.32,0.56,0.80]
labs = ["Pre-impact","First contact","Load transfer","Settle"]
caps = ["Active policy\nshapes posture","Passive pad\nabsorbs impact","Reduced joint /\nmotor stress","Lower total risk"]
for x0,lab,cap in zip(xs,labs,caps):
    axd.add_patch(Rectangle((x0,0.46),0.13,0.22, fill=False))
    axd.text(x0+0.065,0.72,lab,ha="center",fontsize=8)
    axd.text(x0+0.065,0.28,cap,ha="center",fontsize=8)
for i in range(3):
    axd.add_patch(FancyArrowPatch((xs[i]+0.13,0.57),(xs[i+1],0.57),arrowstyle="->",mutation_scale=13))
axd.set_xlim(0,1); axd.set_ylim(0,1)
p5 = save_fig(fig, "figure5_mockup_updated.png")

# Updated Figure 6
fig = plt.figure(figsize=(13,9))
fig.suptitle("Updated Figure 6 mockup — Robustness and system-level validation across diverse falls", fontsize=16, y=0.98, weight="bold")
axa = fig.add_axes([0.05,0.58,0.28,0.28])
grouped_bar(axa, ["Fwd","Back","Left","Right","Trip"],
            [[0.92,0.88,0.84,0.86,0.82],[0.70,0.65,0.62,0.64,0.58]],
            ["Single mechanism","Active + passive"],
            title="(a) Diverse disturbance directions", ylabel="Normalized overall risk")
axb = fig.add_axes([0.38,0.58,0.26,0.28])
grouped_bar(axb, ["Low","Mid","High"],
            [[0.78,0.92,1.12],[0.55,0.66,0.82]],
            ["Single mechanism","Active + passive"],
            title="(b) High-energy / severe-impact tests", ylabel="Risk / failure proxy")
axc = fig.add_axes([0.69,0.58,0.26,0.28]); axc.set_axis_off()
add_box(axc,0,0,1,1,"(c) Real-robot demonstrations")
for i,x in enumerate([0.08,0.39,0.70]):
    axc.add_patch(Rectangle((x,0.30),0.20,0.32, fill=False))
    axc.text(x+0.10,0.22,f"frame {i+1}", ha="center", fontsize=8)
axc.text(0.50,0.10,"indoor / outdoor / dynamic motion", ha="center", fontsize=9)
axc.set_xlim(0,1); axc.set_ylim(0,1)
axd = fig.add_axes([0.18,0.12,0.64,0.28])
grouped_bar(axd, ["Integrity","Survival","Low-risk","Deployment"],
            [[0.74,0.78,0.70,0.66],[0.91,0.94,0.90,0.88]],
            ["Single mechanism","Active + passive"],
            title="(d) System-level robustness summary", ylabel="Higher is better")
p6 = save_fig(fig, "figure6_mockup_updated.png")

print("\n".join([p1,p3,p4,p5,p6]))