"""Runtime breakdown as a stacked horizontal bar chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = "./"
SANS = ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Nimbus Sans",
        "Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": SANS,
    "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.6, "legend.fontsize": 9,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 0,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

methods = ["ALBL", "UCB", "SCRiBLe", "SelectAL", "AutoAL", "FractAL"]
sampling = [1.2, 1.5, 8.8, 3.3, 20.7, 8.8]
reward = [7.3e-5, 7.2e-5, 4.6e-4, 19.8, 28.5, 0.5]
total = [1.2, 1.5, 8.8, 23.1, 49.2, 9.3]

C_SAMP = "#7B8794"
C_REW = "#C0392B"
C_UNSPLIT = "#C8CED6"

y = np.arange(len(methods))[::-1]          # top-to-bottom in listed order

fig, ax = plt.subplots(figsize=(3.3, 1.95))

for i, m in enumerate(methods):
    yy = y[i]
    if np.isnan(sampling[i]):              # AutoAL: no separable phases
        ax.barh(yy, total[i], height=0.62, color=C_UNSPLIT,
                edgecolor="white", linewidth=0.5, hatch="///", zorder=2)
    else:
        ax.barh(yy, sampling[i], height=0.62, color=C_SAMP,
                edgecolor="white", linewidth=0.5, zorder=2)
        ax.barh(yy, reward[i], left=sampling[i], height=0.62, color=C_REW,
                edgecolor="white", linewidth=0.5, zorder=2)
#     ax.text(total[i] + 0.7, yy, f"{total[i]:.1f}", va="center", ha="left",
        #     fontsize=6.5, color="#1A1A1A")

ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.set_xlabel("Runtime per AL run (min)")
ax.set_xlim(0, 52)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#E2E2E2", lw=0.5)

handles = [Patch(facecolor=C_SAMP, label="Acquisition"),
           Patch(facecolor=C_REW, label="Allocation Update")]
        #    Patch(facecolor=C_UNSPLIT, hatch="///", label="Not separable")]
ax.legend(handles=handles, frameon=False, loc="upper right",
          handlelength=1.1, handletextpad=0.4, labelspacing=0.25,
          borderaxespad=0.3)

fig.savefig(f"{OUT}/fig_runtime.pdf")
fig.savefig(f"{OUT}/fig_runtime.png", dpi=300)
plt.close(fig)
print("ok")