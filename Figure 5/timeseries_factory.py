import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from timeseries_overlay import normalise_columns, to_rgba, COL_F, COL_P  # helpers from earlier module

plt.rcParams.update({'font.size': 14})

# ---- Settings -------------------------------------------------------------
VIRION_CSV   = "alpha_distributions_6_virion.csv"     # free virus
PROVIRUS_CSV = "alpha_distributions_6_provirus.csv"
OPACITY = 0.7    # max opacity of each layer (single replicate, so no need to go very low)
# ---------------------------------------------------------------------------

def load(path):
    df = pd.read_csv(path)
    t = df["time"].to_numpy()
    alpha_centers = df.columns[1:].astype(float).to_numpy()
    M = df.iloc[:, 1:].to_numpy(dtype=float).T          # (alpha bins, timepoints)
    return t, alpha_centers, M

t, a, F = load(VIRION_CSV)
t2, a2, P = load(PROVIRUS_CSV)
assert np.array_equal(t, t2) and np.allclose(a, a2), "Time or alpha grids differ between files"

# Independent normalisation of each distribution at every timepoint
F = normalise_columns(F)
P = normalise_columns(P)

dt, da = t[1] - t[0], a[1] - a[0]
extent = [t[0] - dt/2, t[-1] + dt/2, a[0] - da/2, a[-1] + da/2]
kw = dict(origin="lower", aspect="auto", extent=extent, interpolation="nearest")

# Layout: main axis + two thin colourbar axes
fig = plt.figure(figsize=(10, 5))#, dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[40, 1.2, 1.2], wspace=0.3)
ax, caxF, caxP = (fig.add_subplot(gs[0, i]) for i in range(3))

ax.imshow(to_rgba(F, COL_F, OPACITY), **kw)
ax.imshow(to_rgba(P, COL_P, OPACITY), **kw)
ax.set_xlabel("Time")
ax.set_ylabel(r"Cooperation, $\alpha$")
ax.set_ylim(0, 1)
ax.set_title("Free virus (red) and provirus (blue)")

# Colourbars reproduce how each layer looks over white (white -> colour at OPACITY)
def layer_cmap(rgb):
    end = (1 - OPACITY) + OPACITY * np.array(rgb)
    return LinearSegmentedColormap.from_list("layer", [(1, 1, 1), tuple(end)])

norm = Normalize(0, 1)
cbF = fig.colorbar(ScalarMappable(norm, layer_cmap(COL_F)), cax=caxF)
cbP = fig.colorbar(ScalarMappable(norm, layer_cmap(COL_P)), cax=caxP)
for cb, name in ((cbF, "Free\nvirus"), (cbP, "Provirus")):
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=10)
    cb.ax.set_title(name, fontsize=11, pad=6, loc="center")
cbF.ax.set_yticklabels([])                       # tick labels only on the right-hand bar
cbF.ax.tick_params(length=0)
cbP.set_label("Frequency normalised at every timepoint", fontsize=11)

plt.show()
# plt.savefig("single_replicate_timeseries.png", dpi=300, bbox_inches="tight")