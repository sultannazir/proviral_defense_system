"""
Plot the frequency distribution of `alpha` as it evolves over time.

Data format assumed (as in alpha_distributions_1_provirus_1.csv):
    - First column: "time"
    - Remaining columns: bin-center values of alpha (e.g. 0.005, 0.015, ...)
    - Each row: the frequency (probability) of alpha falling in that bin,
      at the given time point. Rows sum to ~1.

Layout:
    - Left: large heatmap of the TOTAL (provirus + virion) alpha distribution,
      normalised per timepoint (row) so the plot shows distribution *shape*.
    - Right (top/bottom): smaller heatmaps decomposing that same total into
      the provirus-only and virion-only contributions. These are NOT
      independently normalised -- they are scaled by the exact same
      per-timepoint factor as the main plot, so at every timepoint
      (provirus panel + virion panel) sums back to the main panel.
    - All three panels share one colourbar (same vmin/vmax).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load data (keep provirus and virion frequencies separate)
# ------------------------------------------------------------------
csv_path = "alpha_distributions_6_provirus.csv"
df = pd.read_csv(csv_path)

time = df["time"].values
alpha_bins = df.columns[1:].astype(float).values      # bin centers
freqp = df.iloc[:, 1:].values                           # provirus, shape: (n_times, n_bins)

csv_path = "alpha_distributions_6_virion.csv"
df = pd.read_csv(csv_path)

freqv = df.iloc[:, 1:].values                           # virion, shape: (n_times, n_bins)
freq = freqv + freqp                                     # total

# ------------------------------------------------------------------
# 1b. Rebin alpha from its native resolution (~0.01) to a coarser 0.025
#     resolution by summing frequencies within each new, wider bin.
#     Apply identically to provirus, virion and total so they stay consistent.
# ------------------------------------------------------------------
new_bin_width = 0.025
orig_bin_width = np.diff(alpha_bins).mean()
group_size = int(round(new_bin_width / orig_bin_width))

n_bins = len(alpha_bins)
n_groups = int(np.ceil(n_bins / group_size))
pad = n_groups * group_size - n_bins


def rebin(arr):
    if pad > 0:
        arr_padded = np.concatenate([arr, np.zeros((arr.shape[0], pad))], axis=1)
    else:
        arr_padded = arr
    return arr_padded.reshape(arr.shape[0], n_groups, group_size).sum(axis=2)


if pad > 0:
    alpha_bins_padded = np.concatenate([alpha_bins, np.full(pad, np.nan)])
else:
    alpha_bins_padded = alpha_bins

freqp = rebin(freqp)
freqv = rebin(freqv)
freq = rebin(freq)

# New bin centers: mean of the original centers in each group (ignoring padding)
alpha_bins = np.nanmean(alpha_bins_padded.reshape(n_groups, group_size), axis=1)

# ------------------------------------------------------------------
# 1c. Normalize each timepoint (row) of the TOTAL by its own maximum, so the
#     main plot shows the *shape* of the alpha distribution at each time,
#     rather than being dominated by timepoints with higher absolute
#     frequencies. Apply the SAME per-row scale factor to the provirus and
#     virion arrays, so that at every timepoint:
#         freqp_norm + freqv_norm == freq_norm
# ------------------------------------------------------------------
row_max = freq.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1.0  # avoid divide-by-zero for any all-zero rows

freq = freq / row_max
freqp = freqp / row_max
freqv = freqv / row_max

# ------------------------------------------------------------------
# 2. Heatmaps: time (x) vs alpha (y), color = frequency
#    Left: total. Right (stacked): provirus, virion. Shared colourbar.
# ------------------------------------------------------------------
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 0.06], wspace=0.35, hspace=0.4)

ax_main = fig.add_subplot(gs[:, 0])
ax_pro = fig.add_subplot(gs[0, 1])
ax_vir = fig.add_subplot(gs[1, 1], sharex=ax_pro, sharey=ax_pro)
cax = fig.add_subplot(gs[:, 2])

# Use pcolormesh so bins are placed correctly (need edges, not just centers)
bin_width = np.diff(alpha_bins).mean()
alpha_edges = np.concatenate([alpha_bins - bin_width / 2,
                               [alpha_bins[-1] + bin_width / 2]])

# For time, since spacing may be uneven (e.g. finer near start), build edges
# from the actual time midpoints so each column is drawn at correct width.
time_edges = np.concatenate([
    [time[0] - (time[1] - time[0]) / 2],
    (time[:-1] + time[1:]) / 2,
    [time[-1] + (time[-1] - time[-2]) / 2]
])

# Shared colour scale across all three panels
vmin, vmax = 0, 0.2
cmap = "viridis"

mesh_main = ax_main.pcolormesh(
    time_edges, alpha_edges, freq.T,
    shading="flat", cmap=cmap, vmin=vmin, vmax=vmax
)
ax_main.set_xlabel("Time", fontsize=12)
ax_main.set_xlim(0, 200000)
ax_main.set_xticks([0,50000,100000,150000,200000])
ax_main.set_ylabel(r"$\alpha$", fontsize=14)
ax_main.set_title(r"Total frequency distribution of $\alpha$ over time", fontsize=12)

ax_pro.pcolormesh(
    time_edges, alpha_edges, freqp.T,
    shading="flat", cmap=cmap, vmin=vmin, vmax=vmax
)
ax_pro.set_xlim(0, 200000)
ax_pro.set_ylabel(r"$\alpha$", fontsize=12)
ax_pro.set_title("Provirus contribution", fontsize=11)
ax_pro.tick_params(labelbottom=False)

ax_vir.pcolormesh(
    time_edges, alpha_edges, freqv.T,
    shading="flat", cmap=cmap, vmin=vmin, vmax=vmax
)
ax_vir.set_xlim(0, 200000)
ax_vir.set_xlabel("Time", fontsize=12)
ax_vir.set_xticks([0,100000,200000])
ax_vir.set_ylabel(r"$\alpha$", fontsize=12)
ax_vir.set_title("Virion contribution", fontsize=11)

cbar = fig.colorbar(mesh_main, cax=cax)
cbar.set_label("Frequency (normalised per timepoint)", fontsize=12)
cbar.ax.locator_params(nbins=4)

fig.tight_layout()
# plt.show()
plt.savefig('branching_factory.png', dpi=400)