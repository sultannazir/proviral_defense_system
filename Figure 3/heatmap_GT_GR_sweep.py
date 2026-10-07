import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---- CONFIG ----
DATA_PATH = "GT_GR_sweep.txt"
OUTPUT_PATH = "gt_gr_heatmap.png"

TIME_MIN = 90000
TIME_MAX = 100000
ALPHA_THRESHOLD = 0.1     # "alpha values below 0.1"
FRACTION_THRESHOLD = 0.2  # "at least 20% of all viruses"

tick_labels = [-2.00, -1.75, -1.50, -1.25, -1.00, -0.75, -0.50, -0.25, 0.00]

def load_rows(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---- AGGREGATE: sum provirus + free_virus histograms per (GT, GR, seed), across
# timepoints in [TIME_MIN, TIME_MAX], to get each replicate's average/combined
# alpha distribution over that time window. Provirus histograms are also summed
# separately, since the condition below only cares about proviruses with low alpha. ----
# key: (GT, GR, seed) -> running sum of (provirus + free_virus) histogram, and count of timepoints
sums = defaultdict(lambda: None)
provirus_sums = defaultdict(lambda: None)
counts = defaultdict(int)
n_bins = None
bin_centers = None

for row in load_rows(DATA_PATH):
    t = row["time"]
    if t < TIME_MIN or t > TIME_MAX:
        continue

    gt = row["GT"]
    gr = row["GR"]
    seed = row["seed"]

    provirus = np.array(row["provirus"], dtype=float)
    free_virus = np.array(row["free_virus"], dtype=float)  # "virion"
    combined = provirus + free_virus

    if n_bins is None:
        n_bins = len(combined)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    key = (gt, gr, seed)
    if sums[key] is None:
        sums[key] = np.zeros(n_bins)
        provirus_sums[key] = np.zeros(n_bins)
    sums[key] += combined
    provirus_sums[key] += provirus
    counts[key] += 1

# ---- For each (GT, GR, seed), compute the *average* alpha distribution over the
# time window, then check whether >= 20% of ALL viruses (provirus + free_virus)
# are proviruses with alpha < 0.1 ----
low_alpha_mask = bin_centers < ALPHA_THRESHOLD

gt_values = sorted(set(k[0] for k in sums.keys()))
gr_values = sorted(set(k[1] for k in sums.keys()))

# key: (GT, GR) -> [bool per seed, whether condition met]
condition_by_cell = defaultdict(list)

for (gt, gr, seed), total_sum in sums.items():
    n_t = counts[(gt, gr, seed)]
    avg_dist = total_sum / n_t              # average combined histogram over timepoints in window
    avg_provirus_dist = provirus_sums[(gt, gr, seed)] / n_t  # average provirus-only histogram

    total_count = avg_dist.sum()
    if total_count == 0:
        continue  # no viruses present in this window for this replicate; skip

    fraction_low_alpha_provirus = avg_provirus_dist[low_alpha_mask].sum() / total_count
    condition_by_cell[(gt, gr)].append(fraction_low_alpha_provirus >= FRACTION_THRESHOLD)

# ---- Build heatmap: fraction of replicates (seeds) meeting the condition, per (GT, GR) ----
heatmap = np.full((len(gt_values), len(gr_values)), np.nan)
for i, gt in enumerate(gt_values):
    for j, gr in enumerate(gr_values):
        flags = condition_by_cell.get((gt, gr), [])
        if flags:
            heatmap[i, j] = np.mean(flags)

# ---- PLOT ----
fig, ax = plt.subplots(figsize=(4, 5))
im = ax.imshow(heatmap, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")

ax.set_xticks([0,4,8])
ax.set_xticklabels([0.01,0.1,1], rotation=45, ha="right")
ax.set_yticks([0,4,8])
ax.set_yticklabels([0.01,0.1,1])
ax.set_xlabel(r"Global host reproduction, $G_R$ (log-scaled)")
ax.set_ylabel(r"Global virus transmission, $G_T$ (log-scaled)")
ax.set_title(
    f"Fraction of runs where defense evolved"
)

# for i in range(len(gt_values)):
#     for j in range(len(gr_values)):
#         val = heatmap[i, j]
#         text = "N/A" if np.isnan(val) else f"{val:.2f}"
#         color = "white" if (not np.isnan(val) and val < 0.5) else "black"
#         ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

cbar = fig.colorbar(im, ax=ax)
# cbar.set_label("Fraction of replicates")

fig.tight_layout()
# plt.show()
fig.savefig(OUTPUT_PATH, dpi=400)
print(f"Saved plot to {OUTPUT_PATH}")