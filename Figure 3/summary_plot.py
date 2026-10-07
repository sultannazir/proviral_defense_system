import json
import numpy as np
import matplotlib.pyplot as plt

# Parameters

plt.rcParams.update({'font.size': 14})

cA_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
seeds = range(50)

# Histogram settings

bin_width = 0.01
bins = np.arange(0, 1 + bin_width, bin_width)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
fig, ax = plt.subplots(figsize=(7, 5), dpi=400)

all_predictions_x = []
all_predictions_y = []
all_prediction_colors = []

# Controls max histogram width around each x location

max_half_width = 0.035

for cA in cA_values:
    for seed in seeds:
        fname = f"sweep_without_mutation/cA{cA}GT0GR0seed{seed+1}.txt"
        with open(fname, "r") as f:
            lines = [json.loads(x) for x in f.readlines()]
        if len(lines) < 100:
            print(f"Skipping {fname}: length = {len(lines)}")
            continue
        last_data = lines[90:100]
        # Shape: (timepoints, alpha bins)
        distsF = np.array([np.array(x["free_virus"]) for x in last_data])
        distsP = np.array([np.array(x["provirus"]) for x in last_data])
        pvals = np.array([
            x["numP"] / (x["numP"] + x["numS"])
            for x in last_data
        ])

        mean_pval = pvals.mean()
        # Original alpha grid
        alpha_bins_fine = np.arange(0, 1, 0.01)
        alpha_centers_fine = alpha_bins_fine + 0.005

        # Average over time
        mean_distF = distsF.mean(axis=0)
        mean_distP = distsP.mean(axis=0)

        # Reconstruct samples

        countsF = np.round(mean_distF * 5000).astype(int)
        countsP = np.round(mean_distP * 5000).astype(int)

        samplesF = np.repeat(alpha_centers_fine, countsF)
        samplesP = np.repeat(alpha_centers_fine, countsP)

        # Histogram

        histF, _ = np.histogram(samplesF, bins=bins, density=False)
        histP, _ = np.histogram(samplesP, bins=bins, density=False)

        # Normalize widths for plotting
        histmax = np.max(histF + histP)
        histF = histF / histmax * max_half_width
        histP = histP / histmax * max_half_width

        # color = seed_colors[seed]
        colorF = 'red'
        colorP = 'blue'

        # Draw horizontal bars centered at cA

        for y, h in zip(bin_centers, histF):

            ax.fill_betweenx(
                [y - bin_width/2, y + bin_width/2],
                cA - h,
                cA + h,
                color=colorF,
                alpha=0.05,
                linewidth=0
            )
        for y, h in zip(bin_centers, histP):

            ax.fill_betweenx(
                [y - bin_width/2, y + bin_width/2],
                cA - h,
                cA + h,
                color=colorP,
                alpha=0.05,
                linewidth=0
            )

        # Prediction point

        pred = 1 / (cA * (2 + mean_pval))
        all_predictions_x.append(cA)
        all_predictions_y.append(pred)


# Average predictions for each c, then plot with a dotted connecting line

all_predictions_x = np.array(all_predictions_x)
all_predictions_y = np.array(all_predictions_y)

mean_x = []
mean_y = []
for cA in cA_values:
    mask = all_predictions_x == cA
    if mask.any():
        mean_x.append(cA)
        mean_y.append(all_predictions_y[mask].mean())

ax.plot(
    mean_x,
    mean_y,
    color="black",
    linestyle=":",
    linewidth=1.5,
    marker="o",
    markersize=5,
    markeredgecolor="white",
    markeredgewidth=1,
    label=r"$\frac{1}{(2+p^*)c}$"
)

# Theory curves

xs = np.linspace(0.45, 1.05, 300)
ysmax = 1 / (2 * xs)
ysmin = 1 / (3 * xs)
omega = 4*((5/4)**(1/20) - 1)/0.25
amin = (1 - np.sqrt(1 - omega*xs)) / (2*xs)
amax = (1 + np.sqrt(1 - omega*xs)) / (2*xs)

ax.plot(xs, amin, color="green", linestyle="dashed")
ax.plot(xs, amax, color="green", linestyle="dashed")

# Labels

ax.set_xlabel(r"Cost of cooperation, $c$")
ax.set_ylabel(r"Cooperation, $\alpha$")
ax.set_title(

   "Local Reproduction and Local Transmission"

)

ax.set_xticks(cA_values)
ax.set_xlim(0.45, 1.05)
ax.set_ylim(0, 1)

# ax.legend(loc='upper right', bbox_to_anchor=(0.5,0.4))

plt.tight_layout()
# plt.show()
plt.savefig("summary_plot_unmix.png", dpi=400)
