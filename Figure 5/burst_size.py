import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Parameters
r = 0.25
c = 0.5
alpha_wt = 0.5      # alpha of the focal free virus (always starts factory)
T_L = 20
n_replicates = 1000


def simulate_factory(e, provirus_alpha, seed=None):
    """
    Simulate one replicate of the virus factory.

    e: probability per timestep that the provirus joins the factory (0 = no provirus)
    provirus_alpha: alpha value of the provirus (0.5 or 0.0); ignored if e == 0
    Returns: (n_wt, n_provirus) final counts, where 'wt' = alpha 0.5 virus
    """
    rng = np.random.default_rng(seed)
    n_wt = 1          # start with one free virus, alpha = 0.5
    n_provirus = 0
    entered = False

    for t in range(T_L):
        # Provirus entry into the factory
        if e > 0 and not entered:
            if rng.random() < e:
                n_provirus += 1
                entered = True

        total = n_wt + n_provirus
        # Weighted mean alpha currently in the factory (the shared public good)
        if provirus_alpha is None:
            alpha_avg = alpha_wt
        else:
            alpha_avg = (n_wt * alpha_wt + n_provirus * provirus_alpha) / total

        # Cost term uses each type's own alpha; benefit term uses the shared alpha_avg
        prob_wt = r * (1 - c * alpha_wt) * alpha_avg
        n_wt += rng.binomial(n_wt, prob_wt)

        if n_provirus > 0:
            prob_provirus = r * (1 - c * provirus_alpha) * alpha_avg
            n_provirus += rng.binomial(n_provirus, prob_provirus)

    return n_wt, n_provirus


def run_scenario(e, provirus_alpha, n_replicates=1000):
    """
    Returns per-replicate counts split by ORIGIN (free virus lineage vs.
    provirus lineage), regardless of each lineage's alpha value.
    """
    results = [simulate_factory(e, provirus_alpha, seed=i) for i in range(n_replicates)]
    n_free_arr = np.array([res[0] for res in results])      # descendants of the focal free virus
    n_provirus_arr = np.array([res[1] for res in results])  # descendants of the provirus

    return n_free_arr, n_provirus_arr


# Scenarios: (e, provirus_alpha). provirus_alpha=None means no provirus at all.
scenarios = {
    'No\nprovirus':            (0.0,  None),
    'e=1\n$\\alpha_p$=0.5':    (1.0,  0.5),
    'e=0.01\n$\\alpha_p$=0.5': (0.01, 0.5),
    'e=1\n$\\alpha_p$=0':      (1.0,  0.0),
    'e=0.01\n$\\alpha_p$=0':   (0.01, 0.0)
}

color_free = '#740efe'        # free-virus contribution to burst size
color_provirus = '#fe7e4a'    # provirus contribution to burst size

# Run all scenarios
results_all = {}
for title, (e_val, palpha) in scenarios.items():
    n_free_arr, n_provirus_arr = run_scenario(e_val, palpha, n_replicates)
    results_all[title] = {
        'n_free': n_free_arr,
        'n_provirus': n_provirus_arr,
        'total': n_free_arr + n_provirus_arr
    }

# Common bin edges across all scenarios (based on total burst size)
all_totals = np.concatenate([res['total'] for res in results_all.values()])
max_count = all_totals.max()
bin_edges = np.arange(0, max_count + 2) - 0.5
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_height = bin_edges[1] - bin_edges[0]

# Max histogram height across scenarios, for normalizing bar half-widths
max_hist_overall = max(
    np.histogram(res['total'], bins=bin_edges)[0].max() for res in results_all.values()
)

fig, ax = plt.subplots(figsize=(5, 4))
x_positions = np.arange(len(scenarios))
max_half_width = 0.35

for x_pos, (title, res) in zip(x_positions, results_all.items()):
    total_arr = res['total']
    n_free_arr = res['n_free']
    n_provirus_arr = res['n_provirus']

    for b_lo, b_hi, center in zip(bin_edges[:-1], bin_edges[1:], bin_centers):
        in_bin = (total_arr > b_lo) & (total_arr <= b_hi)
        count_in_bin = in_bin.sum()
        if count_in_bin == 0:
            continue

        hw = (count_in_bin / max_hist_overall) * max_half_width
        total_width = 2 * hw

        sum_free = n_free_arr[in_bin].sum()
        sum_provirus = n_provirus_arr[in_bin].sum()
        total_virus = sum_free + sum_provirus
        frac_free = sum_free / total_virus if total_virus > 0 else 1.0
        frac_provirus = 1.0 - frac_free

        width_free = total_width * frac_free
        width_provirus = total_width * frac_provirus
        left_edge = x_pos - hw

        if width_free > 0:
            ax.barh(center, width=width_free, left=left_edge, height=bin_height,
                    color=color_free, alpha=0.6, edgecolor='white', linewidth=0.3)
        if width_provirus > 0:
            ax.barh(center, width=width_provirus, left=left_edge + width_free, height=bin_height,
                    color=color_provirus, alpha=0.6, edgecolor='white', linewidth=0.3)

    mean_val = np.mean(total_arr)
    median_val = np.median(total_arr)
    ax.plot([x_pos - max_half_width, x_pos + max_half_width], [mean_val, mean_val],
            color='black', linestyle='-', linewidth=2)
    ax.plot([x_pos - max_half_width, x_pos + max_half_width], [median_val, median_val],
            color='black', linestyle='--', linewidth=2)

ax.set_xticks(x_positions)
ax.set_xticklabels(scenarios.keys(), fontsize=10)
ax.set_ylabel('Burst size', fontsize=12)
ax.set_title(f'Ancestral virus burst size\nDistribution over {n_replicates} virus factories', fontsize=12)
ax.set_xlim(-0.6, len(scenarios) - 0.4)
ax.set_ylim(-0.6, 40.6)

legend_elements = [
    Patch(facecolor=color_free, alpha=0.6, label='Free virus contribution'),
    Patch(facecolor=color_provirus, alpha=0.6, label='Provirus contribution'),
    Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Mean'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Median'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('burst_sizes.png', dpi=400, bbox_inches='tight')
print("Plot saved.")
plt.show()
# Print summary statistics
print("\nSummary statistics (total burst size):")
for title, res in results_all.items():
    total_arr = res['total']
    print(f"{title.replace(chr(10), ' ')}: mean={np.mean(total_arr):.2f}, "
          f"median={np.median(total_arr):.1f}, min={total_arr.min()}, max={total_arr.max()}")