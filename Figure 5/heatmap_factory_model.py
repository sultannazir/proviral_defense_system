import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('lhs3_results.csv')

# Filter: mixed=0 and n_provirus>0
filtered = df[(df['mixed'] == 0) ].copy()
print(len(filtered))

print(f"Total rows: {len(df)}")
print(f"Filtered rows (mixed=0 and n_virion>0): {len(filtered)}")

# Create log-transformed versions
filtered['log_e'] = np.log10(filtered['e'])
filtered['log_diff_rate'] = np.log10(filtered['diff_rate'])

print(f"\nLog parameter ranges:")
print(f"log10(e): {filtered['log_e'].min():.3f} to {filtered['log_e'].max():.3f}")
print(f"log10(diff_rate): {filtered['log_diff_rate'].min():.3f} to {filtered['log_diff_rate'].max():.3f}")

# Create 2D bins for log_e and log_diff_rate
# Using 20 bins for each parameter (adjust as needed)
n_bins = 8

log_e_bins = np.linspace(filtered['log_e'].min(), filtered['log_e'].max(), n_bins + 1)
log_diff_rate_bins = np.linspace(filtered['log_diff_rate'].min(), filtered['log_diff_rate'].max(), n_bins + 1)

# Initialize heatmap
heatmap = np.zeros((n_bins, n_bins))
counts = np.zeros((n_bins, n_bins))

# Fill heatmap
for idx, row in filtered.iterrows():
    e_idx = np.searchsorted(log_e_bins, row['log_e']) - 1
    diff_rate_idx = np.searchsorted(log_diff_rate_bins, row['log_diff_rate']) - 1
    
    # Ensure indices are within bounds
    e_idx = np.clip(e_idx, 0, n_bins - 1)
    diff_rate_idx = np.clip(diff_rate_idx, 0, n_bins - 1)
    
    if (row['outcome'] == 'defense_evolved' or row['outcome'] == 'no_defense'):
        counts[diff_rate_idx, e_idx] += 1
    
    if row['outcome'] == 'defense_evolved':
        heatmap[diff_rate_idx, e_idx] += 1

# Calculate fractions (avoid division by zero)
fraction_heatmap = np.divide(heatmap, counts, where=counts > 0, out=np.full_like(heatmap, np.nan))

# Create the plot
fig, ax = plt.subplots(figsize=(6, 5))

# Create colormap and set bad color to grey
cmap = plt.cm.viridis
cmap.set_bad('lightgrey')

# Create the heatmap
im = ax.imshow(fraction_heatmap, aspect='auto', cmap=cmap, origin='lower',
               extent=[log_e_bins[0], log_e_bins[-1], log_diff_rate_bins[0], log_diff_rate_bins[-1]], vmin=0, vmax=1)

ax.set_xlabel('Provirus excision probability (log-scaled), $\log_{10}e$', fontsize=12)
ax.set_ylabel('Virion diffusion (log-scaled), $\log_{10}D_V$', fontsize=12)
ax.set_title('Fraction of runs with de novo defense evolution', fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
# cbar.set_label('Fraction of runs with defense evolution', fontsize=11)

plt.tight_layout()
plt.savefig('heatmap_e_diffrate.png', dpi=400, bbox_inches='tight')
print("\nHeatmap saved to /mnt/user-data/outputs/heatmap_e_diffrate.png")

plt.show()

# Print statistics
print(f"\nHeatmap statistics:")
print(f"Bins with data: {np.sum(counts > 0)}")
print(f"Mean fraction (where data exists): {np.mean(fraction_heatmap[counts > 0]):.3f}")
print(f"Min fraction: {np.min(fraction_heatmap[counts > 0]):.3f}")
print(f"Max fraction: {np.max(fraction_heatmap[counts > 0]):.3f}")