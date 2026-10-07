
import matplotlib.pyplot as plt
from timeseries_overlay import plot_timeseries   # imports the module made earlier
 
plt.rcParams.update({'font.size': 14})
 
# ---- Settings -------------------------------------------------------------
DATA_DIR  = "mu_sweep_6oct26"
TEMPLATE  = "muA{muA}mustep{mustep}cA0.8GT0GR0seed{seed}.txt"
SEEDS     = range(1, 21)
ALPHA_REP = 0.12
C         = 0.8     # cost of cooperation cA (must match "cA0_5" in the filenames)
# value -> string as it appears in the filename ("0.1" -> "0_1")
MU_A_VALUES    = {0.1: "0.1", 0.01: "0.01"}      # rows
MU_STEP_VALUES = {0.1: "0.1", 0.01: "0.01"}      # columns
# ---------------------------------------------------------------------------
 
fig, axes = plt.subplots(
    len(MU_A_VALUES), len(MU_STEP_VALUES),
    figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
 
for i, (muA, muA_str) in enumerate(MU_A_VALUES.items()):
    for j, (mustep, mustep_str) in enumerate(MU_STEP_VALUES.items()):
        ax = axes[i, j]
        n = plot_timeseries(
            ax, TEMPLATE, c=C, data_dir=DATA_DIR, seeds=SEEDS, alpha_rep=ALPHA_REP,
            verbose=False, muA=muA_str, mustep=mustep_str)
        print(f"muA={muA}, mustep={mustep}: {n} replicates plotted")
        ax.set_title(rf"$\mu_{{\alpha}}={muA}$, $\mu_{{SD}}={mustep}$")
        # if i == 0 and j == 0:
        #     ax.legend(loc="upper right", frameon=True, fontsize=11)
        if j == 0:
            ax.set_ylabel(r"Cooperation, $\alpha$")
        if i == len(MU_A_VALUES) - 1:
            ax.set_xlabel("Time")
 
# plt.show()
plt.savefig("timeseries_4panel_c0.8.png", dpi=400)
 



