"""Reusable module: overlay timeseries of free virus (red) and provirus (blue)
frequency distributions for many replicates on one axis, plus the prediction
1/(c*(2+p)) of every replicate as its own black line."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

COL_F, COL_P = (1, 0, 0), (0, 0, 1)   # free virus = red, provirus = blue

def to_rgba(intensity, rgb, a):
    img = np.zeros(intensity.shape + (4,))
    img[..., :3] = rgb
    img[..., 3] = intensity * a
    return img


def plot_timeseries(ax, fname_template, c, data_dir=".", seeds=range(1, 51),
                    alpha_rep=0.05, dpi = 400, verbose=True, line_kw=None, **fname_kwargs):
    """Draw all replicates onto `ax`, then each replicate's prediction
    1/(c*(2+p)), p = numP/(numS+numP), as a separate black line.
    `fname_template` is a format string with a {seed} field plus any extra
    fields supplied via **fname_kwargs. `c` is the cost of cooperation (cA).
    Returns the number of replicates plotted."""
    n_loaded = 0
    for seed in seeds:
        path = os.path.join(data_dir, fname_template.format(seed=seed, **fname_kwargs))
        if not os.path.exists(path):
            if verbose:
                print(f"Missing {path}, skipping")
            continue
        with open(path) as f:
            lines = [json.loads(x) for x in f if x.strip()]

        t = np.array([x["time"] for x in lines])
        F = np.array([x["free_virus"] for x in lines], dtype=float).T
        P = np.array([x["provirus"] for x in lines], dtype=float).T
        peak = (F+P).max(axis=0, keepdims=True)
        F = np.divide(F, peak, out=np.zeros_like(F), where=peak > 0)
        P = np.divide(P, peak, out=np.zeros_like(P), where=peak > 0)

        dt = t[1] - t[0]
        extent = [t[0] - dt / 2, t[-1] + dt / 2, 0, 1]
        kw = dict(origin="lower", aspect="auto", extent=extent, interpolation="nearest")
        ax.imshow(to_rgba(F, COL_F, alpha_rep), **kw)
        ax.imshow(to_rgba(P, COL_P, alpha_rep), **kw)

        # Prediction for this replicate at every timepoint
        numP = np.array([x["numP"] for x in lines], dtype=float)
        numS = np.array([x["numS"] for x in lines], dtype=float)
        p = numP / (numS + numP)
        pred = 1.0 / (c * (2 + p))
        style = dict(color="black", lw=0.8, alpha=0.5)
        if n_loaded == 0:                          # one legend entry only
            style["label"] = r"$1/(c(2+p))$, each replicate"
        style.update(line_kw or {})
        ax.plot(t, pred, **style)
        n_loaded += 1

    if n_loaded:
        ax.set_xlim(extent[0], extent[1])

    ax.set_ylim(0, 1)
    return n_loaded


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(6, 4), dpi=400)
    n = plot_timeseries(
        ax, "cA1GT0GR0seed{seed}.txt", c=1.0,
        data_dir="sweep_2oct26", muA="0.1", mustep="0.1")
    print(f"Plotted {n} replicates")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Cooperation, $\alpha$")
    # ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("timeseries_overlay_c1.png", dpi=400)