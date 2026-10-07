"""
Plot a 200x200 grid from simulation output files.

Expects, in a data folder, for a given `time`:
    alive_{time}.dat        -> rows: x \t y \t value   (value in {'S','P','I','undefined'})
    provirus_{time}.dat     -> rows: x \t y \t value   (value in [0,1] or 'undefined')
    coinfection_{time}.dat  -> rows: x \t y \t value   (value in [0,1] or 'undefined')

Coloring rule:
    alive == 'undefined' -> black
    alive == 'S'         -> white
    alive == 'P'         -> bluish palette (blue -> green) scaled by provirus value
    alive == 'I'         -> reddish palette (red -> yellow) scaled by coinfection value
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

plt.rcParams.update({'font.size': 20})

GRID_SIZE = 200

# Custom palettes
cmap_P = LinearSegmentedColormap.from_list("blue_green", ["#FFAAFF","#5555AA"])
cmap_I = LinearSegmentedColormap.from_list("red_yellow", ["yellow", "red"])


def load_grid(filepath, size=GRID_SIZE):
    """Load a .dat file into a (size, size) object array indexed as grid[y, x]."""
    grid = np.full((size, size), None, dtype=object)
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # be lenient in case of extra whitespace instead of tabs
                parts = line.split()
                if len(parts) != 3:
                    continue
            x_str, y_str, val = parts
            x, y = int(x_str), int(y_str)
            grid[y, x] = val
    return grid


def build_image(time, data_dir="data", size=GRID_SIZE):
    alive = load_grid(f"{data_dir}/alive_grid_{time}.dat", size)
    provirus = load_grid(f"{data_dir}/provirus_grid_{time}.dat", size)
    coinfection = load_grid(f"{data_dir}/free_virus_grid_{time}.dat", size)

    img = np.zeros((size, size, 3))

    for y in range(size):
        for x in range(size):
            a = alive[y, x]

            if a is None or a == "undefined":
                img[y, x] = (0.0, 0.0, 0.0)

            elif a == "S":
                img[y, x] = (1.0, 1.0, 1.0)

            elif a == "P":
                v = provirus[y, x]
                if v is None or v == "undefined":
                    img[y, x] = (0.0, 0.0, 0.0)
                else:
                    img[y, x] = cmap_P(float(v))[:3]

            elif a == "I":
                v = coinfection[y, x]
                if v is None or v == "undefined":
                    img[y, x] = (0.0, 0.0, 0.0)
                else:
                    img[y, x] = cmap_I(float(v))[:3]

            else:
                # unexpected value, fall back to black
                img[y, x] = (0.0, 0.0, 0.0)

    return img


def plot_time(time, data_dir="data", size=GRID_SIZE, save_path=None):
    img = build_image(time, data_dir, size)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_title(f"Time {time}")
    ax.axis("off")

    # Leave room at the bottom for two stacked, equal-length horizontal colorbars
    fig.subplots_adjust(bottom=0.16)

    cbar_left, cbar_width, cbar_height = 0.25, 0.5, 0.025

    ax_cbar_P = fig.add_axes([cbar_left, 0.1, cbar_width, cbar_height])
    sm_P = ScalarMappable(cmap=cmap_P)
    sm_P.set_array([0, 1])
    cbar_P = fig.colorbar(sm_P, cax=ax_cbar_P, orientation="horizontal")
    cbar_P.set_label(r"Provirus $\alpha$ value (P hosts)")

    ax_cbar_I = fig.add_axes([cbar_left, 0.2, cbar_width, cbar_height])
    sm_I = ScalarMappable(cmap=cmap_I)
    sm_I.set_array([0, 1])
    cbar_I = fig.colorbar(sm_I, cax=ax_cbar_I, orientation="horizontal")
    cbar_I.set_label(r"Free virus $\alpha$ value (I hosts)")


    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage:
    TIME = 100          # change to the time-step you want to plot
    DATA_DIR = "grid_snapshots/exp2"  # folder containing the .dat files

    plot_time(TIME, data_dir=DATA_DIR, save_path=f"grid_snapshots/snaps/exp2_grid_{TIME}.png")
    # To save instead of / in addition to showing:
    # plot_time(TIME, data_dir=DATA_DIR, save_path=f"grid_{TIME}.png")
