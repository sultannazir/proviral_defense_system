#!/usr/bin/env python3
"""One figure: the alpha heatmap over time, with grid snapshots underneath.

The top panel is the alpha-over-time panel (log brightness, alpha 0 at the
bottom, each column a share of its own pool). Underneath sit the grid snapshots
written by run_grids.py, each marked on the heatmap by a dashed line at its
timestep -- so a band appearing in the alpha panel can be read straight off the
grid that produced it.

Two themes. `light` (default) is for print: white page and a white-backed
heatmap, but the grid panels keep display.py's black ground and its cell
colours, so a snapshot in a paper looks like the one on screen. `dark` puts the
whole figure on display.py's dark surface. In both, the grid ramp runs dark
alpha 0 to light alpha 1, as it must against black; the caption spells the
mapping out, so a figure is never ambiguous on its own.

    python3 figure_alpha_grids.py runs/run_1.txt              # svg + png, light
    python3 figure_alpha_grids.py runs/run_1.txt --theme dark
    python3 figure_alpha_grids.py runs/run_1.txt --formats svg --max-panels 6
"""
import argparse
import json
import os
import sys

import numpy as np

import classify as C
import display as D

THEMES = {
    'light': dict(
        bg='#ffffff', fg='#1a1a19', dim='#55534d', spine='#55534d',
        # white page, but the grids keep display.py's black ground and its
        # ramp: against black, alpha 0 is the dark end and alpha 1 the light
        cell_empty=(0, 0, 0),            # unoccupied -- the panel's background
        cell_bare=D.CELL_BARE, cell_inf=D.CELL_INF,
        alpha_lo=D.ALPHA_LO, alpha_hi=D.ALPHA_HI,
        alpha_words='dark alpha 0 to light alpha 1',
        cmap='Blues'),
    'dark': dict(
        bg='#1a1a19', fg='#ffffff', dim='#c3c2b7', spine='#333330',
        cell_empty=D.CELL_EMPTY, cell_bare=D.CELL_BARE, cell_inf=D.CELL_INF,
        alpha_lo=D.ALPHA_LO, alpha_hi=D.ALPHA_HI,
        alpha_words='dark alpha 0 to light alpha 1',
        cmap=None),                      # built from display.py's ramp
}


def heat_cmap(th):
    """The heatmap ramp: matplotlib's white-backed Blues for print, or the
    viewer's own background-to-HEAT_HI ramp for the dark theme."""
    if th['cmap']:
        return th['cmap']
    from matplotlib.colors import LinearSegmentedColormap
    hi = tuple(c / 255 for c in D.HEAT_HI)
    return LinearSegmentedColormap.from_list('display_heat', [th['bg'], hi])


def grid_rgb_from_snapshot(st, a, th):
    """display.grid_rgb, but from stored arrays and in the chosen palette."""
    img = np.empty(st.shape + (3,), np.uint8)
    img[:] = th['cell_empty']
    live = (st == 1) | (st == 2)
    has = live & (a >= 0)
    if has.any():
        t = np.clip(a[has], 0, 1)[:, None]
        img[has] = (th['alpha_lo'] * (1 - t) + th['alpha_hi'] * t).astype(np.uint8)
    img[live & (a < 0)] = th['cell_bare']
    img[st == 2] = th['cell_inf']
    # stored as (nc, nr) = (x, y) like the viewer; transpose so y runs vertically
    return np.transpose(img, (1, 0, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runfile')
    ap.add_argument('--grids', default=None,
                    help='snapshot .npz (default: <runfile stem>_grids.npz)')
    ap.add_argument('-o', '--out', default=None,
                    help='output path or stem; the extension is ignored and '
                         '--formats decides what is written')
    ap.add_argument('--formats', default='svg,png',
                    help='comma-separated output formats (default: svg,png)')
    ap.add_argument('--theme', default='light', choices=tuple(THEMES),
                    help='light (print, default) or dark (display.py colours)')
    ap.add_argument('--pool', default='provirus',
                    choices=('provirus', 'virion', 'pool'),
                    help='which density the heatmap shows (default: provirus)')
    ap.add_argument('--grid-mode', '--grid_mode', default='provirus',
                    choices=('provirus', 'factory'),
                    help="which pool colours the cells (default: provirus)")
    ap.add_argument('--at', default=None,
                    help='draw exactly these snapshot timesteps, comma '
                         'separated; the nearest available one is used for '
                         'each. Overrides --max-panels')
    ap.add_argument('--include-t0', '--include_t0', action='store_true',
                    help='keep the t=0 panel. By default it is dropped when '
                         'the run has any other early snapshot, since t=0 is '
                         'only the seeded state')
    ap.add_argument('--max-panels', '--max_panels', type=int, default=5,
                    help='most grid snapshots to draw (evenly spaced, always '
                         'including the first and last; default 5)')
    ap.add_argument('--decades', type=float, default=4.0,
                    help='log brightness range. Higher shows rarer variants: 3 '
                         'hides the high-alpha tail, 5 saturates the low-alpha '
                         'peak and amplifies single-count noise (default 4)')
    ap.add_argument('--min-count', '--min_count', type=float, default=200)
    ap.add_argument('--every', type=int, default=1,
                    help='keep every Nth record of the run (thins a long run)')
    ap.add_argument('--font-scale', '--font_scale', type=float, default=1.0,
                    help='multiply every font size (default 1.0, sized for a '
                         'single-column publication figure)')
    ap.add_argument('--dpi', type=int, default=160,
                    help='raster resolution; also the resolution of the images '
                         'embedded in an SVG (default 160)')
    a = ap.parse_args()

    th = THEMES[a.theme]
    BG, FG, DIM = th['bg'], th['fg'], th['dim']
    fs = a.font_scale
    F_CAP, F_TITLE, F_LABEL, F_TICK, F_PANEL = (10.5 * fs, 12 * fs, 12 * fs,
                                                11 * fs, 11 * fs)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.fonttype'] = 'none'   # keep SVG text editable

    stem = os.path.splitext(a.runfile)[0]
    gpath = a.grids or (stem + '_grids.npz')
    if not os.path.exists(gpath):
        sys.exit('no snapshots at %s -- run run_grids.py --grid-every first' % gpath)

    lines = [l for l in open(a.runfile).read().splitlines() if l.strip()]
    while lines and not (lines[-1].startswith('{') and lines[-1].endswith('}')):
        lines.pop()
    recs = [json.loads(l) for l in lines][::a.every]
    if not recs:
        sys.exit('no complete records in %s' % a.runfile)

    z = np.load(gpath)
    ts, state = z['t'], z['state']
    alpha = z['alpha_p'] if a.grid_mode == 'provirus' else z['alpha_f']

    pool_idx = np.arange(len(ts))
    if not a.include_t0 and len(ts) > 1 and ts[0] == 0:
        pool_idx = pool_idx[1:]          # t=0 is the seeded state, not a result
    if a.at:
        want = [int(x) for x in a.at.replace(',', ' ').split()]
        idx = np.unique([pool_idx[np.abs(ts[pool_idx] - w).argmin()]
                         for w in want])
    else:
        k = min(a.max_panels, len(pool_idx))
        idx = np.unique(pool_idx[np.linspace(0, len(pool_idx) - 1, k)
                                 .round().astype(int)])

    # ---- the alpha panel, exactly as alpha_matrix.render builds it
    t = np.array([r['time'] for r in recs], float)
    Mx = np.zeros((100, len(recs)))
    for j, r in enumerate(recs):
        P = np.array(r['provirus'], float)
        V = np.array(r['virion'], float)
        h = {'pool': lambda: C.pool_density(P, V, 'mean', a.min_count),
             'provirus': lambda: P, 'virion': lambda: V}[a.pool]()
        tot = h.sum()
        if tot > 0:
            Mx[:, j] = h / tot
    # The t=0 column is a delta -- the whole pool sits in the seeded bin, so its
    # density is 1.0, roughly twice any later peak. Letting it set the
    # normalising maximum spends a quarter of a decade of range on the initial
    # condition, so scale to the run instead and let t=0 clip to full brightness.
    body = Mx[:, t > 0] if (t > 0).any() else Mx
    m = body.max()
    Z = (np.clip(np.log10(Mx / m * (10 ** a.decades - 1) + 1) / a.decades, 0, 1)
         if m > 0 else Mx)

    ncols = len(idx)
    W = max(2.15 * ncols, 9.5)
    cell = W / ncols
    fig = plt.figure(figsize=(W, 3.9 + cell + 0.85))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, ncols, height_ratios=[3.9, cell],
                          hspace=0.30, wspace=0.05,
                          left=0.075, right=0.962, top=0.855, bottom=0.03)

    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(BG)
    ax.imshow(Z, origin='lower', aspect='auto', cmap=heat_cmap(th),
              extent=[t.min(), t.max(), 0, 1], vmin=0, vmax=1,
              interpolation='nearest')
    for i in idx:
        ax.axvline(ts[i], color=DIM, lw=0.9, ls=(0, (4, 3)), alpha=0.8)
    ax.set_ylabel('alpha', color=FG, fontsize=F_LABEL)
    ax.set_xlabel('timestep', color=FG, labelpad=3, fontsize=F_LABEL)
    ax.set_title('%s density  --  log brightness over %g decades'
                 % (a.pool, a.decades), color=FG, fontsize=F_TITLE, loc='left',
                 pad=7)
    # fewer, larger tick labels: crowded small ticks are the main thing that
    # stops a screen figure reading as a printed one
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(colors=DIM, labelsize=F_TICK, length=4.5, width=0.9)
    for sp in ax.spines.values():
        sp.set_color(th['spine'])

    for col, i in enumerate(idx):
        gax = fig.add_subplot(gs[1, col])
        gax.imshow(grid_rgb_from_snapshot(state[i], alpha[i], th),
                   origin='lower', interpolation='nearest')
        gax.set_xticks([]); gax.set_yticks([])
        for sp in gax.spines.values():
            sp.set_color(th['spine'])
        gax.set_title('t = %s' % format(int(ts[i]), ','), color=FG,
                      fontsize=F_PANEL, pad=5)

    params = ''
    ppath = stem + '.params.json'
    if os.path.exists(ppath):
        P = json.load(open(ppath))
        params = '   seed %g, size %d, e %g, lambda %g, diff_rate %g, decay %g' % (
            P['seed'], P['size'], P['e'], P['lam'], P['diff_rate'], P['decay_rate'])
    fig.suptitle('%s%s\ncells: grey = S, no provirus   blue = provirus, %s   '
                 'orange = factory active'
                 % (os.path.basename(a.runfile), params, th['alpha_words']),
                 color=FG, fontsize=F_CAP, x=0.008, ha='left', y=0.992,
                 va='top', linespacing=1.6)

    base = os.path.splitext(a.out)[0] if a.out else (stem + '_alpha_grids')
    written = []
    for ext in [f.strip().lstrip('.') for f in a.formats.split(',') if f.strip()]:
        path = '%s.%s' % (base, ext)
        fig.savefig(path, dpi=a.dpi, facecolor=BG)
        written.append(path)
    plt.close(fig)
    print('wrote %s  (%d records, %d of %d snapshots, %s theme)'
          % (' and '.join(written), len(recs), len(idx), len(ts), a.theme))
    return 0


if __name__ == '__main__':
    sys.exit(main())
