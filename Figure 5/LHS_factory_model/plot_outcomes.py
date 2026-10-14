#!/usr/bin/env python3
"""
Plot how each swept parameter shapes the outcome of the LHS.

Reads the CSV written by classify.py (design parameters joined to outcomes) and
produces:

  outcomes_by_parameter.png  per parameter: a stacked bar of outcome composition
                             across bins of that parameter, and underneath it
                             P(defense present | virus persists) with 95%
                             binomial intervals
  outcome_maps.png           per outcome: where its runs sit in a plane of two
                             parameters, one dot per run
  outcome_maps_smooth.png    the same, kernel-smoothed -- easier to read a trend
                             off, but it interpolates between scattered points,
                             so read it next to the dots

--facet COLUMN splits everything into one block per value of that column rather
than averaging over it, which is the point when the column is something like
perfect_mix that changes the model rather than tuning it.

    python3 plot_outcomes.py lhs_results.csv --facet perfect_mix
    python3 plot_outcomes.py lhs_results.csv --pair e lambda --bins 8
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'
from matplotlib.patches import Patch

# Categorical palette, fixed order, validated for CVD on the adjacent pairlist
# (worst adjacent CVD dE 9.1 light / 8.4 dark).
OUTCOMES = ['host_extinct', 'virus_extinct', 'no_defense', 'defense_evolved']
PRETTY = {
    'host_extinct':    'Host extinct',
    'virus_extinct':   'Virus extinct',
    'no_defense':      'No defense',
    'defense_evolved': 'Defense evolved',
}
THEME = {
    'light': dict(series=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4',
                          '#008300'],
                  surface='#fcfcfb', primary='#0b0b0b', secondary='#52514e',
                  muted='#8a8880', grid='#e5e4e0', faint='#d8d7d2'),
    'dark':  dict(series=['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181',
                          '#008300'],
                  surface='#1a1a19', primary='#ffffff', secondary='#c3c2b7',
                  muted='#8a8880', grid='#333330', faint='#2b2b29'),
}
NOT_PARAMS = {'point', 'rep', 'seed', 'scheme', 'maxtime', 'outfile', 'outcome',
              'established', 'defense', 'defense_present', 'frac_low_pool',
              'finished', 'virus_end', 'vir_time', 't_defense', 'max_frac_low',
              't_end', 'stopped_early', 'num_S', 'num_I', 'n_virion',
              'n_provirus', 'mean_alpha_virion', 'mean_alpha_provirus',
              'frac_provirus_low', 'frac_virion_low', 'mean_provirus_per_host',
              'diffusion', 'pair', 'mixed',
              'perfect_mix', 'perfect_diffusion'}
SURVIVED = ['no_defense', 'defense_evolved']

# Readable names for facet levels. A bare "mixed = 0" says what the column is
# rather than what the condition means; override per figure with --facet-labels.
FACET_LABELS = {
    'mixed':        {0: 'Spatially structured', 1: 'Mixed'},
    'perfect_mix':  {0: 'Spatially structured', 1: 'Mixed'},
}


def facet_label(facet, lev):
    key = int(lev) if float(lev) == int(lev) else lev
    name = FACET_LABELS.get(facet, {}).get(key)
    return name if name else '%s = %g' % (facet, lev)


def detect_params(df, explicit=None, facet=None):
    if explicit:
        return explicit
    out = []
    for c in df.columns:
        if c in NOT_PARAMS or c == facet:
            continue
        v = pd.to_numeric(df[c], errors='coerce')
        if v.notna().sum() > 0.9 * len(df) and v.nunique() > 10:
            out.append(c)
    return out


def is_log(v):
    """Was this parameter sampled on a log scale?  An LHS is uniform on whichever
    scale it was drawn on, so ask which scale the sorted values are uniform in --
    max/min alone would call any wide linear range logarithmic."""
    v = np.asarray(pd.to_numeric(v, errors='coerce'), float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 8 or v.max() / v.min() < 20:
        return False
    s = np.sort(v)
    q = (np.arange(len(s)) + 0.5) / len(s)
    return np.corrcoef(q, np.log(s))[0, 1] > np.corrcoef(q, s)[0, 1]


def bin_edges(v, nbins, log):
    if log:
        return np.exp(np.linspace(math.log(v.min()), math.log(v.max()), nbins + 1))
    return np.linspace(v.min(), v.max(), nbins + 1)


def wilson(k, n, z=1.96):
    """95% Wilson interval -- behaves at k=0 and k=n, unlike the normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_tick(x, log):
    if log and x < 0.01:
        return '1e%d' % round(math.log10(x))
    return '%.2g' % x


# One type scale for every figure in this script, sized for a figure that will
# be shrunk to a journal column. --font-scale multiplies all of it at once.
FS = {'tick': 19.0, 'xtick': 18.0, 'axis': 24.0, 'ylab': 21.0,
      'panel': 26.0, 'legend': 21.0, 'suptitle': 28.0, 'cell': 20.0}


def scale_fonts(k):
    for key in FS:
        FS[key] = FS[key] * k


def _style(a, th):
    a.set_facecolor(th['surface'])
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        a.spines[sp].set_color(th['grid'])
    a.tick_params(colors=th['secondary'], labelsize=FS['tick'], length=7,
                  width=1.6, pad=4)


def _block(df, params, nbins, th, logs, row1, row2, colours, bottom):
    """One facet's pair of rows: composition bars, then P(defense | survived)."""
    stack = [o for o in OUTCOMES if (df['outcome'] == o).any()]
    stack += sorted(set(df['outcome']) - set(OUTCOMES))
    keep = df['outcome'].isin(SURVIVED).values
    dpres = pd.to_numeric(df.get('defense_present', 0), errors='coerce').fillna(0).values

    for k, p in enumerate(params):
        v = pd.to_numeric(df[p], errors='coerce')
        log = logs.get(p, is_log(v))
        edges = bin_edges(v, nbins, log)
        idx = np.clip(np.digitize(v, edges[1:-1]), 0, nbins - 1)
        ax, ax2 = row1[k], row2[k]
        for a in (ax, ax2):
            _style(a, th)
        x = np.arange(nbins)

        bottom_acc = np.zeros(nbins)
        for o in stack:
            frac = np.array([( (df['outcome'].values[idx == b] == o).mean()
                               if (idx == b).sum() else 0.0) for b in range(nbins)])
            ax.bar(x, frac, bottom=bottom_acc, width=0.82, color=colours[o],
                   edgecolor=th['surface'], linewidth=1.1)
            bottom_acc += frac

        pt, lo, hi = [], [], []
        for b in range(nbins):
            m = (idx == b) & keep
            k_, n_ = int(dpres[m].sum()), int(m.sum())
            pt.append(k_ / n_ if n_ else np.nan)
            a_, c_ = wilson(k_, n_)
            lo.append(a_); hi.append(c_)
        ax2.fill_between(x, lo, hi, color=th['series'][0], alpha=0.18, linewidth=0)
        ax2.plot(x, pt, color=th['series'][0], linewidth=3.2, marker='o', markersize=8,
                 markeredgecolor=th['surface'], markeredgewidth=1)
        if keep.any():
            ax2.axhline(dpres[keep].mean(), color=th['muted'], linewidth=1,
                        linestyle=(0, (4, 3)))
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', color=th['grid'], linewidth=0.7)
        ax2.set_axisbelow(True)
        ax.set_ylim(0, 1)

        centres = (edges[:-1] * edges[1:]) ** 0.5 if log else (edges[:-1] + edges[1:]) / 2
        show = sorted(set(np.linspace(0, nbins - 1, 3).round().astype(int).tolist()))
        for a in (ax, ax2):
            a.set_xticks([x[i] for i in show])
            a.set_xticklabels([fmt_tick(centres[i], log) for i in show],
                              fontsize=FS['xtick'])
            a.set_xlim(-0.7, nbins - 0.3)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
        if bottom:
            ax2.set_xlabel(p + (' (log)' if log else ''), color=th['primary'],
                           fontsize=FS['axis'])
        else:
            ax2.set_xticklabels([])
        if k == 0:
            ax.set_ylabel('fraction of runs', color=th['secondary'],
                          fontsize=FS['ylab'])
            ax2.set_ylabel('P(defense)', color=th['secondary'],
                           fontsize=FS['ylab'])
    return stack


def fig_by_parameter(df, params, nbins, th, logs, out, facet=None):
    colours = {o: th['series'][i % len(th['series'])] for i, o in enumerate(OUTCOMES)}
    levels = [None] if facet is None else sorted(df[facet].dropna().unique())
    nb, n = len(levels), len(params)
    fig, axes = plt.subplots(2 * nb, n, figsize=(3.15 * n + 1.9, 4.5 * nb + 1.5),
                             sharey='row',
                             gridspec_kw=dict(height_ratios=[2.4, 1.0] * nb,
                                              hspace=0.42, wspace=0.26))
    axes = np.atleast_2d(axes).reshape(2 * nb, n)
    fig.patch.set_facecolor(th['surface'])

    # A bar row and its P(defense) row belong together; consecutive facets do
    # not. hspace cannot say that -- it spaces every gap alike -- so push the
    # blocks apart here, leaving the pairing inside each block untouched.
    if nb > 1:
        air = 0.030
        for bi in range(nb):
            d = (bi - (nb - 1) / 2.0) * 2 * air
            for a in axes[2 * bi:2 * bi + 2].ravel():
                q = a.get_position()
                a.set_position([q.x0, q.y0 - d, q.width, q.height])

    stack = []
    for bi, lev in enumerate(levels):
        sub = df if lev is None else df[df[facet] == lev]
        s = _block(sub, params, nbins, th, logs, axes[2 * bi], axes[2 * bi + 1],
                   colours, bottom=(bi == nb - 1))
        stack = stack or s
        if lev is not None:
            # place the facet header in the gap above its block, in figure
            # coordinates -- an axes-fraction offset large enough to clear this
            # block lands on top of the panel belonging to the one above
            pos = axes[2 * bi, 0].get_position()
            fig.text(pos.x0, pos.y1 + 0.014, '%s (n=%d)'
                     % (facet_label(facet, lev), len(sub)), color=th['primary'],
                     fontsize=FS['panel'], fontweight='bold', ha='left',
                     va='bottom')
    handles = [Patch(facecolor=colours[o], label=PRETTY.get(o, o)) for o in stack]
    handles.append(plt.Line2D([], [], color=th['series'][0], marker='o', markersize=9,
                              linewidth=3.2,
                              label='P(defense | survived), 95% CI'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.012),
               frameon=False, fontsize=FS['legend'], labelcolor=th['primary'],
               ncol=len(handles), columnspacing=2.4, handlelength=1.5,
               handletextpad=0.7)
    for ext in ('png', 'svg'):
        p = '%s.%s' % (out, ext)
        fig.savefig(p, dpi=160, bbox_inches='tight', facecolor=th['surface'])
        print('wrote', p)
    plt.close(fig)


def fig_maps(df, pair, th, logs, out, facet=None, size=9):
    """One dot per run. No kernel smoothing: a smoothed surface over scattered
    LHS points invents structure between them and hides how thin the sampling
    is, which matters most exactly where an outcome is rare."""
    px, py = pair
    lx = logs.get(px, is_log(df[px]))
    ly = logs.get(py, is_log(df[py]))
    levels = [None] if facet is None else sorted(df[facet].dropna().unique())
    present = [o for o in OUTCOMES if (df['outcome'] == o).any()]
    colours = {o: th['series'][i % len(th['series'])] for i, o in enumerate(OUTCOMES)}

    nb, nc = len(levels), len(present)
    fig, axes = plt.subplots(nb, nc, figsize=(2.5 * nc + 1.0, 2.7 * nb + 0.8),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.10, hspace=0.22))
    axes = np.atleast_2d(axes).reshape(nb, nc)
    fig.patch.set_facecolor(th['surface'])

    for bi, lev in enumerate(levels):
        sub = df if lev is None else df[df[facet] == lev]
        for ci, o in enumerate(present):
            a = axes[bi, ci]
            _style(a, th)
            a.scatter(sub[px], sub[py], s=size * 0.7, c=th['faint'], linewidths=0)
            m = sub[sub['outcome'] == o]
            a.scatter(m[px], m[py], s=size, c=colours[o], linewidths=0)
            if lx:
                a.set_xscale('log')
            if ly:
                a.set_yscale('log')
            if bi == 0:
                a.set_title('%s\n%d of %d' % (PRETTY.get(o, o), len(m), len(sub)),
                            color=th['primary'], fontsize=FS['panel'])
            else:
                a.set_title('%d of %d' % (len(m), len(sub)),
                            color=th['secondary'], fontsize=FS['ylab'])
            if bi == nb - 1:
                a.set_xlabel(px + (' (log)' if lx else ''), color=th['primary'], fontsize=FS['axis'])
        axes[bi, 0].set_ylabel((('%s = %g\n' % (facet, lev)) if lev is not None else '')
                               + py + (' (log)' if ly else ''),
                               color=th['primary'], fontsize=FS['axis'])
    fig.suptitle('Where each outcome lives, over %s and %s  (one dot per run)'
                 % (px, py), color=th['primary'], fontsize=FS['suptitle'], x=0.01, ha='left',
                 y=1.0 + 0.055 / nb)
    for ext in ('png', 'svg'):
        p = '%s.%s' % (out, ext)
        fig.savefig(p, dpi=160, bbox_inches='tight', facecolor=th['surface'])
        print('wrote', p)
    plt.close(fig)



def fig_maps_smooth(df, pair, th, logs, out, facet=None, grid=60, bw=0.16,
                    scale='relative'):
    """Kernel-smoothed P(outcome) over two parameters.

    Complements the dot version rather than replacing it: the smoothing makes a
    trend legible but interpolates between scattered points, so read it next to
    the dots to see how much sampling is actually behind each region. Each panel
    is scaled to its own maximum by default (printed in its title), otherwise a
    rare outcome renders as a blank square.
    """
    px, py = pair
    lx = logs.get(px, is_log(df[px]))
    ly = logs.get(py, is_log(df[py]))
    levels = [None] if facet is None else sorted(df[facet].dropna().unique())
    present = [o for o in OUTCOMES if (df['outcome'] == o).any()]
    nb, nc = len(levels), len(present)
    fig, axes = plt.subplots(nb, nc, figsize=(2.5 * nc + 1.4, 2.7 * nb + 0.8),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.10, hspace=0.26))
    axes = np.atleast_2d(axes).reshape(nb, nc)
    fig.patch.set_facecolor(th['surface'])

    for bi, lev in enumerate(levels):
        sub = df if lev is None else df[df[facet] == lev]
        vx = pd.to_numeric(sub[px], errors='coerce').values
        vy = pd.to_numeric(sub[py], errors='coerce').values
        tx = np.log10(vx) if lx else vx
        ty = np.log10(vy) if ly else vy
        nx = (tx - tx.min()) / (tx.max() - tx.min())
        ny = (ty - ty.min()) / (ty.max() - ty.min())
        gx, gy = np.meshgrid(np.linspace(0, 1, grid), np.linspace(0, 1, grid))
        w = np.exp(-((gx[..., None] - nx) ** 2 + (gy[..., None] - ny) ** 2) / (2 * bw * bw))
        wsum = w.sum(axis=2)
        for ci, o in enumerate(present):
            a = axes[bi, ci]
            _style(a, th)
            z = (w * (sub['outcome'] == o).values).sum(axis=2) / np.maximum(wsum, 1e-12)
            top = float(z.max())
            vmax = max(top, 1e-9) if scale == 'relative' else 1.0
            a.imshow(z, origin='lower', extent=[0, 1, 0, 1], vmin=0, vmax=vmax,
                     cmap='Blues', aspect='auto', interpolation='bilinear')
            ttl = PRETTY.get(o, o) if bi == 0 else ''
            if scale == 'relative':
                ttl += ('\n' if ttl else '') + 'max P = %.2f' % top
            a.set_title(ttl, color=th['primary'] if bi == 0 else th['secondary'],
                        fontsize=FS['cell'])
            ticks = np.linspace(0, 1, 4)
            a.set_xticks(ticks)
            a.set_xticklabels([fmt_tick(10 ** t if lx else t, lx) for t in
                               (tx.min() + ticks * (tx.max() - tx.min()))], fontsize=FS['xtick'])
            a.set_yticks(ticks)
            if ci == 0:
                a.set_yticklabels([fmt_tick(10 ** t if ly else t, ly) for t in
                                   (ty.min() + ticks * (ty.max() - ty.min()))],
                                  fontsize=FS['xtick'])
            if bi == nb - 1:
                a.set_xlabel(px + (' (log)' if lx else ''), color=th['primary'], fontsize=FS['axis'])
        axes[bi, 0].set_ylabel((('%s = %g\n' % (facet, lev)) if lev is not None else '')
                               + py + (' (log)' if ly else ''),
                               color=th['primary'], fontsize=FS['axis'])
    fig.suptitle('Where each outcome lives, over %s and %s  (kernel-smoothed%s)'
                 % (px, py, ', each panel scaled to its own max'
                    if scale == 'relative' else ''),
                 color=th['primary'], fontsize=FS['suptitle'], x=0.01, ha='left',
                 y=1.0 + 0.055 / nb)
    for ext in ('png', 'svg'):
        p = '%s.%s' % (out, ext)
        fig.savefig(p, dpi=160, bbox_inches='tight', facecolor=th['surface'])
        print('wrote', p)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results')
    ap.add_argument('--outdir', '--plot-dir', dest='outdir', default='plots')
    ap.add_argument('--prefix', default='')
    ap.add_argument('--bins', type=int, default=10)
    ap.add_argument('--params', nargs='*', default=None)
    ap.add_argument('--pair', nargs=2, default=['lambda', 'e'],
                    help='the two parameters for the maps (default: lambda and '
                         'e). Pass any two swept parameters')
    ap.add_argument('--facet', default='auto',
                    help="split into one block per value of this column instead "
                         "of averaging over it. 'auto' (default) picks the first "
                         "model-mode column that varies; 'none' disables it")
    ap.add_argument('--facet-files', action='store_true',
                    help='write a separate set of figures per facet level '
                         '(suffixed _<column><value>) instead of stacking the '
                         'levels into one figure')
    ap.add_argument('--log', nargs='*', default=None)
    ap.add_argument('--dot-size', type=float, default=9)
    ap.add_argument('--maps', nargs='+', default=['dots', 'smooth'],
                    choices=('dots', 'smooth'),
                    help='which map figures to draw (default: both)')
    ap.add_argument('--smooth-bw', type=float, default=0.16,
                    help='kernel width of the smoothed maps, in normalised units')
    ap.add_argument('--scale', default='relative', choices=('relative', 'absolute'),
                    help='colour scale of the smoothed maps')
    ap.add_argument('--dark', action='store_true')
    ap.add_argument('--facet-labels', '--facet_labels', default=None,
                    help='rename facet levels, e.g. "0=Spatially structured,'
                         '1=Mixed". Without this the header reads "col = value"')
    ap.add_argument('--defense-label', '--defense_label', default=None,
                    help='legend text for the defense_evolved category. Set it '
                         'to match the rule classify.py was run with, since the '
                         'results CSV records the verdict and not the criterion')
    ap.add_argument('--font-scale', '--font_scale', type=float, default=1.0,
                    help='multiply every font size at once (default 1.0, sized '
                         'for a figure shrunk to a journal column)')
    args = ap.parse_args()
    if args.defense_label:
        PRETTY['defense_evolved'] = args.defense_label
    if args.facet_labels:
        for bit in args.facet_labels.split(','):
            k, _, v = bit.partition('=')
            k = k.strip()
            try:
                k = int(k)
            except ValueError:
                pass
            FACET_LABELS.setdefault(args.facet, {})[k] = v.strip()
    if args.font_scale != 1.0:
        scale_fonts(args.font_scale)

    df = pd.read_csv(args.results)
    if 'outcome' not in df.columns:
        sys.exit('%s has no outcome column -- run classify.py first' % args.results)
    df['defense_present'] = pd.to_numeric(df.get('defense_present', 0),
                                          errors='coerce').fillna(0)
    if args.facet == 'auto':
        # a column that changes the model, not one that tunes it, should never be
        # averaged into the marginals
        args.facet = next((c for c in ('mixed', 'perfect_mix', 'perfect_diffusion',
                                       'scheme', 'diffusion')
                           if c in df.columns and df[c].nunique() > 1), None)
        if args.facet:
            print('faceting on %s (use --facet none to disable)' % args.facet)
    elif args.facet == 'none':
        args.facet = None
    if args.facet:
        if args.facet not in df.columns:
            sys.exit('no column %r in %s' % (args.facet, args.results))
        df[args.facet] = pd.to_numeric(df[args.facet], errors='coerce')

    params = detect_params(df, args.params, args.facet)
    if not params:
        sys.exit('no swept parameters found -- was the design CSV joined in?')
    logs = {p: True for p in (args.log or [])}
    th = THEME['dark' if args.dark else 'light']

    if args.pair:
        pair = tuple(args.pair)
    else:
        spread = {}
        keep = df['outcome'].isin(SURVIVED)
        for p in params:
            v = pd.to_numeric(df[p], errors='coerce')
            e = bin_edges(v, args.bins, logs.get(p, is_log(v)))
            i = np.clip(np.digitize(v, e[1:-1]), 0, args.bins - 1)
            m = [df['defense_present'][(i == b) & keep].mean()
                 for b in range(args.bins) if ((i == b) & keep).sum() >= 5]
            spread[p] = (max(m) - min(m)) if len(m) > 1 else 0.0
        pair = tuple(sorted(params, key=lambda p: -spread[p])[:2])
        print('maps over the two most influential parameters: %s, %s' % pair)

    os.makedirs(args.outdir, exist_ok=True)
    j = lambda stem: os.path.join(args.outdir, args.prefix + stem)

    def draw(sub, suffix='', facet=None):
        fig_by_parameter(sub, params, args.bins, th, logs,
                         j('outcomes_by_parameter' + suffix), facet=facet)
        if 'dots' in args.maps:
            fig_maps(sub, pair, th, logs, j('outcome_maps' + suffix),
                     facet=facet, size=args.dot_size)
        if 'smooth' in args.maps:
            fig_maps_smooth(sub, pair, th, logs, j('outcome_maps_smooth' + suffix),
                            facet=facet, bw=args.smooth_bw, scale=args.scale)

    if args.facet and args.facet_files:
        for lev in sorted(df[args.facet].dropna().unique()):
            draw(df[df[args.facet] == lev], '_%s%g' % (args.facet, lev))
    else:
        draw(df, '', args.facet)


if __name__ == '__main__':
    main()
