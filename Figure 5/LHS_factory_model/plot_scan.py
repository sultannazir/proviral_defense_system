#!/usr/bin/env python3
"""
Plot a controlled 1D scan: outcome probabilities against one parameter.

Unlike plot_outcomes.py, which bins a Latin hypercube, this expects repeated
seeds at each value of one parameter, so every point is a real binomial
proportion with an exact Wilson interval.

    python3 plot_scan.py diffusion_scan/results.csv --param diff_rate --log
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'

from plot_outcomes import THEME, PRETTY, OUTCOMES, wilson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--param', default='diff_rate')
    ap.add_argument('--log', action='store_true')
    ap.add_argument('--out', default='plots/diffusion_scan')
    ap.add_argument('--formats', nargs='+', default=['png', 'svg'])
    ap.add_argument('--dark', action='store_true')
    args = ap.parse_args()

    th = THEME['dark' if args.dark else 'light']
    df = pd.read_csv(args.results)
    df['finished'] = df.get('finished', 1)
    df = df[df['finished'] == 1]
    surv = df['outcome'].isin(['defense_evolved', 'no_defense'])

    xs = sorted(df[args.param].unique())
    S = th['series']
    curves = [
        ('P(survival): virus + host at the end', S[0],
         lambda g: (int(g['_s'].sum()), len(g))),
        ('P(hosts died out)', S[1],
         lambda g: (int((g['outcome'] == 'host_extinct').sum()), len(g))),
        ('P(defense type | survived)', S[2],
         lambda g: (int(g.loc[g['_s'], 'defense_present'].sum()), int(g['_s'].sum()))),
    ]
    df = df.assign(_s=surv)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.35, 1],
                                                   hspace=0.16))
    fig.patch.set_facecolor(th['surface'])
    for a in (ax, ax2):
        a.set_facecolor(th['surface'])
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            a.spines[sp].set_color(th['grid'])
        a.tick_params(colors=th['secondary'], labelsize=9)
        a.grid(color=th['grid'], linewidth=0.7)
        a.set_axisbelow(True)

    for label, colour, fn in curves:
        pt, lo, hi = [], [], []
        for x in xs:
            k, n = fn(df[df[args.param] == x])
            pt.append(k / n if n else np.nan)
            a, b = wilson(k, n)
            lo.append(a); hi.append(b)
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(xs, pt, color=colour, linewidth=2, marker='o', markersize=5,
                markeredgecolor=th['surface'], markeredgewidth=1, label=label)

    # composition among survivors, stacked
    present = [o for o in OUTCOMES if o in ('no_defense', 'defense_evolved')]
    bottom = np.zeros(len(xs))
    width = np.diff(np.log(xs)).min() * 0.8 if args.log else np.diff(xs).min() * 0.8
    for k, o in enumerate(present):
        frac = []
        for x in xs:
            g = df[(df[args.param] == x) & df['_s']]
            frac.append((g['outcome'] == o).mean() if len(g) else 0.0)
        frac = np.array(frac)
        xw = [x * math.exp(width / 2) - x * math.exp(-width / 2) for x in xs] if args.log \
            else [width] * len(xs)
        ax2.bar(xs, frac, bottom=bottom, width=xw, color=S[2 + k],
                edgecolor=th['surface'], linewidth=1.0, label=PRETTY.get(o, o))
        bottom += frac

    if args.log:
        ax.set_xscale('log'); ax2.set_xscale('log')
    ax.set_ylim(-0.03, 1.03); ax2.set_ylim(0, 1)
    ax.set_ylabel('probability', color=th['secondary'], fontsize=10)
    ax2.set_ylabel('composition of survivors', color=th['secondary'], fontsize=10)
    ax2.set_xlabel(args.param + (' (log)' if args.log else ''),
                   color=th['primary'], fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=th['primary'], loc='upper left')
    ax2.legend(frameon=False, fontsize=9, labelcolor=th['primary'],
               loc='upper left', ncol=3)
    n_per = df.groupby(args.param).size().median()
    fig.text(0.01, 0.99, 'Controlled scan over %s  (%d values, %d seeds each, '
             'all other parameters fixed)' % (args.param, len(xs), n_per),
             color=th['primary'], fontsize=12, ha='left', va='top')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for ext in args.formats:
        p = '%s.%s' % (args.out, ext)
        fig.savefig(p, dpi=160, bbox_inches='tight', facecolor=th['surface'])
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    main()
