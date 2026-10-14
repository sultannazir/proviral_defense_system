#!/usr/bin/env python3
"""
Show what the classifier actually saw, run by run.

For a sample of runs (a few from each outcome class by default) this draws three
panels side by side:

  1. population time course     hosts (S and I), free+factory virions, proviruses
  2. low-alpha fraction v time  the quantity the defense call is made on, with
                                the --defense-frac line the classifier compares to
  3. final alpha histograms     provirus and virion pools, with the --low-alpha
                                cut drawn in

The window used for panel 3 is the same one classify.py uses, and the label in
each row's title is classify.py's own verdict, so the figure is an audit of the
classifier rather than a separate reimplementation of it.

    python3 plot_runs.py runs_n1000/lhs_results.csv
    python3 plot_runs.py lhs_results.csv --per-outcome 4
    python3 plot_runs.py lhs_results.csv --runs lhs_out/run_00042.txt
"""

import argparse
import json
import textwrap
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify as C
from plot_outcomes import THEME, PRETTY

# same hue for the same thing in every panel
SLOT = {'num_S': 0, 'num_I': 1, 'virion': 2, 'provirus': 3, 'pool': 4}


def load(path):
    recs = [json.loads(l) for l in open(path).read().splitlines() if l.strip()]
    t = np.array([r['time'] for r in recs], float)
    P = np.array([r['provirus'] for r in recs], float)
    V = np.array([r['virion'] for r in recs], float)
    return dict(t=t, P=P, V=V,
                num_S=np.array([r['num_S'] for r in recs], float),
                num_I=np.array([r['num_I'] for r in recs], float),
                nP=P.sum(axis=1), nV=V.sum(axis=1))


def frac_low(H, cut):
    tot = H.sum(axis=1)
    lo = H[:, C.CENTRES < cut].sum(axis=1)
    return np.divide(lo, tot, out=np.full_like(tot, np.nan), where=tot > 0)


def frac_low_pool(P, V, cut, how='mean', min_count=200):
    """The classifier's own quantity, using classify.py's own pooling."""
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        h = C.pool_density(P[i], V[i], how, min_count)
        if h.sum() > 0:
            out[i] = h[C.CENTRES < cut].sum() / h.sum()
    return out


def pick(df, per_outcome, explicit):
    if explicit:
        return [(f, df[df['outfile'] == f].iloc[0] if (df['outfile'] == f).any() else None)
                for f in explicit]
    out = []
    from plot_outcomes import OUTCOMES
    for o in OUTCOMES:
        g = df[df['outcome'] == o]
        if not len(g):
            continue
        # spread the picks across the group rather than taking the first few
        idx = np.linspace(0, len(g) - 1, min(per_outcome, len(g))).round().astype(int)
        for i in sorted(set(idx.tolist())):
            r = g.iloc[i]
            out.append((r['outfile'], r))
    return out


def draw_row(a1, a2, a3, path, meta, args, th, S, last_row=True):
    """Draw one run: populations, low-alpha fraction, final alpha spectrum."""
    d = load(path)
    for a in (a1, a2, a3):
        a.set_facecolor(th['surface'])
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            a.spines[sp].set_color(th['grid'])
        a.tick_params(colors=th['secondary'], labelsize=7.5, length=3)
        a.grid(color=th['grid'], linewidth=0.6)
        a.set_axisbelow(True)

    # ---- 1. populations
    for name, y in (('num_S', d['num_S']), ('num_I', d['num_I']),
                    ('virion', d['nV']), ('provirus', d['nP'])):
        a1.plot(d['t'], y, color=S[SLOT[name]], linewidth=1.8, label=name,
                marker='o' if len(d['t']) < 3 else None, markersize=4)
    # symlog, not log: a count of exactly zero is the whole point on these
    # plots, and a log axis either drops it or fakes it as a small number.
    a1.set_yscale('symlog', linthresh=1)
    a1.set_ylim(0, max(1e5, d['num_S'].max() * 2, d['nP'].max() * 2))
    a1.set_ylabel('count', color=th['secondary'], fontsize=8)

    # ---- 2. low-alpha fraction over time (what the defense call reads)
    a2.plot(d['t'], frac_low(d['P'], args.low_alpha), color=S[SLOT['provirus']],
            linewidth=1.4, alpha=0.75, label='provirus')
    a2.plot(d['t'], frac_low(d['V'], args.low_alpha), color=S[SLOT['virion']],
            linewidth=1.4, alpha=0.75, label='virion')
    a2.plot(d['t'], frac_low_pool(d['P'], d['V'], args.low_alpha,
                                  args.pool_how, args.min_count),
            color=S[SLOT['pool']], linewidth=2.0, zorder=5, label='pool')
    # the single cut the label is read off
    a2.axhline(args.defense_lo, color=th['muted'], linewidth=1,
               linestyle=(0, (4, 3)))
    a2.text(0.015, args.defense_lo, ' defense evolved above', va='bottom',
            fontsize=6, color=th['muted'], transform=a2.get_yaxis_transform())
    tdv = meta.get('t_defense', '')
    if tdv != '' and tdv == tdv:
        a2.axvline(float(tdv), color=th['primary'], linewidth=1,
                   linestyle=(0, (2, 2)), alpha=0.6)
    a2.set_ylim(-0.03, 1.03)
    a2.set_ylabel('defense fraction\n(alpha < %g)' % args.low_alpha,
                  color=th['secondary'], fontsize=8)

    # ---- 3. final alpha histograms (exactly the window classify.py used)
    survived = d['nV'][-1] > 0 and (d['num_S'][-1] + d['num_I'][-1]) > 0
    sl = slice(-args.window, None) if survived else slice(-1, None)
    drew = False
    above = None
    floor = args.hist_floor
    series = [('provirus', d['P'][sl]), ('virion', d['V'][sl])]
    if args.show_pool:
        # the pool classify.py actually reads the label off
        series.append(('pool', C.pool_density(d['P'][sl].mean(axis=0),
                                              d['V'][sl].mean(axis=0),
                                              args.pool_how, args.min_count)[None, :]))
    for name, H in series:
        h = H.mean(axis=0)
        tot = h.sum()
        if tot <= 0:
            continue
        drew = True
        dens = h / tot
        if name == 'virion':
            above = float(dens[C.CENTRES >= args.low_alpha].sum())
        if name == 'pool':
            a3.step(C.CENTRES, np.maximum(dens, floor) if args.hist_scale == 'log'
                    else dens, where='mid', color=S[SLOT[name]], linewidth=1.6,
                    label=name, zorder=5)
            continue
        if args.hist_scale == 'log':
            # A defense that has taken over puts ~all the density in one bin,
            # which on a linear axis flattens the surviving high-alpha
            # parasite to an invisible line even when it is several percent
            # of the pool. Log y shows both on one axis.
            dens = np.maximum(dens, floor)
        a3.fill_between(C.CENTRES, floor if args.hist_scale == 'log' else 0,
                        dens, color=S[SLOT[name]], alpha=0.5, linewidth=0,
                        step='mid')
        a3.step(C.CENTRES, dens, where='mid', color=S[SLOT[name]],
                linewidth=1.4, label=name)
    a3.axvline(args.low_alpha, color=th['muted'], linewidth=1, linestyle=(0, (4, 3)))
    a3.set_xlim(0, 1)
    if drew and args.hist_scale == 'log':
        a3.set_yscale('log')
        a3.set_ylim(floor, 1.6)
    lab = meta.get('outcome', '')
    fl = meta.get('frac_low_pool', '')
    if fl != '' and fl == fl:
        a3.text(0.98, 0.96, 'defense fraction %.3f' % float(fl),
                ha='right', va='top', transform=a3.transAxes, fontsize=7,
                fontweight='bold' if lab == 'both' else 'normal',
                color=th['primary'] if lab == 'both' else th['muted'])
    if not drew:
        a3.set_ylim(0, 1)
        a3.text(0.5, 0.5, 'nothing left to measure', ha='center', va='center',
                color=th['muted'], fontsize=8.5, transform=a3.transAxes)
    a3.set_xlabel('alpha', color=th['secondary'], fontsize=8)
    a3.set_ylabel('density', color=th['secondary'], fontsize=8)

    if last_row:
        a1.set_xlabel('timestep', color=th['secondary'], fontsize=8)
        a2.set_xlabel('timestep', color=th['secondary'], fontsize=8)

    # ---- verdict + why
    pv = ', '.join('%s=%.3g' % (p, float(meta[p])) for p in args.params
                   if p in meta.index)
    na = lambda v: 'n/a' if (v == '' or v != v) else ('%.3f' % float(v))
    td = meta.get('t_defense', '')
    td = 'never' if (td == '' or td != td) else '%d' % float(td)
    why = ('defense fraction at end %s\nvir_time %s   t_defense %s' % (
        na(meta.get('frac_low_pool', '')), meta.get('vir_time', '-'), td))
    a1.set_title('%s   —   %s' % (os.path.basename(path),
                                  PRETTY.get(meta['outcome'], meta['outcome'])),
                 color=th['primary'], fontsize=9.5, loc='left', fontweight='bold')
    a2.set_title(why, color=th['secondary'], fontsize=7.5, loc='left')
    a3.set_title('\n'.join(textwrap.wrap(pv, 52)), color=th['secondary'],
                 fontsize=6.8, loc='left')



def legend_handles(args, th, S):
    keys = ('num_S', 'num_I', 'virion', 'provirus') + (
        ('pool',) if args.show_pool else ())
    h = [plt.Line2D([], [], color=S[SLOT[k_]], linewidth=2,
                    label='pool (classified on)' if k_ == 'pool' else k_)
         for k_ in keys]
    h.append(plt.Line2D([], [], color=th['muted'], linewidth=1,
                        linestyle=(0, (4, 3)),
                        label='defense cut (%g of pool below alpha %g)'
                              % (args.defense_lo, args.low_alpha)))
    h.append(plt.Line2D([], [], color=th['primary'], linewidth=1,
                        linestyle=(0, (2, 2)), alpha=0.6,
                        label='t_defense'))
    return h


def draw_one(job):
    """Render a single run to its own PNG. Top level so it can be pooled."""
    path, meta, args, th = job
    S = th['series']
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 2.9),
                                     gridspec_kw=dict(width_ratios=[1.5, 1.0, 1.1],
                                                      wspace=0.30))
    fig.patch.set_facecolor(th['surface'])
    draw_row(a1, a2, a3, path, meta, args, th, S, last_row=True)
    fig.legend(handles=legend_handles(args, th, S), loc='lower center', ncol=7,
               frameon=False, fontsize=7.5, labelcolor=th['primary'],
               bbox_to_anchor=(0.5, -0.16))
    out = os.path.splitext(path)[0] + '.png'
    if args.each_dir:
        out = os.path.join(args.each_dir, os.path.basename(out))
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor=th['surface'])
    plt.close(fig)
    return out


def render_each(df, args, th):
    """One PNG per run, written next to its .txt. Skips runs that died out."""
    sel = df[df['outcome'].isin(args.each_outcomes)]
    sel = sel[[os.path.exists(f) for f in sel['outfile']]]
    if not len(sel):
        print('no runs to render individually')
        return
    if args.each_dir:
        os.makedirs(args.each_dir, exist_ok=True)
    jobs = [(r['outfile'], r, args, th) for _, r in sel.iterrows()]
    if args.skip_existing:
        def done(j):
            out = os.path.splitext(j[0])[0] + '.png'
            if args.each_dir:
                out = os.path.join(args.each_dir, os.path.basename(out))
            return os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(j[0])
        before = len(jobs)
        jobs = [j for j in jobs if not done(j)]
        if before - len(jobs):
            print('  skipping %d already up to date' % (before - len(jobs)))
    print('rendering %d per-run figures (%s)...'
          % (len(jobs), ', '.join(args.each_outcomes)))
    n = 0
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for _ in ex.map(draw_one, jobs, chunksize=8):
                n += 1
                if n % 250 == 0:
                    print('  %d/%d' % (n, len(jobs)))
    else:
        for j in jobs:
            draw_one(j)
            n += 1
    print('  wrote %d per-run figures' % n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', help='CSV from classify.py')
    ap.add_argument('--per-outcome', type=int, default=3)
    ap.add_argument('--runs', nargs='*', default=None,
                    help='specific output files instead of a stratified sample')
    ap.add_argument('--low-alpha', type=float, default=0.1,
                    help='alpha below this counts as a defense element')
    ap.add_argument('--defense-lo', type=float, default=0.25,
                    help='a defense evolved if more than this fraction of the '
                         'pool is below --low-alpha')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--out', default='classifier_check',
                    help='output stem; a bare name lands in --plot-dir')
    ap.add_argument('--plot-dir', default='plots',
                    help='where a bare --out stem is written')
    ap.add_argument('--pool', dest='pool_how', default='mean',
                    choices=('mean', 'both', 'provirus', 'virion'),
                    help='how classify.py combined the two histograms')
    ap.add_argument('--min-count', type=float, default=200)
    ap.add_argument('--hist-scale', default='linear', choices=('log', 'linear'),
                    help='y scale of the alpha histograms; log pulls a rare tail '
                         'into view')
    ap.add_argument('--show-pool', action='store_true', default=True,
                    help='overlay the pool the classifier reads')
    ap.add_argument('--no-show-pool', dest='show_pool', action='store_false')
    ap.add_argument('--hist-floor', type=float, default=1e-5,
                    help='lower limit of the log density axis')
    ap.add_argument('--formats', nargs='+', default=['png', 'svg'])
    ap.add_argument('--each', action='store_true',
                    help='also write one PNG per run, next to its .txt file')
    ap.add_argument('--each-dir', default=None,
                    help='write the per-run PNGs here instead of next to the .txt')
    ap.add_argument('--each-outcomes', nargs='+',
                    default=['defense_evolved', 'no_defense'],
                    help='which outcomes get a per-run figure (default: the runs '
                         'that still had a virus at the end)')
    ap.add_argument('--jobs', '-j', type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help='workers for the per-run figures')
    ap.add_argument('--skip-existing', action='store_true',
                    help='do not redraw a per-run PNG that is newer than its .txt')
    ap.add_argument('--only-each', action='store_true',
                    help='skip the combined figure and only write per-run PNGs')
    ap.add_argument('--dark', action='store_true')
    ap.add_argument('--params', nargs='*',
                    default=['e', 'lambda', 'muA', 'step', 'diff_rate',
                             'decay_rate', 'beta'])
    args = ap.parse_args()

    th = THEME['dark' if args.dark else 'light']
    if not os.path.dirname(args.out):
        os.makedirs(args.plot_dir, exist_ok=True)
        args.out = os.path.join(args.plot_dir, args.out)
    df = pd.read_csv(args.results)
    if args.each or args.only_each:
        render_each(df, args, th)
        if args.only_each:
            return
    rows = [(f, r) for f, r in pick(df, args.per_outcome, args.runs)
            if r is not None and os.path.exists(f)]
    if not rows:
        sys.exit('none of the selected output files exist (run from the directory '
                 'that holds the paths in the outfile column)')

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 2.45 * n),
                             gridspec_kw=dict(width_ratios=[1.5, 1.0, 1.1],
                                              hspace=1.05, wspace=0.32))
    if n == 1:
        axes = axes.reshape(1, 3)
    fig.patch.set_facecolor(th['surface'])
    S = th['series']

    for k, (path, meta) in enumerate(rows):
        draw_row(*axes[k], path, meta, args, th, S, last_row=(k == n - 1))

    keys = ('num_S', 'num_I', 'virion', 'provirus') + (
        ('pool',) if args.show_pool else ())
    handles = [plt.Line2D([], [], color=S[SLOT[k_]], linewidth=2,
                          label='pool (classified on)' if k_ == 'pool' else k_)
               for k_ in keys]
    handles.append(plt.Line2D([], [], color=th['muted'], linewidth=1,
                              linestyle=(0, (4, 3)),
                              label='defense cut (%g of pool below alpha %g)'
                                    % (args.defense_lo, args.low_alpha)))
    handles.append(plt.Line2D([], [], color=th['primary'], linewidth=1,
                              linestyle=(0, (2, 2)), alpha=0.6,
                              label='t_defense (when the lineage was first called)'))
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               fontsize=9, labelcolor=th['primary'], bbox_to_anchor=(0.5, -0.02 / max(n, 1)))
    fig.text(0.01, 1.0 - 0.10 / n, 'Classifier check: time course, low-alpha '
             'fraction, and final alpha distribution',
             color=th['primary'], fontsize=12, ha='left', va='top')

    for ext in args.formats:
        p = '%s.%s' % (args.out, ext)
        fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=th['surface'])
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    main()
