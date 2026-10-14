#!/usr/bin/env python3
"""
Extract the raw alpha-versus-time data from a run, for plotting elsewhere.

Every model_numba.py output line already holds the full alpha histogram at that
timestep (100 bins, 0 to 1) for the provirus and virion pools. This reshapes
those into a table, including the pooled density, so display.py's two alpha
panels can be reproduced exactly outside the viewer.

  long (default)  one row per (time, alpha bin):
                  time, alpha, provirus, virion, pool, n_provirus, n_virion
  --wide          one row per timestep, one column per alpha bin, for a single
                  pool chosen with --pool

Densities are fractions of their pool and sum to 1 across the bins at each
timestep (0 when that pool is empty). `pool` is classify.py's pool_density:
each pool normalised then averaged, so neither is drowned by the other.

    python3 alpha_matrix.py lhs_out/run_00000.txt -o alpha.csv
    python3 alpha_matrix.py run.txt --wide --pool pool -o heatmap.csv
    python3 alpha_matrix.py run.txt --wide --pool both -o run.csv   # two files
    python3 alpha_matrix.py run.txt --pool both --png heat.png      # two panels
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import classify as C


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runfile')
    ap.add_argument('-o', '--out', default=None, help='CSV to write (default: stdout)')
    ap.add_argument('--wide', action='store_true',
                    help='matrix layout: one row per timestep, one column per bin')
    ap.add_argument('--pool', default='pool',
                    choices=('pool', 'provirus', 'virion', 'both'),
                    help="which density to use (default: pool). 'both' draws "
                         'the provirus and virion panels separately on one '
                         'shared colour scale; for --wide it is not a valid '
                         'choice, use provirus or virion')
    ap.add_argument('--every', type=int, default=1,
                    help='keep every Nth output line (thins a long run)')
    ap.add_argument('--min-count', type=float, default=200,
                    help='pool smaller than this is left out of the mean, as in '
                         'classify.py')
    ap.add_argument('--png', nargs='?', const='', default=None,
                    help='also render the heatmap, so the CSV can be checked '
                         'against the picture it is meant to reproduce '
                         '(default name: alongside the CSV)')
    ap.add_argument('--decades', type=float, default=3.0,
                    help='log brightness range, matching display.py')
    args = ap.parse_args()

    lines = [l for l in open(args.runfile).read().splitlines() if l.strip()]
    while lines and not (lines[-1].startswith('{') and lines[-1].endswith('}')):
        lines.pop()
    if not lines:
        sys.exit('no complete records in %s' % args.runfile)
    recs = [json.loads(l) for l in lines][::args.every]

    fh = open(args.out, 'w', newline='') if args.out else sys.stdout
    w = csv.writer(fh)
    centres = C.CENTRES

    def write_wide(writer, mode):
        writer.writerow(['time'] + ['%.3f' % a for a in centres])
        for r in recs:
            P = np.array(r['provirus'], float)
            V = np.array(r['virion'], float)
            h = {'pool': lambda: C.pool_density(P, V, 'mean', args.min_count),
                 'provirus': lambda: P, 'virion': lambda: V}[mode]()
            tot = h.sum()
            d = h / tot if tot > 0 else h
            writer.writerow([r['time']] + ['%.6g' % x for x in d])

    if args.wide and args.pool == 'both':
        # one density per column is the whole point of the wide format, so the
        # two pools cannot share a file -- give them one each
        if not args.out:
            sys.exit('--wide --pool both writes two files, so it needs -o')
        stem, ext = os.path.splitext(args.out)
        written = []
        for mode in ('provirus', 'virion'):
            path = '%s_%s%s' % (stem, mode, ext or '.csv')
            with open(path, 'w', newline='') as fh2:
                write_wide(csv.writer(fh2), mode)
            written.append(path)
        fh.close()
        if os.path.exists(args.out) and args.out not in written:
            os.remove(args.out)
        print('wrote %s  (%d timesteps x %d alpha bins each)'
              % (' and '.join(written), len(recs), len(centres)))
        if args.png is not None:
            render(recs, args)
        return

    if args.wide:
        write_wide(w, args.pool)
    else:
        w.writerow(['time', 'alpha', 'provirus', 'virion', 'pool',
                    'n_provirus', 'n_virion'])
        for r in recs:
            P = np.array(r['provirus'], float)
            V = np.array(r['virion'], float)
            nP, nV = P.sum(), V.sum()
            pool = C.pool_density(P, V, 'mean', args.min_count)
            nPool = pool.sum()
            dP = P / nP if nP > 0 else P
            dV = V / nV if nV > 0 else V
            dPool = pool / nPool if nPool > 0 else pool
            for i, a in enumerate(centres):
                w.writerow([r['time'], '%.3f' % a, '%.6g' % dP[i],
                            '%.6g' % dV[i], '%.6g' % dPool[i], int(nP), int(nV)])
    if args.out:
        fh.close()
        print('wrote %s  (%d timesteps x %d alpha bins)' % (args.out, len(recs), 100))

    if args.png is not None:
        render(recs, args)


def render(recs, args):
    """The same heatmaps display.py draws: log brightness, alpha 0 at the bottom.

    With --pool both, the provirus and virion pools get one panel each, sharing a
    single colour ramp and a single normalising maximum. Every column is a share
    of its own pool and sums to 1, so equal brightness means an equal share in
    either panel -- scaling them separately would invent a difference that is not
    in the data. This is the split that matters: a low-alpha defense sits in host
    genomes as provirus while a high-alpha parasite exists as free virions, so
    the pooled panel can hide one behind the other.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.fonttype'] = 'none'

    t = np.array([r['time'] for r in recs], float)
    modes = ['provirus', 'virion'] if args.pool == 'both' else [args.pool]

    mats = []
    for mode in modes:
        M = np.zeros((100, len(recs)))
        for k, r in enumerate(recs):
            P = np.array(r['provirus'], float)
            V = np.array(r['virion'], float)
            h = {'pool': lambda: C.pool_density(P, V, 'mean', args.min_count),
                 'provirus': lambda: P, 'virion': lambda: V}[mode]()
            tot = h.sum()
            if tot > 0:
                M[:, k] = h / tot
        mats.append(M)

    m = max(M.max() for M in mats)          # one maximum across every panel

    fig, axes = plt.subplots(len(mats), 1, figsize=(11, 3.6 * len(mats)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor('#1a1a19')
    for ax, mode, M in zip(axes, modes, mats):
        Z = (np.clip(np.log10(M / m * (10 ** args.decades - 1) + 1) / args.decades,
                     0, 1) if m > 0 else M)
        ax.set_facecolor('#1a1a19')
        ax.imshow(Z, origin='lower', aspect='auto', cmap='Blues',
                  extent=[t.min(), t.max(), 0, 1], vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_ylabel('alpha', color='#ffffff')
        ax.set_title('%s density' % mode, color='#ffffff', fontsize=10, loc='left')
        ax.tick_params(colors='#c3c2b7')
        for sp in ax.spines.values():
            sp.set_color('#333330')
    axes[-1].set_xlabel('timestep', color='#ffffff')
    fig.suptitle('%s  --  log brightness over %g decades, shared scale'
                 % (os.path.basename(args.runfile), args.decades),
                 color='#ffffff', fontsize=10, x=0.005, ha='left')
    fig.tight_layout()

    out = args.png or ((os.path.splitext(args.out)[0] if args.out else
                        os.path.splitext(args.runfile)[0]) + '_heatmap.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#1a1a19')
    plt.close(fig)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
