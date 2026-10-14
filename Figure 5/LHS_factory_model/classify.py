#!/usr/bin/env python3
"""
Classify the outcome of virus-factory runs and summarise a sweep into one CSV.

Each simulation writes JSON lines every --interval steps holding num_S, num_I
and three 100-bin histograms (provirus alpha, virion alpha, proviruses per
host). This reads them and labels each run.

Every run gets one of four labels, decided in this order:

  host_extinct     no hosts left at the end. Nothing else can be said, so this
                   is checked first.
  virus_extinct    hosts remain and the virus is gone in BOTH forms -- zero free
                   virions AND zero proviruses. A population sitting on a bank
                   of prophages with no virions left is NOT extinct: the prophages
                   are still there and (with --sigma > 0) can induce and restart
                   the cycle. Those runs fall through to the defense check below,
                   which is usually where they belong -- driving the free virus
                   out is what a defense system does.
  defense_evolved  more than --defense-lo (0.25) of the element pool has
                   alpha < --low-alpha (0.1)
  no_defense       everything else: the virus is present and stayed high-alpha

WHERE THE CUT SITS matters more than it looks. A real defense type is a distinct
low-alpha mode sitting near 0, not a lower tail. Under the original 0.2/0.2 the
perfectly-mixed arm produced 19 "defense_evolved" runs whose alpha distribution
was in fact a broad unimodal smear with no low mode at all -- they qualified
only because the smear's tail reached past 0.2. Tightening to 0.1/0.25 leaves 2
of those 19 (the genuine ones, including one that fixes at alpha 0) and still
keeps 91% of the spatial arm's defense calls. The 60 spatial calls it drops are
the same artefact: median pool mass below alpha 0.05 is 0.082 for the dropped
runs versus 0.599 for the kept ones.

Also recorded for every run so the categories can be split further afterwards:
`vir_time` (how long free virions lasted), `established`, `t_defense`,
`max_frac_low`, and the raw end-state counts.

WHICH POOL (--pool) is the one subtle part. The two lifestyles live in different
places: a low-alpha defense integrates and sits in every host genome, while a
high-alpha parasite replicates and exists as free virions. Proviruses can
outnumber virions 50:1, so pooling raw COUNTS makes the provirus pool the answer
and erases the parasite. The default, 'mean', normalises each pool to a density
first and averages them, weighing each type by its presence rather than its
census size.

Also recorded for every run, including the excluded ones, so that runs where a
defense evolved and then drove the parasite out can still be pulled out:

  established   the virus persisted past --establish-steps
  defense       the defense fraction crossed --defense-lo for --min-run
                consecutive outputs at any point in the run
  t_defense     when that first happened
  max_frac_low  the highest defense fraction reached at any time

    python3 classify.py lhs_out --design lhs_design.csv -o lhs_results.csv

Add --figures to also write one PNG per surviving run, next to its .txt. Both
are safe to run while the sweep is still going if you pass --finished-only:

    python3 classify.py lhs_out --design lhs_design.csv -o preview_results.csv \
        --finished-only --figures
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

NBINS = 100
CENTRES = (np.arange(NBINS) + 0.5) / NBINS
HERE = os.path.dirname(os.path.abspath(__file__))


def pool_density(P, V, how='mean', min_count=200):
    """Combine the provirus and virion histograms into the pool to classify on.

    'mean' normalises each to a density and averages, so a lytic parasite living
    as a few thousand virions is not drowned by 50x more proviruses. A pool with
    fewer than `min_count` individuals is noise and is left out; if neither
    qualifies, fall back to the raw sum so the caller still gets something.
    """
    if how == 'provirus':
        return P.astype(float)
    if how == 'virion':
        return V.astype(float)
    if how == 'both':
        return P + V
    dens = [X / X.sum() for X in (P, V) if X.sum() >= min_count]
    if not dens:
        return P + V
    return np.mean(dens, axis=0) * (P.sum() + V.sum())


def low_share(P, V, lowmask, rule='pool', pool_how='mean', min_count=200):
    """The statistic the defense call is made on, under one of two rules.

    'pool'            the pooled density: provirus and virion each normalised
                      then averaged, so neither pool is drowned by the other.
                      The share of THAT of elements below low_alpha.
    'provirus-share'  the share of ALL virus copies -- proviruses plus free
                      virions, by raw count -- that are proviruses below
                      low_alpha. A stricter, more literal reading: it demands
                      the defense actually dominate the total virus population,
                      not merely dominate the provirus pool.
    """
    if rule == 'provirus-share':
        tot = P.sum() + V.sum()
        return float(P[lowmask].sum() / tot) if tot > 0 else 0.0
    h = pool_density(P, V, pool_how, min_count)
    return float(h[lowmask].sum() / h.sum()) if h.sum() > 0 else 0.0


def summarise(path, window=10, low_alpha=0.1, defense_lo=0.25, defense_fix=0.95,
              fix_alpha=0.2, establish_steps=2000, min_run=3, pool_how='mean',
              min_count=200, maxtime=200000, defense_rule='pool'):
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    # a live sweep is appending to these files; drop a half-written last line
    while lines and not (lines[-1].startswith('{') and lines[-1].endswith('}')):
        lines.pop()
    if not lines:
        return None
    traj = [json.loads(l) for l in lines]
    last = traj[-1]
    lowmask = CENTRES < low_alpha

    # ---- whole-trajectory quantities
    times = np.array([r['time'] for r in traj])
    Ptraj = np.array([r['provirus'] for r in traj], float)
    Vtraj = np.array([r['virion'] for r in traj], float)
    nVt = Vtraj.sum(axis=1)
    alive = nVt > 0
    vir_time = int(times[alive][-1]) if alive.any() else 0
    established = int(vir_time > establish_steps)

    track = np.empty(len(traj))
    for i in range(len(traj)):
        track[i] = low_share(Ptraj[i], Vtraj[i], lowmask, defense_rule,
                             pool_how, min_count)

    # sustained for min_run consecutive outputs, so a blip is not a lineage
    defense, t_defense = 0, ''
    run = 0
    for i, over in enumerate(track >= defense_lo):
        run = run + 1 if over else 0
        if run >= min_run:
            defense = 1
            t_defense = int(times[i - min_run + 1])
            break

    n_vir_last = sum(last['virion'])
    n_prov_last = sum(last['provirus'])
    n_host_last = last['num_S'] + last['num_I']
    # the virus counts as present if it exists in EITHER form
    n_virus_last = n_vir_last + n_prov_last
    # model_numba.py stops at maxtime, when the hosts die out, when the virus
    # runs out of ways to continue (no virions, and with induction no proviruses
    # either), and when the defense type fixes (--defense-fix). A run that is
    # none of those is still going and has no end state to read.
    # The model stops on ITS OWN --low-alpha (--fix-alpha here), which need not
    # be the alpha the classifier labels on. Read the stop with the model's cut,
    # or every fixation-stopped run -- the strongest defense cases in the sweep --
    # looks unfinished and gets dropped by --finished-only.
    fixmask = CENTRES < fix_alpha
    ph = pool_density(Ptraj[-1], Vtraj[-1], pool_how, min_count)
    ffix = float(ph[fixmask].sum() / ph.sum()) if ph.sum() > 0 else 0.0
    fixed = defense_fix > 0 and ffix >= defense_fix
    finished = ((last['time'] >= maxtime - 100) or n_vir_last <= 0
                or n_host_last <= 0 or fixed)
    # a fixation stop is a deliberate endpoint, so read it off the last line
    # rather than averaging a window that spans the run-up to it
    survived = n_virus_last > 0 and n_host_last > 0 and not fixed

    # ---- end state, averaged over the window for runs that are still going
    recs = traj[-window:] if survived else [last]
    P = np.array([r['provirus'] for r in recs], float).mean(axis=0)
    V = np.array([r['virion'] for r in recs], float).mean(axis=0)
    PN = np.array([r['num_provirus'] for r in recs], float).mean(axis=0)
    num_S = float(np.mean([r['num_S'] for r in recs]))
    num_I = float(np.mean([r['num_I'] for r in recs]))
    n_vir, n_prov = V.sum(), P.sum()

    frac_low_pool = low_share(P, V, lowmask, defense_rule, pool_how, min_count)

    if n_host_last <= 0:
        outcome = 'host_extinct'
    elif n_virus_last <= 0:
        outcome = 'virus_extinct'
    elif frac_low_pool > defense_lo:
        outcome = 'defense_evolved'
    else:
        outcome = 'no_defense'

    mean = lambda h: float((CENTRES * h).sum() / h.sum()) if h.sum() > 0 else ''
    frac = lambda h: round(float(h[lowmask].sum() / h.sum()), 4) if h.sum() > 0 else ''
    return {
        'outfile': path,
        'outcome': outcome,
        'frac_low_pool': round(frac_low_pool, 4),
        'defense_present': int(outcome == 'defense_evolved'),
        'finished': int(finished),
        'defense_fixed': int(fixed),
        'established': established,
        'defense': defense,
        't_defense': t_defense,
        'max_frac_low': round(float(track.max()), 4),
        'vir_time': vir_time,
        'virus_end': int(n_virus_last > 0),
        'virion_end': int(n_vir_last > 0),
        't_end': last['time'],
        'num_S': round(num_S, 1),
        'num_I': round(num_I, 1),
        'n_virion': round(float(n_vir), 1),
        'n_provirus': round(float(n_prov), 1),
        'mean_alpha_virion': round(mean(V), 4) if n_vir > 0 else '',
        'mean_alpha_provirus': round(mean(P), 4) if n_prov > 0 else '',
        'frac_provirus_low': frac(P),
        'frac_virion_low': frac(V),
        'mean_provirus_per_host': round(float((np.arange(NBINS) * PN).sum() / PN.sum()), 3)
            if PN.sum() > 0 else '',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('outdir', help='directory of simulation output .txt files')
    ap.add_argument('--design', default='lhs_design.csv',
                    help='LHS design CSV; joined on the outfile column')
    ap.add_argument('-o', '--out', default='lhs_results.csv')

    g = ap.add_argument_group('classification')
    g.add_argument('--low-alpha', type=float, default=0.1,
                   help='alpha below this counts as defense type')
    g.add_argument('--defense-rule', default='pool',
                   choices=('pool', 'provirus-share'),
                   help="what --defense-lo is a share OF. 'pool' (default) is "
                        "the pooled provirus/virion density; 'provirus-share' "
                        "is the share of all virus copies that are proviruses "
                        "below --low-alpha, counted raw")
    g.add_argument('--defense-lo', type=float, default=0.25,
                   help='a defense evolved if more than this fraction of the '
                        'element pool is below --low-alpha')
    g.add_argument('--defense-fix', type=float, default=0.95,
                   help='the --defense-fix the sweep ran with. A run that hit it '
                        'stopped on purpose and counts as finished')
    g.add_argument('--fix-alpha', type=float, default=0.2,
                   help='the --low-alpha the SWEEP ran with, used only to detect '
                        'that a run stopped on fixation. Separate from '
                        '--low-alpha, which is what the classifier labels on')
    g.add_argument('--pool', dest='pool_how', default='mean',
                   choices=('mean', 'both', 'provirus', 'virion'),
                   help='how to combine the provirus and virion histograms. '
                        '"mean" (default) averages them as densities so neither '
                        'is drowned by the other; "both" sums raw counts')
    g.add_argument('--min-count', type=float, default=200,
                   help='a pool with fewer individuals than this is left out of '
                        'the mean')
    g.add_argument('--window', type=int, default=10,
                   help='average the last N output lines (default 10 = 1000 steps)')
    g.add_argument('--establish-steps', type=int, default=2000,
                   help='the virus counts as established if it persisted longer')
    g.add_argument('--min-run', type=int, default=3,
                   help='consecutive outputs above the defense cut needed before '
                        't_defense is recorded')
    g.add_argument('--maxtime', type=int, default=200000,
                   help='the maxtime the sweep used, to tell a finished run from '
                        'one still in progress')
    g.add_argument('--finished-only', action='store_true',
                   help='drop runs still in progress. Use this to analyse a '
                        'sweep while it is still going')

    f = ap.add_argument_group('figures')
    f.add_argument('--figures', action='store_true',
                   help='also write one PNG per surviving run, next to its .txt')
    f.add_argument('--figures-dir', default=None,
                   help='write those PNGs here instead of next to the .txt')
    f.add_argument('--redraw', action='store_true',
                   help='redraw PNGs that are already up to date')
    f.add_argument('-j', '--jobs', type=int,
                   default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.outdir, '*.txt')))
    if not files:
        sys.exit('no .txt files in %s' % args.outdir)

    design, dcols = {}, []
    if args.design and os.path.exists(args.design):
        with open(args.design) as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames:
                print('warning: %s has no header row -- ignoring it' % args.design)
            elif 'outfile' not in rd.fieldnames:
                print('warning: %s has no outfile column -- cannot join it'
                      % args.design)
            else:
                dcols = [c for c in rd.fieldnames if c != 'outfile']
                for r in rd:
                    design[os.path.normpath(r['outfile'])] = r

    rows = []
    for path in files:
        s = summarise(path, window=args.window, low_alpha=args.low_alpha,
                      defense_lo=args.defense_lo, defense_fix=args.defense_fix,
                      fix_alpha=args.fix_alpha,
                      establish_steps=args.establish_steps, min_run=args.min_run,
                      pool_how=args.pool_how, min_count=args.min_count,
                      maxtime=args.maxtime, defense_rule=args.defense_rule)
        if s is None or (args.finished_only and not s['finished']):
            continue
        d = design.get(os.path.normpath(path), {})
        rows.append({**{c: d.get(c, '') for c in dcols}, **s})
    if not rows:
        sys.exit('no runs to summarise')

    cols = dcols + [c for c in rows[0] if c not in dcols]
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r['outcome']] = counts.get(r['outcome'], 0) + 1
    print('%d runs -> %s' % (len(rows), args.out))
    for k in sorted(counts, key=lambda k: -counts[k]):
        print('  %-16s %4d  (%5.1f%%)' % (k, counts[k], 100.0 * counts[k] / len(rows)))
    keep = [r for r in rows if r['outcome'] in ('defense_evolved', 'no_defense')]
    if keep:
        n = len(keep)
        print('  --')
        print('  virus still present at the end in %d of %d runs' % (n, len(rows)))
        dp = sum(r['defense_present'] for r in keep)
        print('  P(defense evolved | virus persists) = %d/%d = %.3f'
              % (dp, n, dp / n))
        pv = sum(1 for r in keep if not r['virion_end'])
        if pv:
            print('  of those, %d have proviruses but no free virions left' % pv)
    if not design:
        print('  (no design CSV joined -- parameters missing from the output)')

    if args.figures:
        cmd = [sys.executable, os.path.join(HERE, 'plot_runs.py'), args.out,
               '--only-each', '-j', str(args.jobs),
               '--low-alpha', str(args.low_alpha),
               '--defense-lo', str(args.defense_lo),
               '--pool', args.pool_how,
               '--min-count', str(args.min_count),
               '--window', str(args.window)]
        if args.figures_dir:
            cmd += ['--each-dir', args.figures_dir]
        if not args.redraw:
            cmd.append('--skip-existing')
        print()
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
