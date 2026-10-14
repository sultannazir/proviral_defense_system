#!/usr/bin/env python3
"""
Parallel sweep / LHS runner for model_numba.py.

Reads a CSV where every column is a model_numba.py option name (without the
leading '--') and every row is one simulation, then runs the rows across
multiple cores.  A column named 'outfile' is used verbatim as the output path;
otherwise the cacatoo-style filename is used inside --outdir.

    python3 run_sweep.py params.csv --jobs 10 --outdir lhs_out

It can also generate a Latin hypercube design for you:

    python3 run_sweep.py --make-lhs 1000 --vary "lambda=0.01:1,muA=1e-4:0.1:log" \
        --fixed "scheme=1,maxtime=200000" --out params.csv

Ranges are linear by default; append ':log' for a log10-uniform range.
"""

import argparse
import numpy as np
import model_numba as M
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'model_numba.py')
# bookkeeping columns that are not model arguments
def tail_record(path, block=65536):
    """Last complete JSON line of a run, or None."""
    try:
        with open(path, 'rb') as fh:
            fh.seek(0, 2)
            pos = fh.tell()
            buf = b''
            while pos > 0:
                step = min(block, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
                lines = [l for l in buf.split(b'\n')
                         if l.startswith(b'{') and l.endswith(b'}')]
                if lines:
                    return json.loads(lines[-1].decode())
    except (OSError, ValueError):
        pass
    return None


SKIP = {'point', 'rep', 'pair', 'mixed'}


def run_one(row):
    cmd = [sys.executable, MODEL, '--quiet']
    for k, v in row.items():
        if k in SKIP or v is None or str(v) == '':
            continue
        cmd += ['--' + k, str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (' '.join(cmd), r.returncode, r.stderr.strip()[-400:])


def make_lhs(n, vary, fixed, outpath, seed0, design_seed, start_index=0):
    import numpy as np
    rng = np.random.default_rng(design_seed)
    names = list(vary.keys())
    # Latin hypercube: one independent stratified permutation per dimension
    cube = np.empty((n, len(names)))
    for k in range(len(names)):
        perm = rng.permutation(n)
        cube[:, k] = (perm + rng.random(n)) / n

    outdir = fixed.pop('outdir', None)
    rows = []
    for i in range(n):
        row = {'point': start_index + i}
        for k, name in enumerate(names):
            lo, hi, logscale = vary[name]
            if logscale:
                v = 10.0 ** (math.log10(lo) + cube[i, k] * (math.log10(hi) - math.log10(lo)))
            else:
                v = lo + cube[i, k] * (hi - lo)
            row[name] = repr(round(float(v), 8))
        row.update(fixed)
        row['seed'] = seed0 + i
        # every parameter varies, so the cacatoo-style filename (which only
        # encodes scheme/lambda/e/seed) is not unique -> name runs explicitly
        if outdir:
            row['outfile'] = '%s/run_%05d.txt' % (outdir, start_index + i)
        rows.append(row)
    cols = ['point'] + names + sorted(fixed) + ['seed']
    if outdir:
        cols.append('outfile')
    with open(outpath, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print('wrote %d LHS points to %s' % (len(rows), outpath))


def parse_kv(s, as_range=False):
    """'a=1:2,b=1e-4:0.1:log' -> {'a': (1.0, 2.0, False), 'b': (1e-4, 0.1, True)}"""
    out = {}
    if not s:
        return out
    for part in s.split(','):
        k, _, v = part.partition('=')
        if as_range:
            bits = v.split(':')
            if len(bits) == 2:
                out[k.strip()] = (float(bits[0]), float(bits[1]), False)
            elif len(bits) == 3 and bits[2].lower() in ('log', 'log10'):
                lo, hi = float(bits[0]), float(bits[1])
                if lo <= 0:
                    raise SystemExit('log range for %s must have a positive low '
                                     'end, got %g' % (k, lo))
                out[k.strip()] = (lo, hi, True)
            else:
                raise SystemExit('cannot parse range %r (use lo:hi or lo:hi:log)' % v)
        else:
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', nargs='?', help='CSV with one row per simulation')
    ap.add_argument('--jobs', '-j', type=int, default=os.cpu_count() - 2)
    ap.add_argument('--outdir', default=None,
                    help='added as --outdir to every run (unless the CSV has one)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--no-analyse', action='store_true',
                    help='skip the classify + plot step that normally runs '
                         'automatically once the sweep finishes')
    ap.add_argument('--resume', action='store_true',
                    help='skip rows whose output is already complete, and delete '
                         'half-written output before redoing it. Use this to '
                         'restart a sweep that was interrupted -- without it the '
                         'model APPENDS to existing files and corrupts them')
    ap.add_argument('--plot-dir', default='plots',
                    help='where the sweep figures go. Point a test sweep at its '
                         'own directory so it does not overwrite a real one')
    ap.add_argument('--results', default=None,
                    help='summary CSV to write (default: <design>_results.csv)')
    ap.add_argument('--make-lhs', type=int, default=None,
                    help='generate this many LHS points instead of running')
    ap.add_argument('--vary', default='',
                    help='e.g. "lambda=0.01:1,muA=1e-4:0.1:log" '
                         '(linear by default, append :log for log10-uniform)')
    ap.add_argument('--fixed', default='', help='e.g. "scheme=1,maxtime=200000"')
    ap.add_argument('--start-index', type=int, default=0,
                    help='number the first point (and its outfile) from here, so '
                         'a design appended to an existing sweep does not reuse '
                         'run numbers')
    ap.add_argument('--seed0', type=int, default=1,
                    help='seed of the first LHS point; each point gets its own')
    ap.add_argument('--design-seed', type=int, default=20260821,
                    help='RNG seed for the LHS design itself (reproducibility)')
    ap.add_argument('--out', default='params.csv')
    args = ap.parse_args()

    if args.make_lhs:
        make_lhs(args.make_lhs, parse_kv(args.vary, as_range=True),
                 parse_kv(args.fixed), args.out, args.seed0,
                 args.design_seed, args.start_index)
        return 0

    if not args.csv:
        ap.error('need a CSV (or --make-lhs)')

    with open(args.csv) as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if args.outdir and 'outdir' not in row and 'outfile' not in row:
            row['outdir'] = args.outdir

    if args.resume:
        done, redo = [], 0
        for row in rows:
            f = row.get('outfile')
            if not f or not os.path.exists(f):
                continue
            mt = int(row.get('maxtime', 200000))
            rec = tail_record(f)
            if rec is None:                       # empty or unreadable
                if not args.dry_run:
                    os.remove(f)
                redo += 1
                continue
            n_vir = sum(rec['virion'])
            n_host = rec['num_S'] + rec['num_I']
            fixed = M.defense_fraction(np.array(rec['provirus'], float),
                                       np.array(rec['virion'], float)) >= 0.95
            if (rec['time'] >= mt - 100 or n_vir == 0 or n_host == 0 or fixed):
                done.append(f)
            else:
                # interrupted mid-run: the model appends, so the partial file
                # must go or the restarted run writes a second trajectory into it.
                # --dry-run must not touch anything, including a run that is
                # still being written by a sweep in flight.
                if not args.dry_run:
                    os.remove(f)
                redo += 1
        keep = set(done)
        before = len(rows)
        rows = [r for r in rows if r.get('outfile') not in keep]
        print('resume: %d of %d already complete, %d partial%s, %d to run'
              % (len(keep), before, redo,
                 ' (would be removed)' if args.dry_run else ' files removed',
                 len(rows)))
        if not rows:
            print('nothing left to run')
            if not args.no_analyse:
                analyse(args.csv, list(csv.DictReader(open(args.csv))),
                        args.results, args.plot_dir)
            return 0

    if args.dry_run:
        for row in rows:
            print(' '.join('--%s %s' % (k, v) for k, v in row.items()
                           if k not in SKIP and v not in (None, '')))
        return 0

    print('running %d simulations on %d workers' % (len(rows), args.jobs))
    fails = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, r): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(futs):
            cmd, rc, err = fut.result()
            done += 1
            if rc != 0:
                fails += 1
                print('[FAIL %d/%d] %s\n%s' % (done, len(rows), cmd, err))
            elif done % 10 == 0 or done == len(rows):
                print('[%d/%d] done' % (done, len(rows)))
    print('finished, %d failures' % fails)

    if not args.no_analyse:
        analyse(args.csv, rows, args.results, args.plot_dir)
    return 1 if fails else 0


def analyse(design, rows, results, plot_dir='plots'):
    """Classify every run and draw the outcome figures."""
    outdirs = {os.path.dirname(r['outfile']) for r in rows if r.get('outfile')}
    if not outdirs:
        outdirs = {r.get('outdir') for r in rows if r.get('outdir')}
    outdirs = {d for d in outdirs if d}
    if len(outdirs) != 1:
        print('skipping analysis: runs are spread over %d directories %s'
              % (len(outdirs), sorted(outdirs)))
        return
    outdir = outdirs.pop()
    if results is None:
        stem = os.path.splitext(os.path.basename(design))[0]
        results = (stem[:-7] if stem.endswith('_design') else stem) + '_results.csv'

    print('\n--- classifying outcomes ---')
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'classify.py'), outdir,
                         '--design', design, '-o', results]).returncode
    if rc != 0:
        print('classify.py failed; skipping plots')
        return
    print('\n--- plotting ---')
    subprocess.run([sys.executable, os.path.join(HERE, 'plot_outcomes.py'), results,
                    '--plot-dir', plot_dir])
    subprocess.run([sys.executable, os.path.join(HERE, 'plot_runs.py'), results,
                    '--out', 'classifier_check', '--plot-dir', plot_dir])
    # one figure per surviving run, written next to its .txt
    subprocess.run([sys.executable, os.path.join(HERE, 'plot_runs.py'), results,
                    '--only-each'])


if __name__ == '__main__':
    sys.exit(main())
