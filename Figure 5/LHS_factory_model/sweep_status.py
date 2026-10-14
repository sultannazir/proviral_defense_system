#!/usr/bin/env python3
"""
Interim status of a running sweep, without disturbing it.

Reads the tail of every output file and reports how many runs have finished, how
many reached maxtime with BOTH virus and host still present, and an ETA. A run
counts as finished when its last record is either at maxtime or has no virions
left (the only two ways model_numba.py stops).

    python3 sweep_status.py lhs_design.csv
"""
import csv
import model_numba as M
import json
import os
import sys
import time


def started_at(paths):
    """When the sweep began: the birth time of the earliest output file."""
    best = None
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            continue
        t = getattr(st, 'st_birthtime', st.st_mtime)
        if best is None or t < best:
            best = t
    return best


def hms(seconds):
    if seconds != seconds or seconds < 0:
        return '?'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return '%dh%02dm' % (h, m) if h else '%dm%02ds' % (m, s)


def tail_record(path, tail_bytes=8192):
    """Last complete JSON line, or None. Files are being appended to live."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as fh:
            fh.seek(max(0, size - tail_bytes))
            chunk = fh.read().decode('utf-8', 'ignore')
    except OSError:
        return None
    for line in reversed(chunk.splitlines()):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def main():
    design = sys.argv[1] if len(sys.argv) > 1 else 'lhs_design.csv'
    rows = list(csv.DictReader(open(design)))
    total = len(rows)
    maxtime = int(rows[0].get('maxtime', 200000))

    started = done = alive = survivors = host_extinct = virus_gone = 0
    fixed = 0
    steps_done = 0
    existing = []
    for r in rows:
        f = r['outfile']
        if not os.path.exists(f):
            continue
        existing.append(f)
        started += 1
        rec = tail_record(f)
        if rec is None:
            continue
        steps_done += rec['time']
        n_vir = sum(rec['virion'])
        # a prophage bank with no virions is not gone -- with induction it can
        # restart, and the model keeps running
        n_virus = n_vir + sum(rec['provirus'])
        n_host = rec['num_S'] + rec['num_I']
        at_end = rec['time'] >= maxtime - 100
        # a run that hit --defense-fix stopped on purpose with both alive
        import numpy as _np
        is_fixed = M.defense_fraction(_np.array(rec['provirus'], float),
                                      _np.array(rec['virion'], float)) >= 0.95
        if at_end or n_vir == 0 or n_host == 0 or is_fixed:
            done += 1
            if is_fixed:
                fixed += 1
                survivors += 1
            elif at_end and n_virus > 0 and n_host > 0:
                survivors += 1
            elif n_host == 0:
                host_extinct += 1
            else:
                virus_gone += 1
        else:
            alive += 1

    pct = lambda a, b: (100.0 * a / b) if b else 0.0
    print('sweep status: %d/%d started, %d finished, %d still running'
          % (started, total, done, alive))
    print('  ended with BOTH virus and host      : %5d  (%.1f%% of finished)'
          % (survivors, pct(survivors, done)))
    print('  virus gone, hosts remain            : %5d  (%.1f%%)'
          % (virus_gone, pct(virus_gone, done)))
    print('  hosts died out                      : %5d  (%.1f%%)'
          % (host_extinct, pct(host_extinct, done)))
    print('    of the survivors, stopped early on defense fixation: %d' % fixed)
    if done:
        print('  projected survivors at %d runs     : ~%d'
              % (total, round(survivors / done * total)))

    # ---- ETA
    t0 = started_at(existing)
    if not t0 or not done:
        return
    elapsed = time.time() - t0
    # Work is measured in simulated timesteps, not runs: runs differ in length by
    # three orders of magnitude, and the short ones finish first, so counting
    # runs makes an early ETA far too optimistic. Steps already simulated are
    # counted for finished AND in-flight runs.
    mean_steps = steps_done / started if started else 0
    total_steps = mean_steps * total
    rate = steps_done / elapsed if elapsed > 0 else 0
    eta_steps = (total_steps - steps_done) / rate if rate > 0 else float('nan')
    eta_runs = elapsed * (total - done) / done
    print('  --')
    print('  elapsed %s   %.0f runs/h   %.2fM steps/h'
          % (hms(elapsed), done / elapsed * 3600, rate * 3600 / 1e6))
    print('  ETA %s (by simulated steps)   %s (by run count)'
          % (hms(eta_steps), hms(eta_runs)))
    if eta_steps == eta_steps:
        print('  projected finish: %s'
              % time.strftime('%a %H:%M', time.localtime(time.time() + eta_steps)))
    print('  (both are extrapolations; the step-based one is the more reliable, '
          'and\n   early estimates run short because quick runs finish first)')


if __name__ == '__main__':
    main()
