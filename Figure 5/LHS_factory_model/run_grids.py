#!/usr/bin/env python3
"""Headless run that dumps the alpha histograms *and* periodic grid snapshots.

Same kernels and the same parameter defaults as display.py -- it drives
display.Sim directly -- so a run here is bit-for-bit the run the viewer would
show, minus the window. Two things it adds:

  * grid snapshots every --grid-every timesteps, saved as one compressed .npz
    (state plus per-cell mean alpha for both pools), so the grid can be redrawn
    later in display.py's colours without re-running the model
  * a sidecar <dump>.params.json recording every setting, because the dump
    format itself has no header and a run whose settings are lost is a run that
    has to be reverse-engineered

    python3 run_grids.py --seed 1 --size 400 --e 0.01 --diff_rate 1 \
        --maxtime 200000 --grid-every 10000 --dump runs/run_1.txt
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import display as D


def main():
    ap = D.build_parser()
    # --grid-every / --grid-out come from display.build_parser(), so the
    # viewer and this runner cannot drift apart on them
    ap.set_defaults(grid_every=10000)
    ap.add_argument('--quiet', action='store_true',
                    help='no progress line on stderr')
    a = ap.parse_args()

    if not a.dump:
        ap.error('--dump is required (the alpha histograms go there)')
    if a.every <= 0:
        ap.error('--every must be positive')

    d = os.path.dirname(a.dump)
    if d:
        os.makedirs(d, exist_ok=True)
    stem = os.path.splitext(a.dump)[0]
    grid_out = a.grid_out or (stem + '_grids.npz')

    # every setting that reaches the model, so this run is never a mystery later
    params = {k: v for k, v in sorted(vars(a).items())
              if k not in ('screenshot', 'scale', 'grid_px', 'heat_decades',
                           'window', 'quit_at_end')}
    with open(stem + '.params.json', 'w') as fh:
        json.dump(params, fh, indent=2, sort_keys=True)

    sim = D.Sim(a)
    n, nc, nr = sim.n, sim.nc, sim.nr
    buf_p = np.empty(n, np.float64)
    buf_f = np.empty(n, np.float64)

    snaps = {'t': [], 'state': [], 'alpha_p': [], 'alpha_f': []}

    def snapshot():
        D.cell_mean_alpha(sim.p_head, sim.pool.next, sim.pool.alpha, buf_p)
        D.cell_mean_alpha(sim.f_head, sim.pool.next, sim.pool.alpha, buf_f)
        snaps['t'].append(sim.t)
        snaps['state'].append(sim.state.reshape(nc, nr).copy())
        snaps['alpha_p'].append(buf_p.reshape(nc, nr).astype(np.float32))
        snaps['alpha_f'].append(buf_f.reshape(nc, nr).astype(np.float32))

    last = {'S': 0, 'I': 0}

    def record():
        nS, nI, nv, npv = sim.stats()
        last['S'], last['I'] = int(nS), int(nI)
        dump.write(json.dumps(
            {"time": sim.t, "num_S": int(nS), "num_I": int(nI),
             "provirus": sim.hp.tolist(), "virion": sim.hv.tolist(),
             "num_provirus": sim.hn.tolist()},
            separators=(',', ':')) + '\n')

    t0 = time.perf_counter()
    dump = open(a.dump, 'w')
    try:
        record()                       # t = 0, matching the viewer's first frame
        targets = D.snapshot_times(a.grid_every, D.parse_grid_at(a.grid_at),
                                   a.maxtime) if a.grid_every > 0 else []
        ptr = 0
        if targets and targets[0] == 0:
            snapshot()
            ptr = 1
        while not sim.done and sim.t < a.maxtime:
            for _ in range(a.every):
                sim.step()
                # a snapshot boundary inside the sampling stride still lands
                # on the exact timestep, not on the next sampled one
                while ptr < len(targets) and sim.t >= targets[ptr]:
                    snapshot()
                    ptr += 1
                if sim.done or sim.t >= a.maxtime:
                    break
            record()
            if not a.quiet and sim.t % (a.every * 200) == 0:
                sys.stderr.write('\r  t=%d  S=%d  I=%d  %.0fs   '
                                 % (sim.t, last['S'], last['I'],
                                    time.perf_counter() - t0))
                sys.stderr.flush()
    finally:
        dump.close()

    if a.grid_every > 0 and (not snaps['t'] or snaps['t'][-1] != sim.t):
        snapshot()                     # always keep the final state

    if a.grid_every > 0:
        np.savez_compressed(
            grid_out,
            t=np.array(snaps['t'], np.int64),
            state=np.stack(snaps['state']),
            alpha_p=np.stack(snaps['alpha_p']),
            alpha_f=np.stack(snaps['alpha_f']),
            params=np.array(json.dumps(params)))
        sys.stderr.write('\rwrote %s  (%d snapshots of %dx%d)\n'
                         % (grid_out, len(snaps['t']), nc, nr))
    sys.stderr.write('wrote %s  (t=%d, %.1f s)\n'
                     % (a.dump, sim.t, time.perf_counter() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
