#!/usr/bin/env python3
"""
Live viewer for the virus-factory model.

Runs the same Numba kernels as model_numba.py -- identical dynamics, nothing is
re-implemented -- and draws three panels:

  left    the grid: empty / susceptible (shaded by mean provirus alpha) / infected
  top right    population sizes over time, log scale
  right, two   alpha spectrum over time, PROVIRUS and VIRION separately:
               x = timestep, y = alpha bin, brightness = share of that pool
               in that bin

The alpha panels are the point of the thing: a lineage splitting or sweeping
shows up as a band splitting or moving, which a per-frame histogram cannot show.
Keeping the two pools apart matters because they are where the two lifestyles
live -- a low-alpha defense sits in host genomes as provirus while a high-alpha
parasite exists as free virions, so a pooled panel can hide one behind the
other. Both panels use one colour ramp and one shared maximum, so equal
brightness means an equal share of that pool -- the two are directly
comparable.

  space  pause/resume        right arrow  single frame while paused
  +/-    faster/slower       g            cycle grid colouring
  s      save a screenshot   esc / q      quit

  python3 display.py --e 0.1 --lambda 0.1 --decay_rate 0.05 --diff_rate 1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

import model_numba as M
import classify as C

# dark palette (validated categorical steps for the dark surface)
BG        = (26, 26, 25)
PANEL     = (34, 34, 32)
GRIDLINE  = (51, 51, 48)
HEAT_HI   = (158, 203, 255)   # one ramp for both alpha panels
TEXT      = (255, 255, 255)
TEXT2     = (195, 194, 183)
MUTED     = (138, 136, 128)
SERIES = {'num_S': (57, 135, 229), 'num_I': (217, 89, 38),
          'virion': (25, 158, 112), 'provirus': (201, 133, 0)}

CELL_EMPTY = (35, 35, 34)
CELL_BARE  = (74, 74, 72)          # susceptible carrying no provirus
CELL_INF   = (217, 89, 38)         # factory active
ALPHA_LO   = np.array([28, 58, 94], float)     # alpha 0  (dark blue)
ALPHA_HI   = np.array([158, 203, 255], float)  # alpha 1  (light blue)


@njit(cache=True)
def cell_mean_alpha(head, nxt, alpha, out):
    """Mean alpha of each cell's list; -1 where the list is empty."""
    for c in range(head.shape[0]):
        j = head[c]
        if j == -1:
            out[c] = -1.0
            continue
        s = 0.0
        n = 0
        while j != -1:
            s += alpha[j]
            n += 1
            j = nxt[j]
        out[c] = s / n
    return out


class Sim:
    """Owns the model state; one step() is exactly one model_numba timestep."""

    def __init__(self, a):
        self.a = a
        nc = nr = a.size
        self.nc, self.nr, self.n = nc, nr, nc * nr
        n = self.n
        I = M.IDX
        self.state = np.zeros(n, np.int8); self.state_old = np.zeros(n, np.int8)
        self.clock = np.zeros(n, I)
        mk = lambda: (np.full(n, -1, I), np.full(n, -1, I), np.zeros(n, I))
        self.p_head, self.p_tail, self.p_cnt = mk()
        self.f_head, self.f_tail, self.f_cnt = mk()
        self.e_head, self.e_tail, self.e_cnt = mk()
        self.dead = np.zeros(n, I)
        m = np.arange(n, dtype=np.int64)
        self.upd = (((m % nc) << 16) | (m // nc)).astype(I)
        self.pool = M.Pool(max(1 << 18, 4 * n))
        self.rs = np.array([M.seed_rng(a.seed)], dtype=np.uint64)
        self.hp = np.zeros(100, np.int64); self.hv = np.zeros(100, np.int64)
        self.hn = np.zeros(100, np.int64)
        self.alpha_buf = np.zeros(n, np.float64)
        if a.perfect_diffusion:
            a.diff_rate, a.diffusion = 2.0, 'mix'
        self.n_steps = int(a.diff_rate) if a.diff_rate > 1.0 else 0
        self.diff_mix = a.diffusion == 'mix'
        mn = n if a.perfect_mix else 0
        self.mix_perm = np.empty(mn, np.int64)
        self.mix_tmp = np.empty(mn, np.int64)
        k = np.arange(mn, dtype=np.int64)
        self.mix_dest = ((k % nc) * nr + k // nc) if mn else np.empty(0, np.int64)
        M.initialise(nc, nr, self.state, self.clock, self.p_head, self.p_tail,
                     self.p_cnt, self.f_head, self.f_tail, self.f_cnt,
                     self.e_head, self.e_tail, self.e_cnt, self.pool.alpha,
                     self.pool.next, self.pool.free, self.pool.ctr, self.rs, a.alpha)
        self.t = 0
        self.done = False

    def step(self):
        if self.done:
            return
        a = self.a
        self.pool.ensure(2 * int(self.pool.ctr[2]) + 65536)
        if a.perfect_mix:
            M.host_perfect_mix(self.nc, self.nr, self.state, self.clock,
                               self.p_head, self.p_tail, self.p_cnt,
                               self.f_head, self.f_tail, self.f_cnt,
                               self.e_head, self.e_tail, self.e_cnt,
                               a.mix_virions, self.mix_perm, self.mix_dest,
                               self.mix_tmp, self.rs)
        M.host_step(self.nc, self.nr, self.state, self.state_old, self.clock,
                    self.p_head, self.p_tail, self.p_cnt,
                    self.f_head, self.f_tail, self.f_cnt,
                    self.e_head, self.e_tail, self.e_cnt,
                    self.pool.alpha, self.pool.next, self.pool.free, self.pool.ctr,
                    self.dead, self.rs, a.d, a.beta, a.lam, a.scheme, a.e, a.r,
                    a.c_alpha, a.latency, a.muA, a.mut_step, a.loss, a.b, a.cI,
                    a.sigma)
        M.ext_step(self.nc, self.nr, self.upd, self.e_head, self.e_tail, self.e_cnt,
                   self.pool.alpha, self.pool.next, self.pool.free, self.pool.ctr,
                   self.rs, a.diff_rate, a.decay_rate, self.n_steps, self.diff_mix)
        self.t += 1

    def stats(self):
        nS, nI, nvir, nprov = M.collect_stats(
            self.nc, self.nr, self.state, self.p_head, self.f_head, self.p_cnt,
            self.e_head, self.pool.alpha, self.pool.next, self.hp, self.hv, self.hn)
        # same rule as model_numba.py: with induction a prophage bank is not
        # extinct, it can start a new factory, so "no virions" is not the end
        if nS + nI == 0 or (nvir == 0 and (self.a.sigma <= 0.0 or nprov == 0)):
            self.done = True
        return nS, nI, nvir, nprov


def alpha_column(sim, mode='pool'):
    """One column of the alpha-over-time panel: share of the pool per alpha bin."""
    P = sim.hp.astype(float)
    V = sim.hv.astype(float)
    h = {'pool': lambda: C.pool_density(P, V), 'provirus': lambda: P,
         'virion': lambda: V}[mode]()
    tot = h.sum()
    return h / tot if tot > 0 else h


def snapshot_times(grid_every, extra, maxtime):
    """Every timestep a snapshot is wanted at: the regular cadence plus any
    extras, deduplicated and in order."""
    ts = set()
    if grid_every > 0:
        ts.update(range(0, maxtime + 1, grid_every))
    ts.update(x for x in extra if 0 <= x <= maxtime)
    return sorted(ts)


def parse_grid_at(spec):
    return [int(x) for x in str(spec).replace(',', ' ').split()]


def grid_rgb(sim, mode):
    """RGB image of the grid, (nc, nr, 3) so pygame reads it as (w, h)."""
    n = sim.n
    img = np.empty((n, 3), np.uint8)
    img[:] = CELL_EMPTY
    if mode == 'provirus':
        cell_mean_alpha(sim.p_head, sim.pool.next, sim.pool.alpha, sim.alpha_buf)
    else:
        cell_mean_alpha(sim.f_head, sim.pool.next, sim.pool.alpha, sim.alpha_buf)
    a = sim.alpha_buf
    st = sim.state
    live = (st == 1) | (st == 2)
    has = live & (a >= 0)
    if has.any():
        t = np.clip(a[has], 0, 1)[:, None]
        img[has] = (ALPHA_LO * (1 - t) + ALPHA_HI * t).astype(np.uint8)
    img[live & (a < 0)] = CELL_BARE
    img[st == 2] = CELL_INF
    return img.reshape(sim.nc, sim.nr, 3)


def draw_heat(screen, rect, hm, shown, decades, m, title, trange, font):
    """One alpha-over-time panel.

    `m` is the normalising maximum and is shared by both panels, as is the
    colour ramp: each column is a share of its own pool and sums to 1, so the
    same brightness means the same thing in the provirus and the virion panel.
    Scaling or tinting them separately would invent a difference that is not in
    the data.
    """
    rx, hy, RW, hh = rect
    pygame.draw.rect(screen, PANEL, rect)
    cols = hm.shape[1]
    img = np.zeros((cols, 100, 3), np.uint8)
    if m > 0:
        v = np.clip(np.log10(hm[:, :shown] / m * (10 ** decades - 1) + 1) / decades, 0, 1)
        ramp = (np.array(PANEL, float)[None, None, :] * (1 - v.T[..., None])
                + np.array(HEAT_HI, float)[None, None, :] * v.T[..., None])
        img = ramp.astype(np.uint8)
    hs = pygame.surfarray.make_surface(img[:shown, ::-1, :])  # alpha 0 at the bottom
    screen.blit(pygame.transform.scale(hs, (RW - 46, hh - 26)), (rx + 40, hy + 6))
    pygame.draw.rect(screen, GRIDLINE, (rx + 40, hy + 6, RW - 46, hh - 26), 1)
    screen.blit(font.render(title, True, TEXT2), (rx + 44, hy - 14))
    for frac, lab in ((0, '0.0'), (0.5, '0.5'), (1, '1.0')):
        yy = hy + 6 + int((1 - frac) * (hh - 26)) - 6
        screen.blit(font.render(lab, True, MUTED), (rx + 6, yy))
    screen.blit(font.render('time  %s' % trange, True, MUTED), (rx + 44, hy + hh - 14))


def draw_series(surf, rect, hist, keys, font, title):
    x0, y0, w, h = rect
    pygame.draw.rect(surf, PANEL, rect)
    if not hist['t']:
        return
    # scale to the data, not to zero: these series live between 1e2 and 1e5 and
    # anchoring the axis at 0.5 flattens all four onto the top of the panel
    vals = [v for k in keys for v in hist[k] if v > 0]
    if not vals:
        return
    lo = max(1.0, min(vals) / 3.0)
    hi = max(lo * 10, max(vals) * 1.6)
    span = np.log10(hi) - np.log10(lo)
    lg = lambda v: (np.log10(max(v, lo)) - np.log10(lo)) / span
    for gy in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
        if not (lo <= 10 ** gy <= hi):
            continue
        yy = y0 + h - int(lg(10 ** gy) * h)
        pygame.draw.line(surf, GRIDLINE, (x0 + 34, yy), (x0 + w, yy))
        surf.blit(font.render('1e%d' % gy, True, MUTED), (x0 + 3, yy - 7))
    N = len(hist['t'])
    for k in keys:
        pts = [(x0 + 34 + int(i * (w - 34) / max(N - 1, 1)), y0 + h - int(lg(v) * h))
               for i, v in enumerate(hist[k])]
        if len(pts) > 1:
            pygame.draw.lines(surf, SERIES[k], False, pts, 2)
    surf.blit(font.render(title, True, TEXT2), (x0 + 6, y0 + 4))
    for i, k in enumerate(keys):
        surf.blit(font.render(k, True, SERIES[k]), (x0 + w - 78, y0 + 4 + i * 14))


def build_parser():
    """The viewer's command line. Exposed so headless runners can reuse the
    exact same defaults instead of copying them and drifting."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=float, default=1)
    ap.add_argument('--e', type=float, default=0.1)
    ap.add_argument('--lambda', dest='lam', type=float, default=0.1)
    ap.add_argument('--scheme', type=int, default=1)
    ap.add_argument('--size', type=int, default=200)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--b', type=float, default=0.05)
    ap.add_argument('--cI', type=float, default=0.0)
    ap.add_argument('--d', type=float, default=0.01)
    ap.add_argument('--beta', type=float, default=0.5)
    ap.add_argument('--r', type=float, default=0.25)
    ap.add_argument('--c_alpha', type=float, default=0.5)
    ap.add_argument('--latency', type=int, default=20)
    ap.add_argument('--muA', type=float, default=0.001)
    ap.add_argument('--step', dest='mut_step', type=float, default=0.1)
    ap.add_argument('--loss', type=float, default=0.001)
    ap.add_argument('--sigma', type=float, default=0.0001,
                    help='spontaneous prophage induction probability')
    ap.add_argument('--diff_rate', type=float, default=0.5)
    ap.add_argument('--decay_rate', type=float, default=0.2)
    ap.add_argument('--diffusion', default='steps', choices=('steps', 'mix'))
    ap.add_argument('--perfect-diffusion', '--perfect_diffusion',
                    nargs='?', const=1, type=int, default=0,
                    help='ignore diff_rate; move every virion to a random cell')
    ap.add_argument('--mix-virions', '--mix_virions',
                    nargs='?', const=1, type=int, default=0,
                    help='with --perfect-mix, shuffle the extracellular grid by '
                         'the same permutation, so virions travel with their '
                         'location instead of staying put')
    ap.add_argument('--perfect-mix', '--perfect_mix',
                    nargs='?', const=1, type=int, default=0,
                    help='shuffle the whole host grid every timestep')
    ap.add_argument('--every', type=int, default=5,
                    help='model timesteps between frames')
    ap.add_argument('--window', type=int, default=0,
                    help='timesteps of history to show. 0 (default) shows the '
                         'WHOLE run so far, compressing as it grows; a positive '
                         'value scrolls a window of that many timesteps')
    ap.add_argument('--maxtime', type=int, default=200000)
    ap.add_argument('--quit-at-end', '--quit_at_end', action='store_true',
                    help='close the window and exit once --maxtime is reached or '
                         'the run dies out, instead of leaving it open. Needed to '
                         'chain anything after the viewer in a script')
    ap.add_argument('--screenshot', default=None,
                    help='save the window to this PNG once --maxtime is reached '
                         'and quit. Works headless (SDL_VIDEODRIVER=dummy), so a '
                         'figure can be produced without sitting in front of it')
    ap.add_argument('--grid-every', '--grid_every', type=int, default=0,
                    help='timesteps between grid snapshots (0 = off). Needs '
                         '--dump; the snapshots go to a .npz alongside it')
    ap.add_argument('--grid-out', '--grid_out', default=None,
                    help='.npz for the snapshots (default: alongside --dump)')
    ap.add_argument('--grid-at', '--grid_at', default='100',
                    help='extra timesteps to snapshot on top of the regular '
                         'cadence, comma separated. Defaults to 100: t=0 is '
                         'only the seeded state, so an early-but-settled frame '
                         'is the more informative first panel. Empty for none')
    ap.add_argument('--dump', default=None,
                    help='also write the run to this file in model_numba.py '
                         'format, so the alpha-versus-time data can be recovered '
                         'with alpha_matrix.py')
    ap.add_argument('--scale', type=float, default=None,
                    help='pixels per grid cell. May be fractional, so a big grid '
                         'can be shrunk. Default: whatever makes the grid panel '
                         '--grid-px across, so --size does not change the window')
    ap.add_argument('--grid-px', '--grid_px', type=int, default=600,
                    help='on-screen size of the grid panel in pixels when '
                         '--scale is not given (default 600)')
    ap.add_argument('--heat-decades', type=float, default=3.0,
                    help='dynamic range of the alpha panel; lower hides rare '
                         'variants, higher brings them out')
    return ap


def main():
    a = build_parser().parse_args()

    sim = Sim(a)
    dump = None
    if a.dump:
        d = os.path.dirname(a.dump)
        if d:
            os.makedirs(d, exist_ok=True)
        dump = open(a.dump, 'w')
        with open(os.path.splitext(a.dump)[0] + '.params.json', 'w') as fh:
            json.dump({k: v for k, v in sorted(vars(a).items())
                       if k not in ('screenshot', 'scale', 'grid_px',
                                    'heat_decades', 'window', 'quit_at_end')},
                      fh, indent=2, sort_keys=True)

    # Grid snapshots. The viewer's `every` changes under the +/- keys, so the
    # boundary is tested inside the stepping loop rather than once per frame --
    # otherwise a fast setting would step straight over it.
    snaps = {'t': [], 'state': [], 'alpha_p': [], 'alpha_f': []}
    grid_path = None
    buf_p = buf_f = None
    if a.grid_every > 0:
        if not a.dump:
            sys.exit('--grid-every needs --dump (the .npz is named after it)')
        grid_path = a.grid_out or (os.path.splitext(a.dump)[0] + '_grids.npz')
        buf_p = np.empty(sim.n, np.float64)
        buf_f = np.empty(sim.n, np.float64)

    def snapshot():
        cell_mean_alpha(sim.p_head, sim.pool.next, sim.pool.alpha, buf_p)
        cell_mean_alpha(sim.f_head, sim.pool.next, sim.pool.alpha, buf_f)
        snaps['t'].append(sim.t)
        snaps['state'].append(sim.state.reshape(sim.nc, sim.nr).copy())
        snaps['alpha_p'].append(buf_p.reshape(sim.nc, sim.nr).astype(np.float32))
        snaps['alpha_f'].append(buf_f.reshape(sim.nc, sim.nr).astype(np.float32))

    done_saving = []

    def finish():
        """Every way out of the viewer comes through here, so quitting with
        esc, the window button or --quit-at-end all leave the same files."""
        if dump is not None and not dump.closed:
            dump.close()
        if grid_path and not done_saving:
            done_saving.append(True)
            if not snaps['t'] or snaps['t'][-1] != sim.t:
                snapshot()
            np.savez_compressed(
                grid_path, t=np.array(snaps['t'], np.int64),
                state=np.stack(snaps['state']),
                alpha_p=np.stack(snaps['alpha_p']),
                alpha_f=np.stack(snaps['alpha_f']),
                params=np.array(json.dumps(
                    {k: v for k, v in sorted(vars(a).items())
                     if k not in ('screenshot', 'scale', 'grid_px',
                                  'heat_decades', 'window', 'quit_at_end')})))
            print('wrote %s  (%d snapshots of %dx%d)'
                  % (grid_path, len(snaps['t']), sim.nc, sim.nr))

    targets = (snapshot_times(a.grid_every, parse_grid_at(a.grid_at), a.maxtime)
               if grid_path else [])
    ptr = [0]
    if grid_path and targets and targets[0] == 0:
        snapshot()
        ptr[0] = 1

    # the grid panel is a fixed size on screen unless --scale says otherwise, so
    # --size 400 does not produce a 1200px window that runs off the display
    if a.scale is None:
        a.scale = a.grid_px / float(a.size)
    gw = max(64, int(round(a.size * a.scale)))
    PAD, RW = 14, 620
    W, H = PAD * 3 + gw + RW, max(gw + 60, 748)
    pygame.init()
    tag = ''.join([' +perfect-mix' if a.perfect_mix else '',
                   ' +perfect-diffusion' if a.perfect_diffusion else ''])
    pygame.display.set_caption('virus factory  |  e=%g lambda=%g diff=%g decay=%g%s'
                               % (a.e, a.lam, a.diff_rate, a.decay_rate, tag))
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont('menlo,dejavusansmono,monospace', 12)
    big = pygame.font.SysFont('menlo,dejavusansmono,monospace', 15, bold=True)
    clock = pygame.time.Clock()

    cols = max(80, RW - 60)
    heat = {'provirus': np.zeros((100, cols), np.float32),
            'virion':   np.zeros((100, cols), np.float32)}
    hist = {k: [] for k in ('t', 'num_S', 'num_I', 'virion', 'provirus')}
    # whole-run mode: each column covers `span` timesteps. When the array fills,
    # adjacent columns are merged and span doubles, so the panel always shows the
    # entire run at whatever resolution fits. The alternative -- a fixed window --
    # hides everything slower than the window, and this model's interesting
    # dynamics take 10^5 timesteps.
    span, fill, ncol = float(max(a.every, 1)), 0.0, 0
    acc = {'provirus': np.zeros(100, np.float32),
           'virion':   np.zeros(100, np.float32)}
    grid_modes = ['provirus', 'factory']
    gm = 0
    paused, every, single = False, a.every, False
    t0, last_t, fps_txt = time.time(), 0, ''

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                finish()
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    finish()
                    pygame.quit(); return
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_RIGHT:
                    single = True
                if ev.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    every = min(200, every * 2)
                if ev.key == pygame.K_MINUS:
                    every = max(1, every // 2)
                if ev.key == pygame.K_g:
                    gm = (gm + 1) % len(grid_modes)
                if ev.key == pygame.K_s:
                    fn = 'plots/display_t%06d.png' % sim.t
                    os.makedirs('plots', exist_ok=True)
                    pygame.image.save(screen, fn); print('wrote', fn)

        if (not paused or single) and not sim.done and sim.t < a.maxtime:
            for _ in range(every if not single else 1):
                sim.step()
                while (grid_path and ptr[0] < len(targets)
                       and sim.t >= targets[ptr[0]]):
                    snapshot()
                    ptr[0] += 1
                if sim.done or sim.t >= a.maxtime:
                    break
            single = False
            nS, nI, nvir, nprov = sim.stats()
            if dump is not None:
                dump.write(json.dumps(
                    {"time": sim.t, "num_S": int(nS), "num_I": int(nI),
                     "provirus": sim.hp.tolist(), "virion": sim.hv.tolist(),
                     "num_provirus": sim.hn.tolist()},
                    separators=(',', ':')) + '\n')
                dump.flush()
            hist['t'].append(sim.t)
            hist['num_S'].append(nS); hist['num_I'].append(nI)
            hist['virion'].append(nvir); hist['provirus'].append(nprov)
            cols_now = {k: alpha_column(sim, k) for k in ('provirus', 'virion')}
            if a.window > 0:
                keep = max(2, a.window // max(every, 1))
                for k in hist:
                    del hist[k][:-keep]
                for k, hm in heat.items():
                    hm[:, :-1] = hm[:, 1:]
                    hm[:, -1] = cols_now[k]
                ncol = cols
            else:
                # thin the traces so the polylines stay cheap to draw
                if len(hist['t']) > 2 * cols:
                    for k in hist:
                        hist[k][:] = hist[k][::2]
                for k in acc:
                    acc[k] += cols_now[k] * every
                fill += every
                if fill >= span:
                    for k, hm in heat.items():
                        hm[:, ncol] = acc[k] / fill
                        acc[k][:] = 0.0
                    fill = 0.0
                    ncol += 1
                    if ncol >= cols:      # full: merge pairs, halve resolution
                        for hm in heat.values():
                            hm[:, :cols // 2] = 0.5 * (hm[:, 0:cols - 1:2]
                                                       + hm[:, 1:cols:2])
                            hm[:, cols // 2:] = 0.0
                        ncol = cols // 2
                        span *= 2

        screen.fill(BG)
        # ---- grid
        surf = pygame.surfarray.make_surface(grid_rgb(sim, grid_modes[gm]))
        screen.blit(pygame.transform.scale(surf, (gw, gw)), (PAD, PAD))
        pygame.draw.rect(screen, GRIDLINE, (PAD, PAD, gw, gw), 1)

        rx = PAD * 2 + gw
        draw_series(screen, (rx, PAD, RW, 210), hist,
                    ['num_S', 'num_I', 'virion', 'provirus'], font,
                    'populations (log)')

        # ---- alpha over time, provirus and virion kept apart
        shown = max(ncol, 1) if a.window <= 0 else cols
        rng = ('0 - %d' % sim.t) if a.window <= 0 else ('last %d steps' % a.window)
        hh = 185
        hy = PAD + 228
        # one maximum across both panels, so brightness is comparable
        hmax = max(heat['provirus'][:, :shown].max(),
                   heat['virion'][:, :shown].max())
        draw_heat(screen, (rx, hy, RW, hh), heat['provirus'], shown, a.heat_decades,
                  hmax, 'provirus alpha over time', rng, font)
        hy2 = hy + hh + 22
        draw_heat(screen, (rx, hy2, RW, hh), heat['virion'], shown, a.heat_decades,
                  hmax, 'virion alpha over time', rng, font)
        hy, hh = hy2, hh

        # ---- readout
        ty = hy + hh + 12
        nS = hist['num_S'][-1] if hist['t'] else 0
        nI = hist['num_I'][-1] if hist['t'] else 0
        nv = hist['virion'][-1] if hist['t'] else 0
        npv = hist['provirus'][-1] if hist['t'] else 0
        cen = (np.arange(100) + 0.5) / 100
        def _stat(mode):
            c = alpha_column(sim, mode)
            if c.sum() <= 0:
                return float('nan'), float('nan')
            return float((cen * c).sum()), float(c[cen < 0.1].sum())
        mean_p, lo_p = _stat('provirus')
        mean_v, lo_v = _stat('virion')
        if time.time() - t0 > 0.5:
            fps_txt = '%.0f steps/s' % ((sim.t - last_t) / (time.time() - t0))
            t0, last_t = time.time(), sim.t
        lines = [
            't = %-8d %s%s' % (sim.t, fps_txt, '   [PAUSED]' if paused else
                               ('   [ENDED]' if sim.done else '')),
            'S %-7d I %-6d virions %-8d proviruses %-8d prov/host %.2f'
            % (nS, nI, nv, npv, npv / max(nS + nI, 1)),
            'alpha mean  prov %.3f  vir %.3f   |   frac<0.1  prov %.3f  vir %.3f'
            % (mean_p, mean_v, lo_p, lo_v),
            'grid: %s alpha   |  space pause   -/+ speed (%d/frame)   g grid   s save   q quit'
            % (grid_modes[gm], every),
        ]
        for i, s in enumerate(lines):
            screen.blit((big if i == 0 else font).render(s, True, TEXT if i == 0 else TEXT2),
                        (rx, ty + i * 18))
        # grid legend
        lx, ly = PAD, PAD + gw + 8
        for cbox, lab in ((CELL_EMPTY, 'empty'), (CELL_BARE, 'S, no provirus'),
                          (tuple(ALPHA_LO.astype(int)), 'alpha 0'),
                          (tuple(ALPHA_HI.astype(int)), 'alpha 1'),
                          (CELL_INF, 'factory active')):
            pygame.draw.rect(screen, cbox, (lx, ly, 12, 12))
            screen.blit(font.render(lab, True, TEXT2), (lx + 16, ly - 1))
            lx += 20 + font.size(lab)[0] + 12

        pygame.display.flip()
        if (a.screenshot or a.quit_at_end) and (sim.done or sim.t >= a.maxtime):
            if a.screenshot:
                pygame.image.save(screen, a.screenshot)
                print('wrote', a.screenshot)
            finish()
            pygame.quit()
            return
        clock.tick(60)


if __name__ == '__main__':
    main()
