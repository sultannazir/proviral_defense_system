#!/usr/bin/env python3
"""
Numba port of the Cacatoo "Virus factory model" (model.js).

This is a bit-for-bit reimplementation: it reproduces cacatoo's RNG
(the Lehmer/MINSTD generator `seed = seed*48271 % 2147483647`), the exact
order in which random numbers are drawn, the synchronous host update
(including cacatoo's shallow-copy Gridpoint semantics, which make the
per-cell provirus/factory *arrays* effectively live while the scalar
fields are double-buffered), and the asynchronous extracellular update
with its Fisher-Yates shuffled update order.

Output files are byte-identical to those produced by model.js.

Usage:  python3 model_numba.py --seed 1 --lambda 0.1 --e 0.01 --scheme 1
See command_examples.txt.
"""

import argparse
import json
import math
import os
import sys
import time as _time

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Moore neighbourhood, in cacatoo's order (index 0 = self, 1..8 = N W E S NW NE SW SE)
# ---------------------------------------------------------------------------
MOORE_X = np.array([0, 0, -1, 1, 0, -1, 1, -1, 1], dtype=np.int64)
MOORE_Y = np.array([0, -1, 0, 0, 1, -1, -1, 1, 1], dtype=np.int64)

EPSILON = 2.220446049250313e-16          # Number.EPSILON in JS

EMPTY, SUSC, INFECTED = 0, 1, 2

IDX = np.int32                           # index dtype for lists / grids


# ===========================================================================
#  Random number generator: splitmix64
#
#  Everything between the RNG-BLOCK markers is the *only* part of this file
#  that knows which generator is in use. make_reference.py swaps this block for
#  cacatoo's MINSTD to regenerate a bit-identical reference implementation, so
#  validate.sh can still byte-compare the model logic against model.js.
#
#  splitmix64 advances its state with a single 64-bit add, so the serial
#  dependency chain is one cycle; the avalanche mix that turns the state into a
#  number is off the chain and pipelines freely. Period 2**64, and separate
#  seeds land ~2**63 apart, so replicate runs draw from disjoint stretches of
#  the cycle (cacatoo's MINSTD has a period of only 2**31-2, which this model
#  exhausts every ~21000 timesteps).
# ===========================================================================
# ----- RNG BLOCK BEGIN -----
# splitmix64, in UNSIGNED arithmetic.
#
# The multiplies below overflow 64 bits by design. In signed arithmetic that is
# undefined behaviour, which LLVM is entitled to exploit -- and it does so
# differently depending on how the code is inlined and which version is doing
# the optimising, so the same source produced different streams on different
# numba versions. Unsigned overflow is DEFINED to wrap, so this is stable across
# versions and machines. The shifts are logical rather than arithmetic for the
# same reason, which also removes the masking the signed version needed.
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S11 = np.uint64(11)
_S27 = np.uint64(27)
_S30 = np.uint64(30)
_S31 = np.uint64(31)
_SEVEN = np.uint64(7)
_INV53 = 1.1102230246251565e-16                  # 2**-53


@njit(inline='always', cache=True)
def _nxt(s):
    """Advance the generator state."""
    return s + _GOLDEN


@njit(inline='always', cache=True)
def _mix(s):
    z = (s ^ (s >> _S30)) * _MIX1
    z = (z ^ (z >> _S27)) * _MIX2
    return z ^ (z >> _S31)


@njit(inline='always', cache=True)
def _u1(s):
    """State -> float in [0,1).  (cacatoo: genrand_real1)"""
    return np.float64(_mix(s) >> _S11) * _INV53


@njit(inline='always', cache=True)
def _u2(s):
    """State -> float in [0,1).  (cacatoo: genrand_real2, used by shuffle)"""
    return np.float64(_mix(s) >> _S11) * _INV53


@njit(inline='always', cache=True)
def _bits3(s):
    """State -> integer in [0,7].  (cacatoo: genrand_int(0,7))"""
    return np.int64(_mix(s) & _SEVEN)


@njit(inline='always', cache=True)
def _randrange(s, n):
    """State -> integer in [0,n-1].  (cacatoo: genrand_int(0,n-1))"""
    return np.int64(_mix(s) % np.uint64(n))


def seed_rng(seed):
    """Turn a user seed into a starting state, scrambled so that consecutive
    seeds start far apart in the cycle."""
    s = int(seed) & 0xFFFFFFFFFFFFFFFF
    for _ in range(3):
        s = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = ((s ^ (s >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        s = z ^ (z >> 31)
    return s
# ----- RNG BLOCK END -----


@njit(inline='always', cache=True)
def _gauss_step(s, A, D):
    """model.js gaussianStep(A, D, 0, 1): Box-Muller draw, scaled, truncated."""
    s = _nxt(s)
    u1 = _u1(s)
    s = _nxt(s)
    u2 = _u1(s)
    if u1 < EPSILON:
        u1 = EPSILON
    newA = A + D * (math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))
    if newA < 0.0:
        newA = 0.0
    if newA > 1.0:
        newA = 1.0
    return s, newA


# ===========================================================================
#  Virus pool: singly linked lists with tail pointers (ordered, O(1) append)
#     pool_alpha[j]  : alpha value of virus j
#     pool_next[j]   : next virus in the list, or -1
#     free_stack     : LIFO stack of free slots
#     pctr           : [free_top, bump, n_live]
# ===========================================================================
@njit(inline='always', cache=True)
def _alloc(free_stack, pctr):
    if pctr[0] > 0:
        pctr[0] -= 1
        j = free_stack[pctr[0]]
    else:
        j = pctr[1]
        pctr[1] += 1
    pctr[2] += 1
    return j


@njit(inline='always', cache=True)
def _release(j, free_stack, pctr):
    free_stack[pctr[0]] = j
    pctr[0] += 1
    pctr[2] -= 1


@njit(inline='always', cache=True)
def _push(head, tail, cnt, c, a, pool_alpha, pool_next, free_stack, pctr):
    j = _alloc(free_stack, pctr)
    pool_alpha[j] = a
    pool_next[j] = -1
    if head[c] == -1:
        head[c] = j
    else:
        pool_next[tail[c]] = j
    tail[c] = j
    cnt[c] += 1


@njit(inline='always', cache=True)
def _append_node(head, tail, cnt, c, j, pool_next):
    pool_next[j] = -1
    if head[c] == -1:
        head[c] = j
    else:
        pool_next[tail[c]] = j
    tail[c] = j
    cnt[c] += 1


@njit(inline='always', cache=True)
def _free_chain(head, tail, cnt, c, pool_next, free_stack, pctr):
    j = head[c]
    while j != -1:
        nj = pool_next[j]
        _release(j, free_stack, pctr)
        j = nj
    head[c] = -1
    tail[c] = -1
    cnt[c] = 0


# ===========================================================================
#  Host grid: perfect mixing  (== cacatoo Gridmodel.perfectMix())
#
#  Shuffles whole gridpoints -- state, clock, provirus list and factory list all
#  travel together -- and writes them back in cacatoo's order, which transposes
#  as it goes.
#
#  By default only the host grid moves, which is what cacatoo's perfectMix does
#  and keeps this byte-identical to model.js at mixinp=1. That leaves the free
#  virion pool spatially structured, so virion diffusion still has local
#  structure to act on even though the hosts do not -- an asymmetry, not a
#  property of a well-mixed world. --mix-virions applies the SAME permutation to
#  the extracellular grid, so every location's whole contents move together and
#  diffusion has nothing local left to do.
# ===========================================================================
@njit(inline='always', cache=True)
def _permute(arr, perm, dest, tmp, n):
    for k in range(n):
        tmp[k] = arr[perm[k]]
    for k in range(n):
        arr[dest[k]] = tmp[k]


@njit(cache=True)
def host_perfect_mix(nc, nr, state, clock, p_head, p_tail, p_cnt,
                     f_head, f_tail, f_cnt, e_head, e_tail, e_cnt,
                     mix_virions, perm, dest, tmp, rs):
    s = rs[0]
    n = nc * nr
    for i in range(n):
        perm[i] = i
    # cacatoo builds a fresh list every call, so this is a fresh shuffle, not a
    # reshuffle of last step's order
    i = n
    while i > 0:
        i -= 1
        s = _nxt(s)
        ri = int(_u2(s) * (i + 1))
        t = perm[i]
        perm[i] = perm[ri]
        perm[ri] = t

    _permute(state, perm, dest, tmp, n)
    _permute(clock, perm, dest, tmp, n)
    _permute(p_head, perm, dest, tmp, n)
    _permute(p_tail, perm, dest, tmp, n)
    _permute(p_cnt, perm, dest, tmp, n)
    _permute(f_head, perm, dest, tmp, n)
    _permute(f_tail, perm, dest, tmp, n)
    _permute(f_cnt, perm, dest, tmp, n)
    if mix_virions:
        _permute(e_head, perm, dest, tmp, n)
        _permute(e_tail, perm, dest, tmp, n)
        _permute(e_cnt, perm, dest, tmp, n)
    rs[0] = s


# ===========================================================================
#  Host grid: one synchronous sweep  (== sim.host.synchronous())
# ===========================================================================
@njit(cache=True)
def host_step(nc, nr, state, state_old, clock,
              p_head, p_tail, p_cnt, f_head, f_tail, f_cnt,
              e_head, e_tail, e_cnt,
              pool_alpha, pool_next, free_stack, pctr,
              dead_buf, rs,
              d, beta, lam, scheme, e_exc, r, c_alpha, latency,
              muA, mut_step, loss, b, cI, sigma):

    s = rs[0]
    n = nc * nr
    for c in range(n):
        state_old[c] = state[c]

    n_dead = 0
    inf_buf = np.empty(256, dtype=np.float64)
    off_buf = np.empty(256, dtype=np.float64)
    idx_buf = np.empty(256, dtype=IDX)
    keep_buf = np.empty(256, dtype=np.uint8)

    for x in range(nc):
        for y in range(nr):
            c = x * nr + y
            st = state_old[c]

            # -------------------------------------------------- susceptible
            if st == SUSC:
                s = _nxt(s)
                if _u1(s) < d:
                    state[c] = EMPTY
                    dead_buf[n_dead] = c
                    n_dead += 1
                    clock[c] = 0
                    continue

                # pick up extracellular viruses with probability beta
                j = e_head[c]
                if j != -1:
                    if e_cnt[c] > inf_buf.shape[0]:
                        inf_buf = np.empty(2 * e_cnt[c], dtype=np.float64)
                    n_inf = 0
                    nh = -1
                    nt = -1
                    ncnt = 0
                    while j != -1:
                        nj = pool_next[j]
                        s = _nxt(s)
                        if _u1(s) < beta:
                            inf_buf[n_inf] = pool_alpha[j]
                            n_inf += 1
                            _release(j, free_stack, pctr)
                        else:
                            pool_next[j] = -1
                            if nh == -1:
                                nh = j
                            else:
                                pool_next[nt] = j
                            nt = j
                            ncnt += 1
                        j = nj
                    e_head[c] = nh
                    e_tail[c] = nt
                    e_cnt[c] = ncnt

                    if n_inf > 0:
                        s = _nxt(s)
                        u = _u1(s)
                        if scheme == 1:
                            make_factory = u > lam
                        elif scheme == 2:
                            make_factory = u < 0.2 * n_inf
                        else:
                            make_factory = u > 0.2 * n_inf

                        if make_factory:
                            state[c] = INFECTED
                            for k in range(n_inf):
                                _push(f_head, f_tail, f_cnt, c, inf_buf[k],
                                      pool_alpha, pool_next, free_stack, pctr)
                            clock[c] = 1
                        else:
                            for k in range(n_inf):
                                _push(p_head, p_tail, p_cnt, c, inf_buf[k],
                                      pool_alpha, pool_next, free_stack, pctr)

                # Spontaneous prophage induction. A host still susceptible after
                # the environmental route -- whether nothing was picked up, or
                # what was picked up integrated instead of forming a factory --
                # can start a factory from its own prophages. Only reached when
                # sigma > 0, so at sigma = 0 no random number is drawn here and
                # the model still matches model.js exactly.
                if sigma > 0.0 and state[c] == SUSC and p_cnt[c] > 0:
                    s = _nxt(s)
                    if _u1(s) < sigma:
                        # every prophage joins the factory, one copy each
                        j = p_head[c]
                        while j != -1:
                            nj = pool_next[j]
                            _append_node(f_head, f_tail, f_cnt, c, j, pool_next)
                            j = nj
                        p_head[c] = -1
                        p_tail[c] = -1
                        p_cnt[c] = 0
                        state[c] = INFECTED
                        clock[c] = 1

            # ------------------------------------------------------ infected
            elif st == INFECTED:
                # proviruses may excise and join the factory (reverse order)
                m = p_cnt[c]
                if m > 0:
                    if m > idx_buf.shape[0]:
                        idx_buf = np.empty(2 * m, dtype=IDX)
                        keep_buf = np.empty(2 * m, dtype=np.uint8)
                    k = 0
                    j = p_head[c]
                    while j != -1:
                        idx_buf[k] = j
                        keep_buf[k] = 1
                        k += 1
                        j = pool_next[j]
                    any_moved = False
                    for v in range(m - 1, -1, -1):
                        s = _nxt(s)
                        if _u1(s) < e_exc:
                            _push(f_head, f_tail, f_cnt, c,
                                  pool_alpha[idx_buf[v]],
                                  pool_alpha, pool_next, free_stack, pctr)
                            keep_buf[v] = 0
                            any_moved = True
                    if any_moved:
                        nh = -1
                        nt = -1
                        ncnt = 0
                        for v in range(m):
                            jj = idx_buf[v]
                            if keep_buf[v] == 1:
                                pool_next[jj] = -1
                                if nh == -1:
                                    nh = jj
                                else:
                                    pool_next[nt] = jj
                                nt = jj
                                ncnt += 1
                            else:
                                _release(jj, free_stack, pctr)
                        p_head[c] = nh
                        p_tail[c] = nt
                        p_cnt[c] = ncnt

                # virus replication inside the factory
                nf = f_cnt[c]
                tot = 0.0
                j = f_head[c]
                while j != -1:
                    tot += pool_alpha[j]
                    j = pool_next[j]
                alpha_avg = tot / nf

                if nf > off_buf.shape[0]:
                    off_buf = np.empty(2 * nf, dtype=np.float64)
                n_off = 0
                j = f_head[c]
                for _v in range(nf):
                    a = pool_alpha[j]
                    s = _nxt(s)
                    if _u1(s) < r * alpha_avg * (1.0 - c_alpha * a):
                        na = a
                        s = _nxt(s)                   # Virus.copy(mutate=true)
                        if _u1(s) < muA:
                            s, na = _gauss_step(s, a, mut_step)
                        off_buf[n_off] = na
                        n_off += 1
                    j = pool_next[j]
                for k in range(n_off):
                    _push(f_head, f_tail, f_cnt, c, off_buf[k],
                          pool_alpha, pool_next, free_stack, pctr)

                clock[c] += 1
                if clock[c] == latency:              # burst
                    j = f_head[c]
                    while j != -1:
                        _push(e_head, e_tail, e_cnt, c, pool_alpha[j],
                              pool_alpha, pool_next, free_stack, pctr)
                        j = pool_next[j]
                    state[c] = EMPTY
                    dead_buf[n_dead] = c
                    n_dead += 1
                    clock[c] = 0

            # --------------------------------------------------------- empty
            else:
                s = _nxt(s)
                rnd = 1 + _bits3(s)
                nx = x + MOORE_X[rnd]
                if nx < 0:
                    nx += nc
                elif nx >= nc:
                    nx -= nc
                ny = y + MOORE_Y[rnd]
                if ny < 0:
                    ny += nr
                elif ny >= nr:
                    ny -= nr
                nb = nx * nr + ny
                if state_old[nb] == SUSC:
                    s = _nxt(s)
                    if _u1(s) < b * (1.0 - cI * p_cnt[nb]):
                        state[c] = SUSC
                        _free_chain(p_head, p_tail, p_cnt, c,
                                    pool_next, free_stack, pctr)
                        _free_chain(f_head, f_tail, f_cnt, c,
                                    pool_next, free_stack, pctr)
                        clock[c] = 0
                        j = p_head[nb]
                        while j != -1:
                            s = _nxt(s)
                            if _u1(s) >= loss:
                                a = pool_alpha[j]
                                s = _nxt(s)           # Virus.copy(mutate=true)
                                if _u1(s) < muA:
                                    s, a = _gauss_step(s, a, mut_step)
                                _push(p_head, p_tail, p_cnt, c, a,
                                      pool_alpha, pool_next, free_stack, pctr)
                            j = pool_next[j]

    # cacatoo keeps the arrays of cells that died this step alive until the
    # whole synchronous sweep is finished (they live on in the back-up grid).
    for k in range(n_dead):
        c = dead_buf[k]
        _free_chain(p_head, p_tail, p_cnt, c, pool_next, free_stack, pctr)
        _free_chain(f_head, f_tail, f_cnt, c, pool_next, free_stack, pctr)

    rs[0] = s


# ===========================================================================
#  Extracellular grid: one asynchronous sweep  (== sim.ext.asynchronous())
#
#  upd_order holds the grid coordinates *packed* as (x << 16) | y rather than
#  the raw index m = y*nc + x. The shuffle only permutes values, so this gives
#  exactly cacatoo's update order while avoiding a division per cell.
# ===========================================================================
@njit(cache=True)
def ext_step(nc, nr, upd_order, e_head, e_tail, e_cnt,
             pool_alpha, pool_next, free_stack, pctr, rs,
             diff_rate, decay_rate, n_steps, diff_mix):

    s = rs[0]
    n = nc * nr

    # Fisher-Yates shuffle exactly as cacatoo's shuffle()
    i = n
    while i > 0:
        i -= 1
        s = _nxt(s)
        ri = int(_u2(s) * (i + 1))
        tmp = upd_order[i]
        upd_order[i] = upd_order[ri]
        upd_order[ri] = tmp

    # Nodes teleported back into their own cell. model.js pushes them onto the
    # end of the very array it is iterating and re-evaluates the length each
    # step, so such a virion is visited AGAIN in the same sweep -- and can be
    # moved again. This FIFO reproduces that; it only ever fills in the
    # perfect-mixing branch, since Moore-8 never returns the focal cell.
    self_buf = np.empty(64, dtype=np.int64)

    for k in range(n):
        packed = upd_order[k]
        x = packed >> 16
        y = packed & 65535
        c = x * nr + y

        j = e_head[c]
        if j == -1:
            continue
        nh = -1
        nt = -1
        ncnt = 0
        sn = 0
        si = 0
        while True:
            if j == -1:
                if si < sn:
                    j = self_buf[si]
                    si += 1
                    nj = -1
                else:
                    break
            else:
                nj = pool_next[j]

            s = _nxt(s)
            if _u1(s) < decay_rate:
                _release(j, free_stack, pctr)
            elif diff_rate > 1.0 and diff_mix:
                # model.js: diff_rate > 1 is a switch to perfect mixing -- the
                # virion is teleported to a uniformly random grid cell. No
                # uniform is drawn in this branch. Reachable with --diffusion mix.
                s = _nxt(s)
                px = _randrange(s, nc)
                s = _nxt(s)
                qy = _randrange(s, nr)
                nb = px * nr + qy
                if nb == c:
                    if sn >= self_buf.shape[0]:
                        bigger = np.empty(2 * self_buf.shape[0], dtype=np.int64)
                        bigger[:sn] = self_buf[:sn]
                        self_buf = bigger
                    self_buf[sn] = j
                    sn += 1
                else:
                    _append_node(e_head, e_tail, e_cnt, nb, j, pool_next)
            elif diff_rate > 1.0:
                # --diffusion steps (default): diff_rate > 1 means the virion
                # takes floor(diff_rate) Moore-8 steps this timestep instead of
                # one, so diffusion keeps scaling past the single-step ceiling.
                # Decay is still applied once per timestep, not once per step.
                nx = x
                ny = y
                for _s in range(n_steps):
                    s = _nxt(s)
                    rnd = 1 + _bits3(s)
                    nx += MOORE_X[rnd]
                    if nx < 0:
                        nx += nc
                    elif nx >= nc:
                        nx -= nc
                    ny += MOORE_Y[rnd]
                    if ny < 0:
                        ny += nr
                    elif ny >= nr:
                        ny -= nr
                nb = nx * nr + ny
                if nb == c:
                    # the walk came home; it has already diffused this timestep
                    pool_next[j] = -1
                    if nh == -1:
                        nh = j
                    else:
                        pool_next[nt] = j
                    nt = j
                    ncnt += 1
                else:
                    _append_node(e_head, e_tail, e_cnt, nb, j, pool_next)
            else:
                s = _nxt(s)
                if _u1(s) < diff_rate:
                    s = _nxt(s)
                    rnd = 1 + _bits3(s)
                    nx = x + MOORE_X[rnd]
                    if nx < 0:
                        nx += nc
                    elif nx >= nc:
                        nx -= nc
                    ny = y + MOORE_Y[rnd]
                    if ny < 0:
                        ny += nr
                    elif ny >= nr:
                        ny -= nr
                    _append_node(e_head, e_tail, e_cnt, nx * nr + ny, j, pool_next)
                else:
                    pool_next[j] = -1
                    if nh == -1:
                        nh = j
                    else:
                        pool_next[nt] = j
                    nt = j
                    ncnt += 1
            j = nj
        e_head[c] = nh
        e_tail[c] = nt
        e_cnt[c] = ncnt

    rs[0] = s


# ===========================================================================
#  Statistics  (== the sim.time%100==0 block of sim.host.update())
# ===========================================================================
@njit(cache=True)
def collect_stats(nc, nr, state, p_head, f_head, p_cnt, e_head,
                  pool_alpha, pool_next, hist_P, hist_V, hist_PN):
    hist_P[:] = 0
    hist_V[:] = 0
    hist_PN[:] = 0
    num_S = 0
    num_I = 0
    n_virion = 0
    n_provirus = 0
    for c in range(nc * nr):
        if state[c] == SUSC:
            num_S += 1
            j = p_head[c]
            while j != -1:
                x = pool_alpha[j]
                if 0.0 <= x <= 1.0:
                    ix = int(math.floor(x / 0.01))
                    if ix == 100:
                        ix = 99
                    hist_P[ix] += 1
                n_provirus += 1
                j = pool_next[j]
            np_ = p_cnt[c]
            if np_ < 100:
                hist_PN[np_] += 1
            else:
                hist_PN[99] += 1
        elif state[c] == INFECTED:
            num_I += 1
            j = f_head[c]
            while j != -1:
                x = pool_alpha[j]
                if 0.0 <= x <= 1.0:
                    ix = int(math.floor(x / 0.01))
                    if ix == 100:
                        ix = 99
                    hist_V[ix] += 1
                n_virion += 1
                j = pool_next[j]
        j = e_head[c]
        while j != -1:
            x = pool_alpha[j]
            if 0.0 <= x <= 1.0:
                ix = int(math.floor(x / 0.01))
                if ix == 100:
                    ix = 99
                hist_V[ix] += 1
            n_virion += 1
            j = pool_next[j]
    return num_S, num_I, n_virion, n_provirus


# ===========================================================================
#  Initialisation  (== initialGrid + the two init loops in model.js)
# ===========================================================================
@njit(cache=True)
def initialise(nc, nr, state, clock, p_head, p_tail, p_cnt,
               f_head, f_tail, f_cnt, e_head, e_tail, e_cnt,
               pool_alpha, pool_next, free_stack, pctr, rs, alpha0):
    s = rs[0]
    n = nc * nr
    # sim.initialGrid(sim.host, "alive", 'empty', "S", 1.0)
    for _ in range(n):
        s = _nxt(s)                       # rand < 1.0 always -> "S"
    for c in range(n):
        state[c] = SUSC

    # host provirus / factory initialisation
    for x in range(nc):
        for y in range(nr):
            s = _nxt(s)
            if _u1(s) < 0.5:
                _push(p_head, p_tail, p_cnt, x * nr + y, alpha0,
                      pool_alpha, pool_next, free_stack, pctr)

    # sim.initialGrid(sim.host, "clock", 0)  -- still burns one draw per cell
    for _ in range(n):
        s = _nxt(s)
    for c in range(n):
        clock[c] = 0

    # extracellular viruses
    for x in range(nc):
        for y in range(nr):
            s = _nxt(s)
            if _u1(s) < 0.01:
                _push(e_head, e_tail, e_cnt, x * nr + y, alpha0,
                      pool_alpha, pool_next, free_stack, pctr)
    rs[0] = s


# ===========================================================================
#  Driver
# ===========================================================================
def js_num(v):
    """Format a number the way JavaScript's String(Number) does."""
    f = float(v)
    if f == int(f) and abs(f) < 1e21:
        return str(int(f))
    return np.format_float_positional(f, unique=True, trim='-')


class Pool:
    def __init__(self, cap):
        self.cap = cap
        self.alpha = np.zeros(cap, dtype=np.float64)
        self.next = np.full(cap, -1, dtype=IDX)
        self.free = np.zeros(cap, dtype=IDX)
        self.ctr = np.zeros(3, dtype=np.int64)   # free_top, bump, n_live

    def ensure(self, need):
        if need <= self.cap:
            return
        newcap = self.cap
        while newcap < need:
            newcap *= 2
        if newcap > 2 ** 31 - 2:
            raise MemoryError('virus pool would exceed the int32 index range')
        alpha = np.zeros(newcap, dtype=np.float64)
        alpha[:self.cap] = self.alpha
        self.alpha = alpha
        nxt = np.full(newcap, -1, dtype=IDX)
        nxt[:self.cap] = self.next
        self.next = nxt
        fr = np.zeros(newcap, dtype=IDX)
        fr[:self.cap] = self.free
        self.free = fr
        self.cap = newcap


# Fraction of the element pool below `cut`, computed exactly as
# classify.pool_density(how='mean') does it: each histogram is normalised to a
# density before averaging, so a lytic parasite living as a few thousand virions
# is not drowned by 50x more proviruses. The (P.sum()+V.sum()) factor in
# pool_density cancels in the ratio, so this is just the mean of the per-pool
# low-fractions over the pools big enough to be more than noise.
def defense_fraction(hist_P, hist_V, cut=0.2, min_count=200):
    nbins = len(hist_P)
    lo = int(round(cut * nbins))
    fr = [X[:lo].sum() / X.sum() for X in (hist_P, hist_V) if X.sum() >= min_count]
    if fr:
        return float(np.mean(fr))
    tot = hist_P.sum() + hist_V.sum()
    if tot <= 0:
        return 0.0
    return float((hist_P[:lo].sum() + hist_V[:lo].sum()) / tot)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Numba port of the Cacatoo virus-factory model (model.js)")
    ap.add_argument('--seed', type=float, required=True,
                    help='RNG seed (same meaning as model.js --seed)')
    ap.add_argument('--lambda', dest='lam', type=float, default=0.0,
                    help='integration probability (scheme 1 only)')
    ap.add_argument('--e', dest='e', type=float, required=True,
                    help='provirus excision probability')
    ap.add_argument('--scheme', type=int, required=True, choices=(1, 2, 3),
                    help='1: MOI-independent, 2: less likely at high MOI, '
                         '3: more likely at high MOI')
    ap.add_argument('--maxtime', type=int, default=200000)
    ap.add_argument('--size', type=int, default=200)
    ap.add_argument('--outdir', default='sweep_11aug26')
    ap.add_argument('--outfile', default=None,
                    help='explicit output path (overrides the cacatoo-style '
                         'filename built from scheme/lambda/e/seed)')
    ap.add_argument('--interval', type=int, default=100,
                    help='output interval (model.js hard-codes 100)')
    # model parameters (defaults identical to model.js)
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
                    help='spontaneous prophage induction probability per host '
                         'per timestep. 0 disables it, which is what model.js '
                         'did and what validate.sh compares against')
    ap.add_argument('--diff_rate', type=float, default=0.5,
                    help='<=1: probability of one Moore-8 step per timestep. '
                         '>1: with --diffusion steps (default) the virion takes '
                         'floor(diff_rate) steps per timestep; with --diffusion '
                         'mix it is teleported to a random cell, as model.js does')
    ap.add_argument('--diffusion', default='steps', choices=('steps', 'mix'),
                    help='what diff_rate > 1 means. "mix" reproduces model.js')
    ap.add_argument('--perfect-diffusion', '--perfect_diffusion',
                    nargs='?', const=1, type=int, default=0,
                    help='ignore diff_rate: every surviving virion is moved to a '
                         'uniformly random cell each timestep')
    ap.add_argument('--mix-virions', '--mix_virions',
                    nargs='?', const=1, type=int, default=0,
                    help='with --perfect-mix, shuffle the extracellular grid by '
                         'the same permutation, so virions travel with their '
                         'location instead of staying put')
    ap.add_argument('--perfect-mix', '--perfect_mix',
                    nargs='?', const=1, type=int, default=0,
                    help='shuffle the whole host grid every timestep, destroying '
                         'spatial structure among hosts')
    ap.add_argument('--decay_rate', type=float, default=0.2)
    ap.add_argument('--defense-fix', '--defense_fix', type=float, default=0.95,
                    help='stop once the defense type has effectively fixed: the '
                         'fraction of the element pool below --low-alpha is at '
                         'or above this on two consecutive outputs. What happens '
                         'after fixation is not the question being asked, and '
                         'those runs are the expensive ones. 0 disables it')
    ap.add_argument('--low-alpha', '--low_alpha', type=float, default=0.2,
                    help='alpha below this counts as defense type, for '
                         '--defense-fix. Matches classify.py --low-alpha')
    ap.add_argument('--pool', type=int, default=1 << 18,
                    help='initial virus pool size (grows automatically)')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--stdout', action='store_true',
                    help='write output lines to stdout instead of a file')
    args = ap.parse_args(argv)

    nc = nr = args.size
    if not 1 < nc <= 65535:
        ap.error('--size must be between 2 and 65535')
    n = nc * nr

    # ---- filename, exactly as model.js builds it
    if args.scheme == 1:
        fname = 'scheme1_lambda%s_ex%s_seed%s.txt' % (
            js_num(args.lam), js_num(args.e), js_num(args.seed))
    else:
        fname = 'scheme%s_ex%s_seed%s.txt' % (
            js_num(args.scheme), js_num(args.e), js_num(args.seed))
    path = args.outfile if args.outfile else os.path.join(args.outdir, fname)
    if not args.stdout:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    # ---- state
    state = np.zeros(n, dtype=np.int8)
    state_old = np.zeros(n, dtype=np.int8)
    clock = np.zeros(n, dtype=IDX)
    p_head = np.full(n, -1, dtype=IDX); p_tail = np.full(n, -1, dtype=IDX)
    p_cnt = np.zeros(n, dtype=IDX)
    f_head = np.full(n, -1, dtype=IDX); f_tail = np.full(n, -1, dtype=IDX)
    f_cnt = np.zeros(n, dtype=IDX)
    e_head = np.full(n, -1, dtype=IDX); e_tail = np.full(n, -1, dtype=IDX)
    e_cnt = np.zeros(n, dtype=IDX)
    dead_buf = np.zeros(n, dtype=IDX)
    hist_P = np.zeros(100, dtype=np.int64)
    hist_V = np.zeros(100, dtype=np.int64)
    hist_PN = np.zeros(100, dtype=np.int64)

    # cacatoo's update order is 0..n-1 over m, with x = m%nc, y = m//nc.
    # Store (x<<16)|y instead; the shuffle permutes values, so the order is
    # unchanged but no division is needed per cell.
    m = np.arange(n, dtype=np.int64)
    upd_order = (((m % nc) << 16) | (m // nc)).astype(IDX)

    if args.perfect_diffusion:
        # reuse the validated teleport branch: it fires on diff_rate > 1 + mix
        args.diff_rate = 2.0
        args.diffusion = 'mix'
    n_diff_steps = int(args.diff_rate) if args.diff_rate > 1.0 else 0
    diff_mix = args.diffusion == 'mix'
    mn = n if args.perfect_mix else 0
    mix_perm = np.empty(mn, dtype=np.int64)
    mix_tmp = np.empty(mn, dtype=np.int64)
    # cacatoo writes gridpoint k back to (k % nc, k // nc); constant, so build once
    k = np.arange(mn, dtype=np.int64)
    mix_dest = ((k % nc) * nr + k // nc) if mn else np.empty(0, np.int64)

    pool = Pool(max(args.pool, 4 * n))
    rs = np.array([seed_rng(args.seed)], dtype=np.uint64)

    initialise(nc, nr, state, clock, p_head, p_tail, p_cnt,
               f_head, f_tail, f_cnt, e_head, e_tail, e_cnt,
               pool.alpha, pool.next, pool.free, pool.ctr, rs, args.alpha)

    out = sys.stdout if args.stdout else open(path, 'a')
    t0 = _time.perf_counter()
    t = 0
    stop = False
    prev_fixed = False
    try:
        while True:
            # keep at least 2x headroom so the pool can never overflow mid-step
            pool.ensure(2 * int(pool.ctr[2]) + 65536)

            if args.perfect_mix:
                host_perfect_mix(nc, nr, state, clock, p_head, p_tail, p_cnt,
                                 f_head, f_tail, f_cnt, e_head, e_tail, e_cnt,
                                 args.mix_virions, mix_perm, mix_dest, mix_tmp, rs)

            host_step(nc, nr, state, state_old, clock,
                      p_head, p_tail, p_cnt, f_head, f_tail, f_cnt,
                      e_head, e_tail, e_cnt,
                      pool.alpha, pool.next, pool.free, pool.ctr,
                      dead_buf, rs,
                      args.d, args.beta, args.lam, args.scheme, args.e,
                      args.r, args.c_alpha, args.latency,
                      args.muA, args.mut_step, args.loss, args.b, args.cI,
                      args.sigma)

            if t % args.interval == 0:
                num_S, num_I, n_vir, n_prov = collect_stats(
                    nc, nr, state, p_head, f_head, p_cnt, e_head,
                    pool.alpha, pool.next, hist_P, hist_V, hist_PN)
                out.write(json.dumps(
                    {"time": t, "num_S": num_S, "num_I": num_I,
                     "provirus": hist_P.tolist(),
                     "virion": hist_V.tolist(),
                     "num_provirus": hist_PN.tolist()},
                    separators=(',', ':')) + '\n')
                # With induction a pool of prophages is not extinct -- it can
                # start a new factory -- so stopping at "no virions" would cut
                # those runs short. At sigma = 0 this is model.js's rule.
                if n_vir == 0 and (args.sigma <= 0.0 or n_prov == 0):
                    stop = True

                # Defense fixation. Required on two consecutive outputs so a
                # single noisy sample on a small pool cannot end the run.
                if args.defense_fix > 0.0:
                    hit = defense_fraction(hist_P, hist_V,
                                           args.low_alpha) >= args.defense_fix
                    if hit and prev_fixed:
                        stop = True
                    prev_fixed = hit

            ext_step(nc, nr, upd_order, e_head, e_tail, e_cnt,
                     pool.alpha, pool.next, pool.free, pool.ctr, rs,
                     args.diff_rate, args.decay_rate, n_diff_steps, diff_mix)

            t += 1
            if t >= args.maxtime or stop:
                break
    finally:
        if not args.stdout:
            out.close()

    dt = _time.perf_counter() - t0
    if not args.quiet:
        sys.stderr.write("Completed %d steps in %.3f s (%.4f ms/step) -> %s\n"
                         % (t, dt, 1000.0 * dt / max(t, 1),
                            'stdout' if args.stdout else path))
    return 0


if __name__ == '__main__':
    sys.exit(main())
