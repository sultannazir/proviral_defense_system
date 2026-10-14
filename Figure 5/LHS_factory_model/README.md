# Virus factory model — fast version

This is S.A. Nazir's Cacatoo "virus factory" model (`original_model/model.js`) reimplemented with the help of Claude Opus 4.6 into Python with Numba. It is the same model, but much faster, and with additional scripts to run parameter sweeps and classify the outcomes in terms of defense systems. 

The port is not an approximation. We confirmed it reproduces Cacatoo's results bit-by-bit if the same random number generator is used. 

## Setup

    python3 -m pip install -r requirements.txt
    python3 -m pip install pygame        # only for the live viewer

To run the original JavaScript for comparison you also need node and yargs:

    cd original_model && npm install yargs@17

(yargs 18+ is ESM-only and `model.js` uses `require()`, so the major
version matters!)

The first Python run compiles the kernels (~4 s) and caches them; later runs
start immediately, and a very fast due to Numba's low-level (C) implementation.

## One simulation

    python3 model_numba.py --seed 1 --lambda 0.1 --e 0.01 --scheme 1

Writes one JSON object per line, every 100 timesteps, to
`sweep_<DATE>/scheme1_lambda0.1_ex0.01_seed1.txt` — the same filenames and
format `model.js` produced. Each line holds `time`, `num_S`, `num_I` and three
100-bin histograms: provirus alpha, virion alpha, and proviruses per host.

Unlike `model.js`, **every parameter is a command-line flag** (`--beta`, `--r`,
`--muA`, `--diff_rate`, `--decay_rate`, `--latency`, …), each defaulting to the
`model.js` value. 

Two behaviours go beyond `model.js`, both off by default and documented in
[LHS.md](LHS.md): `--diff_rate` above 1 means multiple diffusion steps per
timestep, and `--perfect-mix` / `--perfect-diffusion` are separate flags for the
two halves of what `model.js` did with `mixinp = 1`.

## Watching a run

    python3 display.py --e 0.1 --lambda 0.1 --decay_rate 0.2

Grid on the left; population sizes and an alpha-spectrum-over-time heatmap on
the right. `space` pauses, `+`/`-` changes speed, `s` saves a screenshot,
`q` quits. `--dump FILE` records the run while you watch.


## The LHS sweep

    # build a Latin hypercube design
    python3 run_sweep.py --make-lhs 1000 --out design.csv \
        --vary "e=0.01:1:log,lambda=0.001:0.1:log,decay_rate=0.01:0.25,\
                sigma=1e-5:1e-3:log,beta=0.1:1.0" \
        --fixed "scheme=1,maxtime=200000,outdir=out"

    # run it, then classify and plot automatically
    python3 run_sweep.py design.csv --jobs 10

One simulation uses one core; parallelism comes from running many at once.
While it runs, `./preview.sh` gives progress, an ETA and interim figures.

`classify.py` labels each run by what survived and what the alpha distribution
looks like at the end (`host_extinct`, `virus_extinct`, `no_defense`,
`defense_evolved`) and writes one summary CSV. `plot_outcomes.py` draws the
marginal outcome composition per parameter and two-parameter outcome maps;
`plot_runs.py` draws a per-run check figure so any label can be audited against
the trajectory it came from. `alpha_matrix.py` pulls the raw alpha-versus-time
data out of a run for plotting elsewhere.
