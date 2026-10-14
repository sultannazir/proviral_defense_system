#!/bin/bash
# Snapshot a sweep while it is still running: progress + ETA, classification of
# everything finished so far, the sweep-level figures, and a classifier check.
# Safe to run at any time -- it only reads, and skips runs still in progress.
#
#   ./preview.sh                                  # the current sweep (lhs2)
#   ./preview.sh lhs_design.csv lhs_out           # or point it somewhere else
#   PREVIEW_FIGS=1 ./preview.sh                   # also draw one PNG per run
#
# Per-run PNGs are off by default: there are thousands of them and they are the
# slow part. They are written incrementally (--skip-existing), so turning them
# on costs full price only the first time.
set -u
cd "$(dirname "$0")"
DESIGN=${1:-lhs3_design.csv}
OUT=${2:-lhs3_out}
RESULTS=${3:-preview_results.csv}
PLOTS=${4:-plots_preview}
FIGS=${PREVIEW_FIGS:-0}

python3 sweep_status.py "$DESIGN"
echo
FIGARG=""
[ "$FIGS" = 1 ] && FIGARG="--figures"
# shellcheck disable=SC2086
python3 classify.py "$OUT" --design "$DESIGN" -o "$RESULTS" --finished-only $FIGARG
echo
mkdir -p "$PLOTS"
python3 plot_outcomes.py "$RESULTS" --plot-dir "$PLOTS"
python3 plot_runs.py "$RESULTS" --plot-dir "$PLOTS" --out classifier_check
echo
echo "figures in $PLOTS/"
