#!/bin/bash
# Submit one diagnostic-write-up job per field.
#
#   ./submit_diagnostic_writeup.sh                 # every field in the registry
#   ./submit_diagnostic_writeup.sh brick cloudc    # named fields only
#
# One job per field rather than one job for everything: the per-tile
# astrometric tie is the expensive step and it scales with the number of
# filters, so brick (ten filters) and arches (two) have very different
# runtimes and should not share a wall-clock limit.  The job name carries the
# target and the stage as the standing convention requires, and it is set at
# submit time so a pending job is identifiable in the queue.

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}"
LOGDIR="${LOGDIR:-/blue/adamginsburg/adamginsburg/logs/jwst/diagnostic_writeup}"
mkdir -p "$LOGDIR"

if [ "$#" -gt 0 ]; then
    FIELDS=("$@")
else
    mapfile -t FIELDS < <(PYTHONPATH="$REPO" "$PYTHON" \
        "$REPO/scripts/analysis/make_diagnostic_writeup.py" --list)
fi

for FIELD in "${FIELDS[@]}"; do
    sbatch \
        --account=astronomy-dept \
        --qos=astronomy-dept-b \
        --job-name="${FIELD}-diagwriteup" \
        --output="${LOGDIR}/${FIELD}-diagwriteup_%j.log" \
        --ntasks=1 --cpus-per-task=4 --mem=64gb --time=08:00:00 \
        --wrap "PYTHONPATH='$REPO' '$PYTHON' -u \
                '$REPO/scripts/analysis/make_diagnostic_writeup.py' \
                --field '$FIELD' --skip-empty"
done
