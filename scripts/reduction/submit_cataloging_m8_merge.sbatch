#!/bin/bash
# ---------------------------------------------------------------------------
# Column-merge the per-filter m8 partials into the combined
# ..._resbgsub_m8.fits and de-duplicate it -> ..._resbgsub_m8_dedup.fits.
# afterok on all the per-filter submit_cataloging_m8_partial jobs.  Refuses to
# write a half-merged m8 if a partial is missing (no silent band fallback).
# ---------------------------------------------------------------------------
#SBATCH --job-name=catalog_m8merge
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%j.out

source ~/.bash_profile 2>/dev/null

TARGET=${TARGET:-sickle}
MODULES=${MODULES:-nrcb}
FILTERS=${FILTERS:-"F187N F210M F335M F470N F480M"}
CATDIR=${CATDIR:-/orange/adamginsburg/jwst/${TARGET}/catalogs}
M7=${M7:-}
MERGE_EXTRA=${MERGE_EXTRA:-}

PIPE_ROOT=${PIPE_ROOT:-}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"
PY=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}-catalog-m8merge" 2>/dev/null || true

FILT_CSV=$(echo "$FILTERS" | tr ' ' ',')
M7_ARG=""
[ -n "$M7" ] && M7_ARG="--m7=$M7" || M7_ARG="--catdir=$CATDIR --module=$MODULES"

echo "M8_MERGE start $(date -u +%FT%TZ) NODE=$(hostname) target=$TARGET filters=$FILT_CSV"
"$PY" "$HERE/m8_merge_partials.py" $M7_ARG --filters="$FILT_CSV" --execute $MERGE_EXTRA
echo "M8_MERGE_DONE rc=$? $(date -u +%FT%TZ)"
