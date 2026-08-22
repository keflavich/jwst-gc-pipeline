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
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"; export TMP="$TMPDIR" TEMP="$TMPDIR"; mkdir -p "$TMPDIR"

TARGET=${TARGET:-sickle}
MODULES=${MODULES:-nrcb}
FILTERS=${FILTERS:-"F187N F210M F335M F470N F480M"}
CATDIR=${CATDIR:-/orange/adamginsburg/jwst/${TARGET}/catalogs}
M7=${M7:-}
MERGE_EXTRA=${MERGE_EXTRA:-}

PIPE_ROOT=${PIPE_ROOT:-}
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/_pipe_root.sh"
PY=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
# Locate m8_merge_partials.py robustly.  Under `sbatch` SLURM copies the batch
# script to a spool dir, so ${BASH_SOURCE[0]} points there and $HERE/<sibling>.py
# does not exist (silent rc=2 merge failure, w51 m8).  Resolve via the pipeline
# repo root instead: PIPE_ROOT if given, else the installed package location.
if [ -n "$PIPE_ROOT" ] && [ -f "$PIPE_ROOT/scripts/reduction/m8_merge_partials.py" ]; then
    MERGE_PY="$PIPE_ROOT/scripts/reduction/m8_merge_partials.py"
else
    REPO="$("$PY" -c 'import jwst_gc_pipeline,os;print(os.path.dirname(os.path.dirname(jwst_gc_pipeline.__file__)))' 2>/dev/null)"
    MERGE_PY="$REPO/scripts/reduction/m8_merge_partials.py"
fi
if [ ! -f "$MERGE_PY" ]; then
    echo "FATAL: cannot locate m8_merge_partials.py (tried PIPE_ROOT and package root '$REPO'); set PIPE_ROOT to the repo." >&2
    exit 66
fi

[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}-catalog-m8merge" 2>/dev/null || true

FILT_CSV=$(echo "$FILTERS" | tr ' ' ',')
M7_ARG=""
[ -n "$M7" ] && M7_ARG="--m7=$M7" || M7_ARG="--catdir=$CATDIR --module=$MODULES"

echo "M8_MERGE start $(date -u +%FT%TZ) NODE=$(hostname) target=$TARGET filters=$FILT_CSV"
"$PY" "$MERGE_PY" $M7_ARG --filters="$FILT_CSV" --execute $MERGE_EXTRA
rc=$?
echo "M8_MERGE_DONE rc=$rc $(date -u +%FT%TZ)"
# Propagate the real exit code: a trailing echo previously masked a failed merge
# as SLURM COMPLETED (the merge could silently not run and the job still passed).
exit $rc
