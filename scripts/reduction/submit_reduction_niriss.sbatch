#!/bin/bash
# ---------------------------------------------------------------------------
# NIRISS reduction SLURM submitter (single detector; jwst_gc_pipeline).
#
# Runs reduction/PipelineRerunNIRISS.py (Detector1 + Image2 from uncal, then
# per-exposure fix_alignment + Image3 with abs_refcat -> VIRAC2).  NIRISS is a
# single detector (no nrca/nrcb module split, no destreak), so there is no
# -m/MODULES argument.
#
# Array over filters: one task per filter.
#   sbatch --array=0-3 --job-name=sgrc4147-o012-reduce-niriss \
#          scripts/reduction/submit_reduction_niriss.sbatch
#   # single filter smoke test:
#   sbatch --array=0-0 --export=ALL,FILTERS="F200W" \
#          --job-name=sgrc4147-o012-reduce-niriss-F200W \
#          scripts/reduction/submit_reduction_niriss.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=reduce-niriss
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/reduce_%x_%A_%a.out

# ~/.bash_profile sources ~/.bashrc, which exports CRDS_PATH and STPSF_PATH on
# this account -- so sourcing it OVERWRITES whatever the caller set, and a bare
# ${VAR:-default} below would then read .bashrc's value rather than the caller's.
# Remember what the caller passed, first.
_caller_CRDS_PATH=${CRDS_PATH:-}
_caller_CRDS_SERVER_URL=${CRDS_SERVER_URL:-}
source ~/.bash_profile 2>/dev/null
export CRDS_PATH=${_caller_CRDS_PATH:-${CRDS_PATH:-/orange/adamginsburg/jwst/crds}}
export GC_ALLOW_DEV="${GC_ALLOW_DEV:-1}"
export CRDS_SERVER_URL=${_caller_CRDS_SERVER_URL:-${CRDS_SERVER_URL:-https://jwst-crds.stsci.edu}}

export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# --- parameters (override via --export=ALL,VAR=...) ---
PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
# Reuse existing *_cal.fits (-s) if SKIP=1; SKIP=0 re-fits ramps from uncal.
SKIP=${SKIP:-0}
FILTERS=${FILTERS:-"F158M F200W F356W F480M"}
# skymatch method: match+subtract by default (guards outlier_detection NaN holes
# on few-frame mosaics).  Set to "off" for record-only (no subtraction).
SKYMATCH=${SKYMATCH:-match}

# Pin to this worktree/branch by default (the NIRISS pipeline is not yet on main).
PIPE_ROOT=${PIPE_ROOT:-/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-niriss}
if [ -n "$PIPE_ROOT" ]; then
    export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"
    PIPE="$PIPE_ROOT/jwst_gc_pipeline/reduction/PipelineRerunNIRISS.py"
else
    PIPE="$(python -c 'import os,jwst_gc_pipeline as p; print(os.path.join(os.path.dirname(p.__file__),"reduction","PipelineRerunNIRISS.py"))')"
fi

read -r -a FILT_ARR <<< "$FILTERS"
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILT=${FILT_ARR[$IDX]}
if [ -z "$FILT" ]; then
    echo "No filter for array index $IDX (FILTERS='$FILTERS'). Submit --array=0-$(( ${#FILT_ARR[@]} - 1 ))."
    exit 1
fi

SKIP_FLAG=""
[ "$SKIP" = "1" ] && SKIP_FLAG="-s"

if [ "${SLURM_JOB_NAME:-reduce-niriss}" = "reduce-niriss" ]; then
    scontrol update JobId="${SLURM_JOB_ID:-0}" Name="sgrc${PROPOSAL}-o${FIELD}-reduce-niriss-${FILT}" 2>/dev/null || true
fi

PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
echo "REDUCE(NIRISS) start: proposal=$PROPOSAL field=$FIELD filter=$FILT skip=$SKIP skymatch=$SKYMATCH"
echo "PIPE=$PIPE"
"$PYTHON" "$PIPE" -p "$PROPOSAL" -d "$FIELD" -f "$FILT" $SKIP_FLAG --skymatch-method="$SKYMATCH"
rc=$?
echo "REDUCE(NIRISS) done: filter=$FILT rc=$rc"
exit $rc
