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

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=${CRDS_PATH:-/orange/adamginsburg/jwst/crds}
export GC_ALLOW_DEV="${GC_ALLOW_DEV:-1}"
export CRDS_SERVER_URL=${CRDS_SERVER_URL:-https://jwst-crds.stsci.edu}
# CRDS transient-outage retries.  The client's own defaults are ONE attempt with
# a ZERO-second delay (crds.core.config.get_client_retry_count/_delay_seconds),
# so a single 504 from jwst-crds.stsci.edu kills the whole job -- even when the
# reference it wants is already in CRDS_PATH.  Observed 2026-08-06 across three
# consecutive campaign cycles:
#
#   cloudc  4 of 4 merged reduces   FileNotFoundError .../pars-tweakregstep_0051.asdf
#   sgrc    4 of 8 reduce tasks     ClientResponseError 504 .../_0039.asdf
#   sgrb2   3 of 11 reduce tasks    504 / FileNotFoundError .../_0041,_0056,_0035.asdf
#
# Every one of those files was already cached; the server was answering its root
# at 200 while timing out on file GETs, and recovered within minutes each time.
# Retrying is provenance-neutral -- same context, same files, same bytes -- so
# this changes nothing about what a product is calibrated against.  It is NOT a
# reason to pin CRDS_CONTEXT or set CRDS_READONLY_CACHE; both of those DO change
# provenance and neither is needed to survive a blip.
export CRDS_CLIENT_RETRY_COUNT=${CRDS_CLIENT_RETRY_COUNT:-8}
export CRDS_CLIENT_RETRY_DELAY_SECONDS=${CRDS_CLIENT_RETRY_DELAY_SECONDS:-20}

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
    . "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")/_pipe_root.sh"
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
