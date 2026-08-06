#!/bin/bash
# ---------------------------------------------------------------------------
# NIRISS cataloging SLURM submitter (single detector; jwst_gc_pipeline).
#
# Runs the per-exposure manual PSF-photometry pipeline
# (crowdsource_catalogs_long.py --each-exposure) on the NIRISS reduced crf.
# NIRISS is a single detector: --modules=nis, --each-suffix=o012_crf, and
# --instrument=niriss (which redirects the tree to <target>/niriss/<FILTER>/ and
# forces the NIRISS PSF/SVO/FWHM branches -- NIRISS filter names are shared with
# NIRCam so the instrument cannot be inferred).
#
# Prereq: scripts/reduction/stage_niriss_basepath.sh <target> (once) + the NIRISS
# reduction (submit_reduction_niriss.sbatch) producing *_nis_o012_crf.fits.
#
# Usage:
#   sbatch --array=0-3 --job-name=sgrc4147-o012-cat-niriss \
#          scripts/reduction/submit_cataloging_niriss.sbatch
#   sbatch --array=0-0 --export=ALL,FILTERS="F200W" \
#          --job-name=sgrc4147-o012-cat-niriss-F200W \
#          scripts/reduction/submit_cataloging_niriss.sbatch
# ---------------------------------------------------------------------------
#SBATCH --job-name=cat-niriss
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=96:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%A_%a.out

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
TARGET=${TARGET:-sgrc}
INSTRUMENT=${INSTRUMENT:-niriss}
MODULES=${MODULES:-nis}
EACH_SUFFIX=${EACH_SUFFIX:-o012_crf}
FILTERS=${FILTERS:-"F158M F200W F356W F480M"}
# LW/dense GC field: cap overlap-group size so the joint LevMar PSF fit cost
# (~cubic in group size) cannot blow up the walltime.
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-12}
PARALLEL_WORKERS=${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Pin this worktree/branch (NIRISS support is not yet on main).
PIPE_ROOT=${PIPE_ROOT:-/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-niriss}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"

read -r -a FILT_ARR <<< "$FILTERS"
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILT=${FILT_ARR[$IDX]}
if [ -z "$FILT" ]; then
    echo "No filter for array index $IDX (FILTERS='$FILTERS')."
    exit 1
fi

if [ "${SLURM_JOB_NAME:-cat-niriss}" = "cat-niriss" ]; then
    scontrol update JobId="${SLURM_JOB_ID:-0}" Name="${TARGET}${PROPOSAL}-o${FIELD}-cat-niriss-${FILT}" 2>/dev/null || true
fi

# The manual pipeline refuses to run with SLURM_ARRAY_TASK_ID set (phases must
# not be split across array tasks); we use the array index ONLY to pick this
# task's filter, then run ALL phases for that filter in-process.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID

PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
echo "CATALOG(NIRISS) start: $TARGET $PROPOSAL/$FIELD filter=$FILT modules=$MODULES suffix=$EACH_SUFFIX"
"$PYTHON" -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames="$FILT" \
    --modules="$MODULES" \
    --each-exposure \
    --proposal_id="$PROPOSAL" \
    --field="$FIELD" \
    --target="$TARGET" \
    --instrument="$INSTRUMENT" \
    --each-suffix="$EACH_SUFFIX" \
    --max-group-size="$MAX_GROUP_SIZE" \
    --parallel-workers="$PARALLEL_WORKERS" \
    $EXTRA_ARGS
rc=$?
echo "CATALOG(NIRISS) done: filter=$FILT rc=$rc"
exit $rc
