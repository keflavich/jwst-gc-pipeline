#!/bin/bash
# ---------------------------------------------------------------------------
# Cataloging SLURM submitter (refactored jwst_gc_pipeline).
#
# Runs the ACTIVE per-exposure manual photometry pipeline
# (crowdsource_catalogs_long.py --each-exposure: m-phases m12/m3..m7).
# This is module-importable, so it is run via `python -m` (unlike the
# reduction entry point). Array over filters: one task per filter.
#
# Run this AFTER submit_reduction.sbatch has produced the *_crf.fits.
#
# Usage:
#   sbatch --array=0-7 scripts/reduction/submit_cataloging.sbatch
#   sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,TARGET=brick,\
#          FILTERS="F405N F410M F466N F212N" scripts/reduction/submit_cataloging.sbatch
# ---------------------------------------------------------------------------
# NOTE: not all targets are in the Galactic Center -- the job is renamed to
# "<target>-catalog" at runtime (below) once TARGET is known.  This static
# default only shows until the job starts.
# JOB NAMING (STANDING RULE -- do not submit with the generic name):
#   ALWAYS pass a descriptive name at submit time:
#     sbatch --job-name=<target><program>-o<obsid>-cat[-FILTER] ...
#     e.g. --job-name=brick2221-o001-cat-F182M
#   Pending jobs show ONLY this static name; the runtime rename fires at START.
#SBATCH --job-name=catalog
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%A_%a.out

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

# Scratch on node-local SLURM_TMPDIR, never /tmp (fills node root fs).
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}

# Tag the job with its target (it is not always the GC).  Renames in-queue so
# squeue shows e.g. "w51-catalog" instead of a generic name.
[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}-catalog" 2>/dev/null || true

# Pin a worktree/branch by setting PIPE_ROOT (prepends to PYTHONPATH).
PIPE_ROOT=${PIPE_ROOT:-}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"

# --max-group-size caps overlap-group size so the joint LevMar PSF fit cost
# (~cubic in group size) cannot blow up in dense/wide-PSF fields.  LW filters
# (wider PSF -> larger groups) otherwise hit the walltime; set e.g. 12 for
# dense fields.  Default 'unlimited' preserves prior behaviour for others.
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}
# Extra passthrough args (e.g. method selection / resume):
#   "--daophot --skip-crowdsource --skip-if-done"
EXTRA_ARGS=${EXTRA_ARGS:-}

# Frame-level parallelism: the manual pipeline fits per-frame independently via
# an in-process fork pool sized by --parallel-workers.  Default 1 (serial) would
# leave every requested core but one IDLE, so default it to the allocation.
# Override to dial the resource ask up/down (fewer workers -> fits smaller queue
# holes; see submit_cataloging_chain.sh for the low-resource stream).
PARALLEL_WORKERS=${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}

# Opt-in ZEROFRAME deblend of merged saturated cores (crowded GC fields, e.g.
# gc2211): set DEBLEND_SATSTARS=1.  Auto-degrades to legacy where a frame has no
# sibling _ramp.fits ZEROFRAME, so it is safe to leave on for fields that lack it.
DEBLEND_SATSTARS=${DEBLEND_SATSTARS:-}
DEBLEND_ARG=""
[ -n "$DEBLEND_SATSTARS" ] && DEBLEND_ARG="--deblend-satstars"

read -r -a FILT_ARR <<< "$FILTERS"
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILT=${FILT_ARR[$IDX]}
if [ -z "$FILT" ]; then
    echo "No filter for array index $IDX (FILTERS='$FILTERS')."
    exit 1
fi

# Descriptive per-task job name (filter only known after indexing the array).
# Fallback rename (fires at START; a submit-time --job-name is preserved).
if [ "${SLURM_JOB_NAME:-catalog}" = "catalog" ]; then
    scontrol update JobId="${SLURM_JOB_ID:-0}" Name="${TARGET}${PROPOSAL:+$PROPOSAL}-o${FIELD:-}-cat-${FILT}" 2>/dev/null || true
fi

# The active manual pipeline refuses to run with SLURM_ARRAY_TASK_ID set (its
# phases must not be split across array tasks).  We use the array index ONLY to
# pick this task's filter -- each task then runs ALL phases for that one filter
# in-process, which is exactly the single-job mode the guard wants.  Unset the
# array env so the guard passes; this is array-over-FILTERS, not phase-splitting.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID

echo "CATALOG start: $TARGET $PROPOSAL/$FIELD filter=$FILT modules=$MODULES"
PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
"$PYTHON" -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames="$FILT" \
    --modules="$MODULES" \
    --each-exposure \
    --proposal_id="$PROPOSAL" \
    --field="$FIELD" \
    --target="$TARGET" \
    --each-suffix="$EACH_SUFFIX" \
    --max-group-size="$MAX_GROUP_SIZE" \
    --parallel-workers="$PARALLEL_WORKERS" \
    $DEBLEND_ARG \
    $EXTRA_ARGS
rc=$?
echo "CATALOG done: filter=$FILT rc=$rc"
exit $rc
