#!/bin/bash
# ---------------------------------------------------------------------------
# Reduction + alignment SLURM submitter (refactored jwst_gc_pipeline).
#
# Runs PipelineRerunNIRCAM-LONG.py (Image3 imaging + per-exposure
# fix_alignment + post-resample realign_to_catalog; see
# jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md).
#
# Submitted as a SLURM ARRAY over filters: one array task per filter, each
# task processing all requested modules (nrca,nrcb,merged) for that filter.
# This parallelizes the per-filter loop across nodes.
#
# IMPORTANT (refactor compatibility):
#   * The entry point is a SCRIPT FILE, not an importable module. Its name
#     contains a hyphen (PipelineRerunNIRCAM-LONG.py), so `python -m ...`
#     CANNOT be used -- it must be run by full path.
#   * The script does `from jwst_gc_pipeline.reduction...`, so the
#     jwst_gc_pipeline package must be importable. It is pip-installed
#     (editable), so a bare `python /path/to/PipelineRerunNIRCAM-LONG.py`
#     resolves the package automatically -- no PYTHONPATH needed when running
#     the installed checkout. To run a worktree/branch instead, set
#     PYTHONPATH to that checkout root (see PIPE_ROOT below).
#   * The OLD _bench/cloudef_o002_rereduce.sbatch points at
#     brick-jwst-2221/brick2221/... -- that is the pre-refactor package and
#     is STALE. Use this script instead.
#
# Usage:
#   sbatch --array=0-7 scripts/reduction/submit_reduction.sbatch
#   # or override target via --export:
#   sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,FILTERS="F405N F410M F466N F212N" \
#          scripts/reduction/submit_reduction.sbatch
#
# Defaults below target Sgr C (proposal 4147, field 012), all 8 filters,
# reusing existing *_cal.fits (skip_step1and2 / -s).
# ---------------------------------------------------------------------------
# renamed to "red_<prop>_<field>_<filt>" at runtime (target is not always the GC)
# JOB NAMING (STANDING RULE -- do not submit with the generic name):
#   ALWAYS pass a descriptive name at submit time:
#     sbatch --job-name=<target><program>-o<obsid>-reduce[-FILTER] ...
#     e.g. --job-name=brick2221-o001-reduce-F182M  /  cloudc2221-o002-reduce
#   Multiple reduce/catalog jobs are almost always in flight; a pending job
#   shows ONLY this static name (the runtime rename below fires at START).
#   Include the obsid whenever the program has more than one observation.
#SBATCH --job-name=reduce
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/reduce_%x_%A_%a.out

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=/orange/adamginsburg/jwst/crds
# Run guard (jwst_gc_pipeline.versioning.tags.assert_runnable_version): the
# imaging/cataloging entry points HARD-BLOCK an untagged or dirty tree.  The live
# reduction runs from the working tree (a development run), so permit it by
# default -- it stamps a dev tag, so its products can never masquerade as a
# release.  A RELEASE run should check out a release tag: a tagged, clean tree
# passes as production regardless of this flag.  Set GC_ALLOW_DEV=0 to force the
# block (e.g. to guarantee a run is on a tagged checkout).
export GC_ALLOW_DEV="${GC_ALLOW_DEV:-1}"
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

# Keep all scratch (jwst/astropy/tempfile) on the node-local SLURM scratch, NOT
# /tmp -- the pipeline's intermediates otherwise fill the node root fs (/dev/md0)
# and wedge the node.  SLURM_TMPDIR is per-job node-local; fall back to a large
# shared path (never /tmp).
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# --- parameters (override via --export=ALL,VAR=...) ---
PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
MODULES=${MODULES:-nrca,nrcb,merged}
# Reuse existing *_cal.fits by default (-s / skip_step1and2). Set SKIP=0 to
# re-download + re-fit ramps from MAST (much slower; overwrites all products).
SKIP=${SKIP:-1}
# Full Sgr C filter set. Override with --export=ALL,FILTERS="F405N F470N".
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}

# Pin to a specific checkout (worktree/branch) by setting PIPE_ROOT; otherwise
# use the pip-installed package location.
PIPE_ROOT=${PIPE_ROOT:-}
if [ -n "$PIPE_ROOT" ]; then
    export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"
    PIPE="$PIPE_ROOT/jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py"
else
    PIPE="$(python -c 'import os,jwst_gc_pipeline as p; print(os.path.join(os.path.dirname(p.__file__),"reduction","PipelineRerunNIRCAM-LONG.py"))')"
fi

# Pick this array task's filter.
read -r -a FILT_ARR <<< "$FILTERS"
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILT=${FILT_ARR[$IDX]}
if [ -z "$FILT" ]; then
    echo "No filter for array index $IDX (FILTERS='$FILTERS'). Submit --array=0-$(( ${#FILT_ARR[@]} - 1 ))."
    exit 1
fi

SKIP_FLAG=""
[ "$SKIP" = "1" ] && SKIP_FLAG="-s"

# Optional Image3 skymatch (level-match per-exposure background pedestals ->
# removes mosaic seams/stripes; sickle F470N).  Set SKYMATCH=match (or global/
# local).  Empty (default) preserves historical NIRCam behavior (skymatch off).
SKYMATCH=${SKYMATCH:-}
SKYMATCH_FLAG=""
[ -n "$SKYMATCH" ] && SKYMATCH_FLAG="--skymatch-method=$SKYMATCH"

# Descriptive per-task job name (the #SBATCH --job-name is necessarily generic
# because the filter is only known after indexing the array here).  Rename this
# task so squeue shows what it actually is, e.g. "red_4147_012_F470N".
# Fallback rename (fires at START; a submit-time --job-name is preserved).
if [ "${SLURM_JOB_NAME:-reduce}" = "reduce" ]; then
    scontrol update JobId="${SLURM_JOB_ID:-0}" Name="jw${PROPOSAL}-o${FIELD}-reduce-${FILT}" 2>/dev/null || true
fi

PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
echo "REDUCE start: proposal=$PROPOSAL field=$FIELD filter=$FILT modules=$MODULES skip=$SKIP skymatch=${SKYMATCH:-off}"
echo "PIPE=$PIPE"
"$PYTHON" "$PIPE" -p "$PROPOSAL" -d "$FIELD" -f "$FILT" -m "$MODULES" $SKIP_FLAG $SKYMATCH_FLAG
rc=$?
echo "REDUCE done: filter=$FILT rc=$rc"
exit $rc
