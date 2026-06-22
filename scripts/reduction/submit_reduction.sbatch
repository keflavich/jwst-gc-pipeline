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
#SBATCH --job-name=gc_reduce
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/reduce_%x_%A_%a.out

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

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

echo "REDUCE start: proposal=$PROPOSAL field=$FIELD filter=$FILT modules=$MODULES skip=$SKIP"
echo "PIPE=$PIPE"
python "$PIPE" -p "$PROPOSAL" -d "$FIELD" -f "$FILT" -m "$MODULES" $SKIP_FLAG
rc=$?
echo "REDUCE done: filter=$FILT rc=$rc"
exit $rc
