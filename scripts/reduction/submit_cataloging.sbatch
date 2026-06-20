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
#SBATCH --job-name=gc_catalog
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%A_%a.out

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}

# Pin a worktree/branch by setting PIPE_ROOT (prepends to PYTHONPATH).
PIPE_ROOT=${PIPE_ROOT:-}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"

read -r -a FILT_ARR <<< "$FILTERS"
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILT=${FILT_ARR[$IDX]}
if [ -z "$FILT" ]; then
    echo "No filter for array index $IDX (FILTERS='$FILTERS')."
    exit 1
fi

echo "CATALOG start: $TARGET $PROPOSAL/$FIELD filter=$FILT modules=$MODULES"
python -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames="$FILT" \
    --modules="$MODULES" \
    --each-exposure \
    --proposal_id="$PROPOSAL" \
    --field="$FIELD" \
    --target="$TARGET" \
    --each-suffix="$EACH_SUFFIX"
rc=$?
echo "CATALOG done: filter=$FILT rc=$rc"
exit $rc
