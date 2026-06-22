#!/bin/bash
# ---------------------------------------------------------------------------
# Cross-band (m7) finalize job for the manual photometry pipeline.
#
# Runs the cross-band merge phase ONLY, over ALL filters at once, reusing the
# per-filter m12..m6 products already on disk (built by the per-filter jobs in
# submit_cataloging.sbatch / submit_cataloging_chain.sh).  Uses the existing
# --manual-start-phase m7, which reconstructs the m6 background map from disk.
#
# Typically submitted by submit_cataloging_chain.sh with an afterok dependency
# on the per-filter array, but can be run standalone once m6 is complete.
#
# KNOWN FIDELITY CAVEAT: a standalone m7 job does NOT re-apply the out-of-FOV
# saturated-star flux pins computed at m12 (they live in memory within the
# monolithic run).  This affects only off-FOV bright-star photometry in the
# final cross-band catalog; in-FOV science is unaffected.  For full fidelity on
# off-FOV satstars, run the monolithic multifilter job instead.
# ---------------------------------------------------------------------------
#SBATCH --job-name=gc_catalog_m7
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%j.out

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=/orange/adamginsburg/jwst/crds
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
# ALL filters that participate in the cross-band merge (space-separated).
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}
PARALLEL_WORKERS=${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
# Optional: pin the cross-band astrometric reference filter.
CROSSBAND_REF=${CROSSBAND_REF:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

PIPE_ROOT=${PIPE_ROOT:-}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"

# crowdsource_catalogs_long takes a comma-separated filter list.
FILT_CSV=$(echo "$FILTERS" | tr ' ' ',')
REF_ARG=""
[ -n "$CROSSBAND_REF" ] && REF_ARG="--manual-crossband-ref-filter=$CROSSBAND_REF"

scontrol update JobId="${SLURM_JOB_ID:-0}" Name="cat_${TARGET}_m7" 2>/dev/null || true

echo "CATALOG m7 start: $TARGET $PROPOSAL/$FIELD filters=$FILT_CSV modules=$MODULES"
PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
"$PYTHON" -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames="$FILT_CSV" \
    --modules="$MODULES" \
    --each-exposure \
    --proposal_id="$PROPOSAL" \
    --field="$FIELD" \
    --target="$TARGET" \
    --each-suffix="$EACH_SUFFIX" \
    --max-group-size="$MAX_GROUP_SIZE" \
    --parallel-workers="$PARALLEL_WORKERS" \
    --manual-start-phase=m7 \
    $REF_ARG \
    $EXTRA_ARGS
rc=$?
echo "CATALOG m7 done: rc=$rc"
exit $rc
