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
# FIDELITY: intended to match a monolithic run.  The out-of-FOV satstar flux
# pins from m12's reconcile are persisted to disk (*_satstar_reconciled_m12.fits)
# and reconstructed by --manual-start-phase, so this finalize re-applies them
# like the monolith does.  Off-FOV satstars additionally use the Spitzer-prior
# (_spitzer.reg) + model<=data clamp, loaded fresh every phase in both paths --
# believed to make the m12 satstar_overrides pins vestigial for off-FOV stars
# (unverified, but moot given the persisted round-trip).  See cataloging.py
# run_manual_pipeline start_phase note.  The ZEROFRAME satstar deblend is plumbed
# via DEBLEND_SATSTARS below so a re-made satstar catalog at m7 matches m12.
# ---------------------------------------------------------------------------
# renamed to "<target>-catalog-m7" at runtime (target is not always the GC)
#SBATCH --job-name=catalog_m7
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
# m7 cross-band finalize is I/O + table-stack bound, NOT cpu-parallel (it streams
# the merged columns one at a time).  Asking few cpus lets it backfill into small
# queue holes instead of waiting tens of hours for a large-cpu node.  mem/time are
# deliberately left generous -- shrinking THOSE risks an OOM/timeout kill mid-run
# and losing the whole finalize, whereas a small cpu ask only changes start time.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%j.out

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
# Scratch on node-local SLURM_TMPDIR, never /tmp (fills node root fs).
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
# MODULES must be set together with the target env (brick 2221 = merged;
# a stray nrcb default produced per-module products the m7 merge cannot use).
if [ -n "${PROPOSAL:-}${FIELD:-}${TARGET:-}${EACH_SUFFIX:-}" ] && [ -z "${MODULES:-}" ]; then
    echo "FATAL: MODULES not set but other target env vars are -- set it (e.g. MODULES=merged for brick 2221)." >&2
    exit 64
fi
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}-catalog-m7" 2>/dev/null || true
# ALL filters that participate in the cross-band merge (space-separated).
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}
PARALLEL_WORKERS=${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
# Optional: pin the cross-band astrometric reference filter.
CROSSBAND_REF=${CROSSBAND_REF:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

# ZEROFRAME satstar deblend MUST match the per-filter (m12..m6) array stage: the
# m7 partial-start re-scans frames, and load_or_make_satstar_catalog re-makes a
# frame's satstar catalog when off-FOV pixels force overwrite -- a re-make WITHOUT
# this flag would drop the m12 deblend and diverge from a monolithic run.  Mirror
# submit_cataloging.sbatch so the finalize is functionally identical to the
# monolith.  Auto-degrades to legacy where a frame has no sibling _ramp.fits.
DEBLEND_SATSTARS=${DEBLEND_SATSTARS:-}
DEBLEND_ARG=""
[ -n "$DEBLEND_SATSTARS" ] && DEBLEND_ARG="--deblend-satstars"

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
    $DEBLEND_ARG \
    $REF_ARG \
    $EXTRA_ARGS
rc=$?
echo "CATALOG m7 done: rc=$rc"
exit $rc
