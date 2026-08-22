#!/bin/bash
# ---------------------------------------------------------------------------
# Single-band m8 forced cross-band fill -> partial catalog
# (..._resbgsub_m8_partial_<FILT>.fits).
#
# The monolithic m8 fill sweeps EVERY frame in one job and can overrun the wall
# (on sickle: F187N+F210M = 192 frames ate an 18h budget, F470N/F480M never ran).
# forced_fill_band writes only this band's columns, so the bands fill
# independently; scripts/reduction/m8_merge_partials.py column-merges the
# partials into the final ..._resbgsub_m8.fits (+ dedup).
#
# Reads the m7 merged catalog + per-frame m7 products already on disk
# (--manual-start-phase=m8 reconstructs the m6/m7 context).
#
# Args:  $1 = FILTER (one band, e.g. F335M)   $2 = each-suffix for that band
# Usually launched by submit_cataloging_m8.sh, which knows the per-filter suffix.
# ---------------------------------------------------------------------------
#SBATCH --job-name=catalog_m8p
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
# Low cpu ask backfills into small queue holes (queue delay = large-cpu-node
# scarcity, not memory); mem/time kept generous against an OOM/timeout kill.
#SBATCH --cpus-per-task=6
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%j.out

source ~/.bash_profile 2>/dev/null
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"; export TMP="$TMPDIR" TEMP="$TMPDIR"; mkdir -p "$TMPDIR"
export CRDS_PATH=${CRDS_PATH:-/orange/adamginsburg/jwst/crds}
# Run guard (jwst_gc_pipeline.versioning.tags.assert_runnable_version): the
# imaging/cataloging entry points HARD-BLOCK an untagged or dirty tree.  The live
# reduction runs from the working tree (a development run), so permit it by
# default -- it stamps a dev tag, so its products can never masquerade as a
# release.  A RELEASE run should check out a release tag: a tagged, clean tree
# passes as production regardless of this flag.  Set GC_ALLOW_DEV=0 to force the
# block (e.g. to guarantee a run is on a tagged checkout).
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

FILT=${1:?need a single FILTER as arg 1}
SUFFIX=${2:?need each-suffix as arg 2}
PROPOSAL=${PROPOSAL:-3958}
FIELD=${FIELD:-007}
TARGET=${TARGET:-sickle}
MODULES=${MODULES:-nrcb}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-10}
GROUP_MIN_SEP=${GROUP_MIN_SEP:-3.0}
EXTRA_ARGS=${EXTRA_ARGS:-}

PIPE_ROOT=${PIPE_ROOT:-}
# Locate the helper.  sbatch COPIES the batch script to a spool dir, so
# ${BASH_SOURCE[0]} at runtime need not be the repo copy -- hence the ordered
# fallback.  If PIPE_ROOT is set and the helper cannot be found we STOP: a
# missing pin is the failure this exists to prevent, so it must not be skipped.
_PR_H=""
for _d in "${GC_SCRIPTS_DIR:-}" "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" \
          "${PIPE_ROOT:-}/scripts/reduction" "${SLURM_SUBMIT_DIR:-}/scripts/reduction"; do
    [ -n "$_d" ] && [ -r "$_d/_pipe_root.sh" ] && { _PR_H="$_d/_pipe_root.sh"; break; }
done
if [ -n "$_PR_H" ]; then . "$_PR_H"
elif [ -n "${PIPE_ROOT:-}" ]; then
    echo "[pipe-root] PIPE_ROOT=$PIPE_ROOT set but _pipe_root.sh not found -- refusing" >&2
    exit 2
fi
PY=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}

# Rename ONLY when submitted bare, i.e. the name is still the #SBATCH
# placeholder.  submit_cataloging_m8.sh passes a proper submit-time
# --job-name; overwriting it here would DEGRADE it (the runtime form
# carries no program and no obsid) and would only fire once the job
# starts, long after the pending queue was read.
if [ "${SLURM_JOB_NAME:-catalog_m8p}" = "catalog_m8p" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update JobId="$SLURM_JOB_ID" \
        JobName="${TARGET}${PROPOSAL}-o${FIELD}-m8-${FILT}" 2>/dev/null || true
fi

echo "NODE=$(hostname) nproc=$(nproc) FILT=$FILT SUFFIX=$SUFFIX start=$(date -u +%FT%TZ)"
"$PY" -m jwst_gc_pipeline.photometry.crowdsource_catalogs_long \
    --filternames="$FILT" --modules="$MODULES" \
    --proposal_id="$PROPOSAL" --field="$FIELD" --target="$TARGET" \
    --each-suffix="$SUFFIX" --each-exposure --daophot --skip-crowdsource \
    --manual-iterations --manual-start-phase=m8 --manual-m8-partial \
    --group --max-group-size="$MAX_GROUP_SIZE" \
    --manual-group-min-sep-fwhm="$GROUP_MIN_SEP" \
    --parallel-workers="${SLURM_CPUS_PER_TASK:-6}" \
    $EXTRA_ARGS
echo "CATALOG_M8_PARTIAL_${FILT}_DONE rc=$? end=$(date -u +%FT%TZ)"
