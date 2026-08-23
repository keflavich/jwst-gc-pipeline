#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 3: merge the per-exposure catalogs (jwst_gc_pipeline.photometry.merge_catalogs).
#
# The merge has two parts, and they must run in order:
#
#   1. one per-(program, filter) merge per filter -- parallelisable, an ARRAY
#   2. the all-filter merges, which read EVERY filter's part-1 output -- ONE job
#
# So it is two submissions.  Ask for the index -> (program, filter) map first;
# it runs nothing and tells you the array bounds:
#
#     python -m jwst_gc_pipeline.photometry.merge_catalogs \
#            --target=brick --merge-singlefields --list-jobs
#
# Then, for an 11-entry map:
#
#     jid=$(sbatch --parsable --array=0-10 \
#           --export=ALL,TARGET=brick,MERGE_SINGLEFIELDS=1 \
#           --job-name=brick2221-o001-merge scripts/reduction/submit_merge.sbatch)
#     sbatch --dependency=afterok:$jid --export=ALL,TARGET=brick \
#            --job-name=brick2221-o001-mergeall scripts/reduction/submit_merge.sbatch
#
# An array task that merged one filter stops before part 2, so submitting only
# the array leaves the all-filter catalogs unwritten.
#
# On one node instead: drop --array and set MERGE_SINGLEFIELDS=1.  That runs
# every per-filter job serially and then the all-filter merges, in one task.
# ---------------------------------------------------------------------------
#SBATCH --job-name=merge
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/merge_%x_%A_%a.out

# No core dumps.  A python SIGBUS in a 60+ GB reduction writes a core file
# the size of the process (43 GiB measured, 2026-08-06) into the working
# directory, and /orange runs at 97% full.  HiPerGator sets
# PropagateResourceLimits=NONE, so a `ulimit -c 0` in the submitting shell
# does NOT reach the compute node -- it has to be set here, inside the batch
# script itself (jwst-gc-pipeline#428).
ulimit -c 0

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=${CRDS_PATH:-/orange/adamginsburg/jwst/crds}
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
export STPSF_PATH=${STPSF_PATH:-/orange/adamginsburg/jwst/stpsf-data}
# The entry points refuse to run on a checkout that is not on a release tag.
# The live merge runs from the working tree, so permit it; the products are
# stamped with a dev tag and can never be mistaken for a release.
export GC_ALLOW_DEV="${GC_ALLOW_DEV:-1}"

export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

TARGET=${TARGET:-brick}
MERGE_SINGLEFIELDS=${MERGE_SINGLEFIELDS:-0}
# --merge-workers must not exceed --cpus-per-task above.
MERGE_WORKERS=${MERGE_WORKERS:-4}
MERGE_EXTRA=${MERGE_EXTRA:-}

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

SINGLE_ARG=""
[ "$MERGE_SINGLEFIELDS" = "1" ] && SINGLE_ARG="--merge-singlefields"

# Only rename when the job was submitted under the placeholder name: a name
# passed at submit time identifies target + program + obsid, which this cannot
# reconstruct, and overwriting it breaks the naming convention.
if [ "${SLURM_JOB_NAME:-merge}" = "merge" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol update JobId="$SLURM_JOB_ID" \
        JobName="${TARGET}-merge${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}" 2>/dev/null || true
fi

echo "MERGE start $(date -u +%FT%TZ) NODE=$(hostname) target=$TARGET task=${SLURM_ARRAY_TASK_ID:-none}"
"$PY" -m jwst_gc_pipeline.photometry.merge_catalogs \
      --target="$TARGET" $SINGLE_ARG --merge-workers="$MERGE_WORKERS" $MERGE_EXTRA
rc=$?
echo "MERGE_DONE rc=$rc $(date -u +%FT%TZ)"
exit $rc
