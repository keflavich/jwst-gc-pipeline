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

source ~/.bash_profile 2>/dev/null
export CRDS_PATH=${CRDS_PATH:-/orange/adamginsburg/jwst/crds}
export CRDS_SERVER_URL=${CRDS_SERVER_URL:-https://jwst-crds.stsci.edu}
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
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"
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
