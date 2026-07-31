#!/bin/bash
# ---------------------------------------------------------------------------
# Stage 3: merge the per-exposure catalogs (jwst_gc_pipeline.photometry.merge_catalogs).
#
# Two ways to run it, and the SLURM form you use depends on which:
#
#   * Per-(program, filter) merges -- submit as an ARRAY, one task per job.
#     Ask the module for the index -> (program, filter) map first, so you know
#     the array bounds:
#         python -m jwst_gc_pipeline.photometry.merge_catalogs \
#                --target=brick --merge-singlefields --list-jobs
#     Then, for an 11-entry map:
#         sbatch --array=0-10 --export=ALL,TARGET=brick,MERGE_SINGLEFIELDS=1 \
#                --job-name=brick2221-o001-merge scripts/reduction/submit_merge.sbatch
#
#   * All-filter merges only -- one task, no --array.
#         sbatch --export=ALL,TARGET=brick --job-name=brick2221-o001-merge \
#                scripts/reduction/submit_merge.sbatch
#
# Without --array, MERGE_SINGLEFIELDS=1 still runs every per-(program, filter)
# job, serially, in the one task.
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

[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}-merge${SLURM_ARRAY_TASK_ID:+-$SLURM_ARRAY_TASK_ID}" 2>/dev/null || true

echo "MERGE start $(date -u +%FT%TZ) NODE=$(hostname) target=$TARGET task=${SLURM_ARRAY_TASK_ID:-none}"
"$PY" -m jwst_gc_pipeline.photometry.merge_catalogs \
      --target="$TARGET" $SINGLE_ARG --merge-workers="$MERGE_WORKERS" $MERGE_EXTRA
rc=$?
echo "MERGE_DONE rc=$rc $(date -u +%FT%TZ)"
exit $rc
