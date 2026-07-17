#!/bin/bash
# ---------------------------------------------------------------------------
# One manual-pipeline PHASE as either a per-frame fan-out array task or the
# per-phase finalize barrier.  Driven by submit_cataloging_perframe.sh; not
# usually launched by hand.
#
#   MODE=fanout   : array task; fits frame-shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}
#                   of PHASE, writes per-frame completion markers, STOPS before the
#                   barrier  (--manual-skip-finalize).  Tiny cpu ask -> backfills.
#   MODE=finalize : single job; reconstructs prior-phase state from disk, verifies
#                   every frame's marker, then runs the barrier (reconcile/merge/
#                   vet/residual/bg, + cross-band on the final phase)
#                   (--manual-finalize-only).
#
# Any NSHARDS gives full frame coverage: the shard predicate (index %% N == I)
# partitions each filter's exposure list completely, so NSHARDS only sets
# granularity (more shards = smaller/faster tasks).  A missing shard => the
# finalize hard-crashes on the absent marker (no silent exposure drop).
# ---------------------------------------------------------------------------
# renamed to "<target>-pf-<phase>" at runtime (target is not always the GC)
#SBATCH --job-name=pf
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%A_%a.out

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

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}
CROSSBAND_REF=${CROSSBAND_REF:-}
PHASE=${PHASE:?set PHASE=m12|m3|m4|m5|m6|m7}
MODE=${MODE:?set MODE=fanout|finalize}
NSHARDS=${NSHARDS:-16}
[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}-pf-${PHASE}-${MODE}" 2>/dev/null || true
EXTRA_ARGS=${EXTRA_ARGS:-}

# Pin a worktree/branch (this feature is new -> usually needed until merged).
PIPE_ROOT=${PIPE_ROOT:-}
[ -n "$PIPE_ROOT" ] && export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"

# crowdsource_catalogs_long wants a comma-separated filter list.
FILT_CSV=$(echo "$FILTERS" | tr ' ' ',')
PARALLEL_WORKERS=${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
REF_ARG=""
[ -n "$CROSSBAND_REF" ] && REF_ARG="--manual-crossband-ref-filter=$CROSSBAND_REF"

if [ "$MODE" = "fanout" ]; then
    SHARD="${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}"
    MODE_ARGS="--manual-skip-finalize --manual-frame-shard=$SHARD"
    scontrol update JobId="${SLURM_JOB_ID:-0}" \
        Name="pf_${TARGET}_${PHASE}_s${SLURM_ARRAY_TASK_ID:-0}" 2>/dev/null || true
elif [ "$MODE" = "finalize" ]; then
    MODE_ARGS="--manual-finalize-only"
    scontrol update JobId="${SLURM_JOB_ID:-0}" \
        Name="pf_${TARGET}_${PHASE}_fin" 2>/dev/null || true
else
    echo "MODE must be 'fanout' or 'finalize' (got '$MODE')"; exit 1
fi

# The manual pipeline refuses to run with SLURM_ARRAY_TASK_ID set (it must not be
# phase-split by the array).  We already consumed the array index into the shard
# string above, so unset it: each task runs one PHASE in-process for its shard.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID

echo "PERFRAME $MODE: $TARGET $PROPOSAL/$FIELD phase=$PHASE filters=$FILT_CSV modules=$MODULES ${MODE_ARGS}"
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
    --manual-start-phase="$PHASE" \
    --manual-stop-after-phase="$PHASE" \
    $MODE_ARGS \
    $REF_ARG \
    $EXTRA_ARGS
rc=$?
echo "PERFRAME $MODE done: phase=$PHASE rc=$rc"
exit $rc
