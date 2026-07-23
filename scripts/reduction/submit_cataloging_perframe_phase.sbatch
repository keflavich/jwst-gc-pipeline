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
    # Lustre metadata coherence lag: the finalize launches (afterok) the instant
    # the last fan-out shard exits, but the per-frame completion markers those
    # shards just wrote may not yet be visible on the finalize node's MDT view.
    # The marker check then HARD-CRASHES on a "missing" marker that in fact exists
    # (verified: flagged frames' markers present minutes later).  Settle first so
    # the directory listing is coherent before the strict all-markers verify.
    echo "finalize: sleeping 180s for Lustre marker-metadata coherence before verify."
    sleep 180
else
    echo "MODE must be 'fanout' or 'finalize' (got '$MODE')"; exit 1
fi

# The manual pipeline refuses to run with SLURM_ARRAY_TASK_ID set (it must not be
# phase-split by the array).  We already consumed the array index into the shard
# string above, so unset it: each task runs one PHASE in-process for its shard.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_JOB_ID

echo "PERFRAME $MODE: $TARGET $PROPOSAL/$FIELD phase=$PHASE filters=$FILT_CSV modules=$MODULES ${MODE_ARGS}"
PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}

# Destagger imports: a large fan-out array launches dozens of tasks at once, all
# scanning the shared conda env's package metadata simultaneously.  That races in
# importlib_metadata (observed: MetadataNotFound from photutils' optional-deps /
# tqdm import) and kills the odd task ~30 s in.  A short random sleep spreads the
# import storm; a one-shot retry rescues any task that still loses the race.  Both
# are pure environment hardening -- they do not touch the science.
sleep $(( (RANDOM % 25) + 1 ))

# Node-local copy of the run's output so the retry gate below can inspect the
# failure (also still streamed to the normal SLURM log via tee).
RUNLOG="${SLURM_TMPDIR:-/tmp}/pf_${TARGET}_${PHASE}_${MODE}_$$.log"

run_pipeline() {
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
        $EXTRA_ARGS 2>&1 | tee "$RUNLOG"
    return "${PIPESTATUS[0]}"
}

# Retry ONCE, and ONLY for the import race described above -- a genuine science
# failure (missing marker, bad data, code bug) must fail fast, not run twice.
# The race presents as a startup-import traceback (importlib_metadata /
# optional-deps discovery), so gate the retry on that signature in the log tail;
# if the log is unreadable, fall back to timing (the race kills a task during
# startup imports, ~30 s in -- never hours into the science).
IMPORT_RACE_RE='ModuleNotFoundError|ImportError|PackageNotFoundError|MetadataNotFound|bad magic number|marshal data too short'

start_s=$SECONDS
run_pipeline
rc=$?
if [ $rc -ne 0 ]; then
    elapsed=$(( SECONDS - start_s ))
    race=0
    if [ -s "$RUNLOG" ]; then
        tail -n 100 "$RUNLOG" | grep -qE "$IMPORT_RACE_RE" && race=1
    elif [ "$elapsed" -le 120 ]; then
        race=1
    fi
    if [ "$race" -eq 1 ]; then
        echo "PERFRAME $MODE: rc=$rc matches import-race signature (elapsed=${elapsed}s); retrying once."
        sleep $(( (RANDOM % 30) + 10 ))
        run_pipeline
        rc=$?
    else
        echo "PERFRAME $MODE: rc=$rc is not the import race (elapsed=${elapsed}s); NOT retrying."
    fi
fi
rm -f "$RUNLOG"
echo "PERFRAME $MODE done: phase=$PHASE rc=$rc"
exit $rc
