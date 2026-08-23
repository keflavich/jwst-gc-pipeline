#!/bin/bash
# ---------------------------------------------------------------------------
# One manual-pipeline PHASE as either a per-frame fan-out array task or the
# per-phase finalize barrier.  Driven by submit_cataloging_perframe.sh.
#
# Launching it by hand is supported (re-running a failed finalize without
# repeating its fan-out is the main reason to); the #SBATCH lines below carry a
# usable resource floor for that case.
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
# Resource FLOOR for a hand-launched job.  submit_cataloging_perframe.sh passes
# --cpus-per-task/--mem/--time on the command line, which override these, so a
# driver-submitted job is unchanged.  Without them a direct `sbatch` inherits
# the CLUSTER default -- about 10 minutes -- and dies mid-phase having done
# partial work: wd2's m7 finalize was re-run by hand to avoid repeating a 3-hour
# fan-out and hit TIMEOUT at 00:10:18 (job 39963352, 2026-08-22), taking its 18
# chained m8 jobs down with it.
#
# Re-running ONE phase by hand is a legitimate thing to do -- it is how you
# recover a finalize without redoing its fan-out -- so the bare path needs to be
# survivable rather than merely discouraged.  These match the driver's finalize
# slice (FINALIZE_CPUS/MEM/TIME).
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=12:00:00

source ~/.bash_profile 2>/dev/null
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

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}

# Per-filter suffix override (see submit_cataloging.sbatch), e.g.
# EACH_SUFFIX_OVERRIDES="F187N:destreak_o007_crf,F210M:destreak_o007_crf".
# Optional; a field whose destreak policy splits by wavelength has no single
# EACH_SUFFIX that serves every filter.
EACH_SUFFIX_OVERRIDES=${EACH_SUFFIX_OVERRIDES:-}
EACH_SUFFIX_OVERRIDES_ARG=""
[ -n "$EACH_SUFFIX_OVERRIDES" ] && \
    EACH_SUFFIX_OVERRIDES_ARG="--each-suffix-overrides=$EACH_SUFFIX_OVERRIDES"
CROSSBAND_REF=${CROSSBAND_REF:-}
PHASE=${PHASE:?set PHASE=m12|m3|m4|m5|m6|m7}
MODE=${MODE:?set MODE=fanout|finalize}
NSHARDS=${NSHARDS:-16}
# JOB NAMING.  submit_cataloging_perframe.sh already passes a submit-time
# --job-name carrying target+program+obsid+stage, which is the standing rule --
# and a pending job shows ONLY that name, which is exactly when the queue is
# being watched.  The runtime renames below used to fire unconditionally and
# DEGRADED it: "${TARGET}-pf-${PHASE}-${MODE}" drops the program and the obsid,
# so a queue with brick/cloudc (both 2221) or several gc2211 observations in
# flight could not be told apart.
#
# On an ARRAY they are worse than useless.  `scontrol update JobId=<task>` does
# not reliably address a single element, so the 16 shards race and some lose:
# sgrb2 m12 fanout 38867646 came out `pf_sgrb2_m12_s15` on all 16 tasks, and
# sgrc 38851171 had tasks 13 and 15 both reading `pf_sgrc_m12_s15` (2026-08-07).
# The shard is already unambiguous in the array-task id (`38867646_7`), so the
# name never needed to carry it.
#
# So: rename only when the job was submitted bare, i.e. the name is still the
# `pf` placeholder from the #SBATCH directive above.  A named submission keeps
# the name it was given.  (Same idiom as submit_reduction_miri.sbatch:95.)
#
# The bare-path fallback keeps the `pf_<target>_<phase>` shape rather than the
# dashed one, because that is what the monitor can read: monitoring/jobs.py's
# _NAME_PF gives `pf_sgrb2_m12` and `pf_sgrb2_m12_fin` both stage='m12', whereas
# a dashed `sgrb2-pf-m12-fanout` has no `-o<obsid>-` segment to bound the stage
# and falls through to _NAME_LOOSE as stage='pf-m12-fanout' -- not a stage the
# monitor buckets.  Only the shard suffix, which is the actual defect, is gone.
_pf_rename_wanted() { [ "${SLURM_JOB_NAME:-pf}" = "pf" ] && [ -n "${SLURM_JOB_ID:-}" ]; }
EXTRA_ARGS=${EXTRA_ARGS:-}

# Pin a worktree/branch (this feature is new -> usually needed until merged).
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

# crowdsource_catalogs_long wants a comma-separated filter list.
FILT_CSV=$(echo "$FILTERS" | tr ' ' ',')
PARALLEL_WORKERS=${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
REF_ARG=""
[ -n "$CROSSBAND_REF" ] && REF_ARG="--manual-crossband-ref-filter=$CROSSBAND_REF"

if [ "$MODE" = "fanout" ]; then
    SHARD="${SLURM_ARRAY_TASK_ID:-0}/${NSHARDS}"
    MODE_ARGS="--manual-skip-finalize --manual-frame-shard=$SHARD"
    _pf_rename_wanted && scontrol update JobId="$SLURM_JOB_ID" \
        JobName="pf_${TARGET}_${PHASE}" 2>/dev/null || true
elif [ "$MODE" = "finalize" ]; then
    MODE_ARGS="--manual-finalize-only"
    _pf_rename_wanted && scontrol update JobId="$SLURM_JOB_ID" \
        JobName="pf_${TARGET}_${PHASE}_fin" 2>/dev/null || true
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
        $EACH_SUFFIX_OVERRIDES_ARG \
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

# One retry is not enough when the whole campaign is resident: a re-tie wave
# starts 16 shards per field across six fields at once, they all import against
# the same conda env on the same NFS mount, and the retry lands back in the same
# storm.  Observed repeatedly (2026-07-31/08-01) -- e.g. sgrb2 shard 14 logged
# "matches import-race signature (elapsed=0s); retrying once" and the retry hit
# the identical ModuleNotFoundError.  Each loss costs a whole re-tie iteration,
# because one dead shard leaves the afterok finalize permanently unsatisfied.
#
# Retry up to IMPORT_RACE_RETRIES times with exponential backoff plus jitter, so
# successive attempts spread out instead of re-colliding.  The gate is unchanged:
# ONLY the import-race signature retries; a genuine science failure still fails
# fast on the first attempt.
IMPORT_RACE_RETRIES=${IMPORT_RACE_RETRIES:-3}

attempt=0
while : ; do
    start_s=$SECONDS
    run_pipeline
    rc=$?
    [ $rc -eq 0 ] && break
    elapsed=$(( SECONDS - start_s ))
    race=0
    if [ -s "$RUNLOG" ]; then
        tail -n 100 "$RUNLOG" | grep -qE "$IMPORT_RACE_RE" && race=1
    elif [ "$elapsed" -le 120 ]; then
        race=1
    fi
    if [ "$race" -ne 1 ]; then
        echo "PERFRAME $MODE: rc=$rc is not the import race (elapsed=${elapsed}s); NOT retrying."
        break
    fi
    attempt=$(( attempt + 1 ))
    if [ "$attempt" -gt "$IMPORT_RACE_RETRIES" ]; then
        echo "PERFRAME $MODE: rc=$rc still the import race after ${IMPORT_RACE_RETRIES} retries;"
        echo "  giving up.  This strands the afterok finalize -- rerun this shard and"
        echo "  re-point the finalize (scontrol update jobid=<fin> dependency=...)."
        break
    fi
    backoff=$(( (1 << (attempt - 1)) * 30 + (RANDOM % 30) ))
    echo "PERFRAME $MODE: rc=$rc matches import-race signature (elapsed=${elapsed}s);" \
         "retry ${attempt}/${IMPORT_RACE_RETRIES} in ${backoff}s."
    sleep "$backoff"
done
rm -f "$RUNLOG"
echo "PERFRAME $MODE done: phase=$PHASE rc=$rc"
exit $rc
