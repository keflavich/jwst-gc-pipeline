#!/bin/bash
# ---------------------------------------------------------------------------
# Reduction + alignment SLURM submitter (refactored jwst_gc_pipeline).
#
# Runs PipelineRerunNIRCAM-LONG.py (Image3 imaging + per-exposure
# fix_alignment + post-resample realign_to_catalog; see
# jwst_gc_pipeline/reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md).
#
# Submitted as a SLURM ARRAY over filters: one array task per filter, each
# task processing all requested modules (nrca,nrcb,merged) for that filter.
# This parallelizes the per-filter loop across nodes.
#
# IMPORTANT (refactor compatibility):
#   * The entry point is a SCRIPT FILE, not an importable module. Its name
#     contains a hyphen (PipelineRerunNIRCAM-LONG.py), so `python -m ...`
#     CANNOT be used -- it must be run by full path.
#   * The script does `from jwst_gc_pipeline.reduction...`, so the
#     jwst_gc_pipeline package must be importable. It is pip-installed
#     (editable), so a bare `python /path/to/PipelineRerunNIRCAM-LONG.py`
#     resolves the package automatically -- no PYTHONPATH needed when running
#     the installed checkout. To run a worktree/branch instead, set
#     PYTHONPATH to that checkout root (see PIPE_ROOT below).
#   * The OLD _bench/cloudef_o002_rereduce.sbatch points at
#     brick-jwst-2221/brick2221/... -- that is the pre-refactor package and
#     is STALE. Use this script instead.
#
# Usage:
#   sbatch --array=0-7 scripts/reduction/submit_reduction.sbatch
#   # or override target via --export:
#   sbatch --array=0-3 --export=ALL,PROPOSAL=2221,FIELD=001,FILTERS="F405N F410M F466N F212N" \
#          scripts/reduction/submit_reduction.sbatch
#
# Defaults below target Sgr C (proposal 4147, field 012), all 8 filters,
# reusing existing *_cal.fits (skip_step1and2 / -s).
# ---------------------------------------------------------------------------
# renamed to "red_<prop>_<field>_<filt>" at runtime (target is not always the GC)
# JOB NAMING (STANDING RULE -- do not submit with the generic name):
#   ALWAYS pass a descriptive name at submit time:
#     sbatch --job-name=<target><program>-o<obsid>-reduce[-FILTER] ...
#     e.g. --job-name=brick2221-o001-reduce-F182M  /  cloudc2221-o002-reduce
#   Multiple reduce/catalog jobs are almost always in flight; a pending job
#   shows ONLY this static name (the runtime rename below fires at START).
#   Include the obsid whenever the program has more than one observation.
# RESOURCES: sized from the measured distribution, not a round number, and NOT
# from the median alone -- the tails are what kill a campaign.
#
# 644 reduce batch steps since 2026-07-01 (sacct MaxRSS, joined to the parent's
# job name):
#
#     mean  7.5 GB          > 16 GB : 64  (9.9%)
#     worst 57.0 GB         > 32 GB :  9  (1.4%)   all sgrb2 + brick SW
#                           > 48 GB :  3  (0.5%)   sgrb25365-o001, 57.0 GB
#                           > 64 GB :  0
#     longest 10:11 (sgrc4147-o012-reduce-niriss-F200W, only 1.4 GB)
#     > 6 h : 2             > 12 h : 0
#
# The previous 128gb/24h was ~17x the mean and blocks backfill: a 128 GB ask only
# fits on a node with 128 GB free, so four reduce arrays sat in Priority behind a
# QOS carrying 5196 running CPUs and 76 pending jobs.
#
# 64gb/12h keeps almost all of that benefit -- 64 GB fits far more nodes than 128,
# which is the actual mechanism -- while clearing BOTH observed worst cases with
# headroom. A first pass at 32gb/6h would have OOM-killed 9 of 644 and timed out
# 2, and an OOM at 57 GB does not fail early: it fails ~5 hours in, after the
# reduce has done most of its work. Trading queue time for a dead job plus a
# re-queue at the back of the same queue is the wrong direction.
#
# A per-field exception does NOT need this file edited. `#SBATCH` directives
# cannot read a shell variable, but sbatch's own environment can: setting
# SBATCH_MEM / SBATCH_TIMELIMIT before submitting overrides the directives here,
# and the wrapper honours REDUCE_MEM / REDUCE_TIME for that purpose --
#     REDUCE_MEM=96gb REDUCE_TIME=24:00:00 sbatch ... submit_reduction.sbatch
# Use it for sgrb2 (57 GB observed) if its footprint grows, rather than raising
# the default for all 644 jobs to cover 3 of them.
#
# cpus-per-task stays at 16 deliberately -- the default `adamginsburg` QOS caps
# cpu=10 and would leave a 16-cpu task permanently QOSGrpCpuLimit, which is why
# --qos is pinned below.
#SBATCH --job-name=reduce
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/reduce_%x_%A_%a.out

# No core dumps.  A python SIGBUS in a 60+ GB reduction writes a core file
# the size of the process (43 GiB measured, 2026-08-06) into the working
# directory, and /orange runs at 97% full.  HiPerGator sets
# PropagateResourceLimits=NONE, so a `ulimit -c 0` in the submitting shell
# does NOT reach the compute node -- it has to be set here, inside the batch
# script itself (jwst-gc-pipeline#428).
ulimit -c 0

# Per-field resource override (see the RESOURCES note above).  Exported here so a
# caller that sets REDUCE_MEM/REDUCE_TIME gets them applied by sbatch itself on
# any RESUBMIT this script performs; for the initial submission the caller must
# export SBATCH_MEM/SBATCH_TIMELIMIT or pass --mem/--time.
export SBATCH_MEM="${REDUCE_MEM:-${SBATCH_MEM:-}}"
export SBATCH_TIMELIMIT="${REDUCE_TIME:-${SBATCH_TIMELIMIT:-}}"
[ -z "$SBATCH_MEM" ] && unset SBATCH_MEM
[ -z "$SBATCH_TIMELIMIT" ] && unset SBATCH_TIMELIMIT

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

# Keep all scratch (jwst/astropy/tempfile) on the node-local SLURM scratch, NOT
# /tmp -- the pipeline's intermediates otherwise fill the node root fs (/dev/md0)
# and wedge the node.  SLURM_TMPDIR is per-job node-local; fall back to a large
# shared path (never /tmp).
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# --- parameters (override via --export=ALL,VAR=...) ---
PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
MODULES=${MODULES:-nrca,nrcb,merged}
# Reuse existing *_cal.fits by default (-s / skip_step1and2). Set SKIP=0 to
# re-download + re-fit ramps from MAST (much slower; overwrites all products).
# SKIP=0 re-fits every exposure whatever is on disk; each exposure is processed
# once per run, shared across the module passes (#417).  Set STAGE12_RESUME=1 to
# also skip exposures whose _cal/_ramp are newer than their _uncal AND carry
# this run's CRDS_CTX/CAL_VER, which resumes a killed SKIP=0 job (leave it
# unset for a forcing re-reduce).  The stamp comparison is #433: after a CRDS
# repin or a jwst bump a resume re-fits the whole filter directory rather than
# leaving one Image3 input set straddling two calibrations.
SKIP=${SKIP:-1}
# Declared (and exported) here so the value the log reports below is exactly the
# one `stage12_selection.stage12_resume_enabled` reads out of the environment.
# 0 and unset mean the same thing to it, so this changes no behaviour; what it
# adds is that a job log answers "was this reduction forced, or resumed off
# products already on disk?".
export STAGE12_RESUME=${STAGE12_RESUME:-0}
# Full Sgr C filter set. Override with --export=ALL,FILTERS="F405N F470N".
FILTERS=${FILTERS:-"F115W F162M F182M F212N F360M F405N F470N F480M"}

# Pin to a specific checkout (worktree/branch) by setting PIPE_ROOT; otherwise
# use the pip-installed package location.
PIPE_ROOT=${PIPE_ROOT:-}
if [ -n "$PIPE_ROOT" ]; then
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
    PIPE="$PIPE_ROOT/jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py"
else
    PIPE="$(python -c 'import os,jwst_gc_pipeline as p; print(os.path.join(os.path.dirname(p.__file__),"reduction","PipelineRerunNIRCAM-LONG.py"))')"
fi

# Pick this array task's filter.
read -r -a FILT_ARR <<< "$FILTERS"
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILT=${FILT_ARR[$IDX]}
if [ -z "$FILT" ]; then
    echo "No filter for array index $IDX (FILTERS='$FILTERS'). Submit --array=0-$(( ${#FILT_ARR[@]} - 1 ))."
    exit 1
fi

SKIP_FLAG=""
[ "$SKIP" = "1" ] && SKIP_FLAG="-s"

# Optional Image3 skymatch (level-match per-exposure background pedestals ->
# removes mosaic seams/stripes; sickle F470N).  Set SKYMATCH=match (or global/
# local).  Empty (default) preserves historical NIRCam behavior (skymatch off).
SKYMATCH=${SKYMATCH:-}
SKYMATCH_FLAG=""
[ -n "$SKYMATCH" ] && SKYMATCH_FLAG="--skymatch-method=$SKYMATCH"

# Descriptive per-task job name (the #SBATCH --job-name is necessarily generic
# because the filter is only known after indexing the array here).  Rename this
# task so squeue shows what it actually is, e.g. "red_4147_012_F470N".
# Fallback rename (fires at START; a submit-time --job-name is preserved).
if [ "${SLURM_JOB_NAME:-reduce}" = "reduce" ]; then
    scontrol update JobId="${SLURM_JOB_ID:-0}" Name="jw${PROPOSAL}-o${FIELD}-reduce-${FILT}" 2>/dev/null || true
fi

PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
echo "REDUCE start: proposal=$PROPOSAL field=$FIELD filter=$FILT modules=$MODULES skip=$SKIP resume=$STAGE12_RESUME skymatch=${SKYMATCH:-off}"
echo "PIPE=$PIPE"
"$PYTHON" "$PIPE" -p "$PROPOSAL" -d "$FIELD" -f "$FILT" -m "$MODULES" $SKIP_FLAG $SKYMATCH_FLAG
rc=$?
echo "REDUCE done: filter=$FILT rc=$rc"
exit $rc
