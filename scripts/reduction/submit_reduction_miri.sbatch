#!/bin/bash
# ---------------------------------------------------------------------------
# MIRI reduction SLURM submitter (jwst_gc_pipeline).
#
# Runs reduction/PipelineMIRI.py: Detector1 + Image2 from uncal (or reuse the
# *_cal.fits already on disk with -s), then per-exposure fix_alignment and
# Image3.  MIRI is a single imager, so there is no module split and no
# -m/MODULES argument.
#
# Array over filters: one task per filter.
#   sbatch --array=0-0 --job-name=brick2221-o002-reduce-F2550W \
#          scripts/reduction/submit_reduction_miri.sbatch
#   # pick the observation and filters explicitly:
#   sbatch --array=0-1 --export=ALL,PROPOSAL=2221,FIELD=002,FILTERS="F1130W F2550W" \
#          --job-name=brick2221-o002-reduce-miri \
#          scripts/reduction/submit_reduction_miri.sbatch
#
# `python -m jwst_gc_pipeline.run_pipeline --proposal 2221 --obsid 002
#  --instrument miri` submits this for you, with the resources and the log
# directory from config.yaml.
# ---------------------------------------------------------------------------
# RESOURCES: sized from the measured distribution, with headroom over the worst
# observed job rather than over the median.
#
# 68 MIRI reduce array tasks submitted through THIS script (2026-09-07..10;
# every one a plain `sbatch ... submit_reduction_miri.sbatch` with no --mem, so
# the directive below is what they actually asked for), sacct MaxRSS joined to
# the parent job:
#
#     median 0.93 GB    > 4 GB : 3 (4.4%)
#     mean   1.26 GB    > 6 GB : 0
#     p90    3.37 GB    > 8 GB : 0
#     worst  4.33 GB (cloudef2092-o004 F2100W, 41673827_1)
#     longest 02:22:07 (cloudef2092-o004 F770W) -- 9.9% of the 24 h limit
#
# 22 hand-launched PipelineMIRI jobs from an older checkout (2026-06) reach a
# little higher: worst 5.86 GB (cloudc-2526-G0-F770W, 35443480).  Nothing that
# has ever run PipelineMIRI has exceeded 6 GB.
#
# 128gb was ~100x the median and ~22x the worst.  What that costs here is NOT
# node placement -- every hpg-default node has RealMemory=1004 GB and all 233
# report >=128 GB free, so a 128 GB ask has never restricted which node a MIRI
# reduce can land on.  It costs QOS headroom: astronomy-dept-b carries
# GrpTRES=cpu=6336,mem=49500G with DenyOnLimit, so the ask is charged against a
# shared 49500 GB pool.  At 128gb/16cpu a job sits at 8.0 GB per cpu against a
# pool ratio of 7.81 GB per cpu (49500/6336), so memory becomes co-binding with
# cpu the moment the QOS saturates; at 32gb/16cpu it is 2.0 GB per cpu and
# stops being a limit at all.  10678 puts a MIRI F770W parallel on every one of
# its 139 tiles, which is when that pool gets tested.
#
# Measured today, the QOS is nowhere near saturation (26 running jobs, 76 cpu,
# 1712 GB; the queue is dependency-bound, 162 of 197 pending on Dependency), so
# this buys nothing at this instant.  It is headroom for the fan-out, and an
# ask the measurements support.
#
# 32gb is 5.5x the worst MIRI reduce ever observed and 7.4x the worst through
# this script; the 4.33 GB job would have used 13.5% of it, the 5.86 GB one
# 18.3%.  The largest MIRI ramp on disk (198 MB, NGROUPS=5 NINTS=15 FULL,
# cloudef F770W) is the one that produced that 4.33 GB peak, so reaching 32 GB
# needs an exposure several times larger than anything the survey has seen.
# No job that has ever run PipelineMIRI has been OUT_OF_MEMORY.
#
# The walltime is NOT changed here.  It is loose by the same measure (24 h
# against a 2:22 worst case), but an over-long limit costs only backfill
# priority, and the argument for shortening it is a separate one.
#
# A field that needs more does NOT need this file edited: `#SBATCH` cannot read
# a shell variable, but sbatch's own environment can --
#     SBATCH_MEM=64gb sbatch ... submit_reduction_miri.sbatch
# or pass --mem= on the command line.  Either overrides the directive below.
# ---------------------------------------------------------------------------
#SBATCH --job-name=reduce-miri
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/reduce_%x_%A_%a.out

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
# The run guard hard-blocks an untagged or dirty tree.  The live reduction runs
# from the working tree, so permit it by default; it stamps a dev tag, so its
# products cannot masquerade as a release.  Set GC_ALLOW_DEV=0 to force a
# release-tag checkout.
export GC_ALLOW_DEV="${GC_ALLOW_DEV:-1}"

export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# --- parameters (override via --export=ALL,VAR=...) ---
PROPOSAL=${PROPOSAL:-2221}
FIELD=${FIELD:-002}
TARGET=${TARGET:-brick}
# Reuse existing *_cal.fits (-s) if SKIP=1; SKIP=0 re-fits ramps from uncal.
# 0 like the NIRISS submitter: a first reduction of an observation has no _cal
# files to reuse, and SKIP=1 would silently skip Detector1 and Image2.
SKIP=${SKIP:-0}
FILTERS=${FILTERS:-"F2550W"}

PIPE_ROOT=${PIPE_ROOT:-}
if [ -n "$PIPE_ROOT" ]; then
    # ${PYTHONPATH:+:...} so an unset PYTHONPATH leaves no trailing
    # colon, which would put the cwd on sys.path.
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
    PIPE="$PIPE_ROOT/jwst_gc_pipeline/reduction/PipelineMIRI.py"
else
    PIPE="$(python -c 'import os,jwst_gc_pipeline as p; print(os.path.join(os.path.dirname(p.__file__),"reduction","PipelineMIRI.py"))')"
fi

read -r -a FILT_ARR <<< "$FILTERS"
IDX=${SLURM_ARRAY_TASK_ID:-0}
FILT=${FILT_ARR[$IDX]}
if [ -z "$FILT" ]; then
    echo "No filter for array index $IDX (FILTERS='$FILTERS'). Submit --array=0-$(( ${#FILT_ARR[@]} - 1 ))."
    exit 1
fi

SKIP_FLAG=""
[ "$SKIP" = "1" ] && SKIP_FLAG="-s"

# The name a pending job shows is the one given at submit time; this rename only
# fires once the job STARTS.  Pass --job-name=<target><program>-o<obsid>-reduce.
if [ "${SLURM_JOB_NAME:-reduce-miri}" = "reduce-miri" ]; then
    scontrol update JobId="${SLURM_JOB_ID:-0}" \
        Name="${TARGET}${PROPOSAL}-o${FIELD}-reduce-miri-${FILT}" 2>/dev/null || true
fi

PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
echo "REDUCE(MIRI) start: proposal=$PROPOSAL field=$FIELD filter=$FILT skip=$SKIP"
echo "PIPE=$PIPE"
"$PYTHON" "$PIPE" -p "$PROPOSAL" -d "$FIELD" -f "$FILT" $SKIP_FLAG
rc=$?
echo "REDUCE(MIRI) done: filter=$FILT rc=$rc"
exit $rc
