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
#SBATCH --job-name=reduce-miri
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/reduce_%x_%A_%a.out

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
