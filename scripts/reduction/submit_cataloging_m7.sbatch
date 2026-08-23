#!/bin/bash
# ---------------------------------------------------------------------------
# Cross-band (m7) finalize job for the manual photometry pipeline.
#
# Runs the cross-band merge phase ONLY, over ALL filters at once, reusing the
# per-filter m12..m6 products already on disk (built by the per-filter jobs in
# submit_cataloging.sbatch / submit_cataloging_chain.sh).  Uses the existing
# --manual-start-phase m7, which reconstructs the m6 background map from disk.
#
# Typically submitted by submit_cataloging_chain.sh with an afterok dependency
# on the per-filter array, but can be run standalone once m6 is complete.
#
# FIDELITY: intended to match a monolithic run.  The out-of-FOV satstar flux
# pins from m12's reconcile are persisted to disk (*_satstar_reconciled_m12.fits)
# and reconstructed by --manual-start-phase, so this finalize re-applies them
# like the monolith does.  Off-FOV satstars additionally use the Spitzer-prior
# (_spitzer.reg) + model<=data clamp, loaded fresh every phase in both paths --
# believed to make the m12 satstar_overrides pins vestigial for off-FOV stars
# (unverified, but moot given the persisted round-trip).  See cataloging.py
# run_manual_pipeline start_phase note.  The ZEROFRAME satstar deblend is plumbed
# via DEBLEND_SATSTARS below so a re-made satstar catalog at m7 matches m12.
# ---------------------------------------------------------------------------
# renamed to "<target><prop>-o<obsid>-m7" at runtime ONLY when submitted bare
#SBATCH --job-name=catalog_m7
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
# m7 cross-band finalize is I/O + table-stack bound, NOT cpu-parallel (it streams
# the merged columns one at a time).  Asking few cpus lets it backfill into small
# queue holes instead of waiting tens of hours for a large-cpu node.  mem/time are
# deliberately left generous -- shrinking THOSE risks an OOM/timeout kill mid-run
# and losing the whole finalize, whereas a small cpu ask only changes start time.
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/catalog_%x_%j.out

# No core dumps.  A python SIGBUS in a 60+ GB reduction writes a core file
# the size of the process (43 GiB measured, 2026-08-06) into the working
# directory, and /orange runs at 97% full.  HiPerGator sets
# PropagateResourceLimits=NONE, so a `ulimit -c 0` in the submitting shell
# does NOT reach the compute node -- it has to be set here, inside the batch
# script itself (jwst-gc-pipeline#428).
ulimit -c 0

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
# Scratch on node-local SLURM_TMPDIR, never /tmp (fills node root fs).
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

PROPOSAL=${PROPOSAL:-4147}
FIELD=${FIELD:-012}
TARGET=${TARGET:-sgrc}
# MODULES must be set together with the target env (brick 2221 = merged;
# a stray nrcb default produced per-module products the m7 merge cannot use).
if [ -n "${PROPOSAL:-}${FIELD:-}${TARGET:-}${EACH_SUFFIX:-}" ] && [ -z "${MODULES:-}" ]; then
    echo "FATAL: MODULES not set but other target env vars are -- set it (e.g. MODULES=merged for brick 2221)." >&2
    exit 64
fi
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o012_crf}
# ALL filters that participate in the cross-band merge (space-separated).
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
PARALLEL_WORKERS=${PARALLEL_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}
# Optional: pin the cross-band astrometric reference filter.
CROSSBAND_REF=${CROSSBAND_REF:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

# ZEROFRAME satstar deblend MUST match the per-filter (m12..m6) array stage: the
# m7 partial-start re-scans frames, and load_or_make_satstar_catalog re-makes a
# frame's satstar catalog when off-FOV pixels force overwrite -- a re-make WITHOUT
# this flag would drop the m12 deblend and diverge from a monolithic run.  Mirror
# submit_cataloging.sbatch so the finalize is functionally identical to the
# monolith.  Auto-degrades to legacy where a frame has no sibling _ramp.fits.
DEBLEND_SATSTARS=${DEBLEND_SATSTARS:-}
DEBLEND_ARG=""
[ -n "$DEBLEND_SATSTARS" ] && DEBLEND_ARG="--deblend-satstars"

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

# crowdsource_catalogs_long takes a comma-separated filter list.
FILT_CSV=$(echo "$FILTERS" | tr ' ' ',')
REF_ARG=""
[ -n "$CROSSBAND_REF" ] && REF_ARG="--manual-crossband-ref-filter=$CROSSBAND_REF"

# Rename ONLY when this was submitted bare (no --job-name), i.e. the name is
# still the #SBATCH placeholder.  Same idiom as
# submit_cataloging_perframe_phase.sbatch:93 (#326) and
# submit_reduction_miri.sbatch:95.  This script used to rename itself TWICE,
# unconditionally, over whatever it was given -- submit_cataloging_chain.sh
# passes a proper submit-time name and the second write (`cat_<target>_m7`)
# won.
#
# The bare-path fallback is the FULL shape, not `cat_<target>_m7`.  That was an
# open question on #330 -- whether monitoring/jobs.py reads the underscore form
# better than the dashed one -- and the answer is that it reads it not at all:
#
#     parse_job_name('cat_brick_m7')      -> None          (unattributed)
#     parse_job_name('brick-catalog-m7')  -> stage 'catalog-m7', no obsid
#     parse_job_name('brick2221-o001-m7') -> target brick, proposal 2221,
#                                            obsid 001, stage m7
#
# _NAME_PF only accepts a `pf_` head, so the underscore form falls through every
# shape and _resolve_head cannot find a registered field inside it.  The full
# shape is what submit_cataloging_perframe.sh:88 and run_field_retie_loop.sh:161
# already use.
_m7_rename_wanted() { [ "${SLURM_JOB_NAME:-catalog_m7}" = "catalog_m7" ] && [ -n "${SLURM_JOB_ID:-}" ]; }
_m7_rename_wanted && scontrol update JobId="$SLURM_JOB_ID" \
    JobName="${TARGET}${PROPOSAL}-o${FIELD}-m7" 2>/dev/null || true

echo "CATALOG m7 start: $TARGET $PROPOSAL/$FIELD filters=$FILT_CSV modules=$MODULES suffix=$EACH_SUFFIX overrides=${EACH_SUFFIX_OVERRIDES:-none}"
PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
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
    --manual-start-phase=m7 \
    $DEBLEND_ARG \
    $REF_ARG \
    $EXTRA_ARGS
rc=$?
echo "CATALOG m7 done: rc=$rc"
exit $rc
