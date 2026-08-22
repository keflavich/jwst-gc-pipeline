#!/bin/bash
# Refuse a submission that duplicates a PHASE already queued for this field.
# Source this, do not execute it:
#
#     PHASES="m12 m3" . "$(...)/_refuse_duplicate_chain.sh"
#
# Reads TARGET, PROPOSAL, FIELD and PHASES from the environment.  Exits 3 on a
# duplicate unless GC_ALLOW_DUPLICATE_CHAIN=1.
#
# Why
# ---
# Two full chains for one field were queued for sgrc (2026-08-21 20:35 and
# 2026-08-22 01:34), neither with a dependency, so both would have started on
# priority and run 12 jobs each over the SAME per-frame products and the same
# merged catalogs -- two writers, one output tree.  Nothing noticed: the
# collision guards in `cataloging` protect one run's frames from each other, not
# one field from two concurrent runs, and SLURM has no opinion.
#
# Per PHASE, not per field
# ------------------------
# Adding phases to a field that already has a chain is the normal way to extend
# one -- cloudef's m4-m7 went on behind its queued m3, and sgrb2's m3-m7 behind
# eleven per-filter m12 finalizes.  A per-field check would refuse both.  Only
# re-submitting a phase that is ALREADY queued is the mistake.
#
# Shared because the exposure was never unique to one driver: the per-frame
# chain got this guard first (#483), while `submit_cataloging_m8.sh` fans 18
# jobs per field with no check at all -- and an m8 fan is the more likely source
# of the next duplicate, since it is the driver used to re-arm a field after a
# failed m7.  The uniform job-name convention (#481) is what makes one
# implementation portable across drivers.
#
# GC_ALLOW_DUPLICATE_CHAIN=1 overrides, for a queued chain known to be dead and
# about to be cancelled.  Exactly "1" -- '0', '', 'no', 'false' do not open it,
# because an escape hatch that any truthy-looking value opens becomes the
# default.

_JOB_PREFIX="${TARGET}${PROPOSAL}-o${FIELD}-"
_QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
_dupes=""
for _ph in ${PHASES:-}; do
    # The trailing '-' is what keeps 'm1' from matching 'm12-fanout'.
    if printf '%s\n' "$_QUEUED" | grep -q "^${_JOB_PREFIX}${_ph}-"; then
        _dupes="$_dupes $_ph"
    fi
done
if [ -n "$_dupes" ]; then
    echo "REFUSING: ${TARGET} ${PROPOSAL}/${FIELD} already has queued jobs for phase(s):${_dupes}" >&2
    # List from the names already captured, and never let a no-match grep under
    # `set -e` kill the script before it reaches its own exit code -- that
    # turned the intended `exit 3` into a bare `exit 1`.
    printf '%s\n' "$_QUEUED" | grep "^${_JOB_PREFIX}" | sed 's/^/    /' >&2 || true
    if [ "${GC_ALLOW_DUPLICATE_CHAIN:-0}" != "1" ]; then
        echo "Two chains over one field write the same products concurrently." >&2
        echo "Cancel the queued chain first, or set GC_ALLOW_DUPLICATE_CHAIN=1." >&2
        exit 3
    fi
    echo "GC_ALLOW_DUPLICATE_CHAIN=1 -- submitting anyway." >&2
fi
