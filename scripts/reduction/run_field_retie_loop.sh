#!/bin/bash
# ---------------------------------------------------------------------------
# Automated reduce -> catalog-to-m2 -> re-tie loop (Fix C, 2026-07-16).
#
# Closes the astrometry checkpoint the hands-off way for tweakreg/fix_alignment
# fields (sgrc, cloudef, ... -- anything outside the brick VIRAC2locked path):
#
#   iter i:
#     1. REDUCE (Image3 + fix_alignment).  On iter>=2 fix_alignment applies the
#        consensus offsets table seeded by the previous iter's m2 checkpoint.
#     2. CATALOG TO m2 ONLY (PHASES="m12") with ASTROM_CHECKPOINT_APPLY=1.
#        - checkpoint PASSES (no exposure > 2 mas off consensus)  -> converged.
#        - checkpoint measures misalignment -> it SEEDS/updates the consensus
#          offsets table (seed_offsets_table_from_consensus / update_offsets_
#          table), stale-tags the im0 mosaics, and the finalize exits non-zero.
#     3. Converged?  break.  Else re-reduce (the applied table removes the
#        per-exposure jitter) and repeat, capped at MAXITER.
#   after convergence: run the FULL cataloging chain (m3..m7).
#
# The m2 checkpoint stays a hard gate throughout -- this loop does NOT demote it
# (never sets ASTROM_CHECKPOINT_WARN_ONLY); it makes the SANCTIONED remediation
# (seed table -> re-reduce -> re-catalog) automatic instead of manual.
#
# Usage:
#   PROPOSAL=4147 FIELD=012 TARGET=sgrc FILTERS="F115W F162M F182M F212N F360M F405N F470N F480M" \
#     PIPE_ROOT=/orange/adamginsburg/repos/jwst-gc-pipeline \
#     scripts/reduction/run_field_retie_loop.sh
#
# Optional:
#   RETIE_PROVENANCE_ONLY=1   print the checkout provenance and exit 0 without
#                    submitting anything.  This is the ONLY safe way to inspect
#                    the run: MAXITER=0 is not a no-op (see below).
#   RETIE_UPSTREAM   branch to report the checkout distance against
#                    (default origin/main).  Reporting only -- see warn_if_behind.
#   RETIE_ACCEPT_RESIDUAL_MAS
#                    ceiling under which a FIXED POINT is accepted rather than
#                    stopped on (default 0 = accept none, the behaviour before
#                    this existed).  A repeating residual below it is the
#                    SIAF/DVA-class systematic a per-exposure offsets table
#                    cannot express; the loop raises the m2 correction floor to
#                    just above the measured value and runs m3-m7 over it, with
#                    every residual still measured and recorded.  A repeating
#                    residual AT or above it is a correction that is not
#                    reaching the frame, and still stops the run.
#
# MAXITER must be >= 1.  Values below 1 skip the iteration loop and fall through
# to the FULL m3-m7 cataloging submission; that submitted twelve unintended jobs
# on 2026-08-09.  The script refuses them unless RETIE_PROVENANCE_ONLY=1.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROPOSAL=${PROPOSAL:?set PROPOSAL}
FIELD=${FIELD:?set FIELD}
TARGET=${TARGET:?set TARGET}
FILTERS=${FILTERS:?set FILTERS}
# `${FILTERS:?}` rejects "" and passes " ".  FILTERS is not only the reduce
# list -- it is also the coverage DECLARATION handed to `--expect-filters`, and
# a whitespace-only value declares nothing while looking set.
read -r -a _FILTERS_CHECK <<< "$FILTERS"
if [ "${#_FILTERS_CHECK[@]}" -eq 0 ]; then
    echo "FILTERS is set but holds no filter names (whitespace only)." >&2
    echo "It is both the reduce list and the acceptance coverage declaration;" >&2
    echo "an empty one disables the coverage check without saying so." >&2
    exit 1
fi
MODULES=${MODULES:-nrcb}
EACH_SUFFIX=${EACH_SUFFIX:-destreak_o${FIELD}_crf}
MAX_GROUP_SIZE=${MAX_GROUP_SIZE:-unlimited}
PIPE_ROOT=${PIPE_ROOT:-}
MAXITER=${MAXITER:-3}
# A fixed point needs DEFAULT_REPEATS (3) passes before it can be judged, so the
# check first has an opinion at iteration 3 -- which at MAXITER=3 is the LAST
# one.  Accepting there has nowhere to re-reduce, so the run would end with the
# offsets table ahead of the frames and the mosaics stale-tagged: precisely the
# state the acceptance path exists to avoid, reached by running out of
# iterations instead of by breaking.  So acceptance requires headroom.
if [ "${RETIE_ACCEPT_RESIDUAL_MAS:-0}" != "0" ] && [ "$MAXITER" -lt 4 ]; then
    echo "REFUSING: RETIE_ACCEPT_RESIDUAL_MAS is set but MAXITER=$MAXITER."
    echo "          A fixed point cannot be judged before iteration 3, and"
    echo "          accepting one needs a further pass to re-reduce under the"
    echo "          raised floor.  Re-run with MAXITER=4 or higher."
    exit 2
fi
# MAXITER < 1 skips the iteration loop entirely and falls straight through to the
# FULL m3-m7 cataloging submission at the bottom -- so `MAXITER=0`, the obvious
# way to ask "just show me what this would run", submits a dozen jobs instead.
# That happened on 2026-08-09: MAXITER=0 against sgrc 4147/012 submitted jobs
# 39044950 and 39044953-39044961, ten of which were RUNNING before they were
# cancelled.  There is no dry-run mode, so the only safe way to read the
# provenance banner is RETIE_PROVENANCE_ONLY=1, added below.
if ! [ "$MAXITER" -ge 1 ] 2>/dev/null && [ "${RETIE_PROVENANCE_ONLY:-0}" != 1 ]; then
    echo "REFUSING: MAXITER=$MAXITER -- must be an integer >= 1."
    echo "  Values below 1 do NOT make this a no-op: they skip the iteration"
    echo "  loop and fall through to the full m3-m7 cataloging submission."
    echo "  To inspect the checkout provenance without submitting anything, use"
    echo "  RETIE_PROVENANCE_ONLY=1 (which overrides this check, so it works"
    echo "  whatever MAXITER is set to)."
    exit 2
fi
QOS=${QOS:-astronomy-dept-b}
BASE=${BASE:-/orange/adamginsburg/jwst/${TARGET}}
# Actionability floor for the m2 checkpoint (see cataloging.py ~L3064): per-detector
# residuals of a few mas are SIAF/DVA-class systematics the module-locked/consensus
# offsets table cannot express, so correcting on their detector means never converges.
# Setting this ABOVE the residual scatter lets the loop stop on a sub-floor PASS while
# still measuring+recording every residual.
#
# Do NOT default it here.  The floor is PER FIELD now, derived from each field's
# own scatter and held in `jwst_gc_pipeline.photometry.m2_correction_floors`
# (sgrc/cloudc 8.0, w51 6.0, brick/sgrb2/sickle/cloudef/sgra 4.0).
# `m2_correction_floor` resolves the environment BEFORE that table, so any value
# exported here overrides every per-field entry.  A default of 0 is the worst of
# those: 0 is a VALUE, not "unset" -- it resolves to `(0.0, 'env')` and disables
# the floor outright, so every residual above the 2 mas consensus tolerance
# becomes actionable and m2 corrects a field's intrinsic per-exposure scatter
# (the brick-1182 failure: 35 corrections and an AstrometryCorrectionRequiredError
# on F115W's own 2-3.3 mas jitter).  That never fired only because the jicama
# runner's common.sh happened to export 4.0 first and shadow it.
#
# So leave it UNSET unless the caller set one, and let the per-field table answer.
# When this loop started, in the checkpoint records' own stamp format.  The
# fixed-point check counts only records written at/after it: brick, cloudc and
# cloudef all carry repeating histories from earlier campaigns, and without this
# the first re-run of any of them would stop at iteration 2 citing passes from
# a different campaign.
RETIE_RUN_START=$(date -u +%Y%m%dT%H%M%SZ)

# `floor_env` is how the value reaches every child.  It is EMPTY when the caller
# set nothing, so the child inherits no variable at all and the per-field table
# answers.  An array rather than a bare `VAR=$VAR` because `set -u` aborts the
# script on an unset expansion, and because a conditional string would have to be
# re-split at every use site.
floor_env=()
if [ -n "${ASTROM_M2_CORRECTION_FLOOR_MAS:-}" ]; then
    export ASTROM_M2_CORRECTION_FLOOR_MAS
    floor_env=(ASTROM_M2_CORRECTION_FLOOR_MAS="$ASTROM_M2_CORRECTION_FLOOR_MAS")
    echo "[floor] ASTROM_M2_CORRECTION_FLOOR_MAS=$ASTROM_M2_CORRECTION_FLOOR_MAS (explicit override)"
else
    echo "[floor] ASTROM_M2_CORRECTION_FLOOR_MAS unset -- using the per-field table"
fi

# What the field WOULD get with nothing exported.  Resolved once, here, where
# TARGET is in scope: the bounded-fixed-point branch below needs it as the "never
# lowered" baseline, and that branch is extracted and run standalone by
# test_retie_accept_bounded_branch.py with only a handful of variables defined --
# so it must not reach for TARGET, PIPE_ROOT or python itself.
PER_FIELD_FLOOR=$(PYTHONPATH="${PIPE_ROOT:-}:${PYTHONPATH:-}" python -c "
from jwst_gc_pipeline.photometry.m2_correction_floors import m2_correction_floor
print(m2_correction_floor('${TARGET:-}', env={})[0])" 2>/dev/null || echo 0)
# Which offsets table does m2 REWRITE for this field?  The before/after md5sum
# check below is only meaningful against that one file, and the answer depends on
# the field's CONFIGURED CHANNEL -- not on which tables happen to exist.
#
# Guessing has failed twice.  A hardcoded "_consensus.csv" missed every locked
# field (sgrc, cloudc, sgrb2, quintuplet, gc2211).  A locked-before-consensus
# preference order then missed cloudef obs002, whose bulk is a recorded constant
# but whose per-exposure jitter the checkpoint writes to "_consensus.csv" -- a
# stale "_VIRAC2locked.csv" sat beside it and won the preference.  Both produced
# the same silent symptom: watch a file nobody writes, see no change, stop with
# "this is NOT a checkpoint re-tie" while the correction had in fact been made.
#
# alignment_config.offsets_table_path is the single source of truth, and it is
# what cataloging's own _astrom_offsets_channel delegates to.  OFFSETS_TBL
# overrides it for a field that uses some third table.
CONSENSUS_TBL="${OFFSETS_TBL:-}"
if [ -z "$CONSENSUS_TBL" ]; then
    CONSENSUS_TBL=$(PYTHONPATH="${PIPE_ROOT:-}:${PYTHONPATH:-}" python -c "
from jwst_gc_pipeline.reduction.alignment_config import offsets_table_path
print(offsets_table_path('$BASE', '$PROPOSAL', '$FIELD'))" 2>/dev/null)
fi
if [ -z "$CONSENSUS_TBL" ]; then
    echo "REFUSING: alignment_config declares no table-driven correction channel for"
    echo "  proposal=$PROPOSAL field=$FIELD.  The m2 checkpoint cannot record a"
    echo "  correction for this field, so a re-tie loop can never converge."
    echo "  Add an entry to jwst_gc_pipeline/reduction/alignment_config.py, or set"
    echo "  OFFSETS_TBL explicitly if it uses some other table."
    exit 2
fi
echo "offsets table watched for changes: $CONSENSUS_TBL"

# Digest the table's SHIFT VALUES, so the "did this iteration re-tie anything?"
# test below answers that question rather than "did any byte change".
#
# This was an md5sum of the whole file.  The m2 checkpoint re-stamps `prov_date`
# on rows it did not move, so a round that wrote no correction still changed the
# file, the loop concluded a re-tie had been made, and it re-reduced and
# re-measured an identical residual -- issue #272, where three consecutive
# rounds each reported 15 corrections against 0 changed `dra`/`ddec` cells.  The
# guard was aimed at the opposite mistake (stopping while a correction HAD been
# made), and this is that condition inverted.
#
# A digest failure must NOT read as "unchanged": that would stop a loop that is
# working.  offsets_value_digest.py exits 2 on a table it cannot parse, and this
# then emits a per-call unique token so the comparison reports movement and the
# loop continues, with the reason on stderr.
#
# Scoped to THIS observation.  A table is not always one field's: 10678 registers
# fields=None, so all 139 treasury tiles write into one consensus table, and an
# unscoped digest would let a neighbouring tile's correction -- written while
# this one was in its m2 -- read as this tile's re-tie.  Same false positive as
# the prov_date re-stamp, a different writer.  The fixed-point check below is
# already scoped this way (`--obs-token o$FIELD`).
table_value_digest () {
    local path="$1" out rc=0
    out=$(PYTHONPATH="${PIPE_ROOT:-}:${PYTHONPATH:-}" python \
          "$HERE/offsets_value_digest.py" "$path" \
          --observation "$FIELD" 2>&1) || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "WARNING: could not digest $path ($out); treating this iteration as" >&2
        echo "         having changed the table, which is the fail-open side." >&2
        echo "undigestible-$(date -u +%s%N)-$RANDOM"
        return 0
    fi
    echo "$out"
}

read -r -a _FA <<< "$FILTERS"
NF=${#_FA[@]}
export_common="ALL,PROPOSAL=$PROPOSAL,FIELD=$FIELD,TARGET=$TARGET,FILTERS=$FILTERS"
[ -n "$PIPE_ROOT" ] && export_common="$export_common,PIPE_ROOT=$PIPE_ROOT"

# Poll a SLURM job to a terminal state; echo COMPLETED/FAILED/...
wait_job () {
    local jid="$1"
    while :; do
        local st
        st=$(sacct -j "$jid" --format=State -Pn 2>/dev/null | head -1 | tr -d ' ')
        case "$st" in
            ''|RUNNING|PENDING|REQUEUED|RESIZING|SUSPENDED|COMPLETING) sleep 30;;
            *) echo "$st"; return 0;;
        esac
    done
}

# reduce_fully_succeeded <array-jobid> <ntasks> [sbatch-rc]
#
# True when every task of the reduce array reached COMPLETED.  Returns 1 (and
# explains) otherwise, INCLUDING when no job id could be parsed at all.
#
# `sbatch --wait` blocks until every array task is terminal, but its exit status
# alone is not a safe gate: a partially-failed reduce must NOT be cataloged.
# The filters that failed keep the PREVIOUS iteration's WCS, so the m12 merge
# would combine this iteration's frames for some bands with last iteration's for
# others, and the m2 checkpoint would then "measure" a difference that is really
# just two iterations mixed together -- and write it into the consensus table as
# a correction.  That is a corruption, not a lost job.
#
# Not hypothetical: sgrc iteration 3 (38870453, 2026-08-07) lost all four LW
# filters to CRDS 504s four minutes in (#327) while the four SW filters
# completed, and the loop went straight on to catalog the mixture.
#
# `grep -c COMPLETED` is safe against COMPLETING, which does not contain the
# string.  The comparison is `-ne` rather than `-lt` deliberately: a requeued
# task can appear twice, making n_done EXCEED ntasks, and stopping on that fails
# in the safe direction (a spurious stop, never a spurious catalog).
reduce_fully_succeeded () {
    local jid="$1" ntasks="$2" rc="${3:-}"
    if [ -z "$jid" ]; then
        echo "reduce: could not parse a job id from sbatch (rc=${rc:-?}) -- STOPPING."
        return 1
    fi
    local states n_done n_bad
    states=$(sacct -j "$jid" -o State -X -n 2>/dev/null || true)
    # `grep -c` exits 1 when the count is 0, and the script runs under `set -e`,
    # so an unguarded count kills the loop on exactly the cases we care about:
    # zero COMPLETED (total failure) and zero non-COMPLETED (total success).
    n_done=$(printf '%s\n' "$states" | grep -c COMPLETED || true)
    n_bad=$(printf '%s\n' "$states" | grep -cvE 'COMPLETED|^[[:space:]]*$' || true)
    echo "reduce $jid: $n_done/$ntasks completed, $n_bad not"
    if [ "$n_done" -ne "$ntasks" ]; then
        echo "REDUCE DID NOT FULLY SUCCEED -- STOPPING before cataloging."
        sacct -j "$jid" -o JobID%18,JobName%34,State,Elapsed -X 2>/dev/null \
            | grep -vE 'COMPLETED'
        echo "  Cataloging now would merge this iteration's frames for the"
        echo "  filters that succeeded with the PREVIOUS iteration's frames"
        echo "  for the ones that failed."
        echo "  Re-run the failed filters (FILTERS=\"...\" sbatch"
        echo "  $HERE/submit_reduction.sbatch), then restart the loop."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Which code is this loop actually running?
#
# A loop runs for days -- MAXITER=12 at ~7 h a pass is a fortnight -- and bash
# parses each command as it reaches it and does not re-read what it has already
# run, so a loop is effectively frozen at the state its checkout was in when it
# started, and a safety guard merged the next morning
# never reaches it.  That is not hypothetical: sgrc's loop ran 2026-08-07 to
# 08-09 out of a checkout that predates `reduce_fully_succeeded`, so for its
# whole life it cataloged without checking that its reduce had succeeded --
# the exact guard whose own comment cites that field's failure (#364).  Nothing
# in its output said so; the missing guard is invisible precisely because the
# code that would have printed it is the code that is missing.
#
# TWO checkouts are in play and they need not be the same one:
#   * this script's own, from BASH_SOURCE -- the loop's control flow and its
#     guards;
#   * PIPE_ROOT -- the jwst_gc_pipeline package the reduce and the cataloging
#     import, which is what actually writes the offsets table.
# Both are reported, at launch and at every iteration.

# _last_fetch <dir> -- when the local copy of the upstream ref was last updated.
#
# Printed on EVERY provenance line, including the "0 commit(s) behind" one.  The
# distance is measured against the LOCAL remote-tracking ref, so a checkout that
# has not fetched for a month reads "0 behind" while being a month stale -- and
# that is the case this design turns on, so the qualifier has to be on the line
# a reader actually sees, not only on the warning that fires when the distance
# is non-zero.
_last_fetch () {
    local f="$1/.git/FETCH_HEAD" gd
    gd=$(git -C "$1" rev-parse --git-dir 2>/dev/null) || { echo "unknown"; return 0; }
    case "$gd" in /*) f="$gd/FETCH_HEAD";; *) f="$1/$gd/FETCH_HEAD";; esac
    if [ -f "$f" ]; then
        date -u -r "$f" +%Y-%m-%d 2>/dev/null || echo "unknown"
    else
        echo "never"
    fi
}

# checkout_provenance <dir> -- one line: HEAD, date, dirty, distance behind the
# upstream branch.  Never fails; a non-repository or a git-less environment
# reports what it could not determine rather than stopping the loop.
checkout_provenance () {
    local dir="$1" head date dirty behind
    if [ ! -d "$dir" ]; then echo "$dir (does not exist)"; return 0; fi
    if ! git -C "$dir" rev-parse --git-dir >/dev/null 2>&1; then
        echo "$dir (not a git checkout -- provenance unknown)"; return 0
    fi
    head=$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || echo '?')
    date=$(git -C "$dir" log -1 --format=%cs 2>/dev/null || echo '?')
    # every substitution below is guarded: under `set -e` a bare
    # `var=$(cmd)` whose command fails exits the script, and a pipeline under
    # `pipefail` fails if ANY stage does.  Reporting an unknown is the job here;
    # stopping a two-week loop because git had an opinion is not.
    # `|| echo` on the whole pipeline is not enough: `wc` has already printed 0
    # by the time git fails, so the variable becomes "0" and then "?" on the
    # next line -- guard the pipeline itself.  On a bare repo `git status`
    # returns 128 and this is the line that would otherwise take the function
    # down if it were ever called outside a command substitution.
    if ! dirty=$(git -C "$dir" status --porcelain 2>/dev/null); then
        dirty="?"
    else
        dirty=$(printf '%s' "$dirty" | grep -c . || true)
    fi
    behind=$(git -C "$dir" rev-list --count "HEAD..${RETIE_UPSTREAM:-origin/main}" \
             2>/dev/null || echo '?')
    echo "$dir @ $head ($date), ${dirty} uncommitted, ${behind} commit(s) behind ${RETIE_UPSTREAM:-origin/main} (local ref, last fetched $(_last_fetch "$dir"))"
}

# commits_behind <dir> -- the count alone, or '?' when it cannot be determined.
commits_behind () {
    git -C "$1" rev-list --count "HEAD..${RETIE_UPSTREAM:-origin/main}" 2>/dev/null \
        || echo '?'
}

# Report -- never refuse -- how far behind its upstream a checkout is.
#
# Reporting, not gating, and the reason is worth writing down because the
# obvious stronger designs were tried and measured and both fail:
#
#   * REFUSE WHEN BEHIND origin/main.  Refuses the workflow this project
#     mandates.  CLAUDE.md requires pipeline work to happen on worktree
#     branches; of the 400 worktrees of this repository 395 are behind
#     origin/main and 156 are deliberately both ahead and behind -- a topic
#     branch under test, which is how a re-tie is normally driven.  It also
#     refuses a detached HEAD at a release tag, which is the operational recipe
#     #364 recommends.  A rule that refuses 99% of checkouts trains the operator
#     to set the override every time, which disables it for the real case.
#     And it cannot even be trusted as an all-clear: `rev-list HEAD..origin/main`
#     compares against the LOCAL copy of that ref, so a checkout that has not
#     fetched for a month reads "0 behind".
#
#   * REFUSE WHEN A NAMED GUARD IS ABSENT (#364's "minimum-version assertion").
#     Tried against the actual checkout from #364: it PASSES, because the guard
#     that was missing there -- `reduce_fully_succeeded` -- lives in this shell
#     script, not in the Python package, and a script cannot check itself for a
#     guard whose absence also removes the check -- a script cannot check itself.
#     Meanwhile it refused a
#     legitimate topic-branch worktree.  Wrong on both sides.
#
# The general point, which #364 states and which no self-check can get around:
# the code that would report a missing guard is part of what is missing.  What
# is left is to make the gap VISIBLE -- which is #364's own first suggestion --
# and to keep the operational rule that long loops are restarted after a safety
# merge.
warn_if_behind () {
    local dir behind
    for dir in "$@"; do
        [ -d "$dir" ] || continue
        git -C "$dir" rev-parse --git-dir >/dev/null 2>&1 || continue
        behind=$(commits_behind "$dir")
        if [ "$behind" = '?' ]; then
            echo "  NOTE: cannot tell how far $dir is behind"
            echo "        (no ${RETIE_UPSTREAM:-origin/main} locally; try 'git -C $dir fetch origin')"
        elif [ "$behind" -gt 0 ]; then
            echo "  NOTE: $dir is $behind commit(s) behind its LOCAL"
            echo "        ${RETIE_UPSTREAM:-origin/main}, which is only as fresh as the last fetch."
            echo "        Being behind is normal on a topic branch.  What it means here is that"
            echo "        any safety guard merged in those commits is NOT in this run and will"
            echo "        not arrive while it lasts -- restart the loop to pick them up."
        fi
    done
    return 0
}

# Has the script changed on disk since bash read it?
#
# bash reads a script INCREMENTALLY, by byte offset.  The iteration loop is one
# compound command and is fully parsed before it runs, so an edit cannot change
# the loop mid-flight -- but there is code AFTER the loop, and an edit that
# shifts byte offsets makes bash resume mid-token there.  So the effect of a
# mid-run edit is not "nothing changes": it is that this run stops being
# predictable, and the file on disk stops describing what is executing.
#
# Reported, not acted on: whether a loop may deliberately adopt code mid-run is
# a decision about its contract, not a bug fix, and #364 leaves it open.
_here_top_or_here=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null \
                    || echo "$HERE")
SELF_PATH="${BASH_SOURCE[0]}"
SELF_SUM_AT_LAUNCH=$(md5sum "$SELF_PATH" 2>/dev/null | cut -d' ' -f1 || echo '?')

warn_if_self_changed () {
    local now
    now=$(md5sum "$SELF_PATH" 2>/dev/null | cut -d' ' -f1 || echo '?')
    if [ "$now" != "$SELF_SUM_AT_LAUNCH" ]; then
        echo "WARNING: $SELF_PATH has changed on disk since this loop started."
        echo "  bash reads a script INCREMENTALLY, by byte offset, so an edit that"
        echo "  shifts offsets can make it resume mid-token in code after the loop."
        echo "  This run is now unpredictable and the file no longer describes it."
        echo "  Do not edit a running script; restart the loop instead."
    fi
}

# Exposed so tests can source the helpers without running the loop.
[ -n "${RETIE_LOOP_SOURCE_ONLY:-}" ] && return 0 2>/dev/null

echo "=================  CHECKOUT PROVENANCE  ================="
echo "  loop script : $(checkout_provenance "$_here_top_or_here")"
if [ -n "$PIPE_ROOT" ]; then
    echo "  pipeline    : $(checkout_provenance "$PIPE_ROOT")"
    _pipe_top=$(git -C "$PIPE_ROOT" rev-parse --show-toplevel 2>/dev/null \
                || echo "$PIPE_ROOT")
    if [ "$_here_top_or_here" != "$_pipe_top" ]; then
        echo "  NOTE: the loop and the pipeline package come from DIFFERENT checkouts,"
        echo "        so they can be at different commits and each carry different guards."
    fi
else
    echo "  pipeline    : PIPE_ROOT unset -- whichever jwst_gc_pipeline is on PYTHONPATH"
fi
warn_if_behind "$_here_top_or_here" ${PIPE_ROOT:+"$PIPE_ROOT"}
if [ -z "${PIPE_ROOT:-}" ]; then
    echo "  NOTE: PIPE_ROOT is unset, so this log cannot say which jwst_gc_pipeline"
    echo "        the reduce and the cataloging will import -- whatever is on"
    echo "        PYTHONPATH wins, and it is not recorded anywhere."
fi
if [ "${RETIE_PROVENANCE_ONLY:-0}" = 1 ]; then
    echo "RETIE_PROVENANCE_ONLY=1 -- reported the provenance above and submitted"
    echo "  nothing.  Unset it to run the loop."
    exit 0
fi

for ((it=1; it<=MAXITER; it++)); do
    echo "=================  RE-TIE ITER $it / $MAXITER  ($TARGET $PROPOSAL/$FIELD)  ================="
    # Restated every iteration, not only at launch: an iteration's log is what
    # anyone reads when its results are questioned days later, and "which code
    # produced this" has to be answerable from that log alone.
    echo "[iter $it] running: $(checkout_provenance "$_here_top_or_here")"
    warn_if_self_changed

    # --- 1. reduce (blocks until the whole array finishes) ---
    echo "[iter $it] reducing (fix_alignment applies consensus table if present: $([ -f "$CONSENSUS_TBL" ] && echo yes || echo no))"
    # --job-name at SUBMIT time (standing rule, CLAUDE.md): the in-script runtime
    # rename only fires when the job STARTS, and a quota-bound retie sits PENDING
    # for hours under the generic name -- which is exactly when the queue is being
    # watched, and when several reduce arrays are in flight at once.
    # `|| red_rc=$?` is NOT decoration.  Under `set -e` a plain
    # `var=$(cmd)` assignment exits the script when cmd fails, and
    # `sbatch --wait` returns nonzero as soon as ANY array task fails -- which
    # is exactly the case reduce_fully_succeeded exists to diagnose.  So the
    # loop died here, before `red_rc=$?`, before `echo "$red_out"`, and before
    # the guard: it stopped safely but SILENTLY.  cloudef (2026-08-09, F360M of
    # 4 filters failed) left a log ending at "[iter 1] reducing" with no sbatch
    # output and no reason, and the operator has to go to sacct to find out
    # that anything happened at all.
    red_rc=0
    red_out=$(sbatch --wait --parsable --array=0-$((NF-1)) --qos="$QOS" \
        --job-name="${TARGET}${PROPOSAL}-o${FIELD}-reduce-retie${it}" \
        --export="${export_common}" \
        "$HERE/submit_reduction.sbatch" 2>&1) || red_rc=$?
    echo "$red_out"
    # `|| true` for the same reason as above, with `pipefail` making it sharper:
    # a non-matching grep fails the whole pipeline.  A SUBMISSION failure (QOS
    # limit, bad partition, malformed --export) prints an error with no leading
    # job id, so this grep matches nothing and the loop died HERE -- still one
    # line before the guard, still without printing why.  An empty red_jid with
    # a nonzero rc is a case reduce_fully_succeeded already reports.
    red_jid=$(echo "$red_out" | grep -oE '^[0-9]+' | head -1) || true

    # `sbatch --wait` blocks until every array task is terminal, but its exit
    # status alone is not a safe gate: a partially-failed reduce must NOT be
    # cataloged.  The filters that failed keep the PREVIOUS iteration's WCS, so
    # the m12 merge would combine this iteration's frames for some bands with
    # last iteration's for others, and the m2 checkpoint would then "measure" a
    # difference that is really just two iterations mixed together.
    #
    # This is not hypothetical: sgrc iteration 3 (38870453, 2026-08-07) lost all
    # four LW filters to CRDS 504s four minutes in (issue #327) while the four
    # SW filters completed, and the loop went straight on to catalog the mixture.
    if ! reduce_fully_succeeded "$red_jid" "$NF" "$red_rc"; then
        exit 1
    fi

    # --- 2. catalog to m2 only, with auto-apply ON ---
    echo "[iter $it] cataloging to m2 (ASTROM_CHECKPOINT_APPLY=1)"
    tbl_before=$(table_value_digest "$CONSENSUS_TBL")
    # submit_cataloging_perframe.sh self-sbatches the chain; PHASES="m12" stops at
    # the m2 merge+checkpoint.  Capture the finalize (stage B) job id it prints.
    export ASTROM_CHECKPOINT_APPLY=1
    chain_rc=0
    # `env` is REQUIRED here, not decoration.  bash fixes the boundary of an
    # assignment prefix at PARSE time, before "${floor_env[@]}" expands, so a
    # quoted array expansion sitting after literal VAR=value tokens becomes the
    # COMMAND WORD.  With the array non-empty this tries to RUN
    # `ASTROM_M2_CORRECTION_FLOOR_MAS=4.0` as a program -- rc=127, and the loop
    # stops at iteration 1 having reduced but cataloged nothing:
    #
    #   $ fe=(FOO=1); BAR=2 "${fe[@]}" echo hi
    #   bash: FOO=1: command not found              (rc 127)
    #   $ fe=(FOO=1); env BAR=2 "${fe[@]}" echo hi
    #   hi                                          (rc 0)
    #
    # An EMPTY array expands to nothing and the prefix parses normally, which is
    # why every field taking its floor from PER_FIELD_FLOOR_MAS was unaffected
    # and only an explicit env override hit it (gc2211_o049, 2026-08-27).  That
    # is also the path the manual-restart note tells operators to use.
    # common.sh already prefixes `env` at its own forwarding site.
    chain_out=$(env PROPOSAL=$PROPOSAL FIELD=$FIELD TARGET=$TARGET MODULES=$MODULES \
        EACH_SUFFIX=$EACH_SUFFIX FILTERS="$FILTERS" MAX_GROUP_SIZE=$MAX_GROUP_SIZE \
        PHASES="m12" PIPE_ROOT=$PIPE_ROOT \
        "${floor_env[@]}" \
        bash "$HERE/submit_cataloging_perframe.sh") || chain_rc=$?
    echo "$chain_out"
    if [ "${chain_rc:-0}" -ne 0 ]; then
        echo "[iter $it] the cataloging submission FAILED (rc=$chain_rc) -- STOPPING."
        echo "           Its output is above; nothing was queued to wait on."
        exit 1
    fi
    # Both of these need `|| true`: the fallback exists for "the first pattern
    # did not match", which under `set -o pipefail` is exactly when a bare
    # assignment exits the script -- so the fallback could never run.
    fin_jid=$(echo "$chain_out" | grep -oE 'finalize[^0-9]*[0-9]+' | grep -oE '[0-9]+' | tail -1) || true
    if [ -z "$fin_jid" ]; then
        fin_jid=$(echo "$chain_out" | grep -oE '[0-9]{6,}' | tail -1) || true
    fi
    if [ -z "$fin_jid" ]; then
        echo "[iter $it] could not parse a finalize job id from the cataloging"
        echo "           submission output above -- STOPPING rather than waiting"
        echo "           on nothing and calling the result converged."
        exit 1
    fi
    echo "[iter $it] waiting on m2 finalize job $fin_jid"
    st=$(wait_job "$fin_jid")
    echo "[iter $it] m2 finalize state: $st"

    # --- 3. converged? ---
    tbl_after=$(table_value_digest "$CONSENSUS_TBL")
    if [ "$st" = "COMPLETED" ]; then
        echo "[iter $it] m2 checkpoint PASSED -- converged after $it iter(s)."
        break
    fi
    if [ "$tbl_after" = "$tbl_before" ]; then
        echo "[iter $it] m2 finalize failed ($st) but no SHIFT VALUE in the consensus"
        echo "           table changed for observation $FIELD (provenance re-stamps,"
        echo "           re-serialisation and another observation's rows do not"
        echo "           count -- see offsets_value_digest.py)."
        echo "           This is NOT a checkpoint re-tie (some other failure) -- STOPPING."
        echo "           Inspect logs/catalog_pf_${fin_jid}*.out before retrying."
        exit 1
    fi
    # --- 3b. is it repeating rather than converging? ---
    # The md5 check above only catches a table that did not change AT ALL.  A
    # loop at a fixed point (or oscillating between two states) rewrites the
    # last decimal place every pass, so the md5 differs and the loop runs to
    # MAXITER measuring the same thing -- sgrc spent 4 passes at ~7 h each doing
    # exactly that.  Ask the checkpoint records whether re-tying is changing
    # what the next pass MEASURES.
    if [ "${RETIE_FIXED_POINT_CHECK:-1}" = "1" ] && [ "$it" -ge 2 ]; then
        # Same interpreter + path convention as the CONSENSUS_TBL lookup above.
        # Captured rather than streamed: rc=4 needs the floor the check printed,
        # and re-running it to read that would judge a different set of records.
        # `|| fp_rc=$?` for the same reason as the reduce above: under `set -e` a
        # bare `var=$(cmd)` exits the script when cmd fails, and a NONZERO rc is
        # this check's whole output -- rc 3 and rc 4 are its two verdicts.
        # Without it the loop would die here, one line before the branch that
        # reads them, with the report captured and never printed.
        # Validate the operator-supplied ceiling BEFORE handing it to argparse.
        # A typo makes argparse exit 2, which lands in the `-ne 0` arm below and
        # is reported as "continuing" -- so a mistyped ceiling would not merely
        # fail to accept, it would switch OFF the fixed-point STOP for the rest
        # of the run and grind to MAXITER at ~7 h a pass with a usage message
        # buried in the log.
        case "${RETIE_ACCEPT_RESIDUAL_MAS:-0}" in
            ''|*[!0-9.]*|*.*.*)
                echo "REFUSING: RETIE_ACCEPT_RESIDUAL_MAS=${RETIE_ACCEPT_RESIDUAL_MAS}"
                echo "          is not a number of milliarcseconds."
                exit 2 ;;
        esac
        fp_rc=0
        fp_out=$(PYTHONPATH="${PIPE_ROOT:-}:${PYTHONPATH:-}" python \
                -m jwst_gc_pipeline.photometry.retie_fixed_point \
                --record-dir "${BASE}/astrometry_checkpoints" \
                --obs-token "o${FIELD}" --since "$RETIE_RUN_START" \
                --accept-below-mas "${RETIE_ACCEPT_RESIDUAL_MAS:-0}" \
                --expect-filters "$FILTERS" 2>&1) \
                || fp_rc=$?
        echo "$fp_out"
        if [ "$fp_rc" -eq 3 ]; then
            echo "[iter $it] STOPPING: the re-tie is repeating itself (see above)."
            echo "           More iterations cannot resolve this; the residual"
            echo "           needs a decision, not another pass."
            echo "           Set RETIE_FIXED_POINT_CHECK=0 to override."
            exit 3
        elif [ "$fp_rc" -eq 4 ]; then
            # A BOUNDED fixed point: the residual repeats and is small enough
            # that it is the systematic a per-exposure table cannot express, not
            # a correction failing to reach the frame.  Proceed over it rather
            # than holding the field -- but only by raising the m2 CORRECTION
            # floor to just above it, so the residual is still MEASURED and
            # RECORDED every stage.  The consensus-to-reference tie is never
            # floored, and the gross reference-tie gates are untouched.
            fp_floor=$(echo "$fp_out" | sed -n 's/^ASTROM_M2_CORRECTION_FLOOR_MAS=//p' | tail -1)
            if [ -z "$fp_floor" ]; then
                echo "[iter $it] STOPPING: the fixed-point check reported a bounded"
                echo "           residual but printed no floor to run it under."
                exit 3
            fi
            # Raise the floor and take ANOTHER PASS -- do not break out here.
            # Reaching this point means m2 just wrote new corrections into the
            # offsets table and stale-tagged this filter's _i2d mosaics.  The
            # frames on disk still carry the PREVIOUS pass's baked RAOFFSET, so
            # breaking now would leave the table describing frames that were
            # never re-drizzled (the frame/table disagreement behind brick-1182
            # v001) and the mosaics renamed _im0_badastrom for good.  Another
            # pass re-reduces with the table applied, re-drizzles, and m2 passes
            # under the raised floor -- and the loop then exits by its normal
            # converged branch, with table, frames and mosaics agreeing.
            # Never LOWER an operator-set floor.  Eight fields run at 4.0
            # today; a computed 0.5 would make the next m2 correct everything
            # above 0.5 mas and the loop would never converge.
            #
            # Computed BEFORE the announcement, and both messages interpolate
            # the EXPORTED value.  They used to print `$fp_floor` while the run
            # continued at the max -- so with a preset of 4.0 and a computed
            # 0.6 the log said the run was continuing at 0.6, and the
            # last-iteration line then handed a human the exact lowering this
            # max exists to prevent.
            # Baseline for "a floor is never lowered".  With nothing exported the
            # effective floor is the field's PER-FIELD value, not 0, and taking 0
            # here would let the raise LOWER a field whose table entry is above
            # what the fixed-point check computed (sgrc/cloudc 8.0, w51 6.0).
            _prev_floor=${ASTROM_M2_CORRECTION_FLOOR_MAS:-${PER_FIELD_FLOOR:-0}}
            _effective_floor=$(awk -v a="$_prev_floor" -v b="$fp_floor" \
                'BEGIN{print (a>b)?a:b}')
            echo "[iter $it] BOUNDED fixed point -- re-reducing once more with"
            echo "           ASTROM_M2_CORRECTION_FLOOR_MAS=$_effective_floor (was"
            echo "           ${ASTROM_M2_CORRECTION_FLOOR_MAS:-unset}; the check"
            echo "           computed $fp_floor and a floor is never lowered), so"
            echo "           the frames and the offsets table end up agreeing."
            ASTROM_M2_CORRECTION_FLOOR_MAS="$_effective_floor"
            export ASTROM_M2_CORRECTION_FLOOR_MAS
            # From here the value IS explicit, so it must reach the children --
            # the array is empty until now whenever the caller set nothing.
            floor_env=(ASTROM_M2_CORRECTION_FLOOR_MAS="$ASTROM_M2_CORRECTION_FLOOR_MAS")
            if [ "$it" -eq "$MAXITER" ]; then
                echo "[iter $it] ...but this is the last iteration (MAXITER=$MAXITER)."
                echo "           Re-run with MAXITER=$((MAXITER + 1)) and"
                echo "           ASTROM_M2_CORRECTION_FLOOR_MAS=$_effective_floor to finish."
                exit 2
            fi
        elif [ "$fp_rc" -ne 0 ]; then
            # rc 2 is argparse/usage: the check never ran, so treating it as
            # "no fixed point" would silently disable this whole branch.
            echo "[iter $it] STOPPING: the fixed-point check itself failed"
            echo "           (exit $fp_rc) -- it did not report a verdict, so"
            echo "           continuing would run blind.  Set"
            echo "           RETIE_FIXED_POINT_CHECK=0 to proceed without it."
            exit 3
        fi
    fi

    echo "[iter $it] consensus table updated -> re-reduce + re-catalog."
    if [ "$it" -eq "$MAXITER" ]; then
        echo "REACHED MAXITER=$MAXITER without the checkpoint passing."
        echo "The per-exposure scatter is not closing via the consensus tie alone;"
        echo "inspect ${CONSENSUS_TBL} and the m2 checkpoint records under"
        echo "${BASE}/astrometry_checkpoints/ (residual centroiding/distortion?)."
        exit 2
    fi
done

# --- final: full cataloging (m3..m7) now that im0 is self-consistent ---
echo "=================  FULL CATALOGING (m3..m7)  ================="
unset ASTROM_CHECKPOINT_APPLY   # m3+ must be a FROZEN solution; no more corrections
# Named explicitly rather than left to inheritance.  The chain re-runs m12 and m2
# runs inside it, so this value decides whether the checkpoint raises -- and a
# variable that changes a gate's verdict should be visible at the call site
# rather than reaching the job by a route the reader has to reconstruct.  It also
# makes this invocation symmetric with the m12 one above.
#
# NOT a fix for a propagation failure: the value already arrives.
# submit_cataloging_perframe.sh's --export list begins with ALL, so the job
# inherits the submitting environment plus the named variables, and `export` at
# the top of this script is sufficient on its own.  Verified directly:
#
#   $ export ASTROM_M2_CORRECTION_FLOOR_MAS=4.0
#   $ sbatch --export="ALL,PROPOSAL=p,FIELD=f" --wrap='echo "FLOOR=[$ASTROM_M2_CORRECTION_FLOOR_MAS]"'
#   FLOOR=[4.0] PROPOSAL=[p]
#
# cloudef obs005's m3..m7 chain died here on 2026-08-04, and this is NOT why --
# see #281.  Do not treat this line as having closed that.
# `env` for the same parse-time reason as the m12 submission above: without it a
# non-empty "${floor_env[@]}" is read as the command word, not an assignment.
env PROPOSAL=$PROPOSAL FIELD=$FIELD TARGET=$TARGET MODULES=$MODULES \
    EACH_SUFFIX=$EACH_SUFFIX FILTERS="$FILTERS" MAX_GROUP_SIZE=$MAX_GROUP_SIZE \
    PIPE_ROOT=$PIPE_ROOT \
    "${floor_env[@]}" \
    bash "$HERE/submit_cataloging_perframe.sh"
echo "Submitted full cataloging chain for $TARGET $PROPOSAL/$FIELD."
