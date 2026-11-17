#!/bin/bash
# ---------------------------------------------------------------------------
# List, and optionally remove, the pinned worktrees run_field_retie_loop.sh
# leaves behind (#679).
#
# A pin is the snapshot a retie loop and every job it submits import from.  The
# loop does not delete it at exit, because the m3-m7 chain it submits at the end
# is still queued and imports from it when it starts.  A pin is therefore safe
# to remove only once no job of its field is queued or running.
#
# Pins are named  retie-<TARGET><PROPOSAL>-o<FIELD>-<UTC stamp>-<sha7>, and the
# retie and cataloging jobs are named  <TARGET><PROPOSAL>-o<FIELD>-<phase>...,
# so a pin is KEPT while any job of this user whose name starts with
# <TARGET><PROPOSAL>-o<FIELD>- is in the queue.  A second retie of the same
# field cannot be told apart and errs on the side of keeping.
#
# A loop that is still running is also kept: between iterations it can have no
# job in the queue while it works on the login node, and its script lives in the
# pin.  The lock reason written at launch ("retie loop <host> pid <pid>") names
# the loop; a pin whose loop is alive on this host is kept, and a pin locked
# from another host is kept too because its pid cannot be checked here
# (RETIE_PRUNE_ASSUME_LOOPS_DEAD=1 skips that second rule).
#
# Dry run by default.  If squeue fails nothing is removed.  With --apply a pin
# that cannot be read or removed is reported and the rest are still handled;
# the exit status is then 1.  A symlink named like a pin is skipped: git would
# remove the worktree it points at.
#
#   scripts/reduction/prune_retie_pins.sh            # list
#   scripts/reduction/prune_retie_pins.sh --apply    # remove the finished ones
#
# Environment:  RETIE_PIN_DIR  (default /blue/adamginsburg/adamginsburg/repos/pinned)
#               RETIE_PRUNE_ASSUME_LOOPS_DEAD=1  see above
# ---------------------------------------------------------------------------
set -euo pipefail

apply=0
case "${1:-}" in
    "") ;;
    --apply) apply=1 ;;
    *) echo "usage: $0 [--apply]" >&2; exit 2 ;;
esac

PIN_DIR=${RETIE_PIN_DIR:-/blue/adamginsburg/adamginsburg/repos/pinned}
if [ ! -d "$PIN_DIR" ]; then
    echo "no pin directory at $PIN_DIR; nothing to do"
    exit 0
fi
# Resolve it: the lock-reason lookup below compares against the path git prints,
# which is the resolved one, so a trailing slash, "//" or a symlinked
# RETIE_PIN_DIR would otherwise make a running loop's pin look removable.
PIN_DIR=$(cd -P "$PIN_DIR" && pwd)

if ! queued=$(squeue -u "${USER:-$(id -un)}" -h -o %j 2>/dev/null); then
    echo "squeue failed; cannot tell which pins are in use, removing nothing." >&2
    exit 1
fi

kept=0; removable=0; problems=0
shopt -s nullglob
for pin in "$PIN_DIR"/retie-*; do
    if [ -L "$pin" ]; then
        echo "skip     $pin (a symlink, not a pin)"
        continue
    fi
    [ -e "$pin/.git" ] || continue          # not a worktree: leave it alone
    name=$(basename "$pin")
    token=$(printf '%s' "$name" \
        | sed -nE 's/^retie-(.+-o[^-]+)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]+$/\1/p')
    if [ -z "$token" ]; then
        echo "skip     $pin (name does not parse)"
        continue
    fi
    if printf '%s\n' "$queued" \
           | awk -v p="${token}-" 'index($0, p) == 1 { f = 1 } END { exit !f }'; then
        echo "in use   $pin (a ${token}-* job is queued or running)"
        kept=$((kept + 1))
        continue
    fi
    if ! reason=$(git -C "$pin" worktree list --porcelain 2>/dev/null \
        | awk -v w="$pin" '$1 == "worktree" { on = ($2 == w) }
                           on && $1 == "locked" && !done { sub(/^locked */, ""); print; done = 1 }'); then
        echo "skip     $pin (cannot read the worktree; its source repository has moved or gone?)"
        problems=$((problems + 1))
        continue
    fi
    if [[ "$reason" =~ ^retie\ loop\ (.+)\ pid\ ([0-9]+)$ ]]; then
        loop_host=${BASH_REMATCH[1]}; loop_pid=${BASH_REMATCH[2]}
        if [ "$loop_host" = "$(hostname)" ]; then
            if kill -0 "$loop_pid" 2>/dev/null; then
                echo "in use   $pin (its loop, pid $loop_pid, is running)"
                kept=$((kept + 1))
                continue
            fi
        elif [ "${RETIE_PRUNE_ASSUME_LOOPS_DEAD:-0}" != 1 ]; then
            echo "in use?  $pin (loop ran on $loop_host; pid $loop_pid cannot be checked here)"
            kept=$((kept + 1))
            continue
        fi
    fi
    removable=$((removable + 1))
    if [ "$apply" = 0 ]; then
        echo "removable $pin"
        continue
    fi
    if ! common=$(cd "$pin" && cd "$(git rev-parse --git-common-dir)" && pwd); then
        echo "FAILED   $pin (cannot find its source repository)" >&2
        problems=$((problems + 1))
        continue
    fi
    repo=$common
    [ "$(basename "$common")" = ".git" ] && repo=$(dirname "$common")
    # A pin that is not locked (a failed `worktree lock` at launch, or a manual
    # unlock) makes `worktree unlock` fail; that is not a reason to keep it.
    git -C "$repo" worktree unlock "$pin" 2>/dev/null || true
    if ! git -C "$repo" worktree remove --force "$pin"; then
        echo "FAILED   $pin (git worktree remove failed)" >&2
        problems=$((problems + 1))
        continue
    fi
    echo "removed  $pin"
done
echo "$removable removable, $kept in use"
if [ "$problems" -gt 0 ]; then
    echo "$problems pin(s) could not be handled; see the lines above." >&2
fi
[ "$apply" = 0 ] && [ "$removable" -gt 0 ] && echo "re-run with --apply to remove them"
[ "$problems" -gt 0 ] && exit 1
exit 0
