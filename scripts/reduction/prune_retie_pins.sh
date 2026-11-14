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
# Dry run by default.  If squeue fails nothing is removed.
#
#   scripts/reduction/prune_retie_pins.sh            # list
#   scripts/reduction/prune_retie_pins.sh --apply    # remove the finished ones
#
# Environment:  RETIE_PIN_DIR  (default /blue/adamginsburg/adamginsburg/repos/pinned)
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

if ! queued=$(squeue -u "${USER:-$(id -un)}" -h -o %j 2>/dev/null); then
    echo "squeue failed; cannot tell which pins are in use, removing nothing." >&2
    exit 1
fi

kept=0; removable=0
shopt -s nullglob
for pin in "$PIN_DIR"/retie-*; do
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
    removable=$((removable + 1))
    if [ "$apply" = 0 ]; then
        echo "removable $pin"
        continue
    fi
    common=$(cd "$pin" && cd "$(git rev-parse --git-common-dir)" && pwd)
    repo=$common
    [ "$(basename "$common")" = ".git" ] && repo=$(dirname "$common")
    git -C "$repo" worktree unlock "$pin"
    git -C "$repo" worktree remove --force "$pin"
    echo "removed  $pin"
done
echo "$removable removable, $kept in use"
[ "$apply" = 0 ] && [ "$removable" -gt 0 ] && echo "re-run with --apply to remove them"
exit 0
