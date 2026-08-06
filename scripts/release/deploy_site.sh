#!/bin/bash
# Push the generated data-release site to the Apache docroot.
#
#   https://starformation.astro.ufl.edu/jwst-gc/
#
# The source is whatever `make_webpage.py --out` last wrote. The sync carries
# `--delete`, because a field dropped from the release should stop being served.
#
# WHY THIS SCRIPT EXISTS
#
# `htdocs/jwst-gc/` holds two independently generated trees:
#
#     jwst-gc/index.html, <field>.html, assets/   <- this script (releases/site)
#     jwst-gc/monitor/                            <- scripts/monitoring/deploy_monitor.sh
#
# `releases/site/` has no `monitor/` in it, so a plain
#
#     rsync -rz --delete releases/site/ starformation:.../htdocs/jwst-gc/
#
# deletes the entire 194 MB monitor tree as "extraneous". That is exactly what
# happened on 2026-08-06: the release site was republished at 09:28 and
# https://starformation.astro.ufl.edu/jwst-gc/monitor/ returned 404 until it was
# rsynced back at 14:45. The `protect` filter below is the fix; run the release
# sync through this script rather than by hand and it cannot recur.
#
# Usage:  scripts/release/deploy_site.sh [--dry-run] [source-dir]
set -euo pipefail

DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then DRY="--dry-run -i --stats"; shift; fi

SRC="${1:-/orange/adamginsburg/jwst/releases/site}"
HOST="${RELEASE_WEB_HOST:-starformation}"
DEST="${RELEASE_WEB_DIR:-/h/cnswww-starformation.astro/starformation.astro.ufl.edu/htdocs/jwst-gc}"
URL="${RELEASE_WEB_URL:-https://starformation.astro.ufl.edu/jwst-gc/}"

# Two ways to get this wrong, both destructive: syncing into `monitor/` (which
# would replace the monitor with the release pages), and syncing into `htdocs/`
# itself (which would --delete every other project on the server).
case "$DEST" in
  */jwst-gc|*/jwst-gc/) ;;
  *)
    echo "refusing to sync to '$DEST': the destination must end in /jwst-gc." >&2
    echo "the monitor lives in jwst-gc/monitor and is deployed separately by" >&2
    echo "scripts/monitoring/deploy_monitor.sh." >&2
    exit 2 ;;
esac

if [[ ! -f "$SRC/index.html" ]]; then
  echo "no index.html in $SRC -- run make_webpage.py --out '$SRC' first" >&2
  exit 3
fi

# Was the monitor there before we started? If it was, it has to be there after.
had_monitor=0
if ssh "$HOST" "test -f '$DEST/monitor/index.html'"; then had_monitor=1; fi

ssh "$HOST" "mkdir -p '$DEST'"

# --filter 'protect monitor/': exempt the monitor tree from --delete. An
#   --exclude alone would do it today, but only because rsync protects excluded
#   files by default -- adding --delete-excluded later would silently re-arm the
#   deletion. `protect` says the thing we actually mean.
# --no-perms --omit-dir-times: the docroot is group-writable and setgid; letting
#   rsync stamp HiPerGator's modes onto it fights the server's own defaults.
rsync -rltz --no-perms --omit-dir-times --delete \
      --filter='protect monitor/' --exclude='monitor/' \
      --human-readable $DRY \
      "$SRC/" "$HOST:$DEST/"

if [[ -n "$DRY" ]]; then exit 0; fi

ssh "$HOST" "chmod -R a+rX '$DEST'"

# The protect filter is one line and one typo away from being wrong, so do not
# take it on faith: the run that removes the monitor must fail loudly, while it
# is still the last thing anyone did.
if [[ $had_monitor -eq 1 ]] && ! ssh "$HOST" "test -f '$DEST/monitor/index.html'"; then
  echo "FAILED: $DEST/monitor/index.html was there before this sync and is gone" >&2
  echo "now. The release sync deleted the monitor tree. Restore it with:" >&2
  echo "  scripts/monitoring/deploy_monitor.sh" >&2
  exit 4
fi

echo "deployed -> $URL"
exit 0
