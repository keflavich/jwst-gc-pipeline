#!/bin/bash
# Push the generated monitor to the Apache docroot.
#
# `--publish-dir` only assembles the page set on HiPerGator; nothing on
# HiPerGator is web-served. Apache runs on a separate host whose docroot is a
# different filesystem, so getting the pages in front of a browser is a copy,
# not a link.
#
#   https://starformation.astro.ufl.edu/jwst-gc/monitor/
#
# Two things about the destination are worth knowing before editing it:
#
#   * `htdocs/jwst-gc/index.html` is the **public data-release landing page**.
#     The monitor deliberately lands in the `monitor/` subdirectory; syncing to
#     `jwst-gc/` itself would overwrite the release page with a 2 MB internal
#     status report. The guard below refuses any destination not ending in
#     `/monitor`.
#   * The `diagnostics-<field>` entries in the publish directory are symlinks
#     into /orange and /blue. Those paths do not exist on the web host, so the
#     sync dereferences them (`--copy-links`) and ships the writeups themselves
#     -- roughly 60 MB on top of the 130 MB of figures.
#
# Usage:  scripts/monitoring/deploy_monitor.sh [--dry-run] [source-dir]
set -euo pipefail

# A silent dry run is useless, so it also itemizes: without -i, rsync prints
# nothing at all under --dry-run and reads as "no work to do".
DRY=""
if [[ "${1:-}" == "--dry-run" ]]; then DRY="--dry-run -i --stats"; shift; fi

SRC="${1:-/orange/adamginsburg/web/public/jwst-gc}"
HOST="${MONITOR_WEB_HOST:-starformation}"
DEST="${MONITOR_WEB_DIR:-/h/cnswww-starformation.astro/starformation.astro.ufl.edu/htdocs/jwst-gc/monitor}"
URL="${MONITOR_WEB_URL:-https://starformation.astro.ufl.edu/jwst-gc/monitor/}"

case "$DEST" in
  */monitor|*/monitor/) ;;
  *)
    echo "refusing to sync to '$DEST': the destination must end in /monitor." >&2
    echo "htdocs/jwst-gc/index.html is the public data-release page." >&2
    exit 2 ;;
esac

# Exit codes are read by refresh_monitor.sh, which treats 1 as "the archive has
# failing runs" -- a monitor finding, not a job failure. Nothing here is that,
# so every failure of this script exits >= 2 and is loud.
if [[ ! -f "$SRC/index.html" ]]; then
  echo "no index.html in $SRC -- run the monitor with --publish-dir first" >&2
  exit 3
fi

ssh "$HOST" "mkdir -p '$DEST'"

# --copy-links: dereference the diagnostics-* symlinks (see above).
# --delete: the destination holds nothing but this sync's output, so pruning
#           removed fields and stale figures there is safe.
rsync -rlptz --copy-links --delete --human-readable $DRY \
      "$SRC/" "$HOST:$DEST/"
rc=$?

# 23 is "some files could not be transferred", which --copy-links returns for a
# single dangling symlink -- one missing writeup directory would otherwise fail
# the whole refresh. Report it, keep the pages that did land, and do not make it
# the caller's failure.
if [[ $rc -eq 23 ]]; then
  echo "WARNING: some files were not transferred (rsync 23), most likely a" \
       "dangling diagnostics symlink. The pages themselves are deployed." >&2
  rc=0
fi
[[ $rc -ne 0 ]] && exit $rc

if [[ -z "$DRY" ]]; then
  ssh "$HOST" "chmod -R a+rX '$DEST'"
  echo "deployed -> $URL"
fi
exit 0
