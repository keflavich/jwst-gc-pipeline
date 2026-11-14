#!/bin/bash
# Daily: rebuild the sky viewer's derived data files and publish them.
#
# `jwst_gc_focal_plane.json` (the as-flown focal plane) and
# `jwst_gc_cat_guidestars.json` (the guide stars) are built from inputs that keep
# changing -- delivered exposure headers, MAST guide-star exposures, the
# footprints the monitor refreshes hourly -- and nothing rebuilt them, so the
# viewer showed the state of whenever somebody last ran the builders by hand.
#
# All or nothing.  Both files are built beside their published names, checked,
# and only then renamed into place, so a failed builder or a bad file leaves
# BOTH published copies as they were.  The check (`viewer_data.py check`)
# refuses an empty or unparseable file and one whose counts have shrunk against
# the published copy; `n_as_flown` is in that list because the pointing count
# is complete while the as-flown attitudes are stale.  ALLOW_SHRINK=1 overrides.
#
# The guide-star builder reads MAST and needs MAST_API_TOKEN for proprietary
# exposures.  If MAST fails the run aborts: `--source frames` cannot see the
# visits that never flew, so it is not an automatic fallback.
#
# Installed via scrontab, after the monitor's hourly footprints rebuild; see
# jwst_gc_pipeline/monitoring/UPDATING.md.
#
#   DRY_RUN=1      build and check, publish nothing, keep the staged files
#   DEPLOY=0       update the data.rc docroot only, skip the starformation copy
#   EXPOSURES=...  delivered frames to read (default: the newest release)
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DOCROOT=${DOCROOT:-/orange/adamginsburg/web/public/avm_images}
FOOTPRINTS=${FOOTPRINTS:-/orange/adamginsburg/jwst/monitor/footprints.json}
PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
# Runs the check and summary helpers; separate from PYTHON so the builders can
# be replaced without replacing the check.
CHECK_PYTHON=${CHECK_PYTHON:-$PYTHON}
RSYNC=${RSYNC:-rsync}
DEPLOY=${DEPLOY:-1}
DRY_RUN=${DRY_RUN:-0}
ALLOW_SHRINK=${ALLOW_SHRINK:-0}

FOCAL=jwst_gc_focal_plane.json
GUIDE=jwst_gc_cat_guidestars.json
SCRIPTS="$REPO/scripts/monitoring"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "VIEWER DATA refresh start: $(date -Is)  docroot=$DOCROOT"

# ---- refuse to PUBLISH from a checkout that is behind its remote ----------
# The same guard as refresh_monitor.sh: generating from an old builder is
# harmless, publishing its output over a current file is not.  Make the
# checkout current first (--ff-only, clean tree only); a fetch failure is not
# staleness and does not block the run.  ALLOW_STALE_CHECKOUT=1 overrides.
if [ "${ALLOW_STALE_CHECKOUT:-0}" != "1" ] && [ -d "$REPO/.git" -o -f "$REPO/.git" ]; then
    if git -C "$REPO" fetch --quiet origin 2>/dev/null \
       && git -C "$REPO" diff --quiet 2>/dev/null \
       && git -C "$REPO" diff --cached --quiet 2>/dev/null; then
        git -C "$REPO" merge --ff-only --quiet origin/main 2>/dev/null || true
    fi
    if git -C "$REPO" fetch --quiet origin 2>/dev/null; then
        behind=$(git -C "$REPO" rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
        if [ "${behind:-0}" -gt 0 ]; then
            echo "VIEWER DATA refresh ABORTED: $REPO is $behind commit(s) behind" \
                 "origin/main and could not be fast-forwarded; the published" \
                 "files are NOT being updated. Fast-forward it, or set" \
                 "ALLOW_STALE_CHECKOUT=1 to override." >&2
            exit 3
        fi
    else
        echo "NOTE: could not fetch in $REPO; staleness not checked" >&2
    fi
fi

mkdir -p "$DOCROOT" || exit 1
focal_tmp="$DOCROOT/.$FOCAL.$$"
guide_tmp="$DOCROOT/.$GUIDE.$$"
if [ "$DRY_RUN" != "1" ]; then
    trap 'rm -f "$focal_tmp" "$guide_tmp"' EXIT
fi

extra=()
[ -n "${EXPOSURES:-}" ] && extra=(--exposures "$EXPOSURES")

"$PYTHON" "$SCRIPTS/build_focal_plane.py" --footprints "$FOOTPRINTS" \
    --out "$focal_tmp" ${extra[@]+"${extra[@]}"}
focal_rc=$?
if [ $focal_rc -ne 0 ]; then
    echo "VIEWER DATA refresh ABORTED: build_focal_plane.py rc=$focal_rc;" \
         "published files untouched" >&2
    exit $focal_rc
fi

# `--source mast` is the builder's default; the exposures option is not passed
# because the frames source is never used here.
"$PYTHON" "$SCRIPTS/build_guidestars.py" --footprints "$FOOTPRINTS" \
    --out "$guide_tmp"
guide_rc=$?
if [ $guide_rc -ne 0 ]; then
    echo "VIEWER DATA refresh ABORTED: build_guidestars.py rc=$guide_rc;" \
         "published files untouched" >&2
    exit $guide_rc
fi

shrink=()
[ "$ALLOW_SHRINK" = "1" ] && shrink=(--allow-shrink)
bad=0
"$CHECK_PYTHON" "$SCRIPTS/viewer_data.py" check focal "$focal_tmp" \
    "$DOCROOT/$FOCAL" ${shrink[@]+"${shrink[@]}"} || bad=1
"$CHECK_PYTHON" "$SCRIPTS/viewer_data.py" check guide "$guide_tmp" \
    "$DOCROOT/$GUIDE" ${shrink[@]+"${shrink[@]}"} || bad=1
if [ $bad -ne 0 ]; then
    echo "VIEWER DATA refresh ABORTED: a rebuilt file failed its check;" \
         "published files untouched" >&2
    exit 4
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "VIEWER DATA dry run: nothing published; staged $focal_tmp $guide_tmp"
    "$CHECK_PYTHON" "$SCRIPTS/viewer_data.py" summary "$focal_tmp" "$guide_tmp"
    exit 0
fi

# Same-directory renames, so each file appears whole.  The two renames are not
# one atomic step, but both files have already passed their checks.
mv -f "$focal_tmp" "$DOCROOT/$FOCAL" && mv -f "$guide_tmp" "$DOCROOT/$GUIDE" || {
    echo "VIEWER DATA refresh FAILED while moving the files into $DOCROOT" >&2
    exit 5
}
chmod a+r "$DOCROOT/$FOCAL" "$DOCROOT/$GUIDE" 2>/dev/null

deploy_rc=0
if [ "$DEPLOY" = "1" ]; then
    if [ -z "${WEB_HOST:-}" ] || [ -z "${WEB_DIR:-}" ]; then
        # The host and directory live in publish_hips_layers.py, so the two
        # publishers cannot drift apart.
        target=$("$CHECK_PYTHON" "$SCRIPTS/viewer_data.py" web-target) || target=
        WEB_HOST=$(printf '%s\n' "$target" | sed -n 1p)
        WEB_DIR=$(printf '%s\n' "$target" | sed -n 2p)
    fi
    if [ -z "$WEB_HOST" ] || [ -z "$WEB_DIR" ]; then
        echo "VIEWER DATA deploy FAILED: no web host/dir" >&2
        deploy_rc=6
    else
        "$RSYNC" -t "$DOCROOT/$FOCAL" "$DOCROOT/$GUIDE" "$WEB_HOST:$WEB_DIR/"
        deploy_rc=$?
        if [ $deploy_rc -ne 0 ]; then
            echo "VIEWER DATA deploy FAILED: rsync rc=$deploy_rc; the docroot" \
                 "copy is new, the $WEB_HOST copy is not" >&2
        fi
    fi
fi

echo "VIEWER DATA refresh done: $(date -Is)  " \
     "$("$CHECK_PYTHON" "$SCRIPTS/viewer_data.py" summary \
        "$DOCROOT/$FOCAL" "$DOCROOT/$GUIDE")  deploy=$deploy_rc"
exit $deploy_rc
