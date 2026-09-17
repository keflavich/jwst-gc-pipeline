#!/bin/bash
# Daily: re-stage the offsets table, rebuild the release pages, deploy.
#
# The offsets table is NOT frozen at release time. Every merge stage re-measures
# the exposure-to-reference-frame tie and writes new rows, so the copy staged
# when a release was cut is stale within a day and gains rows for exposures that
# had none. This keeps the published table and the page's summary current.
#
# Installed via scrontab; see jwst_gc_pipeline/monitoring/UPDATING.md for the
# monitor's equivalent line and the rationale for scrontab over cron.
set -uo pipefail

REPO=${REPO:-/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline}
RELEASE_DIR=${RELEASE_DIR:-/orange/adamginsburg/jwst/releases/v1.8-2026.09/gc-treasury}
OFFSETS=${OFFSETS:-/orange/adamginsburg/jwst/gc-treasury/offsets/Offsets_JWST_Brick10678_consensus.csv}
SITE=${SITE:-/orange/adamginsburg/jwst/releases/site}
PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
# Every field with a page, so rebuilding does not drop the others' pages.
FIELDS=${FIELDS:-"arches brick cloudc cloudef_controlfield gc2211 m4 m92 ngc6397 quintuplet sgra sgrb2 sgrc sickle w51 wd1 wd2 gc-treasury"}
DEPLOY=${DEPLOY:-1}

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "OFFSETS refresh start: $(date -Is)  release=$RELEASE_DIR"

# Self-update, the same way the monitor's job does: --ff-only and only from a
# clean tree, so a checkout carrying real work is left alone rather than
# clobbered.
if [ -d "$REPO/.git" ]; then
    if [ -z "$(git -C "$REPO" status --porcelain)" ]; then
        git -C "$REPO" fetch --quiet origin 2>/dev/null
        if git -C "$REPO" merge --ff-only --quiet origin/main 2>/dev/null; then
            echo "  repo fast-forwarded to $(git -C "$REPO" rev-parse --short HEAD)"
        fi
    else
        echo "  repo has local changes; left at $(git -C "$REPO" rev-parse --short HEAD)"
    fi
fi

"$PYTHON" "$REPO/scripts/release/build_offsets_summary.py" \
    --release-dir "$RELEASE_DIR" --table "$OFFSETS"
summary_rc=$?

if [ $summary_rc -ne 0 ]; then
    # Rebuilding the page against a stale or absent summary would republish
    # yesterday's numbers under today's date, which is worse than not
    # rebuilding: the page's own claim is that it is refreshed daily.
    echo "OFFSETS refresh ABORTED: summary rc=$summary_rc, page not rebuilt" >&2
    exit $summary_rc
fi

"$PYTHON" "$REPO/scripts/release/make_webpage.py" --fields $FIELDS --out "$SITE"
page_rc=$?

deploy_rc=0
if [ $page_rc -eq 0 ] && [ "$DEPLOY" = "1" ]; then
    bash "$REPO/scripts/release/deploy_site.sh" "$SITE"
    deploy_rc=$?
fi

echo "OFFSETS refresh done: $(date -Is)  summary=$summary_rc page=$page_rc deploy=$deploy_rc"
exit $(( page_rc | deploy_rc ))
