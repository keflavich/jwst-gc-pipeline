#!/bin/bash
# Rebuild the pipeline monitor and refresh the served copy.
#
# Deployed via `scrontab` (HiPerGator has no per-user crond on the login nodes);
# see jwst_gc_pipeline/monitoring/UPDATING.md for the crontab line.
#
# The field scan and the probe-cutout scan run in SEQUENCE, in one job, because
# they write into the same output directory -- two overlapping runs would
# interleave their writes.
set -uo pipefail

REPO=${REPO:-/orange/adamginsburg/repos/jwst-gc-pipeline}
OUTDIR=${OUTDIR:-/orange/adamginsburg/jwst/monitor}
PUBDIR=${PUBDIR:-/orange/adamginsburg/web/public/jwst-gc}
PYTHON=${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
CUTOUT_LABEL=${CUTOUT_LABEL:-monitor5as}
#: The survey whose footprints the sky view draws.  Set FOOTPRINTS=0 to skip
#: the rebuild (offline runs, or a deliberately pinned footprints.json).
FOOTPRINT_PROGRAM=${FOOTPRINT_PROGRAM:-10678}
FOOTPRINTS=${FOOTPRINTS:-1}

# Pin a worktree/branch by setting PIPE_ROOT (prepended to PYTHONPATH).
PIPE_ROOT=${PIPE_ROOT:-$REPO}
export PYTHONPATH="$PIPE_ROOT:${PYTHONPATH:-}"

cd "$REPO" || exit 1

echo "MONITOR refresh start: $(date -Is)  outdir=$OUTDIR pub=$PUBDIR"

# ---- refuse to PUBLISH from a checkout that is behind its remote ----------
#
# The monitor reverted to an old page three times over two days, each time
# hours after the fix had merged.  The cause was not the code: this cron runs
# with REPO pointing at a THIRD checkout (`jwst-gc-pipeline-schedule`), which
# sat on a feature branch 599 commits behind main with MONITOR_DEPLOY=1.  Every
# hour it regenerated a months-old page and published it over the current one.
# Pulling the other two checkouts, which is what I did twice, changed nothing.
#
# Generating from a stale tree is harmless; PUBLISHING from one is what
# overwrites good pages, so that is what this refuses.  A fetch failure is not
# staleness and must not block the run -- the check is skipped and says so.
#
# ALLOW_STALE_CHECKOUT=1 overrides, for a deliberate run from a branch.
# FIRST, try to make the checkout current, because refusing to publish is not a
# fix on its own: main takes several commits a day, nothing fast-forwards this
# checkout on a schedule, so a distance-based gate reaches `behind > 0` within a
# day and then freezes the page -- announced by one stderr line in a cron log,
# with the job still reporting success.  "The monitor silently stopped
# publishing" is the same class of failure as "the monitor silently published an
# old page", which is what this is meant to end.
#
# `--ff-only`, and only with a clean tree: a checkout carrying real work is
# never touched.  It simply fails here and falls through to the guard below,
# which is what the guard is for.
stale_checkout=0
if [ "${ALLOW_STALE_CHECKOUT:-0}" != "1" ]; then
    if git -C "$REPO" fetch --quiet origin 2>/dev/null \
       && git -C "$REPO" diff --quiet 2>/dev/null \
       && git -C "$REPO" diff --cached --quiet 2>/dev/null; then
        if git -C "$REPO" merge --ff-only --quiet origin/main 2>/dev/null; then
            :
        else
            echo "NOTE: $REPO could not be fast-forwarded to origin/main" \
                 "(branch $(git -C "$REPO" rev-parse --abbrev-ref HEAD));" \
                 "checking whether it is stale" >&2
        fi
    fi
    if git -C "$REPO" fetch --quiet origin 2>/dev/null; then
        behind=$(git -C "$REPO" rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
        if [ "${behind:-0}" -gt 0 ]; then
            stale_checkout=1
            echo "WARNING: $REPO is $behind commit(s) behind origin/main" \
                 "($(git -C "$REPO" rev-parse --short HEAD), branch" \
                 "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)) --" \
                 "generating but NOT publishing, because publishing from a" \
                 "stale checkout overwrites the live page with an old one." \
                 "Fast-forward it, or set ALLOW_STALE_CHECKOUT=1 to override." >&2
        fi
    else
        echo "NOTE: could not fetch in $REPO; staleness not checked" >&2
    fi
fi

# The sky view's footprints, rebuilt BEFORE the pages that embed them.
#
# This step did not exist, and its absence was invisible: the pages regenerated
# every hour while `footprints.json` sat at whatever date somebody last ran the
# builder by hand.  The visit statuses in it -- executed / skipped / scheduled --
# come from a LIVE STScI fetch rather than from the APT, so they go stale on
# their own schedule.  Measured 2026-09-12: the served page said 3 executed
# while STScI said 12, having drifted 9 tiles in nine hours.
#
# No `--force`: the APT is cached and changes rarely, and the half that changes
# every few hours is the visit-status table, which is fetched live on every run
# regardless.  Re-downloading 1.1 MB hourly would buy nothing.
#
# Written to a temp file and moved into place.  The builder dumps JSON straight
# to `--out`, so one that dies mid-write leaves a TRUNCATED footprints.json --
# and the sky view parses it client-side, which makes that a broken map rather
# than a stale one.  A failed rebuild keeps the previous file instead.
footprints_rc=0
if [ "$FOOTPRINTS" = "1" ]; then
    fp_tmp="$OUTDIR/.footprints.json.$$"
    "$PYTHON" "$REPO/scripts/monitoring/build_footprints.py" \
        "$FOOTPRINT_PROGRAM" --out "$fp_tmp"
    footprints_rc=$?
    # A rebuild that lost the visit statuses is as much a bad file as a
    # truncated one, and it does NOT fail: `fetch_visit_status` returns {} on an
    # unresolvable host, a 404, or a page-shape change, `build` then exits 0 with
    # `status_counts={}` and a perfectly well-formed file.  Without this an STScI
    # hiccup replaces a good file hourly and the published page loses every
    # status and every link.  (Review of #852.)
    _n_status () {
        python3 -c "import json,sys
try:
    print(len(json.load(open(sys.argv[1])).get('status_counts') or {}))
except Exception:
    print(0)" "$1" 2>/dev/null || echo 0
    }
    fp_new=$(_n_status "$fp_tmp")
    fp_prev=$(_n_status "$OUTDIR/footprints.json")
    if [ "$footprints_rc" -eq 0 ] && [ -s "$fp_tmp" ] \
       && { [ "$fp_new" -gt 0 ] || [ "$fp_prev" -eq 0 ]; }; then
        mv -f "$fp_tmp" "$OUTDIR/footprints.json"
    elif [ "$footprints_rc" -eq 0 ] && [ -s "$fp_tmp" ]; then
        rm -f "$fp_tmp"
        echo "WARNING: rebuild carries no visit statuses (previous file has" \
             "$fp_prev) -- keeping the previous $OUTDIR/footprints.json" >&2
    else
        rm -f "$fp_tmp"
        echo "WARNING: footprints rebuild exited $footprints_rc -- keeping the" \
             "previous $OUTDIR/footprints.json" >&2
    fi
fi

# The field view.  Its exit status is 1 when any run is failing, which is a
# finding about the ARCHIVE, not a failure of this job -- so it is recorded and
# not propagated, or the scheduler would report a broken cron every hour.
"$PYTHON" -m jwst_gc_pipeline.monitoring \
    --outdir "$OUTDIR" --json "$OUTDIR/monitor.json" --publish-dir "$PUBDIR"
fields_rc=$?

# The probe-cutout view (a scan of <base>/cutouts/<label>/).
"$PYTHON" -m jwst_gc_pipeline.monitoring --cutout-label "$CUTOUT_LABEL" \
    --outdir "$OUTDIR" --json "$OUTDIR/monitor_${CUTOUT_LABEL}.json" \
    --publish-dir "$PUBDIR"
cutout_rc=$?

echo "MONITOR refresh done: $(date -Is)  fields_rc=$fields_rc" \
     "cutout_rc=$cutout_rc footprints_rc=$footprints_rc"

# Push to Apache.  OFF by default: this needs outbound ssh to the web host, and
# a SLURM compute node may not have it -- a cron that fails on the network every
# hour trains everyone to ignore it.  Turn on with MONITOR_DEPLOY=1 once you have
# confirmed `ssh starformation true` works from wherever this runs.
deploy_rc=0
if [ "${MONITOR_DEPLOY:-0}" = "1" ] && [ "$stale_checkout" = "1" ]; then
    # Non-zero so the SCHEDULER reports it.  Reaching here now means the
    # fast-forward above could not fix it -- a dirty or diverged checkout,
    # which needs a person -- so this is a real finding rather than routine
    # drift, and a silent skip is how the original bug survived two days.
    echo "MONITOR deploy SKIPPED: checkout is behind origin/main and could" \
         "not be fast-forwarded; the published page is NOT being updated" >&2
    deploy_rc=3
elif [ "${MONITOR_DEPLOY:-0}" = "1" ]; then
    "$REPO/scripts/monitoring/deploy_monitor.sh" "$PUBDIR"
    deploy_rc=$?
    echo "MONITOR deploy rc=$deploy_rc"
fi

# Fail the JOB only if the generator itself broke (rc >= 2) or could not run at
# all (rc 127/126).  rc 1 means "the archive has failing runs", which is the
# monitor working correctly.
# `footprints_rc` is deliberately absent from this loop.  It reports a network
# fetch of a third-party page; that failing is not a reason to stop publishing
# the pipeline's status, and the previous footprints.json is still served.  It
# is reported on the done line and warned about above.
#
# rc 1 from the GENERATOR means "the archive has failing runs", which is the
# monitor working correctly. Anything higher is the generator itself breaking.
for rc in "$fields_rc" "$cutout_rc"; do
    if [ "$rc" -gt 1 ]; then
        echo "FATAL: monitor generator exited $rc" >&2
        exit "$rc"
    fi
done

# The deploy has no such convention -- ANY non-zero is a real failure, and it
# gets its own message. Folding it into the loop above both swallowed its rc 1
# and, when it did trip, blamed the generator for a step it never ran.
if [ "$deploy_rc" -ne 0 ]; then
    echo "FATAL: deploy_monitor.sh exited $deploy_rc" >&2
    exit "$deploy_rc"
fi
exit 0
