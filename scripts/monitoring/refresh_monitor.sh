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

# Pin a worktree/branch by setting PIPE_ROOT (prepended to PYTHONPATH).
PIPE_ROOT=${PIPE_ROOT:-$REPO}
export PYTHONPATH="$PIPE_ROOT:${PYTHONPATH:-}"

cd "$REPO" || exit 1

echo "MONITOR refresh start: $(date -Is)  outdir=$OUTDIR pub=$PUBDIR"

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

echo "MONITOR refresh done: $(date -Is)  fields_rc=$fields_rc cutout_rc=$cutout_rc"

# Push to Apache.  OFF by default: this needs outbound ssh to the web host, and
# a SLURM compute node may not have it -- a cron that fails on the network every
# hour trains everyone to ignore it.  Turn on with MONITOR_DEPLOY=1 once you have
# confirmed `ssh starformation true` works from wherever this runs.
deploy_rc=0
if [ "${MONITOR_DEPLOY:-0}" = "1" ]; then
    "$REPO/scripts/monitoring/deploy_monitor.sh" "$PUBDIR"
    deploy_rc=$?
    echo "MONITOR deploy rc=$deploy_rc"
fi

# Fail the JOB only if the generator itself broke (rc >= 2) or could not run at
# all (rc 127/126).  rc 1 means "the archive has failing runs", which is the
# monitor working correctly.
for rc in "$fields_rc" "$cutout_rc" "$deploy_rc"; do
    if [ "$rc" -gt 1 ]; then
        echo "FATAL: monitor generator exited $rc" >&2
        exit "$rc"
    fi
done
exit 0
