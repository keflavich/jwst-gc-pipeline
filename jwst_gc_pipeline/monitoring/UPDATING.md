# Keeping the monitor up to date

The monitor is a **snapshot**: it reads the tree and the queue at the moment it
runs. Nothing about the page updates itself, so the page is only as current as
the last run.

## By hand

```bash
python -m jwst_gc_pipeline.monitoring \
    --outdir      /orange/adamginsburg/jwst/monitor \
    --json        /orange/adamginsburg/jwst/monitor/monitor.json \
    --publish-dir /orange/adamginsburg/web/public/jwst-gc
```

and the probe-cutout view (a separate scan of the `cutouts/monitor5as/` subtree):

```bash
python -m jwst_gc_pipeline.monitoring --cutout-label monitor5as \
    --outdir      /orange/adamginsburg/jwst/monitor \
    --publish-dir /orange/adamginsburg/web/public/jwst-gc
```

Exit status is **1 when any run is failing**, so a scheduler can act on it.

## On a schedule

HiPerGator has no per-user crond on the login nodes; `scrontab` is the supported
path (data-qa already deploys its MAST monitor that way — see
`data-qa/docs/scrontab.example`). Install with `scrontab -e`:

```cron
# Rebuild the JWST-GC pipeline monitor and refresh the served copy.
# Hourly is the useful cadence: a cataloging stage takes hours, so anything
# faster mostly re-reads an unchanged tree, and anything slower makes the page
# stale enough that people stop trusting it.
#SCRON --account=astronomy-dept --qos=astronomy-dept-b --partition=hpg-default
#SCRON --cpus-per-task=2 --mem=8gb --time=00:40:00
#SCRON --job-name=gc-monitor-refresh
#SCRON --output=/orange/adamginsburg/jwst/logs/gc-monitor-%j.out
0 * * * * /orange/adamginsburg/repos/jwst-gc-pipeline/scripts/monitoring/refresh_monitor.sh
```

`scrontab` runs the line as a SLURM job, so it inherits the compute node's view
of `/orange` and `/blue` and does not need a login shell.

### Walltime

A cold scan of the whole archive took **14 m 17 s wall / 6.4 s user** when
measured on a cold NFS metadata cache — essentially all of it directory-listing
latency, not computation. A warm one is ~25 s. **Do not set the interval below
the cold-start cost**; hourly leaves ample margin. Earlier drafts of this file
said 5–10 minutes, which was a warm-ish measurement (NFS metadata, plus a
bounded sample of FITS headers per filter); a warm one takes **~25 s**. The
40-minute walltime above is deliberate headroom — a scan killed part-way through
writing would leave a truncated page, and because the served copies are
hardlinks they would show the truncation too.

### Two scans, one schedule

`refresh_monitor.sh` runs the field scan and the probe-cutout scan in sequence.
Keep them in one job rather than two cron lines: they write into the same output
directory, and two overlapping runs can interleave their writes.

## Refreshing the probe cutouts themselves

The `monitor5as` cutouts are *data*, not part of the page — they only change when
you re-run them:

```bash
python -m jwst_gc_pipeline.monitoring probe            # show the matrix, submit nothing
python -m jwst_gc_pipeline.monitoring probe --execute  # one ~5" job per field
```

Worth re-running after a pipeline change that could affect cataloging, since a
5-arcsec run exercises the whole per-exposure chain (m12→m6) per field in minutes.
Not worth putting on a timer — nothing changes between runs unless the code does.

## Refreshing the footprints

The sky view reads `footprints.json`, which is generated from the APT file, not
scanned from disk — so the hourly refresh does **not** update it:

```bash
python scripts/monitoring/build_footprints.py 10678 \
    --out /orange/adamginsburg/jwst/monitor/footprints.json
```

Worth re-running when the program's plan changes, and to pick up visits as they
execute (the observed layer comes from the APT visit status). Not worth putting
on a timer — a Flight Ready program's pointings do not move between runs.

## What the schedule does NOT do

* It does not re-run the astrometry paper's validation. That has its own SLURM
  dependency after re-cataloging; the monitor reports its age and flags a verdict
  whose catalogs have since been rewritten.
* It does not regenerate the survey footprints (see above).
* It does not re-measure any astrometry. Every number comes from records the
  pipeline already wrote.

## If the served copy stops updating

The published files are **hardlinks**, and they track regeneration only because
`render.write_html` rewrites in place, preserving the inode. If someone changes
that to an atomic write (temp + rename), the inode changes and the old links
freeze at stale content *while still looking live*. Re-running with
`--publish-dir` repairs it — which is why the refresh script always passes it,
even though it is usually a no-op.

To check by hand that the link is still shared:

```bash
stat -c '%i %n' /orange/adamginsburg/jwst/monitor/monitor.html \
                /orange/adamginsburg/web/public/jwst-gc/monitor.html
```

Two identical inode numbers means the served page is the generated page.
