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
#SCRON --cpus-per-task=2 --mem=8gb --time=00:50:00
#SCRON --job-name=gc-monitor-refresh
#SCRON --output=/orange/adamginsburg/jwst/logs/gc-monitor-%j.out
0 * * * * MONITOR_DEPLOY=1 REPO=<checkout> PATH=<python-env>/bin:$PATH bash <checkout>/scripts/monitoring/refresh_monitor.sh
```

**`REPO` is not optional and its value is not obvious.** `refresh_monitor.sh`
does `cd "$REPO"` and then `python -m jwst_gc_pipeline.monitoring`, and `-m`
puts the working directory first on `sys.path` — so the checkout named by
`REPO` is the code that runs, whatever `PIPE_ROOT` says. The installed job on
2026-08-15 therefore names the checkout that carries this feature, not
`/orange/adamginsburg/repos/jwst-gc-pipeline`, which is a separate clone that
sits on `main` and does not update itself. Point `REPO` at whichever checkout
you intend to serve from, and remember that nothing pulls it.

Read the installed job rather than trusting this file:

```bash
scrontab -l
```

`MONITOR_DEPLOY=1` is on in that line because the two things it needs were
**measured** on a compute node (`srun` onto an existing allocation, 2026-08-15):

```
c0712a-s1.ufhpc
https stsci: 200
ssh starformation: OK
```

Outbound HTTPS reaches STScI (which the schedule panel needs) and outbound ssh
reaches the web host (which the deploy needs). Without the deploy the refresh
rebuilds the page into `/orange` and **nothing on HiPerGator is web-served**, so
the served copy at starformation.astro.ufl.edu stays at whatever the last manual
deploy left -- which is exactly how it came to read *Generated 2026-08-05* on
2026-08-15. The default stays 0 for anyone whose node cannot do this; check
before turning it on, as the note in `refresh_monitor.sh` says.

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

## Getting it in front of a browser

There are **two** web servers.

**Dev** — `/orange/adamginsburg/web/public/` is served directly at
`https://data.rc.ufl.edu/pub/adamginsburg/` (the URL says `/pub/`, not
`/public/`). `--publish-dir` alone puts the monitor at

> https://data.rc.ufl.edu/pub/adamginsburg/jwst-gc/

with no further step. Use it for work in progress.

**Deployment** — `starformation.astro.ufl.edu` is a separate host with its own
filesystem, docroot
`/h/cnswww-starformation.astro/starformation.astro.ufl.edu/htdocs`. Getting there
is a copy rather than a link:

```bash
scripts/monitoring/deploy_monitor.sh --dry-run     # see what would move
scripts/monitoring/deploy_monitor.sh               # ~190 MB the first time
```

That serves the aggregate page at

> **https://starformation.astro.ufl.edu/jwst-gc/monitor/**

with per-field pages at `.../monitor/fields/<field>.html`.

Two constraints the script encodes, both easy to get wrong by hand:

* `htdocs/jwst-gc/index.html` is the **public data-release landing page**. The
  monitor goes in the `monitor/` subdirectory; rsyncing to `jwst-gc/` itself
  replaces the release page with an internal status report. The script refuses a
  destination that does not end in `/monitor`.
* the `diagnostics-<field>` entries are symlinks into `/orange` and `/blue`,
  which do not exist on the web host. The sync dereferences them, which is what
  makes the click-through to the diagnostic figures work off-site.

**The release site can delete the monitor.** `htdocs/jwst-gc/` is written by two
unrelated generators — the release pages by `scripts/release/make_webpage.py`,
`monitor/` by the script above — and `releases/site/` contains no `monitor/`. A
release sync carrying `--delete` therefore sees the whole 194 MB monitor tree as
extraneous and removes it. That is what took the URL offline on **2026-08-06**:
the release site was republished at 09:28, the monitor 404'd until 14:45.
`scripts/release/deploy_site.sh` now runs that sync with
`--filter='protect monitor/'` and fails loudly if the monitor is missing
afterwards; use it rather than rsyncing `releases/site/` by hand. To put the
monitor back after a wipe, re-run `deploy_monitor.sh` — the publish directory on
HiPerGator is the source of truth and survives it.

**A symlink is not servable.** Apache returns **403** for a symlink whose target
leaves the served tree, which on the dev server means anything pointing into
`/blue`. `publish()` therefore *copies* figures it cannot hardlink, but the
`diagnostics-<field>` entries are directory symlinks and stay symlinks — so on
the **dev** URL those writeup links 403 while every figure resolves. They work on
the deployment URL, because the rsync dereferences them. If the dev copy needs
them, copy the writeup directories in by hand.

The published page is public and unauthenticated. It quotes job-log excerpts,
absolute paths, and QA verdicts; nothing there is proprietary, but it is not
linked from the release index either.

## If the published copy stops updating

The files in `--publish-dir` are **hardlinks**, and they track regeneration only
because `render.write_html` rewrites in place, preserving the inode. If someone
changes that to an atomic write (temp + rename), the inode changes and the old
links freeze at stale content *while still looking live*. Re-running with
`--publish-dir` repairs it — which is why the refresh script always passes it,
even though it is usually a no-op.

To check by hand that the link is still shared:

```bash
stat -c '%i %n' /orange/adamginsburg/jwst/monitor/monitor.html \
                /orange/adamginsburg/web/public/jwst-gc/monitor.html
```

Two identical inode numbers means the published page is the generated page. The
copy on the web host is a *copy*: it is only as fresh as the last
`deploy_monitor.sh`, so the refresh cron should run that too.

## The observing schedule panel

The page also reads the **published STScI weekly observing schedule** and shows
program 10678's scheduled visits (`jwst_gc_pipeline/monitoring/schedule.py`).
This is what lets the monitor answer *when does the Treasury start* rather than
only *what is on disk*; the first 10678 visit is `10678:1:1` / `GC_1` at
2026-08-17T08:10:36Z.

```bash
# default: program 10678, live fetch, cached under <outdir>/schedule/
python -m jwst_gc_pipeline.monitoring --outdir ... --publish-dir ...

python -m jwst_gc_pipeline.monitoring --schedule-program 2221   # another program
python -m jwst_gc_pipeline.monitoring --schedule-offline        # cache only
python -m jwst_gc_pipeline.monitoring --schedule-program ''     # no panel
```

Reports are cached under `<outdir>/schedule/` and the parsed result is written
to `schedule.json` beside the page, so the panel survives a network failure and
says on the page that it did. A schedule is a **plan** -- STScI's own note is
that executed observations can differ from those scheduled -- so the panel never
claims a visit happened; a past visit with no data on disk reads *not seen*.
