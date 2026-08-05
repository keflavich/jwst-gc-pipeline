# Pipeline run monitoring

Answers three questions about a `jwst-gc-pipeline` run, from the products on disk
plus the SLURM queue:

* **how far has it got** — the stage ladder, reduction through the cataloging merges;
* **is what it produced coherent** — provenance tags, m2 astrometry checkpoints;
* **what is in flight** — live jobs and the error signatures in their logs.

It works unchanged on a full field and on a small `--cutout-region` probe run,
because a cutout writes the same tree one level down under `cutouts/<label>/`.

```bash
python -m jwst_gc_pipeline.monitoring                          # every field
python -m jwst_gc_pipeline.monitoring --target brick           # one field
python -m jwst_gc_pipeline.monitoring --cutout-label monitor5as   # the probe runs
python -m jwst_gc_pipeline.monitoring probe                    # dry-run the probe matrix
python -m jwst_gc_pipeline.monitoring probe --execute          # submit it
python -m jwst_gc_pipeline.monitoring \
    --publish-dir /orange/adamginsburg/web/public/jwst-gc      # serve it
```

Writes `<outdir>/monitor.html` (aggregate), `<outdir>/monitor_fragment.html` (the
same page without the document wrapper, for publishing), `<outdir>/fields/<target>.html`
(one per field), and optionally a JSON snapshot. Exit status is 1 when any run is
failing, so a cron invocation is actionable.

A full-archive scan is ~30 s.

## Modules

| module | responsibility |
|---|---|
| `scan.py` | on-disk product taxonomy; observation-safe, symlink-safe |
| `jobs.py` | `squeue`/`sacct` state, job-name parsing, log error signatures |
| `checks.py` | verdicts; every threshold **imported** from its enforcing module |
| `probe.py` | plan and submit the ~5″ cutout runs the monitor is exercised on |
| `paper.py` | reads the astrometry paper's own machine-readable verdicts |
| `render.py` | self-contained HTML, light + dark |
| `report.py` | glue and file output |

## Keeping it current

The page is a snapshot — see [UPDATING.md](UPDATING.md) for the `scrontab`
deployment and `scripts/monitoring/refresh_monitor.sh`. A cold full-archive scan
is 5–10 minutes; a warm one is ~25 s.

## Reading a red line

Every finding carries three things, because a number alone is not actionable:

* **what the number means** and which gate it is measured against;
* **cause** — how a run gets into this state, and what to do about it. Where the
  evidence distinguishes two causes, it says which: misalignment confined to one
  detector reads as a detector-local defect, spread across all of them reads as
  the frame having moved; bad tiles confined to the mosaic edge read as thin
  coverage, in the interior as a distortion problem.
* **evidence** behind a *what is affected, and why* disclosure — the affected
  rows themselves (which exposures, which detector, by how much, at what
  contrast), plus drawn diagnostics.

Two diagnostics are **drawn by the monitor** from numbers it already read, as
inline SVG — so they exist for every field rather than only where somebody made
figures, and they work in a published artifact where an external request would be
blocked:

* a **per-tile residual map** of the mosaic, worst cell circled, over-tolerance
  cells outlined — this is what separates "one bad corner" from "a gradient";
* a **per-exposure offset quiver** coloured by detector, with flagged exposures
  solid — one colour pointing away means one chip, a general fan means the frame.

Each field also carries a **diagnostic writeup** (`<field>/diagnostic_writeup/`):
a compiled `main.pdf` plus a fixed figure set D1–D8. Because the set is the same
for every field, a finding links to the figure that actually shows it rather than
to a directory listing — misalignment → D2 (internal repeatability), per-tile and
WCS-provenance findings → D3 (absolute tie), saturation → D8 (colour diagrams).
Links are written as `diagnostics-<field>/figures/D3_....pdf`, which is the
symlink name the served copy carries, so they resolve with no extra publishing.

Diagnostics that already exist on disk (`astrometry_diag/`, `audit_plots/`,
`figures/`, `pngs/`, and the paper's `outputs/*/figures/`) are **linked**, not
embedded — they are megabytes. `--publish-dir` hardlinks them into `figures/` so
the relative links resolve on the served copy. Coverage is uneven and is not
hidden: brick has ~1250 figures, gc2211 has none.

## Sky view

A footprint map showing where the survey plans to observe and what it has.
Footprints come from APT program **10678** — *The JWST/NIRCam Legacy Survey of
the Galactic Center* (Schoedel, Cycle 5, Flight Ready): 139 pointings, NIRCam
prime with MIRI coordinated parallels, ~64′×75′ over the GC.

```bash
python scripts/monitoring/build_footprints.py 10678 --out <outdir>/footprints.json
```

downloads the APT file, parses the targets and `OrientRange`, and projects
aperture corners through the observatory attitude with `pysiaf`. Re-run it when
the program changes — the observed layer fills in from the APT visit status, so
it updates as visits execute.

Three JWST layers, each its own colour and toggle:

| layer | why separate |
|---|---|
| planned NIRCam | the prime — **one outline per pointing, covering its whole primary dither** |
| planned MIRI ∥ | lands ~7.5′ from the prime, so it covers **different sky** — one colour would imply contiguous coverage the survey does not have |
| observed | read from the APT visit status; empty today, rendered as "none yet" with its toggle **on** so the first executed visit appears without anyone enabling it |

ACES and the Roman GBTDS spring/autumn tiles are available as context but **off
by default** — this is a JWST monitor. ACES comes from its real ~655-vertex
outline (`ACES_footprint.reg`, Galactic, converted to ICRS at build time), not a
rectangle, because a rectangle would misstate which pointings it covers.

There is no "target area" layer. That polygon is the Roman file's
`target_area`, which is the **JWST** target area — it was offered under the
Roman heading, where it read as Roman context and was simply mislabelled.

### The dither pattern is measured, not looked up

Each NIRCam outline is the bounding box of one pointing's whole primary dither,
built in the anchor aperture's ideal frame (where the offsets are defined) and
projected out through the attitude. Eight per-detector boxes describe a single
dither position, and the gaps between them are exactly what FULLBOX exists to
fill — drawing them separately advertises holes the observation does not leave.
1112 polygons become 139.

JDox publishes the dither tables only as a download behind a JavaScript-rendered
page, and **pysiaf does not carry them at all** (`grep -r FULLBOX` over the
package returns nothing). So `DITHER_HALF_EXTENT` comes from flight data:

* program **2221**'s APT file records `PrimaryDitherType=FULLBOX` with
  `PrimaryDithers=6TIGHT` — the same pattern 10678 uses (5365, by contrast, is
  plain `FULLBOX/6`, and its pattern is a different size);
* its exposures carry 24 `XOFFSET`/`YOFFSET` positions, 6 primary × 4 subpixel;
* the same subpixel set repeats at each primary, so differencing corresponding
  exposures recovers the primary grid exactly: **3 × 24.3″ in X, 2 × 10.8″ in
  Y** → half-extent 24.3″ × 5.4″.

The result checks itself: in that frame the two NIRCam modules are 41.3″ apart
along X with ~5″ gaps between detectors, and a pattern spanning 48.6″ × 10.8″
covers both — which is what FULLBOX is for. The drawn box grows 308.3″ → 356.9″
and 134.7″ → 145.5″; MIRI's 83″ × 118″ illuminated field becomes 131.9″ × 129.4″.

A pattern not in `DITHER_HALF_EXTENT` is **not** treated as zero: those
pointings fall back to per-detector outlines and the builder says so on stderr,
because silently drawing one dither position under-states the covered area by
the width of a module gap.

Two things the geometry does deliberately: the attitude is anchored on
`NRCALL_FULL` (the aperture APT's target coordinate refers to) and the MIRI
parallel is projected through that *same* attitude, which is what puts it where
the parallel actually observes; and `PA_V3` is a **range** (79–95°) until each
visit is scheduled, so the midpoint is used and the range is stated on the page.

The panel draws the footprints **twice**, by two routes with different
dependencies:

* a **static map** — an inline SVG on a TAN tangent plane, built at render time
  by `skyview.static_map`. No script, no fetch, no tiles: ~100 kB of geometry
  that is part of the HTML, so it renders in a published artifact, from a
  `file://` path, and offline. Per-pointing hover, drag to pan,
  click-then-scroll to zoom, adaptive angular scale bar.
* the **interactive view** — Aladin Lite over HiPS imagery, loaded on request
  from a same-origin copy `publish()` links in. No third-party CDN, and the
  1.8 MB script is not paid for by readers who never ask for it.

Drawn in **Galactic coordinates** (`skyview.DEFAULT_FRAME`), with an RA/Dec
graticule dashed over the l/b one. This is a Galactic-centre survey: in l/b the
tiled strip runs along a grid line rather than diagonally across one, and the
drawn field is 1000×557 instead of 1000×1232 — it fits the panel because it is
in the frame the survey was planned in. Without astropy the transform is
unavailable and the map falls back to ICRS rather than disappearing.

Every layer, Roman included, is drawn in the static map. Roman used to be
Aladin-only, which left its three toggles doing nothing at all until someone
loaded 1.8 MB of script — live-looking buttons that were inert. Roman's tiles
also lie outside the JWST bounding box that sets the home view, so the zoom-out
ceiling rises from 3× to 8× when they are present; otherwise you could enable a
layer you had no way to bring into view.

HiPS tiles are unavoidably remote, so the interactive route is what a strict
content-security policy blocks — a published artifact cannot run it, and neither
can a `file://` page. When it fails the static map stays up and a note beside it
says which part was unavailable. The layer toggles drive both routes, so a view
built on the static map survives the upgrade.

## Page structure

Three kinds of page, because one 2 MB scroll made the monitor's entry point its
slowest document:

| page | holds | does not hold |
|---|---|---|
| `monitor.html` | the sky map **first**, then one status card per observation, then the cutout summary | any field's detail |
| `fields/<target>.html` | that field's stage table, astrometry checkpoint, provenance, queue and findings | the sky map (~100 kB, identical on all 18) |
| `monitor_<label>.html` | the same split for a cutout label | — |

A card on the front page is a link to `fields/<target>.html#<anchor>`; the
anchor picks out the observation within the field. `--no-per-field` turns the
split off, and then the front page keeps its detail inline rather than linking
to pages that were never written.

## Publishing

**Two web servers**, and they want different things:

| | path | URL | how |
|---|---|---|---|
| dev | `/orange/adamginsburg/web/public/jwst-gc` | `https://data.rc.ufl.edu/pub/adamginsburg/jwst-gc/` | `--publish-dir`, served directly |
| deployment | `starformation:.../htdocs/jwst-gc/monitor` | `https://starformation.astro.ufl.edu/jwst-gc/monitor/` | `scripts/monitoring/deploy_monitor.sh` |

The dev path is *served as written* — the URL is `/pub/`, not `/public/`. The
deployment host is a separate machine with its own filesystem, so getting there
is an rsync; that script dereferences the `diagnostics-<field>` symlinks and
refuses any destination not ending in `/monitor`, because
`htdocs/jwst-gc/index.html` is the public data-release landing page. See
`UPDATING.md`.

`--publish-dir` hardlinks the generated pages into a web directory and points
`index.html` at the aggregate page. The `*_fragment.html` outputs are body
fragments for the artifact publisher — no doctype, charset or viewport — so they
are deliberately not served.

A hardlink rather than a copy because `render.write_html` rewrites a page **in
place** (`open(path, 'w')`), which keeps the inode: the served copy tracks every
regeneration with no second copy and no separate publish step.

Across filesystems — about half the figures live on `/blue` while the output is
on `/orange` — a hardlink is impossible and the fallback is a **copy, not a
symlink**. A symlink whose target leaves the served tree is not a file a web
server will hand out: Apache at data.rc.ufl.edu returns 403 for every one of
them, and it fails in the worst way, with the pages fine and only their figures
missing. An unchanged copy is left alone (`_copy_is_current`), so the hourly
refresh does not re-copy ~110 MB; that skip is deliberately restricted to the
cross-filesystem case, since skipping a same-filesystem relink is the stale-page
bug below.

Re-run it after every generation anyway. That is not redundant — it costs nothing
when the inode is unchanged, and it is what keeps the served copy correct if the
writer ever moves to an atomic write (write-temp + rename). An atomic write
replaces the inode, and the old hardlink would then sit frozen at whatever it
first pointed to: a stale dashboard that still looks live. `test_publish_relinks_
after_an_inode_replacing_write` pins that behaviour.

One caveat inherent to in-place rewriting: a reader who loads the page during the
write gets a partial document. The window is the few hundred milliseconds it takes
to write ~400 kB.

## The rules that shape it

These are not stylistic choices; each one is a bug the monitor must not have.

**Thresholds are imported, never copied.** `EXPOSURE_CONSENSUS_TOL_MAS`,
`CROSSFILTER_TOL_MAS`, `LOCAL_CELL_TOL_MAS`, `REFERENCE_AGREE_TOL_MAS`,
`REFERENCE_CROSSCHECK_GROSS_MAS`, `DEFAULT_MIN_CONTRAST` all come from
`visit_consensus` / `astrometry_checkpoint` / `astrometry_offsets`. A monitor
carrying its own copy of “5 mas” drifts away from the gate it claims to watch and
then reports green on a run the pipeline would have refused. The two monitor-owned numbers — `CONSENSUS_SCATTER_WARN_MAS` and
`PAPER_VERDICT_AGE_WARN_DAYS` — are each labelled in the page as monitor
heuristics, not pipeline gates. A test greps the renderer for numeric gate
literals, because asserting identity for a few imported constants does not stop
a new literal being typed inline.

`CROSSFILTER_TOL_MAS`, `STAGE_STABILITY_TOL_MAS` and `REFERENCE_AGREE_TOL_MAS`
are imported and used in message text but **no check reports the m7 cross-filter
gate yet** — the m7 checkpoint record is not read. That is a gap, not a claim.

**Nothing here measures an astrometric offset.** The m2 checkpoint already ran
the sanctioned offset-histogram machinery; the monitor reports its records. An
ad-hoc re-measurement would be a nearest-neighbour median against a dense
reference, which is banned outright (see `CLAUDE.md`, astrometry rule #1).

**Never glob a stage without pinning the observation.** One directory holds every
observation of a field. Where the pipeline itself writes a name with no `_o<obs>`
token — the per-filter merged catalogs — the ambiguity is **reported**
(`scope='ambiguous'`, hatched in the ladder) rather than resolved by guessing.
Ambiguity is claimed only for filters registered to more than one observation
(`shared_filters`): gc2211's five observations share their whole filter list and
ngc6334's two proposals share F200W/F470N, but brick's two observations use
disjoint filters, so flagging brick would bury the real cases.

**A checkpoint is a snapshot, not a live verdict.** At m2 a misalignment corrects
the offsets table and stops the run so the frames can be regenerated. If the
reduced frames are newer than the record, the fix has probably already landed —
that reads as a warning with the reason stated, not a permanent red.

**Registered ≠ globbed.** wd1 lists o001 and o003 but the reduction globs only
`001`; wd2 lists o003/o005 and globs `005`. Those observations have no products
by design and are shown as *not globbed*, not as a field with nothing done.

**Filters are split by instrument.** `fields.yaml` keeps one flat filter list per
proposal covering every instrument, so a NIRCam run must drop the MIRI filters
(via `naming.MIRI_FILTERS`) or grow permanently empty rows.

**Follow symlinks.** `brick`, `cloudc` and `wd1` under `/orange/adamginsburg/jwst`
point into `/blue` and `/orange/adamginsburg/westerlund`; an unresolved scan
reports the flagship fields as empty.

**Names are free, `stat` is not.** A single filter's `pipeline/` holds ~33,000
files. `os.listdir` costs ~0.05 s; statting them all costs ~5 s, and
`scandir` + `entry.is_file()` is the same cost again because NFS returns
`d_type` unknown. Counts come from names; `stat` is spent on a bounded sample for
timestamps, and a sampled timestamp is reported as such.

## The astrometry paper

The Brick astrometry paper (Overleaf `6a521006b63a11a7e0d80fa0`, checked out at
`<brick>/astrometry_paper`) already validates the release products:
`astrometry_paper/scripts/post_recat_validation.py` runs on a SLURM dependency after re-cataloging
and writes `outputs/<date>_postrecat/summary.json`, whose `problems` list is the
verdict, with every threshold pinned in the paper's `astrometry_paper/config.py`.

The monitor **reports that verdict; it does not re-implement it.** The paper's
gates (cross-filter vs anchor > 30 mas, p60/p90 mode flip > 10 mas,
degenerate-pair drift ≥ 0.10 mag) are computed there with the sanctioned
window-swept offset histogram over full catalogs — a second copy here would drift
from the published numbers, and recomputing would mean the monitor crossmatching
catalogs itself.

What the monitor adds is the question that script cannot ask about itself:

* **is the verdict still about the products on disk?** A catalog rewritten under a
  verdict makes a stale pass, which is worse than no verdict.
* **is a missing certifier being read as a pass?** Saturation continuity and
  degenerate-pair flatness are only recorded when a qualifying merged table was
  found; absent means UNKNOWN.
* **does the freshness guard bind?** `astrometry_paper/provenance.py::check_catalog_freshness` raises on
  a catalog older than `MIN_CATALOG_DATE`, so the analysis would refuse to run.

Problems are scoped to the run's programme — the paper validates both brick
programmes in one pass, and a 1182 failure is not a 2221 failure.

Three further checks come from the paper's WCS-provenance and per-tile sections
and need only headers or JSON already on disk:

* **LW filteroffset must match the module** (A→`…_0007.asdf`, B→`…_0008.asdf`). A
  swapped reference displaces every position by up to ~26 mas per module, ~52 mas
  A−B, *anti-symmetrically* — so a run mixing swapped and corrected frames
  manufactures an inter-module offset no reference tie can diagnose.
* **one CRDS context per field.** Frames calibrated against different contexts do
  not share one WCS solution.
* **the per-tile worst cell, not the tile count.** `measure_offset_grid` runs with
  no `max_off_mas`, and `astrometry_offsets` sets `off_ok=True` whenever that is
  `None` — so "36/36 tiles ok" means *36 tiles had a coherent peak*, not *36 tiles
  within tolerance*. Brick F182M reads 36/36 with a **29.1 mas** worst cell while
  its bulk tie is 0.7 mas. The monitor reports `worst_off_mas` against
  `LOCAL_CELL_TOL_MAS`.

## Probe cutouts

`probe.py` plans one ~5″ cutout cataloging run per field: `run_pipeline.resolve`
gives the `each_suffix`, the filter is then chosen as the first preferred filter
whose frames actually carry that suffix (choosing the filter first is what
produced the wd1 F150W failure), and the centre is taken from the **exposure
frames**, not the mosaic — a mosaic's covered area is the union of every visit and
module, so its interior contains points no frame of the observation being probed
covers. The frame glob is pinned to the proposal, because ngc6334's F182M
directory holds both 6778 and 7213 frames under one suffix.

A one-filter cutout runs m12 → m6 and stops: m7 needs two filters, m8 is the
cross-band dedup, and the all-filter merge reads `<basepath>/catalogs/`, which a
cutout never writes.

## Tests

```bash
pytest jwst_gc_pipeline/monitoring/tests/test_monitoring.py
```

The scanner is tested against synthetic trees so the assertions are about the
rules, not about today's archive. The registry-derived tests (`shared_filters`,
`is_globbed`, the MIRI split) read the real `fields.yaml` on purpose — they assert
the shapes the monitor has to handle.
