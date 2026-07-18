# CMZ HiPS mosaic + giant-catalog sharing — plan

Plan for (1) a **growing two-color CMZ HiPS** image that new fields (program
10678 as it rolls in) fold into without a full rebuild, and (2) **sharing the
full-CMZ giant catalog** with hierarchical/HiPS visualization in Aladin.

Written to sit next to the release machinery (`stage_release.py`,
`RELEASE_DEPLOYMENT_CHECKLIST.md`) — the release gate is the natural trigger for
regenerating the HiPS + catalog products.

---

## 0. Grounding — what exists today

**Image HiPS** (in `/orange/adamginsburg/jwst/jwst_scripts/`, the `jwst_rgb`
package — *not* this repo):
- Generator: **`reproject.hips.reproject_to_hips`** (astropy `reproject`), galactic
  frame, 512-px PNG tiles to order 13 (`jwst_rgb/save_rgb.py:191`). **Not** CDS
  `Hipsgen.jar`.
- Input is a **already-composited RGBA PNG** with embedded AVM/WCS — HiPS is built
  from the color PNG, not the FITS.
- **Full rebuild every time** (`shutil.rmtree` + regenerate). **No** incremental
  growth, **no** MOC, **no** catalog HiPS, **no** Allsky.
- The "third color by interpolation" already exists: `make_two_filter_rgb`
  (`scripts/gc2211_rgb_images.py:109`) sets R = long band, B = short band,
  **G = 0.5·(R+B)** of the asinh-stretched channels. Stretch = `simple_norm`
  asinh/log, percentile clips.

**Release** (`scripts/release/`): per-field `-merged_i2d.fits` mosaics + per-field
cross-band catalog (`basic_merged_indivexp_photometry_tables_merged_resbgsub_m7…`),
staged to `/orange/adamginsburg/jwst/releases/<version>/[<group>/]<field>/`, served
via Globus + HTTPS + a static webpage (`make_webpage.py`). **No cross-field CMZ
catalog** and **no footprint/coverage tracking** exist yet. **Program 10678 is
greenfield** — onboard via the `FIELDS` dict in `stage_release.py`.

**Filter reality (decisive for the two-color choice):**
- **F212N** — the CMZ-wide short-wave anchor (blue).
- **F480M** — the **CMZ-wide long band with program 10678** (red). The go-forward
  two-color pair is **F212N + F480M**.
- **F405N** — a **legacy** per-field long-narrow band, present in the older
  programs (2221/1182/5365/4147) but not the go-forward CMZ set. Use it only as a
  fallback for a legacy field that predates F480M coverage.
- ⇒ Two-color = F212N (blue) + F480M (red), with F405N a legacy per-region
  fallback (§1.2).

> Note: a data survey of the *current* disk (pre-10678) shows F405N common and
> F480M sparse — that is the legacy set, not the go-forward one. Key the design on
> 10678's filters (F212N + F480M), not the legacy on-disk mix.

---

## Part 1 — Growing two-color CMZ HiPS

### 1.1 Design principle — separate the incremental substrate from the color

The current pipeline bakes color into a PNG *then* HiPS-tiles it, so any new field
forces a full recomposite + rebuild. Invert that:

> **Build one mono (grayscale, real-flux) HiPS per filter as the incremental
> substrate. Derive the color composite as a thin top layer.**

Because HiPS tiles are HEALPix-addressed, adding a field touches only the tiles it
overlaps (+ their ancestor tiles). Color (2-band + interpolated green) is then a
per-tile combine of the mono HiPS — cheap to re-derive, and re-derivable at
different stretches without recomputing the substrate.

```
per-field i2d ──► mono HiPS: HiPS/F212N   (incremental, real flux, fits tiles)
                  mono HiPS: HiPS/LONG     (F480M where present, else F405N)
                                  │
                                  ▼  (derive; cheap)
        color HiPS: R=LONG, B=F212N, G=0.5·(R+B)   ← 2-color + interpolated 3rd
                                  │
                                  ▼
                    Aladin Lite pane on the release webpage
```

### 1.2 Long-band policy (F480M primary, F405N legacy fallback)

The red channel is **F480M** — CMZ-wide with 10678. For **legacy** fields that
predate F480M coverage, fall back to F405N in those regions only. Concretely:
- `HiPS/F480M` — the primary long-band mono HiPS (grows CMZ-wide as 10678 lands),
- `HiPS/F405N` — a legacy mono HiPS covering only the pre-10678 fields.

The color layer uses F480M where present, F405N only to fill legacy gaps, with the
choice recorded in a per-band coverage MOC (§1.4) so every colored region's
long-band provenance is explicit. As 10678 completes, the F405N fallback shrinks
to nothing and the mosaic is uniformly F212N+F480M.

### 1.3 Tooling: two viable substrates

| | **A. CDS `Hipsgen.jar`** (recommended for the survey substrate) | **B. `reproject.hips`** (keep for per-target quick-looks) |
|---|---|---|
| Incremental add | **Native** — re-run over an index incl. new files; only affected tiles rewrite | none (full rebuild); would need a custom tile-merge (§1.5) |
| Input | FITS i2d directly (real flux, mono) | composited PNG (current path) |
| MOC | **generates coverage MOC** | none |
| Color HiPS | **RGB HiPS from 2–3 mono HiPS** built-in | done upstream in the PNG |
| Allsky / low orders | yes | not written |
| Dep | Java (`Hipsgen.jar`, one jar) | pure Python (already in use) |

**Recommendation: pure-Python (`reproject.hips`) — no Java.** The apparent reason
to reach for Hipsgen (native incremental) is not actually needed: because
`reproject.hips` writes a **correct all-order pyramid per field**, folding a new
field into the master is a **per-order tile combine** with **no pyramid
re-derivation** — so it is incremental *and* avoids the HEALPix nested-child
orientation math (exactly the bug class the repo's HiPS-orientation QA already
polices). §1.5 is therefore the primary path, not a fallback. Keep the Java
Hipsgen driver only for the one thing pure-Python can't do — the progressive
**catalog** HiPS (§2.4) — and as an optional alternative image builder.

### 1.4 Coverage MOC — also the footprint tracker the release lacks

For each field/filter, compute a **MOC** (Multi-Order Coverage, `mocpy`) from the
i2d valid-data footprint. The union MOC per filter = the survey coverage map. This:
- fills the "no footprint/coverage tracking" release gap,
- drives incremental rebuilds (the MOC of a *new* field = exactly the tiles to
  regenerate),
- is published as a `.fits`/`.moc` for Aladin overlay + machine-readable coverage,
- lets the webpage show "what's covered so far" as 10678 rolls in.

### 1.5 Pure-Python incremental merge (the primary path)

`reproject_to_hips` each NEW field into a scratch tree, then merge **per order**:
for each `(Norder, Npix)` tile the field produced, copy it in if the master lacks
it, else combine the two (nan-aware mean/priority) and write back. **No ancestor
regeneration is needed** — reproject.hips already emitted a correct tile at every
order for that field, so the coarse tiles combine directly the same way the deep
ones do. That is what makes this both incremental (only overlapping master tiles
change) and safe (no hand-rolled HEALPix nested-child orientation). FITS (numeric)
tiles keep the combine lossless; the color layer is derived on top (§1.1).

### 1.6 Rollout / steps

1. **`cmz_hips/`** driver package (new, in `jwst_scripts` or this repo):
   `build_mono_hips(field, filter)`, `merge_or_update(master, field)`,
   `derive_color_hips(F212N, LONG)`, `build_coverage_moc(field, filter)`.
2. Backfill the current CMZ fields (F212N + F405N/F480M) into the master mono HiPS
   + MOC.
3. Derive the CMZ-wide color HiPS (2-color + interpolated green).
4. **Wire into the release**: after `stage_release.py` stages a field, call the
   incremental HiPS update for that field's F212N + long band, refresh the color
   HiPS + MOC. Gate on the same astrometry checks (a mis-registered field must not
   pollute the mosaic — reuse `check_interframe_overlap.py`).
5. **10678 onboarding**: add its `FIELDS` entry; the incremental path folds each
   new observation in automatically as it clears the release gate.
6. Embed an **Aladin Lite** pane on the release webpage (color HiPS + coverage MOC).

---

## Part 2 — Sharing the full-CMZ giant catalog

### 2.1 The gap

All catalog merging today is **within a field**. There is no CMZ-wide table and no
hierarchical catalog product. Three things to build: an **assembly**, a
**distribution format**, and a **visualization**.

### 2.2 Assemble the CMZ-wide catalog (new tooling)

`assemble_cmz_catalog.py`: vstack the per-field cross-band `resbgsub_m7` catalogs
into one table, with:
- **de-duplication in field-overlap regions** (same-star match across field
  boundaries within a tolerance; keep the higher-S/N / more-bands detection),
- a **provenance block** per source: `field`, `program`, `obsid`, source pipeline
  tag (ties into the versioning `GCTAG`/sidecar work — the CMZ catalog records
  which tagged run each field came from),
- **unified columns**: `ra`/`dec`, per-filter `mag_ab`/`mag_vega`/`flux_jy` +
  errors, `qfit`, `flags`, and a per-source filter-coverage bitmask (which bands
  actually measured it — filters differ by field).

Guard with the same astrometry sign-off as the release (cross-field seams are
exactly where a per-field tie error shows up).

### 2.3 Distribution format — HATS / Parquet (LSDB-native)

Publish the assembled catalog as a **HATS** (Hierarchical Adaptive Tiling Scheme,
formerly `hipscat`) **Parquet** dataset via `hats-import`: HEALPix-partitioned,
~equal objects per partition, Apache Parquet. Why:
- **LSDB** (LINCC Frameworks) reads it for parallel, out-of-core **cross-matching
  and analytics** — "80 TB from a laptop"; the format Rubin/STScI/IPAC/CDS/ESA are
  standardizing on.
- Cloud/Globus-friendly, partition-pruned queries (no full download to filter by
  sky region or magnitude).
- Keep the flat **FITS + ECSV** (current) *and* add a flat **Parquet** for the
  "just give me one file" user; HATS is the scalable path.

### 2.4 Visualization — HiPS catalog for Aladin

Generate a **HiPS catalog** (progressive catalog) with CDS **`Hipsgen-cat.jar`**
from the assembled table. This is the direct answer to "catalog visualization in
Aladin/HiPS": a zoom-dependent source layer that Aladin Lite/Desktop (and ESASky,
etc.) render natively — pan/zoom over millions of sources with no server. Embed it
in the release webpage's Aladin Lite pane, overlaid on the color image HiPS + the
coverage MOC.

> `mocpy` vs `Hipsgen-cat` vs `HATS`, to be explicit:
> - **`mocpy`** → *coverage* (footprint/where-is-data), not per-source display.
> - **`Hipsgen-cat` (HiPS catalog)** → *visualization* of sources in Aladin.
> - **`HATS`/`hipscat` + LSDB** → *analysis/distribution/cross-match* at scale.
> Use all three; they are complementary layers, not alternatives.

### 2.5 Release integration + steps

1. `assemble_cmz_catalog.py` (vstack + dedup + provenance + unified columns).
2. `hats-import` → HATS Parquet; also emit a flat Parquet + keep FITS/ECSV.
3. `Hipsgen-cat.jar` → HiPS catalog; `mocpy` → catalog coverage MOC.
4. `stage_release.py` gains a `--cmz-catalog` step that (re)assembles when any
   member field re-releases, and stages the HATS tree + HiPS catalog + MOC into the
   release tree next to the images.
5. `make_webpage.py` gains an **Aladin Lite** view: color image HiPS + catalog HiPS
   + coverage MOC, plus HATS/Parquet/FITS download links.

---

## 3. Cross-cutting: provenance tie-in

Both products stamp the **pipeline tag** (`jwst_gc_pipeline.versioning`): the HiPS
`properties` and the catalog HATS metadata record the tag + the MANIFEST of input
fields/tags they were built from. An incremental HiPS/catalog update records which
field/tag was folded in and when — so a growing CMZ product stays fully traceable
(and a re-released field triggers exactly the tiles/partitions it touches).

## 4. Decisions

Resolved:
- **Long band** → **F480M** (CMZ-wide with 10678), F405N legacy per-region
  fallback (§1.2).
- **Substrate** → **pure-Python** `reproject.hips` + per-order incremental merge;
  Java Hipsgen kept only for the catalog HiPS (§1.3/1.5).

Open for the author:
1. **HATS now or later**: ship HATS from the first CMZ-wide release, or start with
   flat Parquet + catalog HiPS and add HATS when catalog size/cross-match demand it?
2. Where the tooling lives: implemented in **this repo** (`jwst_gc_pipeline/cmz/`)
   so it sits with the release/gate machinery and the versioning provenance; the
   per-target press-image RGBs stay in `jwst_scripts`.
