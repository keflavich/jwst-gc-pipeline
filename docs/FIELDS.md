# The field registry

Every target the pipeline knows about is declared in one file:
[`jwst_gc_pipeline/fields.yaml`](../jwst_gc_pipeline/fields.yaml). Adding a
target to the **reduce, catalog and merge stages** is an edit to that file
alone. Four smaller lists elsewhere still name targets; see
[What this does not replace](#what-this-does-not-replace).

`jwst_gc_pipeline/fields.py` loads it and answers the questions the pipeline
asks: which filters a proposal observed, which observation number to build a
filename from, which data tree a target lives in, how many visits it has, which
astrometric reference frame it uses.

## Adding a target

Add a block under `fields:`:

```yaml
  mynewfield:
    root: orange                       # which data tree; see `roots:` at the top
    fov_region: regions_/nircam_mynewfield_fov.reg    # NIRCam realignment only; omit otherwise
    observations:
      '9999':                          # the proposal id
        nvisits: 2
        reference_frame: VIRAC2
        obsids:
          nircam: ['001']              # every observation of this field
        reference_catalog:
          '001': catalogs/gaia_virac2_refcat_epoch2025.5.fits
        filters: [f115w, f212n, f405n]
```

`python -m jwst_gc_pipeline.run_pipeline --proposal 9999 --obsid 001` on an
unregistered proposal prints this block with your values filled in.

That is the whole change. The reduce, catalog and merge stages all read it.

**Write it wherever it reads best.** Order in this file means nothing: the
loader sorts proposals numerically and filters by wavelength, so two files that
differ only in arrangement behave identically. A test enforces this by
shuffling the file and comparing.

## The fields of an observation

| key | meaning |
|---|---|
| `obsids` | Every observation number that images this field, per instrument, as a **list**. `'*'` is the one supported scalar and claims **every** observation of the proposal for that instrument — see [The `'*'` wildcard](#the--wildcard) below. |
| `glob_obsid` | The observation number the merge builds filename patterns from, per instrument. Needed only when `obsids` lists more than one; `'*'` matches several. |
| `joint_obsids` | Tokens naming several observations cataloged in one run, e.g. `'002-998'`. |
| `nvisits` | How many visits the observation has. |
| `filters` | The filters to catalog and merge, lowercase. |
| `niriss_filters` | NIRISS reuses NIRCam filter names on a different pixel scale, so its filters and products are kept apart. |
| `reference_frame` | The astrometric frame token (`VIRAC2`, `Gaia`), used to name a per-proposal offsets table, `offsets/Offsets_JWST_Brick<proposal>_<token>[_average].csv`, on the fallback path taken when the proposal has no locked table. Keyed per proposal: two fields sharing a proposal must agree, and the loader raises if they disagree. **Leave it out** when the proposal aligns from a table declared in `reduction/alignment_config.py` — that is where the frame a product records comes from, and an absent token makes `resolve_shift` refuse to build the fallback filename rather than name a table that should not be used. |
| `reference_catalog` | Observation number → the catalog file the astrometry ties TO, relative to the field directory. Per observation, because different observations of one proposal sit at different epochs. A value may be a list; `reference_catalog_path` takes the first entry present on disk, whatever the instrument. MIRI and NIRISS are simply the ones that register more than one. What happens when none is present differs: the NIRCam driver **raises**, while MIRI falls through to `twomass.fits` and then to running with no reference. |
| `reference_catalog_by_filter` | The rare per-filter override of the above: observation → filter → file. |
| `default_reference_catalog` | The catalog consulted for any observation that has no exact `reference_catalog` key. An exact key still wins, and an observation with neither still raises. This is what makes a wildcard-obsid proposal tie-able: it declares no obsid list, so there is nothing for per-obsid keys to hang on. |
| `offsets_table` | Path to the measured astrometric offsets, relative to the field's directory. Measured from the data once and then fixed. |

### The `'*'` wildcard

`obsids: {nircam: '*'}` claims every observation of the proposal for that
instrument. It is for a field that owns a whole proposal whose plan is long and
provisional — 10678, the GC Treasury, whose 139 observation numbers (001..139,
the same set for NIRCam and MIRI) are published today and are re-issued by
every replan. The wildcard records the ownership, which survives a replan, in
place of a 139-entry list.

**Use a list instead** for any proposal whose observations are split between
fields: the wildcard takes an obsid outside the plan as this field's where a
list would raise. That is the cost it accepts.

Rules the loader and the registry hold to:

- At most one field may hold the wildcard per (proposal, instrument), checked
  when the registry **loads**.
- An explicit number registered by another field wins over the wildcard.
- The wildcard resolves only obsid-**shaped** keys (`fields.is_obsid`: `042`,
  `002-998`), so a typo raises rather than being absorbed.
- Any other scalar (`nircam: '001'`) raises `FieldRegistryError`: a bare string
  would load as its individual characters.

What it means to a consumer that asks "how many observations?": a wildcard
answers "several, count unknown". `filter_observation_count` reports more than
one observation, so the foreign-observation filter at the m2 merge iteration
stays switched on; `monitoring.scan` reports the field as multi-observation and
counts its products across every observation of the proposal; and
`default_field_token` answers `None`. `'*'` is never handed out as an
observation number, because callers write what they get into product filenames.

### Observation numbers are per instrument, and they have to be

Proposal 2221 numbers its NIRCam and MIRI pointings of the same two fields in
opposite order:

| field | NIRCam | MIRI |
|---|---|---|
| brick | 001 | 002 |
| cloudc | 002 | 001 |

Both are right; the products on disk agree with each. Before this file existed,
one number was recorded per (target, proposal) — the NIRCam one — and the merge
used it for MIRI filters too. A Brick F2550W merge therefore looked for
`jw02221-o001*f2550w*i2d.fits`, found none of the 17 files named
`jw02221-o002`, and quietly fell back to a less accurate WCS instead of
stopping.

## Data trees

The `roots:` block at the top of the file is the only place the absolute paths
appear:

```yaml
roots:
  orange: /orange/adamginsburg/jwst
  blue: /blue/adamginsburg/adamginsburg/jwst
```

A field's `root:` names one of them. Pointing the pipeline at another disk is an
edit to those two lines. A test checks that no other absolute path creeps into
the file.

## What this replaced

Eight registries for a NIRCam target, ten with MIRI, spread over six files. Four
sat inside functions, where they could not be imported, printed, or overridden —
you edited the source. Nothing checked that they agreed, and they had drifted:

- **`cloudc/2526` had filters but no observation number**, so that merge raised
  `KeyError('2526')`. The observation is real — 2526 obs 021, MIRI F770W. The
  registry supplies the number.
- **`w51/1182` had an observation number but no filters**, and w51 has no 1182
  data on disk. It was never observed. Dropped.
- **`wd1` and `wd2` were on `/orange` in the catalog driver and `/blue` in the
  merge**, so those targets cataloged to one tree and merged from the other.
- **The MIRI observation numbers**, described above.
- **The reference catalogs were a fourth and fifth registry** in the reduction
  drivers, keyed by (proposal, observation) and, in one case, by filter. The
  field-registry documentation said registration was centralised while
  `PipelineRerunNIRCAM-LONG.py` still did its own exact lookup, so a new field
  registered here still failed at stage 1. They are `reference_catalog` entries
  now.

Each of those is a way for two hand-maintained copies of one fact to disagree,
and each degraded quietly rather than stopping.

## What this does not replace

Four lists outside the three pipeline stages still name targets. A new field
reaches them only if you use the tool that reads them.

| where | list | what it duplicates |
|---|---|---|
| `reduction/alignment_config.py` | `ALIGNMENT_CONFIG` | nothing — it describes how a field is *aligned*, not what a field is, and is already one typed registry with its own tests |
| `scripts/release/stage_release.py` | `FIELDS` | the data directory and observation prefixes |
| `photometry/make_starless_image.py` | `TARGETS` | the base path and filter list |
| `reduction/build_virac2_offsets.py` | `REGION` | the base path |

The last three are worth folding in next; each is a place two copies of one
fact can drift apart, which is what this change is about.

`jwst_gc_pipeline/reduction/make_merged_psf.py` keeps its own out-of-date copies.
It is deprecated and scheduled for removal; nothing imports it.
