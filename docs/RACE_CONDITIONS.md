# Race conditions

This pipeline runs as SLURM arrays: dozens of tasks start within seconds of each
other, on different nodes, against one shared filesystem. Every failure below
came from two of them touching the same thing at the same time, and most of them
produced a wrong answer rather than an error.

That is the reason this page exists. A crash from a race is cheap — you rerun
it. A race that silently substitutes a default is expensive, because the run
completes, the products look ordinary, and the mistake is discovered later or
not at all.

## The rule

**Never write a shared path in place.** Write a temp sibling, then `os.replace`,
which is atomic within a filesystem: a reader sees either the old file or the
new one, never a partial one and never nothing.

```python
tmp = f'{path}.tmp{os.getpid()}'
table.write(tmp, overwrite=True, format='fits')
os.replace(tmp, path)
```

`.tmp{pid}` rather than a fixed `.tmp`, so two writers do not collide in the
temp file either. [`jwst_gc_pipeline/atomic_io.py`](../jwst_gc_pipeline/atomic_io.py)
packages this as `atomic_write`, along with `locked` for a read-modify-write and
`keep_a_copy` for a backup that leaves the original in place;
`merge_catalogs.py:1966-1971` (the consolidated satstar cache) and
`versioning/prov_sidecar.py:64-72` are the same pattern written out by hand.

The corollary: **a missing input is not a measurement.** Code that reads a
shared file must distinguish "the file is not there" from "the file says zero",
and must not proceed on the second reading when it meant the first.
`AlignmentShift.table_present` exists for exactly this, and its comment says
why: *conflating "undeterminable" with "measured zero" is the failure mode this
module exists to remove.*

## Known races

| | where | what happens | status |
|---|---|---|---|
| Import storm | `submit_cataloging_perframe_phase.sbatch` | dozens of tasks import the same modules off shared storage at once; some die ~30 s in | fixed — random 1–25 s stagger, plus a retry gated on the import signature |
| Metadata coherence | same, finalize mode | a finalize lists the marker directory before Lustre has settled and crashes on a marker that does exist | fixed — 180 s settle before the strict verify |
| All-filter merge | `merge_catalogs.py:3022`, `submit_merge.sbatch` | N array tasks each ran the all-filter merge: N writers of one file, most reading inputs their siblings had not written yet | fixed — an array task stops after its own filter; the all-filter merge is a separate job with `afterok` |
| Per-frame product names | `saturated_star_finding.py:4442`, `crowdsource_catalogs_long.py:1741` | concurrent runs differing only by post-processing options wrote the same filenames | fixed — the iteration label is part of the name |
| Offsets / consensus tables | `astrometry_checkpoint.py` | see below | fixed — atomic write, backup by copy, and a lock across the read-modify-write |
| PSF grid cache | `crowdsource_catalogs_long.py:2144-2150, 2251-2255` | see below | **open** |

### Fixed: the offsets and consensus tables

Both writers used to do this:

```python
os.replace(out_path, f'{out_path}.pre_{stage}_{stamp}')   # backup
tbl.write(out_path, overwrite=True)                        # rebuild
```

Between those two calls the table **did not exist**. A concurrent reader in that
window — another filter's cataloging job resolving its shift, or a reduce job —
takes the missing-table branch at `unified_alignment.py:284-290` and aligns its
frame at **(0, 0)**, records `table_present=False`, and carries on. Small
window, invisible failure: the frame is silently left on the raw pointing.

There was a second problem in the same place. `update_offsets_table` reads the
table, corrects rows, and writes it back, with nothing serialising that against
another process doing the same. Two filters' m2 checkpoints correcting the same
per-proposal table means the second write drops the first correction. The table
is shared per proposal, so that is the normal case, not an exotic one — and an
offsets-table curation failure on exactly this file is what left brick-1182
visit-001 about 20″ off (`CLAUDE.md`).

Both writers now keep the backup by **copying**, write through
`atomic_io.atomic_write`, and hold `atomic_io.locked` across the whole
read-modify-write.

### Open: the PSF grid cache

```python
if os.path.exists(_psf_fn):
    grid = to_griddedpsfmodel(_psf_fn)     # may be half-written
...
nrc.psf_grid(..., save=True, outdir=_psf_outdir)   # writes in place
```

`psf_grid(save=True)` writes the FITS directly into the shared cache directory,
so the file exists from the moment it is created until it is complete. A
concurrent task that checks in that window sees it and reads a truncated file.

The same window costs work: array tasks starting together all miss the cache,
all build the same grid (17–20 minutes, ~300 GB peak each), and all write the
same filename.

Fix: build into a temp sibling and `os.replace` into the cache.

## Races that are structural, not bugs

Two things look like races and are the design working:

- **A fan-out shard that dies strands its finalize.** The finalize depends on
  `afterok`, so it never runs, and a shard's missing marker crashes it rather
  than letting the phase proceed on fewer frames. Recovery is manual: rerun the
  shard, then `scontrol update jobid=<finalize> dependency=...`.
- **`afterok` between stages** is what makes the one-command chain correct.
  Submitting the stages without dependencies is the race.

## Adding a writer

Ask, before writing any path that is not unique to one task:

1. **Can two tasks write this?** Then write a temp sibling and `os.replace`.
2. **Is it read-modify-write?** Then a lock, or restructure so one job owns it —
   the all-filter merge took the second route.
3. **What does a reader do when it is missing?** If the answer is "uses a
   default", say so out loud in the reader and make the caller decide. A silent
   zero is the failure this whole page is about.
4. **Does the name include everything that distinguishes the run?** Module,
   detector, observation, iteration label. Two runs that differ in a way the
   filename does not capture will overwrite each other.
