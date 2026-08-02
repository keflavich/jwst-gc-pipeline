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

**Never write a shared path in place.** Write a temp sibling **named for the
writer**, then `os.replace`, which is atomic within a filesystem: a reader sees
either the old file or the new one, never a partial one and never nothing.

For an astropy Table, call `atomic_io.py::write_table_atomic`, which does this.
By hand it is:

```python
tmp = os.path.join(directory, f'.{root}.tmp{os.getpid()}{uuid4().hex[:8]}{ext}')
table.write(tmp, overwrite=True, format='fits')
os.replace(tmp, path)
```

The tag matters as much as the `os.replace`. With a fixed `.tmp`, N writers
share one temp path, interleave inside it, and `os.replace` publishes the
mixture — a well-formed file with wrong contents, which is worse than the
partial read the rule exists to prevent. A bare pid is not a tag either: this
storage is shared and written from many nodes, and pids repeat both across nodes
and over time on one — hence the `uuid4`.

`overwrite=True` is not a substitute. astropy's FITS table writer implements it
by unlinking the file and then opening it *without* the flag, so a second writer
that recreates the file in between kills the first with "File ... already
exists".

`merge_catalogs.py::load_satstar_catalog` (the consolidated satstar cache) and
`prov_sidecar.py::write_sidecar` are the same pattern written out by hand.

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
| All-filter merge | `merge_catalogs.py::main`, `submit_merge.sbatch` | N array tasks each ran the all-filter merge: N writers of one file, most reading inputs their siblings had not written yet | fixed — an array task stops after its own filter; the all-filter merge is a separate job with `afterok` |
| Per-frame product names | `saturated_star_finding.py::remove_saturated_stars`, `crowdsource_catalogs_long.py::load_or_make_satstar_catalog` | concurrent runs differing only by post-processing options wrote the same filenames | fixed — the iteration label is part of the name |
| Per-phase seed caches | `atomic_io.py::write_table_atomic` | several caches are keyed by (filter, module) rather than by shard, so every shard of a phase rebuilt and wrote the same path | fixed — temp sibling named for the writer, then `os.replace` |
| Offsets / consensus tables | `astrometry_checkpoint.py::update_offsets_table` | see below | **open** |
| PSF grid cache | `crowdsource_catalogs_long.py::get_psf_model` | see below | fixed — built in a private directory, moved into the cache when finished |

### Open: the offsets and consensus tables

Both writers do the same two steps:

```python
os.replace(out_path, f'{out_path}.pre_{stage}_{stamp}')   # backup
tbl.write(out_path, overwrite=True)                        # rebuild
```

Between them the table **does not exist**. A concurrent reader in that window —
another filter's cataloging job resolving its shift, or a reduce job — takes the
missing-table branch in `unified_alignment.py::_shift_from_consensus` and aligns its frame at
**(0, 0)**. It records `table_present=False` and carries on. This is the
substitute-a-default shape: the frame is silently left on the raw pointing.

The window is small and the failure is invisible, which is the bad combination.

There is a second problem in the same place. `update_offsets_table` reads the
table, corrects rows, and writes it back, with nothing serialising that against
another process doing the same. Two filters' m2 checkpoints correcting the same
per-proposal table concurrently means the second write drops the first
correction. The table is shared per proposal, so this is the normal case, not an
exotic one — and an offsets-table curation failure on exactly this file is what
left brick-1182 visit-001 about 20″ off (`CLAUDE.md`).

Fix: write atomically as above, and take a lock across the read-modify-write.

### Fixed: the PSF grid cache

```python
if os.path.exists(_psf_fn):
    grid = to_griddedpsfmodel(_psf_fn)     # may be half-written
...
nrc.psf_grid(..., save=True, outdir=_psf_outdir)   # writes in place
```

`psf_grid(save=True)` names its own output files, so it cannot be handed a
temporary path. It used to write them straight into the shared cache, where each
file exists from creation until complete — and a concurrent task checking
`os.path.exists` in that window reads a truncated grid.

The build now goes into a private `.building-<host>-<pid>/` inside the cache,
and each finished file is moved into place with `os.replace`
(`atomic_io.publish_into`), so a file a reader can see is a file that is done.

The window costs work as well as correctness: array tasks starting together all
miss the cache, all build the same grid (17–20 minutes, ~300 GB peak each), and
all write the same name. Publishing atomically does not stop the duplicated
work — only the corrupt read.

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
