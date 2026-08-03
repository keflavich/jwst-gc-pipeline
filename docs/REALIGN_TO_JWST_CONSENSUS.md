# Realigning the catalogs to the JWST reference-filter consensus

**Instructions for the agent picking this up.** The machinery this depends on
(`photometry/consensus_catalog.py`, the m2 checkpoint writing
`catalogs/<filter>_consensus.fits`) is on `main`; read
[`JWST_CONSENSUS_CATALOG.md`](JWST_CONSENSUS_CATALOG.md) first.

## What you are doing, and what you are not

Every filter of a field is currently tied to VIRAC2 independently, spending
VIRAC2's ~40 mas per-star error and its 2014.0 proper-motion propagation once
per filter. Instead: one filter — the reference filter — keeps its VIRAC2 tie,
and every other filter is re-tied to **that filter's** consensus catalog.

**Operate on catalogs only. Do not re-run cataloging.** The fits are unchanged;
only positions move, by an amount expected to be well under a pixel. Re-running
the photometry would be both wasteful and a different experiment.

The one product that does need regenerating afterwards is the **drizzled
mosaics**, and only because the absolute correction they inherit changes. Treat
that as a separate, explicitly-approved step — it is a reduction, not a
realignment.

## The stage names

There is no `m1` or `m2` *photometry* phase. The phase list is:

```
m12, m3, m4, m5, m6     per filter
m7                      cross-band merge (needs every filter in one job)
m8                      forced cross-band fill
```

`m12` is iter1+iter2 fused into one per-frame pass. **`m2` is the name of the
astrometry checkpoint** that runs after the m12 merge, not a phase. So "realign
m1…m8" means: every per-frame catalog from m12 through m6, then the m7 and m8
products built from them.

## Order of work

**1. Confirm the inputs exist.** For the field, every filter needs
`catalogs/<filter>_consensus.fits` from a completed m2 checkpoint, and
`catalogs/jwst_reference_consensus.fits` must exist
(`consensus_catalog.promote_reference_filter`). If the reference filter's
consensus is missing, stop — do not fall back to VIRAC2 silently.

**2. Measure, before changing anything.** For each non-reference filter, run
`consensus_catalog.tie_to_reference_consensus` and record: offset, contrast,
`window_edge_fraction`, number of pairs. Write these to a table and **look at
them before applying any of them**. This is the first time these numbers exist;
they are the evidence for what a sane tolerance is.

Expect them to be small. If any filter reads more than a few tens of mas, that
is a finding about the existing VIRAC2 ties, not a routine correction — report
it rather than applying it.

**3. Map the offset per tile before applying it.** A single number per filter
is exactly the sign-off `CLAUDE.md` forbids: a field-average can read ~0 while
half the field is offset. Use `measure_offset_grid` (≥12×12) and check the
offset **magnitude** per tile, not only the contrast.

**4. Apply to the catalogs.** For each affected catalog, add the offset to the
positions and record the provenance in the table meta — what was applied, from
which reference filter, measured against which consensus file, and when. A
catalog that has been realigned must be distinguishable from one that has not,
or a second run will apply it twice. This is the failure that left brick-1182
visit-001 ~20″ off; the guard is a stamp, and `fix_alignment`'s `RAOFFSET`
idempotency check is the pattern to copy.

Cover `m12`, `m3`–`m6` per-frame catalogs, then the merged per-filter catalogs,
then `m7` and `m8`. The cross-band products must be rebuilt from realigned
inputs rather than shifted themselves — shifting a cross-band catalog after the
match would move sources that were matched at their old positions.

**5. Verify.** Re-measure each filter against the reference consensus after the
correction: it must now read ~0. Then re-run the inter-frame overlap gate
(`scripts/release/check_interframe_overlap.py --field <f> --scan`) — realignment
must not open a seam between overlapping frames.

**6. Mosaics, separately and with approval.** Re-drizzling is the only step that
touches reduction outputs. Get explicit sign-off, and state the expected shift
(< 0.5 pixel) so it can be checked rather than assumed.

## Things to get right

- **Never NN-median.** Every offset here goes through
  `astrometry_offsets.measure_offset` (histogram stacking, window sweep), as
  enforced by `measure_offsets.assert_sparse_reference_for_nn_median` and the
  grep-guard test.
- **The reference filter does not move.** It is the anchor; re-tying it to
  itself is a no-op and re-tying it to VIRAC2 during this work would undo the
  point.
- **Write atomically.** These catalogs are read by concurrent array tasks; use
  `atomic_io.write_table_atomic`. See `RACE_CONDITIONS.md`.
- **One concern per PR**, on a worktree branched from `origin/main`.

## What is deliberately unspecified

The **tolerance** for a filter-to-reference tie. It should be far tighter than a
VIRAC2 tie, and nobody has measured how tight yet. Step 2 produces those
numbers; propose a threshold from them, with the distribution, rather than
picking one first.
