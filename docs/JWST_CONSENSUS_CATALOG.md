# The JWST consensus catalogs

Three catalogs anchor this pipeline's astrometry, and they are not
interchangeable.

| | what it is | who makes it |
|---|---|---|
| **absolute reference catalog** | Gaia DR3 + VIRAC2, PM-propagated to the observation epoch. Defines the frame. | `reduction/build_gaia_virac2_refcat_byquery.py` |
| **per-filter JWST consensus catalog** | One filter's own stars, pooled across its visits. Deeper than VIRAC2 and internally far more precise. | `photometry/consensus_catalog.py`, at the m2 checkpoint |
| **JWST reference-filter consensus** | The per-filter consensus of whichever filter best matches VIRAC2. What the field's *other* filters tie to. | `scripts/reduction/promote_reference_consensus.py` |

## Why the JWST catalogs exist

VIRAC2 is the frame, and it is also the floor on how well anything can be tied
to it: roughly 40 mas per star, propagated from a 2014.0 reference epoch. Tying
six filters to VIRAC2 independently spends that error six times, and the filters
end up related to each other only through it.

The JWST data are far better than that against themselves — the same stars, in
the same filter, minutes apart. So: tie **one** filter to VIRAC2, and tie the
rest to that filter. A filter anchored this way inherits neither VIRAC2's
per-star error nor its proper-motion propagation.

## The per-filter consensus catalog

`catalogs/<filter>[_obstoken]_consensus.fits`, written at the **m2** checkpoint —
after the m12 merge, which is where the first per-frame catalogs exist.

It is built in two steps, and the split matters:

1. **Per `(visit, filter)`**, `visit_consensus.build_visit_consensus` selects
   reliable stars (S/N ≥ 10, qfit ≤ 0.1, not a replaced saturated fit), removes
   each exposure's measured relative offset, associates within 0.2″, and takes a
   per-star median. Grouping by visit is not cosmetic: detecting that one
   exposure is misaligned means comparing it against **its own visit's** other
   exposures.
2. **Pooled across visits** by `consensus_catalog.pool_visit_consensi` into one
   catalog per filter.

### Pooling

Association across visits uses `search_around_sky` — **all** pairs within the
radius — with union-find, not a nearest-partner pass. A nearest-partner pass
caps a group at two members, so a star seen in three visits would come out as
one merged pair plus a leftover duplicate row, and `n_visits` could never
exceed 2. ngc6334/6778 and wd1/1905 both have three visits.

Union is greedy in order of separation and **refuses to merge two groups that
already share a visit**. Association is across visits: two genuinely close stars
in one visit stay two stars however near, and a transitive chain
A(v1)–B(v2)–C(v1) cannot quietly merge A with C.

Positions are averaged as unit vectors, so RA wrap is not a special case.
Magnitudes are averaged in flux, not in magnitudes.

Columns: `RA DEC n_visits n_exposures scatter_mas err_mas refmag skycoord`.
`err_mas` is `scatter_mas / sqrt(n_exposures)`, and stays **NaN** rather than 0
for a single-exposure star — an identically-zero uncertainty free-passes a QC
gate.

### Pooling does not re-tie the visits — but it checks that it needn't

Each visit's consensus is already on the frame the checkpoint that built it
verified, and a second correction here would fold avoidable noise into the
positions that are meant to *be* the reference. But averaging visits that are
**not** on a common frame puts every shared star at the midpoint of the
disagreement — bias is half the offset — while the row count inflates with
unmerged duplicates, and nothing about the output would say so.

So each visit's bulk offset from the anchor visit is **measured and recorded**
(`IVMAXMAS`, `IV_<visit>` in the meta), with the density-immune offset histogram
and the window sweep, never a nearest-neighbour median. A visit beyond
`GROSS_INTER_VISIT_MAS` (100 mas) raises `InterVisitOffsetError` and no catalog
is written. That is a **gross** gate in the sense `CLAUDE.md` uses for the
sparse-Gaia cross-check — it catches "these visits were never tied", not a
few-mas imperfection, and against a dense catalog the histogram peak over-reads
by a few mas anyway. The recorded values are what make a fine threshold
choosable later. A tie that cannot be measured at all (too few stars for a
histogram peak) is recorded as such and counted in `IVNOMEAS`; it is not
silently treated as zero.

**The exposure-vs-consensus tolerance is 2 mas** (`EXPOSURE_CONSENSUS_TOL_MAS`).
These are the same stars, in the same filter, minutes apart, so a disagreement
is not statistical: it is distortion error (measured at < 2.5 mas rms) or a
wrong-source match. The tolerance stays at 2 mas until there is a measured
distribution to tighten it against.

### The observation token

The filename carries the same per-observation disambiguator as every other
per-obs product (`crowdsource_catalogs_long.obs_token`: `_j7213`, `_o023`, …).
This is not decorative. ngc6334's proposals 6778 and 7213 share a target
directory **and** a filter list at reference epochs 1.6 yr apart, and
cloudef/2092 has two obsids under one directory. Without the token the second
m2 checkpoint silently overwrites the first field's reference catalog.

## The reference filter

`consensus_catalog.reference_filter` picks the filter closest to VIRAC2 in the
two senses that matter — wavelength, and which stars it leaves unsaturated:

```
score = |ln(lambda / 2.15um)| + 0.105 * bandwidth_class
        bandwidth_class: N=0, M=1, W=2  (W2 counts as W)
```

which gives **F212N > F210M > F187N > F182M > F200W > F150W**, and also
**F277W > F140M > F115W**.

The two criteria *trade off*; neither dominates. F210M beats F187N because it is
much closer to Ks, while F187N beats F200W because F200W saturates the bright
stars VIRAC2 measures — and no rule that ranks one criterion strictly above the
other reproduces both. One bandwidth class is worth 0.094–0.116 in ln λ; the
code uses 0.105.

Both ends of that window are set by a comparison that decides a real field's
anchor — `F210M vs F187N` (sickle, w51) the upper bound, `F182M vs F200W`
(ngc6334/7213) the lower — so those two fields' picks are effectively the free
parameter. A third pair, `F164N vs F182M`, is separated by 0.00086 in rank and
flips at `w ≥ 0.1045`, so 0.105 sits just past that flip. One step either way
down the ranking is still a reasonable anchor.

Distance is measured in **log** wavelength, and that is not a detail. Linearly,
F277W sits 0.62 µm from Ks and F140M 0.75 µm — so close that no long-wavelength
penalty large enough to be meaningful can put F277W first, which it must be. In
log, `ln(2.77/2.15) < |ln(1.40/2.15)|` with room to spare, and the same measure
puts MIRI last without a separate channel term. A narrow long-wavelength filter
(F323N) can therefore outrank a wide blue one (F150W) on its merits.

Filter names parse as *digits, bandwidth class, optional trailing `2`*: `F150W2`
is 1.50 µm, not 15.02. Stripping every digit instead would rank a 1.5 µm SW
filter below most of MIRI and land it 0.0013 from `F1500W`.

VIRAC2's astrometry comes from VVV Ks monitoring, hence 2.15 µm.

### Promotion is a separate step

The chosen filter's consensus is copied to
`catalogs/jwst_reference[_obstoken]_consensus.fits`, carrying `REFFILT`:

```
python scripts/reduction/promote_reference_consensus.py \
    --basepath /orange/adamginsburg/jwst/brick --field-name brick --proposal-id 2221
```

This is not part of the m2 checkpoint, because the checkpoint runs *per filter*
and cannot know which of the field's filters ranks best. Run it once every
filter's m2 has completed. It refuses rather than falling back when the chosen
filter's consensus is not on disk — tying every other filter to a silently
absent reference is the shape of failure this ladder exists to prevent.

### Relationship to `alignment_config.reference_filter`

`reduction/alignment_config.py` carries a hand-set `FieldAlignment.reference_filter`
— "the band whose visit consensus defines this field's internal frame". That is
the same question, answered by hand. The two are now kept consistent by
`test_the_formula_reproduces_the_hand_set_reference_filters`, and a second test
requires the hand-set value to be a band the proposal actually observes. That
second check is what caught w51/6151 declaring `F200W`, which 6151 does not
observe at all; it is now `F210M`.

## Tying the other filters

`consensus_catalog.tie_to_reference_consensus` measures a filter's offset from
the reference-filter consensus with the same density-immune histogram used
everywhere else (`measure_offset`, with the window sweep).

**No threshold gates this yet, deliberately.** The tie should be far tighter
than a direct VIRAC2 tie, for the reasons above — but "far tighter" is not a
number, and inventing one would either pass everything or fail good data. The
offset is measured and recorded; the tolerance gets set from the measurements.

## What is not here

The catalogs are written and the ties are measurable. **Applying** a
reference-filter tie to the m12–m8 catalogs is a separate piece of work, with
instructions in [`REALIGN_TO_JWST_CONSENSUS.md`](REALIGN_TO_JWST_CONSENSUS.md).
`tie_to_reference_consensus` has no pipeline caller yet — it is the tool that
work starts from, not something running today.
