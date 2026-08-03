# The JWST consensus catalogs

Three catalogs anchor this pipeline's astrometry, and they are not
interchangeable.

| | what it is | who makes it |
|---|---|---|
| **absolute reference catalog** | Gaia DR3 + VIRAC2, PM-propagated to the observation epoch. Defines the frame. | `reduction/build_gaia_virac2_refcat_byquery.py` |
| **per-filter JWST consensus catalog** | One filter's own stars, pooled across its visits. Deeper than VIRAC2 and internally far more precise. | `photometry/consensus_catalog.py`, at the m2 checkpoint |
| **JWST reference-filter consensus** | The per-filter consensus of whichever filter best matches VIRAC2. What the field's *other* filters tie to. | `photometry/consensus_catalog.promote_reference_filter` |

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

`catalogs/<filter>_consensus.fits`, written at the **m2** checkpoint — after the
m12 merge, which is where the first per-frame catalogs exist.

It is built in two steps, and the split matters:

1. **Per `(visit, filter)`**, `visit_consensus.build_visit_consensus` selects
   reliable stars (S/N ≥ 10, qfit ≤ 0.1, not a replaced saturated fit), removes
   each exposure's measured relative offset, associates within 0.2″, and takes a
   per-star median. Grouping by visit is not cosmetic: detecting that one
   exposure is misaligned means comparing it against **its own visit's** other
   exposures.
2. **Pooled across visits** by `consensus_catalog.pool_visit_consensi` into one
   catalog per filter. Stars seen in more than one visit are averaged, and
   `n_visits` records how many saw each.

Pooling does not re-measure an offset between visits: each visit's consensus is
already on the same frame by the time the checkpoint that built it returns, and
a second measurement there would fold avoidable noise into the positions that
are meant to *be* the reference.

**The exposure-vs-consensus tolerance is 2 mas** (`EXPOSURE_CONSENSUS_TOL_MAS`).
These are the same stars, in the same filter, minutes apart, so a disagreement
is not statistical: it is distortion error (measured at < 2.5 mas rms) or a
wrong-source match. The tolerance stays at 2 mas until there is a measured
distribution to tighten it against.

## The reference filter

`consensus_catalog.reference_filter` picks the filter closest to VIRAC2 in the
two senses that matter — wavelength, and which stars it leaves unsaturated:

```
score = |ln(lambda / 2.15um)| + 0.105 * bandwidth_class
        bandwidth_class: N=0, M=1, W=2
```

which gives **F212N > F210M > F187N > F182M > F200W > F150W**, and also
**F277W > F140M > F115W**.

The two criteria *trade off*; neither dominates. F210M beats F187N because it is
much closer to Ks, while F187N beats F200W because F200W saturates the bright
stars VIRAC2 measures — and no rule that ranks one criterion strictly above the
other reproduces both. One bandwidth class is worth 0.094–0.116 in ln λ; the
code uses 0.105.

Distance is measured in **log** wavelength, and that is not a detail. Linearly,
F277W sits 0.62 µm from Ks and F140M 0.75 µm — so close that no long-wavelength
penalty large enough to be meaningful can put F277W first, which it must be. In
log, `ln(2.77/2.15) < |ln(1.40/2.15)|` with room to spare, and the same measure
puts MIRI last without a separate channel term. A narrow long-wavelength filter
(F323N) can therefore outrank a wide blue one (F150W) on its merits.

VIRAC2's astrometry comes from VVV Ks monitoring, hence 2.15 µm.

The chosen filter's consensus is copied to
`catalogs/jwst_reference_consensus.fits`, carrying `REFFILT` in its header.

## Tying the other filters

`consensus_catalog.tie_to_reference_consensus` measures a filter's offset from
the reference-filter consensus with the same density-immune histogram used
everywhere else (`measure_offset`, with the window sweep).

**No threshold gates this yet, deliberately.** The tie should be far tighter
than a direct VIRAC2 tie, for the reasons above — but "far tighter" is not a
number, and inventing one would either pass everything or fail good data. The
offset is measured and recorded; the tolerance gets set from the measurements.

## What is not here

The catalogs are written and the ties are measured. **Applying** a
reference-filter tie to the m12–m8 catalogs is a separate piece of work, with
instructions in [`REALIGN_TO_JWST_CONSENSUS.md`](REALIGN_TO_JWST_CONSENSUS.md).
