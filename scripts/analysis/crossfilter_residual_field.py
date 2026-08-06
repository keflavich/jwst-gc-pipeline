#!/usr/bin/env python
"""Measure the JWST-internal, position-dependent astrometric residual field.

Why this exists
---------------
Per-star astrometric precision on these fields is sub-mas, and the per-detector
and per-exposure terms are sub-mas after the DVA correction (see the m2
checkpoint records).  The astrometric floor that survives over the whole FOV is
something else: a COHERENT, POSITION-DEPENDENT difference between filters.

Two catalogs of the same field in two filters share the frames, the offsets
table, the DVA correction and the reference tie.  Everything that differs
between them is a per-filter WCS term.  The pipeline models that term as a
single constant per filter (the CRDS ``filteroffset``) and the tie that is
applied to the frames is a pure translation, so anything position-dependent
survives by construction.

This script measures what survives, using the pipeline's own sanctioned
same-star estimator.  It never computes a nearest-neighbour median against a
dense catalog: ``local_residual_map`` refuses to run until ``measure_offset``
has verified a small global tie, which is what makes the nearest partner the
right star (ASTROMETRY RULE #1).

Result on the Brick (m7 vetted, S/N>20, qfit<0.1, saturated dropped, 45" cells,
32 cells).  Every amplitude is PER-COMPONENT rms -- the convention
``measure_residual_field`` records in ``rms_convention`` -- and ``absorbed`` is
the dof-corrected fraction against a 9% chance level::

    pair             channels   bulk    rms   after   SEM   absorbed   |J|   matched
    F212N vs F187N   SW-SW      0.85   0.54    0.47  0.03        14%  0.27      0.64
    F405N vs F466N   LW-LW      0.51   0.51    0.30  0.06        60%  0.45      0.69
    F182M vs F187N   SW-SW      0.86   0.98    0.78  0.02        29%  0.88      0.13
    F212N vs F182M   SW-SW      0.17   1.40    1.21  0.03        18%  0.94      0.97
    F212N vs F405N   SW-LW      0.84   2.50    1.68  0.11        50%  2.31      0.58
    F182M vs F466N   SW-LW      0.71   2.53    1.58  0.13        57%  2.00      0.10
    F212N vs F200W   SW-SW      0.81   3.42    2.27  0.08        51%  3.27      0.85
    F182M vs F115W   SW-SW      1.96   4.47    2.18  0.08        74%  5.36      0.28

Read that as: the BULK tie between any two filters is sub-2 mas -- global
alignment is excellent -- while the same two catalogs disagree by 0.5-4.5 mas
rms as a function of position, at 7-45x the per-cell standard error.  A
6-parameter linear fit over the FOV absorbs 14-74% of it above chance.

The amplitude does NOT simply track the SW/LW split: two same-channel,
same-detector pairs (F212N/F200W, F182M/F115W) exceed both SW-LW pairs, and
the largest are those with the widest bandpass separation.  Wavelength
separation is the better predictor.

Watch the ``matched`` column.  F182M/F466N rests on 10% of the F182M list and
F182M/F187N on 13%; anything displaced beyond the 300 mas match radius is
absent from the statistic by construction, so a low fraction is a reason to
read the number cautiously, not a defect of the pair.

Splitting by brightness gives the same field in every magnitude quartile
(1.37-1.46 mas for F212N vs F182M), so this is a WCS-class term, not a
flux-dependent centroid systematic.  Cell size does not move it either.

Usage
-----
    python crossfilter_residual_field.py --field brick \\
        --bands f212n f182m f405n f466n [--cell 45] [--stage resbgsub_m7]

Requires the pipeline installed (it uses ``astrometry_offsets``).
"""
import argparse
import itertools

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
from jwst_gc_pipeline.photometry.astrometry_checkpoint import measure_residual_field

BASE = "/orange/adamginsburg/jwst"


def load(field, band, stage, snr_min=20.0, qfit_max=0.1):
    """Reliable stars from one vetted merged catalog, as SkyCoord + inst mag."""
    path = (f"{BASE}/{field}/catalogs/"
            f"{band}_merged_indivexp_merged_{stage}_dao_basic_vetted.fits")
    tbl = Table.read(path)
    cols = tbl.colnames
    sc = SkyCoord(tbl["skycoord_centroid" if "skycoord_centroid" in cols
                      else "skycoord"])
    keep = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    if "qfit" in cols:
        qfit = np.asarray(tbl["qfit"], float)
        keep &= np.isfinite(qfit) & (qfit <= qfit_max)
    fluxcol = "flux_fit" if "flux_fit" in cols else "flux"
    flux = np.asarray(tbl[fluxcol], float)
    if f"{fluxcol}_err" in cols or "flux_err" in cols:
        err = np.asarray(tbl["flux_err"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = flux / err
        keep &= np.isfinite(snr) & (snr >= snr_min)
    if "replaced_saturated" in cols:
        keep &= ~np.asarray(tbl["replaced_saturated"], bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(np.where(flux > 0, flux, np.nan))
    return sc[keep], np.asarray(mag)[keep]


def pair_field(a, b, cell, label):
    """(bulk, field) for one pair; ``field`` is None when the pair is unusable.

    ``local_residual_map`` raises rather than returning on an unverified tie,
    so a field whose bulk tie fails would traceback out of a survey loop.
    Check the tie here and report the pair as unmeasurable instead.
    """
    bulk = measure_offset(a, b, sweep=True, context=label)
    if bulk is None or not bulk.get("ok") or bulk.get("swept") or bulk["off"] > 100.0:
        return bulk, None
    field = measure_residual_field(a, b, bulk, cell_arcsec=cell,
                                   min_stars=40, context=label)
    return bulk, field


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="brick")
    ap.add_argument("--bands", nargs="+",
                    default=["f212n", "f182m", "f405n", "f466n"])
    ap.add_argument("--stage", default="resbgsub_m7")
    ap.add_argument("--cell", type=float, default=45.0)
    ap.add_argument("--mag-split", action="store_true",
                    help="also split the first pair by magnitude quartile")
    args = ap.parse_args()

    cat = {}
    for band in args.bands:
        cat[band] = load(args.field, band, args.stage)
        print(f"{band}: {len(cat[band][0])} reliable stars", flush=True)

    print(f"\n{'pair':24s} {'bulk':>6s} {'ncell':>5s} {'FOV rms':>8s} "
          f"{'coherent':>9s} {'cell SEM':>9s} {'after affine':>13s} "
          f"{'absorbed*':>10s} {'grad mas/arcmin':>16s}")
    for a, b in itertools.combinations(args.bands, 2):
        bulk, f = pair_field(cat[a][0], cat[b][0], args.cell, f"{a} vs {b}")
        if f is None:
            print(f"{a} vs {b:16s} too few populated cells")
            continue
        print(f"{a} vs {b:16s} {bulk['off']:6.2f} {f['n_cells']:5d} "
              f"{f['rms_mas']:8.2f} {f['coherent_mas']:9.2f} "
              f"{f['median_sem_mas']:9.2f} {f['rms_after_affine_mas']:13.2f} "
              f"{100 * f['affine_absorbed_adjusted']:8.0f}% "
              f"{f['gradient_mas_per_arcmin']:16.2f}")

    if args.mag_split and len(args.bands) >= 2:
        a, b = args.bands[0], args.bands[1]
        sc, mag = cat[a]
        ok = np.isfinite(mag)
        qs = np.nanpercentile(mag[ok], [0, 25, 50, 75, 100])
        print(f"\n--- {a} vs {b}, split by {a} instrumental magnitude quartile ---")
        for i in range(4):
            sel = ok & (mag >= qs[i]) & (mag <= qs[i + 1])
            _, f = pair_field(sc[sel], cat[b][0], args.cell, f"q{i}")
            if f is None:
                continue
            print(f"  Q{i + 1} n={int(sel.sum()):6d} mag {qs[i]:+6.2f}..{qs[i + 1]:+6.2f}"
                  f"  FOV rms {f['rms_mas']:5.2f}  SEM {f['median_sem_mas']:4.2f}"
                  f"  after affine {f['rms_after_affine_mas']:5.2f} mas")


if __name__ == "__main__":
    main()
