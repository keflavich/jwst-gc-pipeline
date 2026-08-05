"""Solve per-(detector, filter) offsets onto the JWST 2-micron consensus frame.

The detector-frame *shape* of the cross-filter residual is small (0.16-0.41 mas
once the anchor's own dither-smearing signature is differenced out).  The term
that matters is the per-detector CONSTANT, because:

  * CRDS ``filteroffset`` is keyed on (CHANNEL, MODULE) -- one entry for the
    four SW detectors of a module -- so a filter offset that varies detector to
    detector inside a module cannot be expressed;
  * the pipeline offsets table is keyed on (visit, exposure, MODULE) too, and
    the tie is a translation, so it cannot remove it either;
  * in a mosaic, different sky positions are covered by different mixes of the
    four detectors, so a per-detector constant becomes a POSITION-DEPENDENT
    sky-frame field -- which is what issue #296 measured.

So: measure, for each (detector, filter), the median residual against the
anchor consensus, with only a per-(exposure, MODULE) median removed -- exactly
what the offsets table can already correct.  What is left is what CRDS and the
table are both blind to.

**SIGN.**  What is measured here is a RESIDUAL, ``(filter - anchor)``.  The
table this writes holds the CORRECTION, which is its NEGATIVE.  ``--write``
does that negation; the printed per-detector numbers are residuals and are
labelled as such.  Getting this backwards doubles the error instead of
removing it, so the two never share a column.

**Frame.**  Sky-frame vectors are only valid at the roll they were solved at.
The portable convention is ``instrument`` = the sky vector rotated by
``+ROLL_REF``; ``filter_frame_correction.instrument_to_sky`` rotates back.
Verified on three observations -- brick jw02221-o001 (roll 89.13), sgrc
jw04147-o012 (91.50), wd2 jw03523-o005 (141.01), two fields and two years
apart: de-rotating collapses them onto one vector set, and a table built from
brick+sgrc alone predicts wd2 -- held out, 52 degrees away in roll -- with a
0.41 mas residual and 95% of the variance explained.  The wrong rotation
direction gives 2.15 mas and explains nothing, so the 1.7-degree degeneracy
between brick and sgrc is broken by wd2.

Estimator: same-star matched pairs through
``astrometry_offsets.local_residual_map`` at a single giant cell, which will
not run until ``measure_offset`` has verified a small global tie and refuses a
swept one.  Nothing here pairs nearest neighbours against a dense catalog by
hand (ASTROMETRY RULE #1).

Usage::

    python solve_filter_frame_offsets.py --field brick --anchor f212n \\
        --bands f182m f187n --max-exp 8
    python solve_filter_frame_offsets.py --field brick --bands f182m \\
        --write out.ecsv          # writes the NEGATED (correction) values
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    local_residual_map, measure_offset)

BASE = "/orange/adamginsburg/jwst"

#: ``local_residual_map`` refuses a global tie larger than radius/3, so this
#: also bounds the largest bulk offset the solve will accept.
MATCH_RADIUS_ARCSEC = 0.15
MIN_PAIRS = 200


def load_merged(field, band, stage, snr_min=20.0, qfit_max=0.1):
    t = Table.read(f"{BASE}/{field}/catalogs/"
                   f"{band}_merged_indivexp_merged_{stage}_dao_basic_vetted.fits")
    c = t.colnames
    sc = SkyCoord(t["skycoord_centroid" if "skycoord_centroid" in c else "skycoord"])
    keep = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    q = np.asarray(t["qfit"], float)
    keep &= np.isfinite(q) & (q <= qfit_max)
    f = np.asarray(t["flux_fit" if "flux_fit" in c else "flux"], float)
    e = np.asarray(t["flux_err"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        keep &= np.isfinite(f / e) & (f / e >= snr_min)
    if "replaced_saturated" in c:
        keep &= ~np.asarray(t["replaced_saturated"], bool)
    return sc[keep]


def exposure_residual(path, anchor_sc, snr_min=20.0, qfit_max=0.1):
    """Median same-star ``(exposure - anchor)`` residual in mas, nothing removed.

    Uses ``local_residual_map`` at one giant cell rather than a hand-rolled
    match-and-median: it enforces the verified-tie precondition, requires the
    partner to be unique, and raises rather than returning a number when the
    tie is not good enough to make nearest-partner the right star.
    """
    t = Table.read(path)
    if "skycoord_centroid" not in t.colnames:
        return None
    q = np.asarray(t["qfit"], float)
    fl = np.asarray(t["flux_fit"], float)
    er = np.asarray(t["flux_err"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = fl / er
    keep = np.isfinite(q) & (q <= qfit_max) & np.isfinite(snr) & (snr >= snr_min)
    if keep.sum() < MIN_PAIRS:
        return None
    sc = SkyCoord(t["skycoord_centroid"][keep])
    sc = sc[np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)]
    if len(sc) < MIN_PAIRS:
        return None

    g = measure_offset(sc, anchor_sc, sweep=True, context=os.path.basename(path))
    if g is None or not g.get("ok") or g.get("swept"):
        return None
    if g["off"] > MATCH_RADIUS_ARCSEC * 1000.0 / 3.0:
        return None
    lrm = local_residual_map(sc, anchor_sc, g, cell_arcsec=1e9,
                             match_radius=MATCH_RADIUS_ARCSEC * u.arcsec,
                             min_stars=MIN_PAIRS, tol_mas=np.inf,
                             context=os.path.basename(path))
    if not lrm["cells"]:
        return None
    c = max(lrm["cells"], key=lambda cc: cc["n"])
    # local_residual_map reports the residual about the applied global tie, so
    # add the tie back to recover the total (exposure - anchor).
    return (g["dra"] + c["dra_mas"], g["ddec"] + c["ddec_mas"], int(c["n"]))


def roll_ref_of(path):
    """``ROLL_REF`` of the frame a per-exposure catalog was measured on."""
    src = Table.read(path).meta.get("FILENAME")
    if not src or not os.path.exists(src):
        return None
    for ext in (("SCI", 1), 0):
        try:
            hdr = fits.getheader(src, ext=ext)
        except (OSError, KeyError, IndexError):
            continue
        if "ROLL_REF" in hdr:
            return float(hdr["ROLL_REF"])
    return None


def run(field, band, anchor_band, stage, max_exp):
    anchor = load_merged(field, anchor_band, stage)
    pat = (f"{BASE}/{field}/{band.upper()}/{band}_nrc*_visit*_vgroup*_exp*_"
           f"{stage}_daophot_basic.fits")
    per = defaultdict(list)
    rolls = []
    seen = set()
    for p in sorted(glob.glob(pat)):
        m = re.search(rf"{band}_(nrc[ab](?:[0-9]|long))_visit(\d+)_vgroup(\w+)_exp(\d+)_",
                      os.path.basename(p))
        if not m:
            continue
        det, visit, expn = m.group(1), m.group(2), m.group(4)
        key = (visit, expn)
        # Cap distinct EXPOSURES.  Testing `len(per[key])` instead would create
        # the key through the defaultdict and make the cap a no-op after the
        # first exposure of a visit.
        if key not in seen and len(seen) >= max_exp:
            continue
        r = exposure_residual(p, anchor)
        if r is None:
            continue
        seen.add(key)
        per[key].append((det, r[0], r[1], r[2]))
        roll = roll_ref_of(p)
        if roll is not None:
            rolls.append(roll)

    # remove a per-(exposure, MODULE) median -- what the offsets table can do
    out = defaultdict(list)
    for rows in per.values():
        for mod in ("nrca", "nrcb"):
            g = [r for r in rows if r[0].startswith(mod)]
            if len(g) < 2:
                continue
            mx = np.median([r[1] for r in g])
            my = np.median([r[2] for r in g])
            for det, dx, dy, _n in g:
                out[det].append((dx - mx, dy - my))
    return out, (float(np.median(rolls)) if rolls else None)


def gauge(vectors):
    """Per-module mean removed, matching ``filter_frame_correction``'s gauge."""
    out = dict(vectors)
    for mod in ("NRCA", "NRCB"):
        members = {d: v for d, v in out.items() if d.upper().startswith(mod)}
        if not members:
            continue
        mean = np.array(list(members.values()), dtype=float).mean(axis=0)
        for d in members:
            out[d] = (out[d][0] - mean[0], out[d][1] - mean[1])
    return out


def sky_residual_to_instrument_correction(vals, roll_ref_deg):
    """Sky ``(filter - anchor)`` residuals -> gauged instrument-frame CORRECTION.

    Rotate by ``+ROLL_REF`` into the instrument frame, negate (residual ->
    correction), then re-gauge so the result honours the module-mean
    convention the file documents.
    """
    t = np.radians(float(roll_ref_deg))
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return gauge({d: tuple(-(rot @ np.asarray(v, dtype=float)))
                  for d, v in vals.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", required=True)
    ap.add_argument("--anchor", default="f212n")
    ap.add_argument("--bands", nargs="+", required=True)
    ap.add_argument("--stage", default="resbgsub_m7")
    ap.add_argument("--max-exp", type=int, default=8)
    ap.add_argument("--roll-ref", type=float, default=None,
                    help="override ROLL_REF (else read from the frames)")
    ap.add_argument("--write", default=None,
                    help="write an instrument-frame ECSV of the NEGATED "
                         "(correction) values")
    args = ap.parse_args()

    rows = []
    for band in args.bands:
        out, roll = run(args.field, band, args.anchor, args.stage, args.max_exp)
        roll = args.roll_ref if args.roll_ref is not None else roll
        print(f"\n### {args.field} {band} - {args.anchor}: per-detector RESIDUAL "
              f"after a per-(exposure, module) median  [mas, sky frame, "
              f"ROLL_REF {roll}]")
        print(f"{'det':10s} {'n':>3s} {'dRA*':>8s} {'dDec':>8s} {'sem':>6s}")
        vals, sems = {}, {}
        for det in sorted(out):
            a = np.array(out[det])
            if len(a) < 2:
                continue
            sem = float(np.hypot(a[:, 0].std(), a[:, 1].std()) / np.sqrt(len(a)))
            print(f"{det:10s} {len(a):3d} {a[:,0].mean():+8.2f} {a[:,1].mean():+8.2f} "
                  f"{sem:6.2f}")
            vals[det.upper()] = (a[:, 0].mean(), a[:, 1].mean())
            sems[det.upper()] = sem
        if vals:
            v = np.array(list(vals.values()))
            print(f"  within-module residual rms: {np.sqrt((v**2).sum(1).mean()):.2f} mas")
        if args.write and vals:
            if roll is None:
                raise SystemExit("--write needs ROLL_REF; pass --roll-ref")
            for det, (dx, dy) in sky_residual_to_instrument_correction(vals, roll).items():
                rows.append((det, band.upper(), args.anchor.upper(), "instrument",
                             round(float(dx), 3), round(float(dy), 3),
                             1, round(sems[det], 3)))

    if args.write and rows:
        tbl = Table(rows=rows, names=("detector", "filter", "anchor", "frame",
                                      "dx_mas", "dy_mas", "n", "sem_mas"))
        tbl.meta["comments"] = [
            "Per-(detector, filter) placement CORRECTIONS onto the JWST "
            "2-micron consensus frame.",
            "dx_mas/dy_mas are the correction to ADD -- the NEGATIVE of the "
            "measured (filter - anchor) residual.",
            "frame=instrument: sky vector rotated by +ROLL_REF; rotate by "
            "-ROLL_REF to apply (filter_frame_correction.instrument_to_sky).",
            "Module-mean removed per module.",
            f"Solved from field {args.field} by solve_filter_frame_offsets.py; "
            f"see issue #296.",
        ]
        tbl.write(args.write, overwrite=True)
        print(f"\nwrote {args.write} ({len(rows)} rows, NEGATED = corrections)")


if __name__ == "__main__":
    main()
