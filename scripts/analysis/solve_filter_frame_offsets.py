"""Solve per-(detector, filter) offsets onto the JWST 2-micron consensus frame.

The detector-frame *shape* turned out to be small (0.16-0.41 mas once the
anchor's own dither-smearing signature is differenced out).  The term that
matters is the per-detector CONSTANT, because:

  * CRDS `filteroffset` is keyed on (CHANNEL, MODULE) -- one value for the four
    SW detectors of a module -- so a filter offset that varies detector to
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

Run on two or more observations, de-rotate each by its own ROLL_REF, and the
vectors collapse onto one instrument-frame set -- verified on brick
jw02221-o001 (roll 89.13), sgrc jw04147-o012 (roll 91.50) and wd2 jw03523-o005
(roll 141.01): 94-98% of the variance explained pairwise, per-detector scatter
0.19 mas on a 1.69 mas signal.  Output feeds
jwst_gc_pipeline/reduction/data/filter_frame_offsets.ecsv.

Usage:
    python solve_filter_frame_offsets.py --field brick --anchor f212n \
        --bands f182m f187n --max-exp 8
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset

BASE = "/orange/adamginsburg/jwst"


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


def exposure_residual(path, anchor_sc, snr_min=20.0, qfit_max=0.1,
                      match_radius=0.15 * u.arcsec):
    """Median same-star (exposure - anchor) offset in mas, NO median removal."""
    t = Table.read(path)
    c = t.colnames
    if "skycoord_centroid" not in c:
        return None
    q = np.asarray(t["qfit"], float)
    fl = np.asarray(t["flux_fit"], float)
    er = np.asarray(t["flux_err"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = fl / er
    keep = np.isfinite(q) & (q <= qfit_max) & np.isfinite(snr) & (snr >= snr_min)
    if keep.sum() < 200:
        return None
    sc = SkyCoord(t["skycoord_centroid"][keep])
    sc = sc[np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)]
    if len(sc) < 200:
        return None
    g = measure_offset(sc, anchor_sc, sweep=True, context=os.path.basename(path))
    if g is None or not g.get("ok") or g["off"] > 300.0:
        return None
    cosd = float(np.cos(np.radians(np.median(sc.dec.deg))))
    al = SkyCoord((sc.ra.deg + g["dra"] / 3.6e6 / cosd) * u.deg,
                  (sc.dec.deg + g["ddec"] / 3.6e6) * u.deg, frame="icrs")
    i1, d2d, _ = al.match_to_catalog_sky(anchor_sc)
    ok = d2d < match_radius
    j2, _, _ = anchor_sc[i1[ok]].match_to_catalog_sky(al)
    idx = np.flatnonzero(ok)[j2 == np.flatnonzero(ok)]
    if len(idx) < 200:
        return None
    dra = (al[idx].ra - anchor_sc[i1[idx]].ra).to(u.mas).value * cosd
    dde = (al[idx].dec - anchor_sc[i1[idx]].dec).to(u.mas).value
    # the histogram tie was applied above, so add it back: we want the total
    return (float(np.median(dra)) - g["dra"], float(np.median(dde)) - g["ddec"],
            len(idx))


def run(field, band, anchor_band, stage, max_exp):
    anchor = load_merged(field, anchor_band, stage)
    pat = (f"{BASE}/{field}/{band.upper()}/{band}_nrc*_visit*_vgroup*_exp*_"
           f"{stage}_daophot_basic.fits")
    per = defaultdict(list)
    for p in sorted(glob.glob(pat)):
        m = re.search(rf"{band}_(nrc[ab](?:[0-9]|long))_visit(\d+)_vgroup(\w+)_exp(\d+)_",
                      os.path.basename(p))
        if not m:
            continue
        det, visit, expn = m.group(1), m.group(2), m.group(4)
        key = (visit, expn)
        if sum(1 for k in per if k[0] == visit) and len(per[(visit, expn)]) >= 10:
            continue
        if len({k[1] for k in per}) > max_exp and (visit, expn) not in per:
            continue
        r = exposure_residual(p, anchor)
        if r is not None:
            per[key].append((det, r[0], r[1], r[2]))

    # remove a per-(exposure, MODULE) median -- what the offsets table can do
    out = defaultdict(list)
    for key, rows in per.items():
        for mod in ("nrca", "nrcb"):
            g = [r for r in rows if r[0].startswith(mod)]
            if len(g) < 2:
                continue
            mx = np.median([r[1] for r in g])
            my = np.median([r[2] for r in g])
            for det, dx, dy, n in g:
                out[det].append((dx - mx, dy - my))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--anchor", default="f212n")
    ap.add_argument("--bands", nargs="+", required=True)
    ap.add_argument("--stage", default="resbgsub_m7")
    ap.add_argument("--max-exp", type=int, default=8)
    args = ap.parse_args()

    for band in args.bands:
        out = run(args.field, band, args.anchor, args.stage, args.max_exp)
        print(f"\n### {args.field} {band} - {args.anchor}: per-detector constant "
              f"AFTER a per-(exposure, module) median  [mas]")
        print(f"{'det':10s} {'n':>3s} {'dRA*':>8s} {'dDec':>8s} {'sem':>6s}")
        vals = []
        for det in sorted(out):
            a = np.array(out[det])
            if len(a) < 2:
                continue
            sem = float(np.hypot(a[:, 0].std(), a[:, 1].std()) / np.sqrt(len(a)))
            print(f"{det:10s} {len(a):3d} {a[:,0].mean():+8.2f} {a[:,1].mean():+8.2f} "
                  f"{sem:6.2f}")
            vals.append([a[:, 0].mean(), a[:, 1].mean()])
        if vals:
            v = np.array(vals)
            print(f"  within-module residual rms: {np.sqrt((v**2).sum(1).mean()):.2f} mas")


if __name__ == "__main__":
    main()
