#!/usr/bin/env python
"""BLOCKING gate: reference-free inter-frame overlap registration.

The brick-1182 F200W seam failure (2026-07-12) proved that our existing gates are
STRUCTURALLY BLIND to a per-visit residual:

- ``registration_failsafes.py`` matches the mosaic against its OWN merged catalog.
  Both are built from the same ``_crf`` frames, so if visit-001 carries a ~90 mas
  residual, the catalog inherits it too -> mosaic and catalog AGREE (offset ~0) ->
  PASS, even though both are wrong. A self-referential truth cannot see a shared
  error. It also searches only +-2.5" (no sweep) so a gross overlap offset reads as
  "no overlap", not FAIL.
- Bulk / coarse-grid vs-reference checks average the two visits together in the
  overlap and read ~0.

The ONLY check that sees it is REFERENCE-FREE and PAIRWISE: detect on each
per-exposure ``_crf`` frame (each on its own corrected GWCS), pool by
exposure-group (visit x module), and histogram-stack every OVERLAPPING pair's
MUTUAL offset. Overlapping same-instrument groups must co-register to < 30 mas.
Non-zero exit on FAIL so it gates a release/staging chain.

Usage::

    python check_interframe_overlap.py --field brick --filter F200W
    python check_interframe_overlap.py --field brick --scan           # all filters
    # optional external absolute cross-check (fine grid vs VIRAC2/Gaia):
    python check_interframe_overlap.py --field brick --filter F200W --refcat <path>
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from photutils.detection import find_peaks

from jwst_gc_pipeline.photometry.interframe_overlap import (
    overlap_offset_grid, DEFAULT_OVERLAP_TOL_MAS)
from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset_grid

BASE = os.environ.get("JWST_BASE", "/orange/adamginsburg/jwst")
TOL_MAS = float(os.environ.get("OVERLAP_TOL_MAS", DEFAULT_OVERLAP_TOL_MAS))
# external-reference fine-grid gate
GRID_N = int(os.environ.get("OVERLAP_GRID_N", 16))
GRID_MAX_OFF_MAS = float(os.environ.get("OVERLAP_GRID_MAX_OFF_MAS", 80.0))


def _detect(path, nsigma=8.0, box=5):
    with fits.open(path, memmap=True) as h:
        d = np.asarray(h["SCI"].data)
        w = WCS(h["SCI"].header)
    fin = np.isfinite(d)
    if fin.sum() < 2000:
        return None
    _, med, std = sigma_clipped_stats(d[fin], sigma=3.0, maxiters=3)
    tb = find_peaks(np.where(fin, d, 0.0), threshold=med + nsigma * std, box_size=box)
    if tb is None or len(tb) < 20:
        return None
    sky = w.pixel_to_world(np.array(tb["x_peak"]), np.array(tb["y_peak"]))
    return SkyCoord(sky.ra, sky.dec)


def _group_key(crf_path):
    """Group per-exposure crf by (visit, module).  Filename like
    jw01182004001_04101_00001_nrca3_destreak_o004_crf.fits -> ('004001','nrca')."""
    base = os.path.basename(crf_path)
    visit = base.split("_")[0][-6:]        # e.g. 004001
    det = ""
    for tok in base.split("_"):
        if tok.startswith("nrc"):
            det = tok
            break
    module = det[:4] if det else "det?"    # nrca / nrcb (SW) or nrcalong/nrcblong
    return f"{visit}:{module}"


def build_groups(field, filt):
    pat = f"{BASE}/{field}/{filt}/pipeline/jw*_*_nrc*_destreak_o*_crf.fits"
    frames = sorted(glob.glob(pat))
    groups = {}
    ndet = {}
    for fn in frames:
        s = _detect(fn)
        if s is None:
            continue
        k = _group_key(fn)
        groups.setdefault(k, []).append(s)
        ndet[k] = ndet.get(k, 0) + len(s)
    pooled = {k: SkyCoord(np.concatenate([c.ra.deg for c in v]) * u.deg,
                          np.concatenate([c.dec.deg for c in v]) * u.deg)
              for k, v in groups.items()}
    return pooled, ndet, len(frames)


def _refcat(path):
    t = Table.read(path)
    cols = {c.lower(): c for c in t.colnames}
    ra, dec = t[cols["ra"]], t[cols["dec"]]
    rc = SkyCoord(ra.data * u.deg if ra.unit is None else ra,
                  dec.data * u.deg if dec.unit is None else dec)
    src = cols.get("source")
    gaia = None
    if src is not None:
        gm = np.array([s in (b"GaiaDR3", "GaiaDR3") for s in t[src]])
        gaia = rc[gm] if gm.any() else None
    return rc, gaia


def check_filter(field, filt, refcat=None, verbose=True):
    pooled, ndet, nframes = build_groups(field, filt)
    if len(pooled) < 2:
        if verbose:
            print(f"  {field} {filt}: only {len(pooled)} group(s) from {nframes} "
                  f"crf -- nothing to overlap-check", flush=True)
        return dict(field=field, filt=filt, PASS=True, note="insufficient groups")

    # PER-TILE (local) reference-free check: a per-visit residual is spatially
    # varying, so a field-pooled single offset can average below tol while a thin
    # seam is ~90 mas off. Grid it.
    res = overlap_offset_grid(pooled, tol_mas=TOL_MAS, nx=GRID_N, ny=GRID_N,
                              maxsep=3.0 * u.arcsec)
    bad = [r for r in res if r["overlap"] and not r["ok"]]
    overlapped = [r for r in res if r["overlap"]]
    if verbose:
        print(f"  {field} {filt}: {nframes} crf -> {len(pooled)} groups, "
              f"{len(overlapped)} overlapping pairs, {len(bad)} FAIL (tol {TOL_MAS:.0f} mas, "
              f"{GRID_N}x{GRID_N} tiles)", flush=True)
        for r in sorted(overlapped, key=lambda r: -(r["worst_off_mas"] or 0))[:8]:
            tag = "FAIL" if not r["ok"] else "ok"
            print(f"      {tag}: {r['a']} | {r['b']}  worst tile off={r['worst_off_mas']:.0f} mas "
                  f"cell={r['worst_off_cell']} ({r['n_ok']}/{r['n_total']} tiles ok)", flush=True)

    ext_fail = False
    if refcat:
        rc, gaia = _refcat(refcat)
        allsrc = SkyCoord(np.concatenate([p.ra.deg for p in pooled.values()]) * u.deg,
                          np.concatenate([p.dec.deg for p in pooled.values()]) * u.deg)
        for rn, rr in (("VIRAC2", rc), ("Gaia", gaia)):
            if rr is None:
                continue
            g = measure_offset_grid(allsrc, rr, nx=GRID_N, ny=GRID_N,
                                    maxsep=3.0 * u.arcsec, max_off_mas=GRID_MAX_OFF_MAS,
                                    context=f"{field}/{filt}/{rn}")
            if verbose:
                wc = g.get("worst_off_cell")
                print(f"      fine grid {GRID_N}x{GRID_N} vs {rn}: clean={g['clean']} "
                      f"worst_off={g['worst_off_mas']:.0f} mas "
                      f"(max {GRID_MAX_OFF_MAS:.0f}) n_ok={g['n_ok']}/{g['n_total']}",
                      flush=True)
            if not g["clean"]:
                ext_fail = True

    return dict(field=field, filt=filt, PASS=bool(not bad and not ext_fail),
                n_fail=len(bad), pairs=res)


def field_filters(field):
    fs = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                 for p in glob.glob(f"{BASE}/{field}/*/pipeline/")})
    return [f for f in fs if f.upper().startswith("F")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", required=True)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--scan", action="store_true", help="every filter of the field")
    ap.add_argument("--refcat", default=None,
                    help="optional external refcat for the fine-grid absolute cross-check")
    args = ap.parse_args(argv)

    filts = field_filters(args.field) if args.scan else [args.filter]
    if not filts or filts == [None]:
        print("ERROR: give --filter or --scan", file=sys.stderr)
        return 2
    any_fail = False
    for f in filts:
        r = check_filter(args.field, f, refcat=args.refcat)
        if not r.get("PASS"):
            any_fail = True
    if any_fail:
        print(f"\nOVERLAP GATE: FAIL for {args.field} -- inter-frame misregistration "
              f"(> {TOL_MAS:.0f} mas). Do NOT stage; re-examine per-visit alignment.",
              flush=True)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
