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
    overlap_offset_grid, pairwise_overlap_offsets, DEFAULT_OVERLAP_TOL_MAS)
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
        if tok.startswith("nrc") or tok.startswith("mirimage"):
            det = tok
            break
    if det.startswith("mirimage"):
        module = "mirimage"
    else:
        module = det[:4] if det else "det?"  # nrca / nrcb (SW) or nrcalong/nrcblong
    return f"{visit}:{module}"


def build_groups(field, filt):
    # NIRCam crf carry the _destreak token (working-copy lineage); MIRI skips
    # destreak so its crf do NOT (jw..._mirimage_o001_crf.fits) -- glob both.
    # Excluding MIRI silently PASSED it, and MIRI visit misregistration is a
    # known failure (the F2550W doubled-star saga).
    pats = (f"{BASE}/{field}/{filt}/pipeline/jw*_*_nrc*_destreak_o*_crf.fits",
            f"{BASE}/{field}/{filt}/pipeline/jw*_*_nrc*_o*_crf.fits",
            f"{BASE}/{field}/{filt}/pipeline/jw*_*_mirimage*_o*_crf.fits")
    frames = sorted({fn for pat in pats for fn in glob.glob(pat)})
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
    # FAIL-CLOSED on "found nothing": a gate that goes green because its glob
    # matched zero files (renamed products, naming drift) is the silent
    # false-agreement class this repo bans.  Distinguish it from a genuine
    # single-group field (frames found, nothing to pairwise-check).
    if nframes == 0:
        if verbose:
            print(f"  {field} {filt}: NO crf frames matched -- cannot verify "
                  f"inter-frame registration (glob mismatch / missing products?)",
                  flush=True)
        return dict(field=field, filt=filt, PASS=False, could_not_verify=True,
                    note="no crf frames matched")
    if not pooled:
        if verbose:
            print(f"  {field} {filt}: {nframes} crf but detection produced NO "
                  f"usable groups -- cannot verify", flush=True)
        return dict(field=field, filt=filt, PASS=False, could_not_verify=True,
                    note="no detections from any crf")
    if len(pooled) < 2:
        if verbose:
            print(f"  {field} {filt}: single exposure-group from {nframes} crf "
                  f"-- nothing to overlap-check", flush=True)
        return dict(field=field, filt=filt, PASS=True,
                    note=f"single exposure-group ({nframes} crf)")

    # PER-TILE (local) reference-free check: a per-visit residual is spatially
    # varying, so a field-pooled single offset can average below tol while a thin
    # seam is ~90 mas off. Grid it.
    # TWO-LAYER frame-vs-frame check, both reference-free:
    #   fine  : per-tile grid on mutual-coverage cells (owns the 30-100 mas
    #           seam regime; blind BY CONSTRUCTION to offsets > its cell
    #           margin, which empty the mutual-coverage cells);
    #   gross : field-pooled SWEPT histogram over the intersection populations
    #           (owns the >margin regime -- the brick-1182 v001 ~20" class --
    #           a gross rigid offset still overlaps in the strip, so the swept
    #           peak recovers it with no reference catalog).
    # A pair verifies only if a layer POSITIVELY measured it; a pair NEITHER
    # layer can measure is could-not-verify -> exit 2 (fail closed: "a
    # failsafe that cannot run is not a passing failsafe").
    res = overlap_offset_grid(pooled, tol_mas=TOL_MAS, nx=GRID_N, ny=GRID_N,
                              maxsep=3.0 * u.arcsec)
    pw = {(r["a"], r["b"]): r for r in pairwise_overlap_offsets(
        pooled, tol_mas=TOL_MAS, maxsep=3.0 * u.arcsec)}
    GROSS_MAS = 3.0 * 3000.0  # beyond the grid's ~3*maxsep cell margin
    bad, unverifiable, overlapped = [], [], []
    for r in res:
        if not r["overlap"]:
            continue
        overlapped.append(r)
        p = pw.get((r["a"], r["b"])) or pw.get((r["b"], r["a"])) or {}
        r["pairwise"] = {k: p.get(k) for k in
                         ("off_mas", "dra_mas", "ddec_mas", "contrast",
                          "n_peak", "measurable", "swept", "ok")}
        if r.get("could_not_verify"):
            # fine layer blind -> the gross layer must decide
            if p.get("measurable"):
                if not p["ok"]:
                    r["fail_reason"] = (f"gross pairwise offset "
                                        f"{p['off_mas']:.0f} mas (swept="
                                        f"{p.get('swept')})")
                    bad.append(r)
            else:
                unverifiable.append(r)
        else:
            if not r["ok"]:
                r["fail_reason"] = "per-tile misregistration"
                bad.append(r)
            elif (p.get("measurable") and p.get("off_mas") is not None
                    and p["off_mas"] > GROSS_MAS):
                # tiles measured fine locally but the pooled swept peak sits
                # beyond the grid's sight -- gross regime, fail
                r["fail_reason"] = f"gross pairwise offset {p['off_mas']:.0f} mas"
                bad.append(r)
    if verbose:
        print(f"  {field} {filt}: {nframes} crf -> {len(pooled)} groups, "
              f"{len(overlapped)} overlapping pairs, {len(bad)} FAIL, "
              f"{len(unverifiable)} could-not-verify (tol {TOL_MAS:.0f} mas, "
              f"{GRID_N}x{GRID_N} tiles + pooled swept)", flush=True)
        for r in sorted(overlapped, key=lambda r: -(r["worst_off_mas"] or 0))[:8]:
            tag = ("FAIL" if r in bad
                   else ("could-not-verify" if r in unverifiable else "ok"))
            _w = r["worst_off_mas"]
            _p = r.get("pairwise", {})
            print(f"      {tag}: {r['a']} | {r['b']}  worst tile off="
                  f"{'n/a' if _w is None else f'{_w:.0f} mas'} "
                  f"({r['n_ok']}/{r['n_total']} tiles ok, "
                  f"{r.get('n_no_coverage', 0)} no-mutual-coverage cells; "
                  f"pooled off={'n/a' if _p.get('off_mas') is None else f'{_p['off_mas']:.0f} mas'}"
                  f"{' MEASURABLE' if _p.get('measurable') else ''}"
                  f"{'; ' + r['fail_reason'] if r.get('fail_reason') else ''})",
                  flush=True)
        for r in unverifiable:
            print(f"      COULD NOT VERIFY: {r['a']} | {r['b']} -- footprints "
                  f"intersect but neither the per-tile grid (no mutual-coverage "
                  f"cells) nor the pooled swept histogram (no measurable peak) "
                  f"could measure the pair.  Fail-closed: requires the external "
                  f"reference map (--refcat) or a fixed reduction to stage.",
                  flush=True)

    ext_fail = False
    ext_ran = False
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
            ext_ran = True
            if verbose:
                wc = g.get("worst_off_cell")
                print(f"      fine grid {GRID_N}x{GRID_N} vs {rn}: clean={g['clean']} "
                      f"worst_off={g['worst_off_mas']:.0f} mas "
                      f"(max {GRID_MAX_OFF_MAS:.0f}) n_ok={g['n_ok']}/{g['n_total']}",
                      flush=True)
            if not g["clean"]:
                ext_fail = True

    # FAIL-CLOSED: a pair NEITHER frame-vs-frame layer could measure only
    # passes when the external reference map ran AND is clean; otherwise the
    # filter is could-not-verify (exit 2 -- refused by stage_release).
    could_not_verify = bool(unverifiable) and not (ext_ran and not ext_fail)
    if unverifiable and ext_ran and not ext_fail and verbose:
        print(f"      could-not-verify pair(s) accepted via the CLEAN external "
              f"reference map", flush=True)
    return dict(field=field, filt=filt,
                PASS=bool(not bad and not ext_fail and not could_not_verify),
                could_not_verify=could_not_verify,
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
    any_noverify = False
    for f in filts:
        r = check_filter(args.field, f, refcat=args.refcat)
        if r.get("could_not_verify"):
            any_noverify = True
        elif not r.get("PASS"):
            any_fail = True
    if any_fail:
        print(f"\nOVERLAP GATE: FAIL for {args.field} -- inter-frame misregistration "
              f"(> {TOL_MAS:.0f} mas). Do NOT stage; re-examine per-visit alignment.",
              flush=True)
        return 1
    if any_noverify:
        # exit 2 = could-not-verify: distinct from a measured FAIL, but still
        # refused by stage_release (its rc != 0 branch) -- fail closed, never
        # green-because-the-glob-matched-nothing.
        print(f"\nOVERLAP GATE: COULD NOT VERIFY {args.field} -- at least one filter "
              f"had no matchable crf frames / no detections. Fix the products or the "
              f"glob; a gate that finds nothing is not a passing gate.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
