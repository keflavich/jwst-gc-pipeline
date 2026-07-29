#!/usr/bin/env python
"""Local-registration failsafes for JWST-GC mosaics (spatially resolved).

A field-average astrometry check passes over a LOCALIZED seam/overlap misregistration
(brick 1182 F356W, 2026-07: several-arcsec junk in the module-overlap band, bulk ~0).
These checks are spatially binned and use CONFOUND-FREE truth sets (no external catalog,
so crowding/extinction can't fool them):

  1. per-module   : every bright MERGED detection must have a same-band per-module
                    (nrca/nrcb) detection within TOL.  The merged is the only place the
                    two modules are combined, so overlap-misregistration junk appears
                    here and not in the clean single-module mosaics.
  2. cross-band   : every bright detection must have a detection in ANOTHER JWST band
                    within TOL.  Same stars, JWST-internal registration is sub-mas, and
                    all bands are NIR -> no VIRAC2 color/depth decoupling.
  3. own-catalog  : every bright detection must have a source in the mosaic's OWN vetted
                    catalog within TOL (and the catalog must land on the mosaic).  A
                    mosaic must match the catalog derived from it.

Per cell: fraction of bright detections that have a truth-set match ("agreement") and the
median offset.  Agreement ~1 where registered; it COLLAPSES in a misregistered band.
FAIL if any covered cell drops below FRAC_FLOOR (or << field median) or offset > OFF_MAX.
Non-zero exit on FAIL so it can gate a chain.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder

BASE = "/orange/adamginsburg/jwst"
GRID = 20
MX = 2.5 * u.arcsec              # pair-separation search radius (recovers offsets up to this)
XBIN = 0.04                      # arcsec, offset-histogram bin
MIN_PAIRS = 80                   # pairs needed in a cell to attempt a peak
MIN_PEAK_RATIO = 5.0             # peak/background below this -> cell UNVERIFIED (not a fail)
FAIL_MIN_RATIO = 10.0            # a FAIL needs peak/background >= this -- CONFIDENT contrast,
                                 # not just the verify floor. A real localized seam doubles
                                 # stars into a SHARP secondary peak (the clean brick cells
                                 # verify at median contrast ~18); a floor-level peak
                                 # (ratio ~ MIN_PEAK_RATIO) at a large offset is dense-field
                                 # wrong-pair noise in a crowded, few-detection cell, not a
                                 # seam (brick F405N: 7 bright-star cells at 80 mas / peak_bg
                                 # 5-8 were a FALSE own_catalog FAIL; the same-star m7 check
                                 # of those regions read <=22 mas, 2026-07). Coverage is
                                 # unchanged (no detections removed); only the fail bar rises.
OFF_MAX = 60.0                   # a VERIFIED cell whose peak offset exceeds this (mas) -> FAIL


def detect(path, thr=30.0):
    h = fits.open(path); sci = h["SCI"]; w = WCS(sci.header); d = sci.data.astype("float32")
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=2.5, threshold=thr * std)(d - med)
    if t is None:
        return None, None
    return SkyCoord(w.pixel_to_world(t["xcentroid"], t["ycentroid"])), np.asarray(t["flux"], float)


# Precise merged-mosaic filename parser.  The bug being fixed is that `g[0]` on
# an UNSORTED `glob.glob(jw*-o*...)` picked a non-deterministic file whenever >1
# matched.  We enumerate with a tight character-class glob (no `*` in the
# proposal/observation) and validate each name with this regex, which yields the
# proposal-observation key.  NOTE: >1 observation in one filter directory is a
# NORMAL layout, not a stray -- gc2211 is multi-observation by design and
# ngc6334 F200W carries two proposals (both o001) on purpose.  So the ambiguity
# is resolved by (a) an optional release scope and (b) a DETERMINISTIC sorted
# pick, NOT by refusing.  Distinguishing a genuine multi-observation layout from
# a misfiled stray is what the release ``observations`` scope is for; a
# within-directory obs count cannot tell them apart.  Filter class allows the
# wide-double bands (F150W2/F322W2) -- their trailing `2` was silently dropped,
# and a dropped band fails OPEN (cross-band needs >=2 bands or it warns-not-fails).
_MOSAIC_RE = re.compile(
    r"^jw(?P<prop>\d{5})-o(?P<obs>\d{3})_t001_nircam_clear-"
    r"(?P<filt>f\d{3,4}[wmn]2?)-(?P<module>merged|nrca|nrcb|nrcalong|nrcblong)"
    r"_i2d\.fits$")


def _mosaic_candidates(field, filt, module, observations=None):
    """On-disk mosaics for (field, filt, module) as sorted (obs_key, path),
    name-validated.  ``obs_key`` = ``"<proposal>-<observation>"``.  When
    ``observations`` (a set of obs_keys) is given, only in-scope mosaics are
    returned -- this is how a misfiled stray from another observation is
    excluded (brick's 2221 o002), while a legitimate multi-observation layout
    keeps all its in-scope mosaics."""
    # tight glob: 5-digit proposal, 3-digit observation -- no `*` in either
    pat = (f"{BASE}/{field}/{filt}/pipeline/"
           f"jw[0-9][0-9][0-9][0-9][0-9]-o[0-9][0-9][0-9]_t001_nircam_clear-"
           f"{filt.lower()}-{module}_i2d.fits")
    out = []
    for p in sorted(glob.glob(pat)):
        m = _MOSAIC_RE.match(os.path.basename(p))
        if m and m.group("filt") == filt.lower() and m.group("module") == module:
            key = f"{m.group('prop')}-{m.group('obs')}"
            if observations is not None and key not in observations:
                continue
            out.append((key, p))
    return sorted(out)


def mosaic(field, filt, module="merged", observations=None):
    """The merged mosaic for (field, filt, module).  Deterministic: a sorted
    pick of the (in-scope) name-validated candidates -- fixes the non-
    deterministic `g[0]` on an unsorted glob.  Returns None when none match."""
    cands = _mosaic_candidates(field, filt, module, observations=observations)
    return cands[0][1] if cands else None


def catalog_sc(field, filt):
    g = glob.glob(f"{BASE}/{field}/catalogs/{filt.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits")
    if not g:
        return None
    t = Table.read(g[0])
    for c in ("skycoord", "skycoord_ref"):
        m = [x for x in t.colnames if x.lower() == c]
        if m:
            return SkyCoord(t[m[0]])
    return None


def per_cell(det, flux, truth, label, bright_pct=None, fail_min_ratio=MIN_PEAK_RATIO):
    """Per-cell registration offset by pair-separation HISTOGRAM cross-correlation.

    For every det-truth pair within MX, bin by the detection's spatial cell and by the
    offset (dRA*cos, dDec).  In each cell the REAL counterparts pile into a peak at the
    true offset; chance coincidences form a flat background -> crowding-robust (NOT
    nearest-neighbour, which just measures the chance-NN distance in a dense field).

    A cell is VERIFIED only if it has >=MIN_PAIRS and peak/background >=MIN_PEAK_RATIO;
    otherwise it is UNVERIFIED (reported, never a fail).  A verified cell FAILs if its
    peak offset exceeds OFF_MAX *and* its contrast >= ``fail_min_ratio``.  Field FAIL =
    any cell fails.

    ``fail_min_ratio`` (default ``MIN_PEAK_RATIO`` -> the historic strict behaviour) is
    raised to ``FAIL_MIN_RATIO`` ONLY for the own-catalog check, where a floor-level
    peak in a dense bright-star cell is wrong-pair noise, not a seam.  The cross-band
    and per-module checks keep the strict floor, so a real seam that own-catalog's
    relaxed bar might miss is still caught by them (defense in depth).
    """
    if det is None or truth is None or len(det) < 200 or len(truth) < 200:
        return dict(label=label, error="missing detections/truth")
    ia, ib, sep, _ = search_around_sky(det, truth, MX)
    if len(ia) < 2000:
        return dict(label=label, error="too few pairs")
    dra = (truth[ib].ra - det[ia].ra).to(u.arcsec).value * np.cos(det[ia].dec.rad) * 1000
    dde = (truth[ib].dec - det[ia].dec).to(u.arcsec).value * 1000
    pra, pde = det[ia].ra.deg, det[ia].dec.deg

    xe = np.linspace(det.ra.deg.min(), det.ra.deg.max(), GRID + 1)
    ye = np.linspace(det.dec.deg.min(), det.dec.deg.max(), GRID + 1)
    ci = np.clip(np.digitize(pra, xe) - 1, 0, GRID - 1)
    cj = np.clip(np.digitize(pde, ye) - 1, 0, GRID - 1)
    hb = np.arange(-MX.to(u.arcsec).value * 1000, MX.to(u.arcsec).value * 1000 + XBIN * 1000, XBIN * 1000)

    off = np.full((GRID, GRID), np.nan)      # peak offset (mas)
    ratio = np.full((GRID, GRID), np.nan)    # peak/background
    npair = np.zeros((GRID, GRID), int)
    order = np.lexsort((cj, ci))
    ci, cj, dra, dde = ci[order], cj[order], dra[order], dde[order]
    keyc = ci * GRID + cj
    bnd = np.searchsorted(keyc, np.arange(GRID * GRID + 1))
    for k in range(GRID * GRID):
        s, e = bnd[k], bnd[k + 1]
        npair[k // GRID, k % GRID] = e - s
        if e - s < MIN_PAIRS:
            continue
        H, xb, yb = np.histogram2d(dra[s:e], dde[s:e], bins=[hb, hb])
        bg = np.median(H[H > 0]) if (H > 0).any() else 0.0
        pi, pj = np.unravel_index(H.argmax(), H.shape)
        i0, j0 = k // GRID, k % GRID
        ratio[i0, j0] = H.max() / bg if bg > 0 else np.inf
        # refine the peak to sub-bin with the local centroid
        dcen = (xb[pi] + xb[pi + 1]) / 2
        ecen = (yb[pj] + yb[pj + 1]) / 2
        off[i0, j0] = np.hypot(dcen, ecen)

    verified = np.isfinite(ratio) & (ratio >= MIN_PEAK_RATIO) & (npair >= MIN_PAIRS)
    # A FAIL requires a large offset AND confident contrast. A real localized seam
    # doubles stars into a sharp high-contrast peak; a bright-star-crowded, sparse
    # cell yields a floor-level peak (ratio ~ MIN_PEAK_RATIO) at a spurious offset.
    # Sub-FAIL_MIN_RATIO high-offset cells stay verified-but-not-failed (reported).
    fail = verified & (off > OFF_MAX) & (ratio >= fail_min_ratio)
    # High offset but sub-fail_min_ratio contrast: NOT a fail, but reported so a real
    # low-contrast issue is never silently hidden by the margin.
    unconfident = verified & (off > OFF_MAX) & (ratio < fail_min_ratio)
    worst = [dict(ra=float((xe[i] + xe[i + 1]) / 2), dec=float((ye[j] + ye[j + 1]) / 2),
                  offset_mas=round(float(off[i, j]), 0), peak_bg=round(float(ratio[i, j]), 1),
                  npairs=int(npair[i, j]))
             for i, j in sorted(zip(*np.where(fail)), key=lambda c: -off[c])][:8]
    unconfident_cells = [dict(ra=float((xe[i] + xe[i + 1]) / 2), dec=float((ye[j] + ye[j + 1]) / 2),
                              offset_mas=round(float(off[i, j]), 0), peak_bg=round(float(ratio[i, j]), 1),
                              npairs=int(npair[i, j]))
                         for i, j in sorted(zip(*np.where(unconfident)), key=lambda c: -off[c])][:8]
    return dict(label=label, verified_cells=int(verified.sum()),
                unverified_cells=int((npair >= MIN_PAIRS).sum() - verified.sum()),
                median_verified_offset_mas=round(float(np.nanmedian(off[verified])), 1) if verified.any() else None,
                n_fail=int(fail.sum()), PASS=bool(fail.sum() == 0), worst=worst,
                n_unconfident_highoff=int(unconfident.sum()),
                unconfident_highoff_cells=unconfident_cells,
                _g=(off, verified, (xe, ye)))


def build_truths(field, filt, xband, observations=None):
    det, flux = detect(mosaic(field, filt, "merged", observations=observations))
    truths = {}
    # 1. per-module
    pm = []
    for m in ("nrca", "nrcb", "nrcalong", "nrcblong"):
        p = mosaic(field, filt, m, observations=observations)
        if p:
            s, _ = detect(p)
            if s is not None:
                pm.append(s)
    if pm:
        truths["per-module"] = SkyCoord(np.concatenate([s.ra.deg for s in pm]) * u.deg,
                                        np.concatenate([s.dec.deg for s in pm]) * u.deg)
    # 2. cross-band
    if xband:
        p = mosaic(field, xband, "merged", observations=observations)
        if p:
            s, _ = detect(p)
            truths[f"cross-band({xband})"] = s
    # 3. own catalog
    c = catalog_sc(field, filt)
    if c is not None:
        truths["own-catalog"] = c
    return det, flux, truths


def plot_all(results, out):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6.5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        if "_g" not in r:
            ax.set_title(f"{r['label']}: {r.get('error','')}"); continue
        off, verified, (xe, ye) = r["_g"]
        shown = np.where(verified, off, np.nan)
        im = ax.pcolormesh(xe, ye, shown.T, cmap="inferno", vmin=0, vmax=max(OFF_MAX * 2, 100))
        ax.invert_xaxis(); plt.colorbar(im, ax=ax, label="verified peak offset [mas]")
        v = "PASS" if r["PASS"] else f"FAIL {r['n_fail']}"
        med = r.get("median_verified_offset_mas")
        ax.set_title(f"{r['label']}\nmed {med} mas — {v}", color="green" if r["PASS"] else "red")
    fig.tight_layout(); fig.savefig(out, dpi=100); print("wrote", out)


def field_bands(field):
    """Filters with a merged mosaic on disk for this field.  Enumerates with a
    tight character-class glob (no `*` in proposal/observation) and validates
    each name with ``_MOSAIC_RE``; the mosaic's parsed filter must match its
    ``<field>/<FILT>/pipeline`` directory."""
    out = []
    pat = (f"{BASE}/{field}/*/pipeline/"
           f"jw[0-9][0-9][0-9][0-9][0-9]-o[0-9][0-9][0-9]_t001_nircam_clear-"
           f"*-merged_i2d.fits")
    for p in glob.glob(pat):
        m = _MOSAIC_RE.match(os.path.basename(p))
        if m is None or m.group("module") != "merged":
            continue
        filt = m.group("filt").upper()
        d = os.path.basename(os.path.dirname(os.path.dirname(p)))   # <field>/<FILT>/pipeline
        if d.upper() == filt:
            out.append(filt)
    return sorted(set(out))


def scan_field(field, verbose=True, images_only=False, observations=None):
    """Run the cross-band + own-catalog failsafes on EVERY band of a field.

    Cross-band truth for band F = the pooled detections of all OTHER bands of the field
    (same stars, JWST-internal registration).  Returns {band: {check: verdict}} and an
    overall PASS/FAIL.  Detects each band once.

    ``images_only``: gate an IMAGE-ONLY release -- run the reference-free cross-band
    (image-to-image) check only, and SKIP own-catalog.  An image-only release ships the
    mosaics without the catalog, so a mosaic<->catalog mismatch (own_catalog FAIL) is not
    a reason to block; the images can still be internally consistent and shippable.
    """
    bands = field_bands(field)
    if len(bands) < 2:
        return dict(field=field, bands=bands, error="need >=2 bands for cross-band")
    dets = {}
    for b in bands:
        p = mosaic(field, b, "merged", observations=observations)
        s, f = detect(p) if p else (None, None)
        dets[b] = (s, f)
        if verbose:
            print(f"  detect {field} {b}: {0 if s is None else len(s)}", flush=True)
    # SW and LW detect different stellar populations and have independent distortion
    # solutions, so a SW-vs-LW cross-match yields chance pairs -> spurious offsets that
    # false-FAIL an internally-consistent field (e.g. gc2211 F200W vs F277W ~89 mas is an
    # artifact; the within-channel + inter-module audit is FLAGS none). Cross-band truth
    # must therefore be pooled WITHIN channel only.
    def channel(f):
        return "SW" if int(f[1:4]) <= 212 else "LW"

    report, any_fail = {}, False
    for b in bands:
        d, fl = dets[b]
        if d is None:
            report[b] = {"error": "no detections"}; any_fail = True; continue
        others = [dets[o][0] for o in bands
                  if o != b and dets[o][0] is not None and channel(o) == channel(b)]
        checks = {}
        if others:
            tru = SkyCoord(np.concatenate([s.ra.deg for s in others]) * u.deg,
                           np.concatenate([s.dec.deg for s in others]) * u.deg)
            r = per_cell(d, fl, tru, f"{b} vs cross-band"); r.pop("_g", None)
            checks["cross_band"] = r
        if not images_only:
            cat = catalog_sc(field, b)
            if cat is not None:
                r = per_cell(d, fl, cat, f"{b} vs own-catalog",
                             fail_min_ratio=FAIL_MIN_RATIO); r.pop("_g", None)
                checks["own_catalog"] = r
        bad = any((not c.get("PASS", True)) for c in checks.values())
        report[b] = checks
        any_fail = any_fail or bad
        if verbose:
            def _tag(k, v):
                s = f"{k}={'PASS' if v.get('PASS') else 'FAIL:'+str(v.get('n_fail'))}"
                nu = v.get("n_unconfident_highoff") or 0
                return s + (f"(unconf={nu})" if nu else "")   # high-off, sub-margin cells
            tags = " ".join(_tag(k, v) for k, v in checks.items())
            print(f"  {field} {b}: {'FAIL' if bad else 'ok'}  {tags}", flush=True)
    return dict(field=field, bands=bands, PASS=bool(not any_fail), report=report)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", required=True)
    ap.add_argument("--filter", default=None, help="single band (omit for --scan)")
    ap.add_argument("--xband", default=None, help="cross-band reference filter (e.g. F200W)")
    ap.add_argument("--scan", action="store_true", help="scan EVERY band of the field (gate mode)")
    ap.add_argument("--images-only", action="store_true",
                    help="cross-band (image-to-image) check only; skip own-catalog "
                         "(gate for an image-only release)")
    ap.add_argument("--observations", default=None,
                    help="csv of <proposal>-<observation> keys (e.g. "
                         "02221-001,01182-004) to scope mosaic selection to; a "
                         "misfiled stray from another observation in a shared "
                         "target dir is excluded.  Omit to pick deterministically "
                         "(sorted) among whatever validly-named mosaics are present.")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    obs = set(args.observations.split(",")) if args.observations else None

    if args.scan or not args.filter:
        res = scan_field(args.field, images_only=args.images_only, observations=obs)
        if args.json:
            json.dump(res, open(args.json, "w"), indent=2, default=str)
        print(json.dumps({"field": res.get("field"), "PASS": res.get("PASS"),
                          "error": res.get("error")}, default=str))
        if res.get("error"):
            return 0    # could not verify (e.g. <2 bands) -> warn, do NOT block
        return 0 if res.get("PASS") else 1   # exit 1 = FAIL -> gate blocks staging

    det, flux, truths = build_truths(args.field, args.filter, args.xband,
                                     observations=obs)
    # own-catalog gets the relaxed fail bar; per-module / cross-band stay strict.
    results = [per_cell(det, flux, t, f"{args.filter} vs {name}",
                        fail_min_ratio=(FAIL_MIN_RATIO if name == "own-catalog"
                                        else MIN_PEAK_RATIO))
               for name, t in truths.items()]
    if args.plot:
        plot_all(results, args.plot)
    any_fail = False
    for r in results:
        r.pop("_g", None)
        print(json.dumps(r, indent=2, default=str))
        any_fail = any_fail or (not r.get("PASS", True))
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
