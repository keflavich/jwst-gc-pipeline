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

A high-offset cell FAILs by any of three INDEPENDENT axes (see ``per_cell``):
  ratio  : peak/background contrast >= FAIL_MIN_RATIO -- the historic, density-COUPLED
           discriminant, kept transitionally.
  sig    : Poisson significance of the peak over the EXPECTED wrong-pair background,
           >= FAIL_MIN_SIG -- density-FLAT, so the bar means the same thing in a sparse
           and a crowded cell.
  contig : the cell is part of a vector-coherent 4-connected patch of >=MIN_SEAM_CELLS
           high-offset cells -- amplitude-free, so it still works where every contrast
           statistic is at the floor.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import label as ndlabel
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
FAIL_MIN_SIG = 55.0              # density-FLAT companion to FAIL_MIN_RATIO: Poisson significance
                                 # of the peak over the EXPECTED wrong-pair background,
                                 # (H.max()-lam)/sqrt(lam).  See ``_peak_stats`` for why lam is
                                 # the MEAN over the search disk and not median(H[H>0]): the
                                 # latter is pinned at exactly 1 count in every cell of every
                                 # real field measured (brick F405N, 372 verified cells), which
                                 # makes the published ratio a bare PEAK-COUNT threshold and
                                 # makes (H.max()-bg)/sqrt(bg) identically ratio-1 -- a
                                 # relabelling, not a new statistic.  With the mean background
                                 # the statistic is genuinely density-flat: peak counts ~ n,
                                 # lam ~ n^2, so sig ~ n/sqrt(n^2) = const while ratio ~ n.
                                 # CALIBRATION (brick, 2026-07; full table in the PR for #170):
                                 #   7 F405N artifact cells (m7 truth <=22 mas, must NOT fail):
                                 #      sig 32.6-49.3   (ratio 5-8)
                                 #   injected +90 mas seams (must fail), per-cell medians:
                                 #      full-field 78.1, half-field 75.4, 10%-band 60.7
                                 # 55 is the log-midpoint of the artifact ceiling (49.3) and the
                                 # HARDEST seam's median (60.7).  At 55: 0/7 artifact cells fail
                                 # and 290 / 130 / 28 seam cells fail.  The two populations do
                                 # OVERLAP at the low end (both start at sig ~32), so this bar
                                 # buys a field-level verdict, not a per-cell one.
MIN_SEAM_CELLS = 3               # contiguity axis: >=this many 4-connected, vector-COHERENT
                                 # high-offset cells is SEAM-SHAPED regardless of contrast; a
                                 # scattered singleton is wrong-pair noise regardless of
                                 # contrast.  Amplitude-free, so it is the one check that still
                                 # works in the crowdedest cells where every contrast statistic
                                 # is at the floor.  3, not the 2 proposed in #170: two of the
                                 # seven brick F405N artifact cells ARE 4-adjacent ((12,13) and
                                 # (13,13)) and -- measured, 2026-07 -- all seven share the same
                                 # peak vector (+80, 0) mas, so vector coherence does NOT reject
                                 # that pair.  A 2-cell bar would therefore re-create exactly
                                 # the false FAIL this thread is about; 3 does not.  The
                                 # injected 90 mas seam labels as ONE 371-cell component, so the
                                 # bar has ~100x headroom on the seam side.
SEAM_OFF_TOL = 40.0              # mas (== one XBIN); cells in a contiguous component must agree
                                 # in their peak offset VECTOR to within this to count toward
                                 # the same seam.  A seam is one rigid displacement shared by
                                 # neighbouring cells; this drops the cells of a component that
                                 # peak somewhere unrelated.


def detect(path, thr=30.0):
    h = fits.open(path); sci = h["SCI"]; w = WCS(sci.header); d = sci.data.astype("float32")
    _, med, std = sigma_clipped_stats(d, sigma=3.0)
    t = DAOStarFinder(fwhm=2.5, threshold=thr * std)(d - med)
    if t is None:
        return None, None
    return SkyCoord(w.pixel_to_world(t["xcentroid"], t["ycentroid"])), np.asarray(t["flux"], float)


def mosaic(field, filt, module="merged"):
    g = glob.glob(f"{BASE}/{field}/{filt}/pipeline/jw*-o*_t001_nircam_clear-{filt.lower()}-{module}_i2d.fits")
    return g[0] if g else None


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


def _offset_hist_bins():
    """(bin edges, number of bins inside the |sep| < MX search disk) for the offset
    histogram.  Chance pairs are uniform over that disk, so the disk bin count is the
    denominator of the expected per-bin background."""
    r = MX.to(u.arcsec).value * 1000
    hb = np.arange(-r, r + XBIN * 1000, XBIN * 1000)
    c = (hb[:-1] + hb[1:]) / 2
    bx, by = np.meshgrid(c, c, indexing="ij")
    return hb, int((np.hypot(bx, by) <= r).sum())


HB, N_BG_BINS = _offset_hist_bins()


def _peak_stats(H, npairs, n_bg_bins=N_BG_BINS):
    """(peak, ratio, sig) for one cell's offset histogram.

    ``ratio`` = H.max()/median(H[H>0]) is the historic discriminant.  MEASURED CAVEAT
    (brick F405N, 2026-07): median(H[H>0]) is exactly 1 in every verified cell, because
    the (2*MX/XBIN)^2 offset bins vastly outnumber the ~1e2-1e3 pairs in a cell, so every
    occupied bin holds one count except the peak.  The "ratio" is therefore a bare
    PEAK-COUNT threshold, and the naive Poisson significance (H.max()-bg)/sqrt(bg)
    evaluates to exactly ratio-1 -- a relabelling with no new information.

    ``sig`` therefore uses the EXPECTED background: chance pairs are uniform in the
    (dRA, dDec) plane inside the |sep| < MX search disk, so their expected count per bin
    is lam = npairs / n_bg_bins -- a fractional number that keeps scaling with density
    where the median saturates at 1.  That restores the density-flat property:
    peak ~ n, lam ~ n^2, so sig ~ n/sqrt(n^2) = const while ratio ~ n.
    """
    peak = float(H.max())
    bg = np.median(H[H > 0]) if (H > 0).any() else 0.0
    ratio = peak / bg if bg > 0 else np.inf
    lam = npairs / float(n_bg_bins) if n_bg_bins else 0.0
    sig = (peak - lam) / np.sqrt(lam) if lam > 0 else np.inf
    return peak, ratio, sig


def _seam_components(highoff, offx, offy, min_cells=MIN_SEAM_CELLS, off_tol=SEAM_OFF_TOL):
    """Cells belonging to a vector-COHERENT contiguous patch of >=``min_cells`` cells.

    Contiguity is the density-INDEPENDENT axis: it uses only the geometry of which cells
    are high-offset, not the peak amplitude, so it still discriminates in the crowdedest
    cells where every contrast statistic is at the floor.  A seam is one rigid
    displacement shared by neighbouring cells, so only the members within ``off_tol`` of
    the component's median offset VECTOR count, and the component fires only if
    ``min_cells`` of them survive.  Requiring a MAJORITY rather than every member matters
    on real data: an injected field-wide 90 mas seam labels as a single 371-cell
    component in which a handful of cells peak elsewhere, and an all-members rule discards
    the entire seam because of them.
    """
    out = np.zeros_like(highoff, dtype=bool)
    if not highoff.any():
        return out
    lab, n = ndlabel(highoff)                     # 4-connectivity (default structure)
    for c in range(1, n + 1):
        m = lab == c
        if m.sum() < min_cells:
            continue
        mx, my = np.nanmedian(offx[m]), np.nanmedian(offy[m])
        coh = m & (np.hypot(offx - mx, offy - my) <= off_tol)
        if coh.sum() >= min_cells:
            out |= coh
    return out


def per_cell(det, flux, truth, label, bright_pct=None, fail_min_ratio=MIN_PEAK_RATIO,
             fail_min_sig=FAIL_MIN_SIG):
    """Per-cell registration offset by pair-separation HISTOGRAM cross-correlation.

    For every det-truth pair within MX, bin by the detection's spatial cell and by the
    offset (dRA*cos, dDec).  In each cell the REAL counterparts pile into a peak at the
    true offset; chance coincidences form a flat background -> crowding-robust (NOT
    nearest-neighbour, which just measures the chance-NN distance in a dense field).

    A cell is VERIFIED only if it has >=MIN_PAIRS and peak/background >=MIN_PEAK_RATIO;
    otherwise it is UNVERIFIED (reported, never a fail).  Field FAIL = any cell fails.

    A verified cell whose peak offset exceeds OFF_MAX FAILs by ANY of three independent
    paths (a cell need satisfy only one):

      ratio  : contrast >= ``fail_min_ratio``.  TRANSITIONAL -- the historic
               density-COUPLED discriminant, kept so this change can only add
               sensitivity, never remove it.  ``fail_min_ratio`` (default
               ``MIN_PEAK_RATIO`` -> historic strict behaviour) is raised to
               ``FAIL_MIN_RATIO`` ONLY for own-catalog, where a floor-level peak in a
               dense bright-star cell is wrong-pair noise, not a seam.
      sig    : Poisson significance (peak-lam)/sqrt(lam) >= ``fail_min_sig``, with lam
               the EXPECTED wrong-pair count per bin (``_peak_stats``).  This is the
               density-FLAT statistic: unlike the ratio it does not degrade as the
               wrong-pair background rises, so the bar means the same thing in a sparse
               and a crowded cell.  See FAIL_MIN_SIG for the calibration.
      contig : the cell belongs to a vector-coherent contiguous patch of
               >=MIN_SEAM_CELLS high-offset cells.  Amplitude-free, so it is the only
               path that still works where every contrast statistic is at the floor.

    Both statistics are reported per cell (``peak_bg`` and ``peak_sig``) and the fail
    paths are broken out in ``n_fail_by_path`` so the two can be compared on real data.
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
    off = np.full((GRID, GRID), np.nan)      # peak offset magnitude (mas)
    offx = np.full((GRID, GRID), np.nan)     # peak dRA*cos  (mas), signed
    offy = np.full((GRID, GRID), np.nan)     # peak dDec     (mas), signed
    ratio = np.full((GRID, GRID), np.nan)    # peak/median(H[H>0])   (density-coupled)
    sig = np.full((GRID, GRID), np.nan)      # (peak-lam)/sqrt(lam)  (density-flat)
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
        H, xb, yb = np.histogram2d(dra[s:e], dde[s:e], bins=[HB, HB])
        pi, pj = np.unravel_index(H.argmax(), H.shape)
        i0, j0 = k // GRID, k % GRID
        _, ratio[i0, j0], sig[i0, j0] = _peak_stats(H, e - s)
        # refine the peak to sub-bin with the local centroid
        dcen = (xb[pi] + xb[pi + 1]) / 2
        ecen = (yb[pj] + yb[pj + 1]) / 2
        offx[i0, j0], offy[i0, j0] = dcen, ecen
        off[i0, j0] = np.hypot(dcen, ecen)

    verified = np.isfinite(ratio) & (ratio >= MIN_PEAK_RATIO) & (npair >= MIN_PAIRS)
    highoff = verified & (off > OFF_MAX)
    # Three INDEPENDENT fail paths (OR'd) -- see the docstring. Each is a different
    # projection of "is this a seam or wrong-pair noise?", and each is weakest in a
    # different regime, so the union is strictly more sensitive than any one.
    fail_ratio = highoff & (ratio >= fail_min_ratio)          # transitional, density-coupled
    fail_sig = highoff & (np.nan_to_num(sig, nan=0.0, posinf=1e30) >= fail_min_sig)
    fail_contig = _seam_components(highoff, offx, offy)       # amplitude-free
    fail = fail_ratio | fail_sig | fail_contig
    # High offset but no fail path triggered: NOT a fail, but reported so a real
    # low-contrast issue is never silently hidden by the margin.
    unconfident = highoff & ~fail

    def _cells(mask):
        return [dict(ra=float((xe[i] + xe[i + 1]) / 2), dec=float((ye[j] + ye[j + 1]) / 2),
                     offset_mas=round(float(off[i, j]), 0),
                     peak_bg=round(float(ratio[i, j]), 1),
                     peak_sig=round(float(sig[i, j]), 1),
                     npairs=int(npair[i, j]),
                     paths=[n for n, m in (("ratio", fail_ratio), ("sig", fail_sig),
                                           ("contig", fail_contig)) if m[i, j]])
                for i, j in sorted(zip(*np.where(mask)), key=lambda c: -off[c])][:8]

    return dict(label=label, verified_cells=int(verified.sum()),
                unverified_cells=int((npair >= MIN_PAIRS).sum() - verified.sum()),
                median_verified_offset_mas=round(float(np.nanmedian(off[verified])), 1) if verified.any() else None,
                n_fail=int(fail.sum()), PASS=bool(fail.sum() == 0), worst=_cells(fail),
                n_fail_by_path=dict(ratio=int(fail_ratio.sum()), sig=int(fail_sig.sum()),
                                    contig=int(fail_contig.sum())),
                n_unconfident_highoff=int(unconfident.sum()),
                unconfident_highoff_cells=_cells(unconfident),
                _g=(off, verified, (xe, ye), ratio, sig, npair, offx, offy))


def build_truths(field, filt, xband):
    det, flux = detect(mosaic(field, filt, "merged"))
    truths = {}
    # 1. per-module
    pm = []
    for m in ("nrca", "nrcb", "nrcalong", "nrcblong"):
        p = mosaic(field, filt, m)
        if p:
            s, _ = detect(p)
            if s is not None:
                pm.append(s)
    if pm:
        truths["per-module"] = SkyCoord(np.concatenate([s.ra.deg for s in pm]) * u.deg,
                                        np.concatenate([s.dec.deg for s in pm]) * u.deg)
    # 2. cross-band
    if xband:
        p = mosaic(field, xband, "merged")
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
        off, verified, (xe, ye) = r["_g"][:3]
        shown = np.where(verified, off, np.nan)
        im = ax.pcolormesh(xe, ye, shown.T, cmap="inferno", vmin=0, vmax=max(OFF_MAX * 2, 100))
        ax.invert_xaxis(); plt.colorbar(im, ax=ax, label="verified peak offset [mas]")
        v = "PASS" if r["PASS"] else f"FAIL {r['n_fail']}"
        med = r.get("median_verified_offset_mas")
        ax.set_title(f"{r['label']}\nmed {med} mas — {v}", color="green" if r["PASS"] else "red")
    fig.tight_layout(); fig.savefig(out, dpi=100); print("wrote", out)


def field_bands(field):
    """Filters with a merged mosaic on disk for this field."""
    out = []
    for p in glob.glob(f"{BASE}/{field}/*/pipeline/jw*-o*_t001_nircam_clear-*-merged_i2d.fits"):
        b = os.path.basename(p)
        try:
            filt = b.split("clear-")[1].split("-merged")[0].upper()
        except IndexError:
            continue
        d = os.path.basename(os.path.dirname(os.path.dirname(p)))   # <field>/<FILT>/pipeline
        if d.upper() == filt:
            out.append(filt)
    return sorted(set(out))


def scan_field(field, verbose=True, images_only=False):
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
        p = mosaic(field, b, "merged")
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
                if not v.get("PASS", True):
                    p = v.get("n_fail_by_path") or {}
                    s += "[" + ",".join(f"{n}:{c}" for n, c in p.items() if c) + "]"
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
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.scan or not args.filter:
        res = scan_field(args.field, images_only=args.images_only)
        if args.json:
            json.dump(res, open(args.json, "w"), indent=2, default=str)
        print(json.dumps({"field": res.get("field"), "PASS": res.get("PASS"),
                          "error": res.get("error")}, default=str))
        if res.get("error"):
            return 0    # could not verify (e.g. <2 bands) -> warn, do NOT block
        return 0 if res.get("PASS") else 1   # exit 1 = FAIL -> gate blocks staging

    det, flux, truths = build_truths(args.field, args.filter, args.xband)
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
