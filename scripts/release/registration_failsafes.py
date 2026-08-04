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

OVERLAP_STRIDE = 16              # pixel stride when sampling a mosaic for module overlap
MIN_OVERLAP_SAMPLES = 50         # sampled positions with real data in BOTH modules before
                                 # the modules count as overlapping.  At the i2d pixel scale
                                 # a stride-16 sample is ~0.5"; 50 samples is a few arcsec of
                                 # genuine shared sky, well under the thinnest real seam
                                 # measured (sgrc F360M, 279) and far above the 0 that two
                                 # abutting modules give (arches F212N/F323N).


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


def field_band_mosaics(field, observations=None):
    """``{FILT: {module_token: path}}`` for every validly-named mosaic on disk.

    ``field_bands`` lists only the bands that have a ``merged`` mosaic, so a band
    drizzled per-module and never merged is invisible to it -- and therefore
    silently ungated.  That is not a rare state: cloudc F182M, sgrc F115W/F162M,
    cloudef F162M/F210M and sgrb2 F150W are all in it (2026-08-03), and arches
    and sickle have no merged mosaic in ANY band.  Enumerate by module instead
    and let the caller decide what it can check with what is present.
    """
    out = {}
    pat = (f"{BASE}/{field}/*/pipeline/"
           f"jw[0-9][0-9][0-9][0-9][0-9]-o[0-9][0-9][0-9]_t001_nircam_clear-"
           f"*_i2d.fits")
    for p in sorted(glob.glob(pat)):
        m = _MOSAIC_RE.match(os.path.basename(p))
        if m is None:
            continue
        filt = m.group("filt").upper()
        d = os.path.basename(os.path.dirname(os.path.dirname(p)))   # <field>/<FILT>/pipeline
        if d.upper() != filt:
            continue
        if observations is not None \
                and f"{m.group('prop')}-{m.group('obs')}" not in observations:
            continue
        out.setdefault(filt, {}).setdefault(m.group("module"), p)
    return out


def module_family(token):
    """``nrca``/``nrcalong`` -> ``'a'``; ``nrcb``/``nrcblong`` -> ``'b'``.

    A field names its per-module mosaics with the SW tokens in both channels
    (arches writes ``f323n-nrca``, not ``f323n-nrcalong``), so the family, not
    the token, is what identifies "the same piece of sky".
    """
    return "a" if token.startswith("nrca") else "b"


def _sampled_valid_sky(path, stride=OVERLAP_STRIDE):
    """(ra, dec) of stride-sampled pixels that carry real data, plus (wcs, data).

    ``i2d`` mosaics are a rectified plain ``RA---TAN`` grid with no SIP, so
    ``WCS(header)`` is exact here -- the GWCS rule (ASTROMETRY RULE #2) exempts
    them explicitly.
    """
    with fits.open(path) as hdul:
        for h in hdul:
            if h.data is not None and h.data.ndim == 2 and h.header.get("CTYPE1"):
                data, hdr = np.asarray(h.data), h.header
                break
        else:
            return None
    ww = WCS(hdr)
    ys, xs = np.mgrid[0:data.shape[0]:stride, 0:data.shape[1]:stride]
    good = np.isfinite(data[ys, xs]) & (data[ys, xs] != 0)
    if not good.any():
        return None
    ra, dec = ww.all_pix2world(xs[good].astype(float), ys[good].astype(float), 0)
    return dict(ra=ra, dec=dec, wcs=ww, data=data)


def modules_overlap(path_a, path_b, stride=OVERLAP_STRIDE):
    """Do the two per-module mosaics share sky where BOTH carry real data?

    Not a bounding-box test: two abutting NIRCam modules drizzled onto their own
    grids can have boxes that touch or overlap while no pixel holds data from
    both.  Sample A's real pixels, map them into B, and count the ones that land
    on real data there.

    Returns ``None`` when either mosaic cannot be read (unknown, not "no").
    """
    a = _sampled_valid_sky(path_a, stride)
    b = _sampled_valid_sky(path_b, stride)
    if a is None or b is None:
        return None
    x, y = b["wcs"].all_world2pix(a["ra"], a["dec"], 0)
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    inside = ((xi >= 0) & (yi >= 0)
              & (xi < b["data"].shape[1]) & (yi < b["data"].shape[0]))
    if inside.any():
        d = b["data"][yi[inside], xi[inside]]
        n_both = int((np.isfinite(d) & (d != 0)).sum())
    else:
        n_both = 0
    return dict(n_sampled=int(len(x)), n_in_bbox=int(inside.sum()),
                n_both=n_both, overlaps=bool(n_both >= MIN_OVERLAP_SAMPLES))


def field_module_geometry(field, observations=None, verbose=False):
    """Whether this field's nrca and nrcb footprints share sky.

    ``mode``:

    * ``'single-module'`` — only one module family was observed (sickle: nrcb
      only).  There is no inter-module seam, so a per-module gate is complete.
    * ``'disjoint'`` — both modules observed, no band shows shared data (arches,
      quintuplet: the two modules image adjacent, non-overlapping sky).  A merged
      mosaic would add nothing the per-module mosaics do not already carry, so a
      per-module gate is again complete.
    * ``'overlapping'`` — some band has real data from both modules.  The seam
      between them is exactly where the misregistration this script exists to
      catch would live, so a band in this field needs its MERGED mosaic to be
      fully gated.
    * ``'merged-only'`` — no per-module mosaics were kept, so the geometry cannot
      be measured from disk.  The merged mosaic is all there is, and gating it is
      both the only option and the right one (brick, w51's single-module bands).
    * ``'unknown'`` — both modules exist but no band had both readable.
    """
    inv = field_band_mosaics(field, observations=observations)
    fams = set()
    for mods in inv.values():
        fams.update(module_family(t) for t in mods if t != "merged")
    if not fams:
        mode = "merged-only" if any("merged" in m for m in inv.values()) else "unknown"
        return dict(mode=mode, families=[], evidence={})
    if len(fams) == 1:
        return dict(mode="single-module", families=sorted(fams), evidence={})
    evidence, seen = {}, False
    for filt in sorted(inv):
        mods = inv[filt]
        pa = mods.get("nrca") or mods.get("nrcalong")
        pb = mods.get("nrcb") or mods.get("nrcblong")
        if not (pa and pb):
            continue
        r = modules_overlap(pa, pb)
        if r is None:
            continue
        seen = True
        evidence[filt] = r
        if verbose:
            print(f"  module overlap {field} {filt}: {r['n_both']} shared "
                  f"samples of {r['n_sampled']} -> "
                  f"{'OVERLAPPING' if r['overlaps'] else 'disjoint'}", flush=True)
    if not seen:
        return dict(mode="unknown", families=sorted(fams), evidence=evidence)
    mode = ("overlapping" if any(r["overlaps"] for r in evidence.values())
            else "disjoint")
    return dict(mode=mode, families=sorted(fams), evidence=evidence)


def _channel(f):
    """SW and LW detect different stellar populations and have independent distortion
    solutions, so a SW-vs-LW cross-match yields chance pairs -> spurious offsets that
    false-FAIL an internally-consistent field (e.g. gc2211 F200W vs F277W ~89 mas is an
    artifact; the within-channel + inter-module audit FLAGS none). Cross-band truth must
    therefore be pooled WITHIN channel only."""
    return "SW" if int(f[1:4]) <= 212 else "LW"


def _scan_view(field, view, band_paths, verbose, images_only):
    """Cross-band + own-catalog checks over one coherent set of mosaics.

    A *view* is a set of same-geometry mosaics that can serve as one another's
    cross-band truth: either every band's ``merged`` mosaic, or every band's
    mosaic for one module family.  Mixing the two would cross-match a merged
    mosaic against a single module's, whose non-overlapping parts have no
    counterpart to find.
    """
    bands = sorted(band_paths)
    if len(bands) < 2:
        return dict(view=view, bands=bands, PASS=None,
                    error=f"need >=2 bands for cross-band, have {len(bands)}",
                    report={})
    dets = {}
    for b in bands:
        s, f = detect(band_paths[b])
        dets[b] = (s, f)
        if verbose:
            print(f"  detect {field} [{view}] {b}: {0 if s is None else len(s)}",
                  flush=True)

    report, any_fail, unchecked = {}, False, []
    for b in bands:
        d, fl = dets[b]
        if d is None:
            # An unreadable/empty mosaic is NOT "locally misregistered" -- calling
            # it FAIL makes stage_release print a diagnosis that is simply wrong
            # about the file.  Could-not-verify, per this script's own tri-state.
            report[b] = {"error": "no detections"}
            unchecked.append(f"{b}: no detections in view {view} (mosaic empty, "
                             f"truncated, or unreadable)")
            continue
        others = [dets[o][0] for o in bands
                  if o != b and dets[o][0] is not None and _channel(o) == _channel(b)]
        checks = {}
        if others:
            tru = SkyCoord(np.concatenate([s.ra.deg for s in others]) * u.deg,
                           np.concatenate([s.dec.deg for s in others]) * u.deg)
            r = per_cell(d, fl, tru, f"{b} vs cross-band [{view}]"); r.pop("_g", None)
            checks["cross_band"] = r
        if not images_only:
            cat = catalog_sc(field, b)
            if cat is not None:
                r = per_cell(d, fl, cat, f"{b} vs own-catalog [{view}]",
                             fail_min_ratio=FAIL_MIN_RATIO); r.pop("_g", None)
                checks["own_catalog"] = r
        # A check that MATCHED NOTHING is not a pass.  ``per_cell`` returns
        # ``dict(error=...)`` with no ``PASS`` key for "too few pairs" / "missing
        # detections", and ``.get("PASS", True)`` used to read those as passes --
        # the same silent-pass hole this script exists to close, one level down.
        # Reachable: gc2211's SW view pools F150W (o028) and F200W (o023) mosaics
        # 13.5 arcmin apart, so there are zero pairs to match.
        errored = {k: c for k, c in checks.items() if "error" in c}
        for k, c in errored.items():
            unchecked.append(f"{b}: {k} could not be evaluated in view {view} "
                             f"({c['error']})")
        graded = {k: c for k, c in checks.items() if k not in errored}
        # Only when NOTHING graded is the band unchecked.  Appending on an empty
        # `others` regardless would block six fields (arches, quintuplet, sgra,
        # gc2211, m4, ngc6397) that have one band per channel as a property of
        # their observing programs -- no re-reduction can ever give them a second
        # SW or LW band, and their own-catalog check runs and passes.  A gate a
        # correct field cannot pass teaches people to reach for the override.
        if not graded:
            unchecked.append(f"{b}: no check available in view {view} "
                             f"(sole {_channel(b)} band"
                             f"{', no own-catalog' if images_only else ''})")
        bad = any((not c.get("PASS", True)) for c in graded.values())
        report[b] = checks
        any_fail = any_fail or bad
        if verbose:
            def _tag(k, v):
                s = f"{k}={'PASS' if v.get('PASS') else 'FAIL:'+str(v.get('n_fail'))}"
                nu = v.get("n_unconfident_highoff") or 0
                return s + (f"(unconf={nu})" if nu else "")   # high-off, sub-margin cells
            tags = " ".join(_tag(k, v) for k, v in checks.items())
            print(f"  {field} [{view}] {b}: {'FAIL' if bad else 'ok'}  {tags}",
                  flush=True)
    return dict(view=view, bands=bands, PASS=bool(not any_fail), report=report,
                unchecked=unchecked)


def scan_field(field, verbose=True, images_only=False, observations=None):
    """Run the cross-band + own-catalog failsafes on EVERY band of a field.

    Cross-band truth for band F = the pooled detections of all OTHER bands of the field
    (same stars, JWST-internal registration).  Detects each band once.

    The mosaics are grouped into *views* according to the field's module geometry:

    * modules that OVERLAP — the seam between them is what this script exists to
      catch, so the ``merged`` mosaic (the only place the two modules are
      combined) is the thing that must be checked.  A band with no merged mosaic
      cannot be fully gated here; it is checked per module for what that is worth
      and reported as ungated.
    * modules that are DISJOINT (arches, quintuplet) or a field that used only
      ONE module (sickle) — there is no seam to catch, and each module's own
      mosaic is a complete object, so the gate is PER MODULE and every module
      must pass on its own.  The merged mosaic is gated TOO where one exists:
      it is not needed for the seam, but it SHIPS, and a merged drizzle that
      places module B at the wrong offset is invisible in the per-module views.
      A merged view covering fewer than 2 bands is dropped rather than gated —
      it has nothing to cross-band-check against — and the drop is printed.

    ``images_only``: gate an IMAGE-ONLY release -- run the reference-free cross-band
    (image-to-image) check only, and SKIP own-catalog.  An image-only release ships the
    mosaics without the catalog, so a mosaic<->catalog mismatch (own_catalog FAIL) is not
    a reason to block; the images can still be internally consistent and shippable.

    ``PASS`` is tri-state.  ``True``/``False`` are a verified pass/fail; ``None``
    means the field could not be verified either way -- no mosaics, no view with
    >=2 bands, a check that errored, or overlapping modules with a band whose
    merged mosaic is missing.  ``None`` BLOCKS: ambiguity is not a pass.

    There is deliberately NO "not covered here, and that is fine" verdict: every
    view admitted is gated, and anything that cannot be gated is either dropped
    before it becomes a view (the <2-band merged case above) or reported as
    ungated, which blocks.  The distinction is made when the view is BUILT, not
    when it is judged, because by judging time a view that cannot be checked is
    indistinguishable from one that failed to be.
    """
    inv = field_band_mosaics(field, observations=observations)
    if not inv:
        return dict(field=field, bands=[], PASS=None,
                    error="no validly-named mosaics on disk")
    geom = field_module_geometry(field, observations=observations, verbose=verbose)
    if verbose:
        print(f"  {field} module geometry: {geom['mode']} "
              f"(families {geom['families']})", flush=True)

    views, ungated = {}, []
    if geom["mode"] in ("disjoint", "single-module"):
        for fam in geom["families"]:
            paths = {}
            for filt, mods in inv.items():
                cand = [t for t in mods
                        if t != "merged" and module_family(t) == fam]
                if cand:
                    paths[filt] = mods[sorted(cand)[0]]
            if paths:
                views[f"module-{fam}"] = paths
        # The per-module views account for the module IMAGES, but the merged
        # product also SHIPS, and a merged drizzle that places module B at the
        # wrong offset -- or writes a wrong output WCS -- is invisible in them.
        # Gating only per-module opened zero merged mosaics for m92 (4 bands),
        # gc2211 (3) and sgra (2), all of which the previous gate did open.
        # This went unnoticed because arches and quintuplet, the two fields the
        # disjoint branch was written for, have NO merged mosaics at all, so
        # there the dict comes back empty and nothing changes.
        merged = {f: m["merged"] for f, m in inv.items() if "merged" in m}
        # >= 2, the same bar the per-module views below use.  A view with ONE band
        # cannot serve as its own cross-band truth, so it is not a view -- and
        # admitting it is not neutral: it lands in `unresolved` and the field
        # verdict becomes None, which BLOCKS.  sickle is the case: one merged
        # mosaic, five bands passing on the module view, and no re-reduction short
        # of producing four more merged mosaics could clear it.  A gate a correct
        # field cannot pass is a gate that teaches people to use the override.
        if len(merged) >= 2:
            views["merged"] = merged
        elif merged:
            print(f"  {field}: only {sorted(merged)} has a merged mosaic -- a "
                  f"one-band merged view cannot be cross-band-checked against "
                  f"anything, so it is not gated here (the module views below "
                  f"still gate every band)", flush=True)
    else:
        # overlapping / merged-only / unknown: merged is the object to gate
        merged = {f: m["merged"] for f, m in inv.items() if "merged" in m}
        if merged:
            views["merged"] = merged
        why = {"overlapping": "this field's modules overlap",
               "merged-only": "no per-module mosaics were kept, so the module "
                              "geometry could not be measured",
               "unknown": "this field's module geometry could not be measured"}
        for filt, mods in sorted(inv.items()):
            if "merged" not in mods:
                ungated.append(
                    f"{filt}: no merged mosaic, and "
                    f"{why.get(geom['mode'], geom['mode'])}"
                    f" -- the inter-module seam of this band is NOT covered here "
                    f"(present: {sorted(mods)})")
        # Only when something is ungated is it worth also running the per-module
        # views: they cannot see the seam (that is what merged is for), so for a
        # fully-merged field they would triple the runtime and add nothing.  When
        # a band HAS no merged mosaic, they are the only look at it available.
        if ungated:
            for fam in geom["families"]:
                paths = {}
                for filt, mods in inv.items():
                    cand = [t for t in mods
                            if t != "merged" and module_family(t) == fam]
                    if cand:
                        paths[filt] = mods[sorted(cand)[0]]
                if len(paths) >= 2:
                    views[f"module-{fam}"] = paths

    if not views:
        return dict(field=field, bands=sorted(inv), PASS=None,
                    geometry=geom["mode"],
                    error="no view with >=2 bands to cross-match")

    results = {name: _scan_view(field, name, paths, verbose, images_only)
               for name, paths in sorted(views.items())}
    # In per-module mode EVERY module must pass on its own -- that is the whole
    # point of accepting the modules separately.
    any_fail = any(r.get("PASS") is False for r in results.values())
    unresolved = [f"view {n}: {r['error']}" for n, r in results.items()
                  if r.get("PASS") is None]
    unresolved += ungated
    for r in results.values():
        unresolved += r.get("unchecked", [])

    if any_fail:
        passed = False
    elif unresolved:
        passed = None
    else:
        passed = True
    if verbose and unresolved:
        for u in unresolved:
            print(f"  {field} UNGATED: {u}", flush=True)
    return dict(field=field, bands=sorted(inv), geometry=geom["mode"],
                module_families=geom["families"],
                overlap_evidence=geom.get("evidence", {}),
                views=results, PASS=passed, unresolved=unresolved,
                # flattened single-view report, for callers that read `report`
                report=(results.get("merged") or
                        list(results.values())[0]).get("report", {}))


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
                          "geometry": res.get("geometry"),
                          "error": res.get("error"),
                          "unresolved": res.get("unresolved")}, default=str))
        # PASS is tri-state and only True is a pass.  `None` (could not verify:
        # no mosaics, <2 bands, a band with no merged mosaic in an overlapping-
        # module field) used to return 0 and let staging proceed -- a gate that
        # goes green because it never ran.  Ambiguity is not a pass; it blocks,
        # and stage_release's --allow-registration-fail + ALLOW_REGISTRATION_FAIL=1
        # is the deliberate, two-key way past it.
        if res.get("PASS") is None:
            for u in (res.get("unresolved") or [res.get("error") or "unspecified"]):
                print(f"UNVERIFIED: {u}", file=sys.stderr)
            return 2    # exit 2 = could-not-verify -> gate blocks staging
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
