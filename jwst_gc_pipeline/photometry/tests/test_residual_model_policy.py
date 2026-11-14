"""QA-image content policy for merged-catalog residual / model i2d mosaics.

The pipeline's QA images obey a strict content contract that downstream
catalog evaluation and the residual-bg feedback loop rely on:

  * merged RESIDUAL i2d  -> background ONLY.  NO stars (saturated OR unsaturated).
  * merged MODEL i2d     -> stars ONLY (saturated AND unsaturated).  NO background.
  * the intermediate model SUBTRACTED to form the residual must EXCLUDE saturated
    stars (they are already removed from the data per-frame); the MODEL written
    to disk must INCLUDE them.

These are data-driven checks against the sickle F480M products and the curated
"must be subtracted" bright-star region file.  They SKIP when those products are
absent (e.g. CI without the data tree), and run as a hard regression on the
analysis machine.  Pin: 2026-06-17 bright stars left in the residual at ~88% of
their data peak (saturated-star model under-fit -> dirty residual).
"""
import datetime
import difflib
import glob
import json
import os
import subprocess
import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u

# `SICKLE_BASEPATH` is a TEST hook, not a configuration knob: it is how the
# no-products case (CI) is reproduced on a machine that has the products.
#   SICKLE_BASEPATH=/nonexistent pytest ... -> 2 passed, 3 skipped
SICKLE = os.environ.get("SICKLE_BASEPATH", "/orange/adamginsburg/jwst/sickle")
REG = f"{SICKLE}/regions_/f480m_brightstar_regression_20260617.reg"


def _read_points(path):
    ra, dec = [], []
    for line in open(path):
        s = line.strip()
        if s.startswith('point('):
            b = s[s.index('(') + 1:s.index(')')]
            r, d = b.split(',')
            ra.append(float(r)); dec.append(float(d))
    return SkyCoord(ra * u.deg, dec * u.deg)


def _latest(pattern):
    fns = sorted(glob.glob(pattern), key=os.path.getmtime)
    return fns[-1] if fns else None


def _product_date(path):
    """The mosaic's ``DATE`` (when it was written), or ``None``."""
    if path is None:
        return None
    from astropy.io import fits as _fits
    try:
        return _fits.getheader(path).get("DATE")
    except (OSError, ValueError):
        return None


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value))
    except ValueError:
        return None


#: The residual and the model are written seconds apart by one m5 finalize (26 s
#: on the 2026-08-21 products).  An hour is far outside that and well inside any
#: single run, so it separates "two products of one chain" from "two chains".
SAME_RUN_HOURS = 1.0


def cross_run_reason(data_date, resid_date, model_date,
                     same_run_hours=SAME_RUN_HOURS):
    """Why these three products are not one reduction generation, else ``None``.

    Item 2 of issue #266, which pinning the globs did not close.  ``STAGE``
    makes each glob match exactly one file, so numerator and denominator can no
    longer come from two *stages*; nothing yet stops them coming from two
    *runs*.  That is the failure that turned this file red on 2026-07-05 with no
    commit behind it:

        data_i2d                DATE = 2026-07-05T19:11:39
        model / residual i2d    DATE = 2026-06-27T18:40:04

    A partial re-run regenerated the data mosaic and never rewrote the QA
    products, so every ratio graded a June model against a July data mosaic.
    The ordering is what makes this checkable without a tolerance: within one
    generation the data mosaic is drizzled BEFORE the catalog chain that
    subtracts from it, so a data mosaic NEWER than the residual it is divided
    into means the residual describes a data mosaic that no longer exists.

    Returns a reason string suitable for ``pytest.skip`` -- an unverifiable
    comparison is not a verdict, in either direction.
    """
    named = (("data i2d", data_date), ("residual i2d", resid_date),
             ("model i2d", model_date))
    parsed = {what: _parse_date(value) for what, value in named}
    unknown = sorted(what for what, when in parsed.items() if when is None)
    if unknown:
        return ("cannot establish that the QA products are one reduction "
                "generation: no readable DATE on " + ", ".join(unknown))
    data, resid, model = (parsed["data i2d"], parsed["residual i2d"],
                          parsed["model i2d"])
    gap_hours = abs((resid - model).total_seconds()) / 3600.0
    if gap_hours > same_run_hours:
        return (f"residual and model i2d are {gap_hours:.1f} h apart "
                f"(> {same_run_hours:.1f} h), so they are not from one "
                f"cataloging run: residual {resid.isoformat()}, "
                f"model {model.isoformat()}")
    older = min(resid, model)
    if data > older:
        return (f"the data i2d ({data.isoformat()}) is NEWER than the QA "
                f"products graded against it ({older.isoformat()}), so a "
                f"re-reduction has moved the denominator out from under them "
                f"(issue #266 item 2)")
    return None


def _img(path, what=""):
    if path is None:
        # Never reached when the skip predicate and the fixture agree; kept so a
        # future divergence fails with the missing product named rather than
        # with astropy's "Empty filename: None".
        pytest.skip(f"required QA product not on disk: {what}")
    h = fits.open(path)
    sci = h['SCI'] if 'SCI' in [x.name for x in h] else h[0]
    return sci.data, WCS(sci.header)


def _box(arr, w, stars, box=3):
    """Yield each star's (2*box+1)^2 core cutout, or ``None`` when off-image."""
    ny, nx = arr.shape
    xs, ys = w.world_to_pixel(stars)
    for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < nx and 0 <= yi < ny):
            yield None
            continue
        yield arr[max(0, yi - box):yi + box + 1, max(0, xi - box):xi + box + 1]


def _peaks(arr, w, stars, box=3):
    return np.array([np.nan if sub is None or not np.isfinite(sub).any()
                     else float(np.nanmax(sub))
                     for sub in _box(arr, w, stars, box)])


def _troughs(arr, w, stars, box=3):
    """Deepest pixel in each star's core box -- the over-subtraction counterpart
    of `_peaks`.  A model carrying more flux than the star leaves a HOLE here,
    and `_peaks` on the residual cannot see one: a crater and a clean
    subtraction both have a small positive maximum."""
    return np.array([np.nan if sub is None or not np.isfinite(sub).any()
                     else float(np.nanmin(sub))
                     for sub in _box(arr, w, stars, box)])


#: Integrated-flux metric (issue #695).  A peak ratio is biased by the model
#: mosaic being sharper than the drizzled data mosaic, so the ratio that carries
#: the claim "the model holds the star's flux" is ``sum(model) / sum(data - sky)``
#: over a box that contains the star.  The box is scaled to the star's own
#: nearest curated neighbour (half the separation), so a close pair does not
#: put one star's flux in the other's aperture, and capped for isolated stars.
BOX_HALFWIDTH_CAP_PX = 12     # 25 x 25 px = 1.6" at the 0.063" LW scale
BOX_HALFWIDTH_FLOOR_PX = 3    # the 7 x 7 core `_peaks` has always used
#: Square sky annulus, in pixels from the star.  The tight alternative
#: (box + 2..6 px) reads PSF wings as sky and was tried and rejected: it moves
#: the median ratio from 0.88 to 0.91 and the spread with it.
SKY_INNER_PX = 30
SKY_OUTER_PX = 45
SKY_MIN_PIXELS = 100

#: Ensemble bounds on the integrated model / data ratio, calibrated on the
#: sickle F480M ``resbgsub_m5`` products of 2026-08-21 (39 curated stars, 38 in
#: frame with a 45 px margin, 32 of them at the box cap):
#:
#:     all in-frame stars      min 0.39  p10 0.77  median 0.88  p90 1.08  max 1.43
#:     isolated (box at cap)   min 0.39            median 0.87            max 1.33
#:
#: The median is nearly independent of the box size (0.88 at 7x7, 15x15 and
#: 25x25 in a separate fixed-box check) and the per-star spread is wide, so single stars are not asserted on: they go to the
#: committed record.  The median band is wide enough for an ordinary
#: re-reduction to stay inside it; the gross bounds are those the peak-ratio
#: tests already use.
INTEGRATED_MEDIAN_BAND = (0.7, 1.3)
INTEGRATED_GROSS_BOUNDS = (0.3, 2.0)
INTEGRATED_GROSS_FRACTION_MAX = 0.1
INTEGRATED_MIN_STARS = 20

RECORD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                           "sickle_f480m_residual_record.json")
#: Set to regenerate the committed record in place.  An environment variable
#: and not a pytest option, because an option needs a ``conftest.py`` and this
#: tree has none.
RECORD_UPDATE_ENV = "RESIDUAL_RECORD_UPDATE"


def nearest_neighbour_arcsec(stars):
    """Separation of each star from its nearest OTHER star in ``stars``.

    ``match_to_catalog_sky(nthneighbor=2)`` on the catalogue itself: neighbour 1
    is the star, neighbour 2 the nearest other.  This is a within-sample
    spacing for sizing a box, not an astrometric offset measurement.
    """
    stars = SkyCoord(stars)
    if stars.isscalar or len(stars) < 2:
        return np.full(1 if stars.isscalar else len(stars), np.inf)
    _, sep, _ = stars.match_to_catalog_sky(stars, nthneighbor=2)
    return sep.to_value(u.arcsec)


def star_halfwidth_px(nn_arcsec, pixscale_arcsec,
                      cap=BOX_HALFWIDTH_CAP_PX, floor=BOX_HALFWIDTH_FLOOR_PX):
    """Box half-width in pixels: half the nearest-neighbour distance, clipped."""
    half = np.floor(0.5 * np.asarray(nn_arcsec, float) / pixscale_arcsec)
    return np.clip(half, floor, cap).astype(int)


def _star_pixels(w, stars):
    xs, ys = w.world_to_pixel(stars)
    return (np.round(np.atleast_1d(xs).astype(float)).astype(int),
            np.round(np.atleast_1d(ys).astype(float)).astype(int))


def annulus_sky(data, w, stars, halfwidths, inner=SKY_INNER_PX,
                outer=SKY_OUTER_PX, min_pixels=SKY_MIN_PIXELS):
    """Per-star sky: median of ``data`` in a square annulus around the star.

    Pixels inside any OTHER star's box are masked, since a neighbour's flux in
    the annulus would raise the sky and lower that star's integrated flux.  A
    star whose annulus is not fully on the image, or leaves fewer than
    ``min_pixels`` finite pixels, gets ``nan``.
    """
    ny, nx = data.shape
    xi, yi = _star_pixels(w, stars)
    halfwidths = np.asarray(halfwidths, int)
    sky = np.full(len(xi), np.nan)
    side = np.arange(-outer, outer + 1)
    ring = np.maximum(np.abs(side[None, :]), np.abs(side[:, None])) >= inner
    for i, (x, y) in enumerate(zip(xi, yi)):
        if not (outer <= x < nx - outer and outer <= y < ny - outer):
            continue
        cut = data[y - outer:y + outer + 1, x - outer:x + outer + 1]
        keep = ring & np.isfinite(cut)
        for j in range(len(xi)):
            if j == i:
                continue
            h = halfwidths[j]
            x0, x1 = xi[j] - h - (x - outer), xi[j] + h + 1 - (x - outer)
            y0, y1 = yi[j] - h - (y - outer), yi[j] + h + 1 - (y - outer)
            if x1 <= 0 or y1 <= 0 or x0 >= cut.shape[1] or y0 >= cut.shape[0]:
                continue
            keep[max(0, y0):y1, max(0, x0):x1] = False
        if keep.sum() >= min_pixels:
            sky[i] = float(np.median(cut[keep]))
    return sky


def integrated_ratio(numerator, data, w, stars, halfwidths, sky):
    """``sum(numerator[box]) / sum(data[box] - sky)`` per star.

    ``numerator`` is the model for the model/data ratio and the residual for
    the residual fraction.  ``numerator``, ``data`` and ``w`` share one pixel
    grid.  ``nan`` for a star whose box is not fully on the image, has no
    finite pixel in common between the two arrays, or whose sky-subtracted data
    sums to zero or less (a ratio against a non-positive flux is not a ratio).
    """
    if numerator.shape != data.shape:
        raise ValueError(f"numerator {numerator.shape} and data {data.shape} "
                         f"are not on one pixel grid")
    ny, nx = data.shape
    xi, yi = _star_pixels(w, stars)
    halfwidths = np.asarray(halfwidths, int)
    sky = np.broadcast_to(np.asarray(sky, float), xi.shape)
    out = np.full(len(xi), np.nan)
    for i, (x, y, h) in enumerate(zip(xi, yi, halfwidths)):
        if not (h <= x < nx - h and h <= y < ny - h) or not np.isfinite(sky[i]):
            continue
        num = numerator[y - h:y + h + 1, x - h:x + h + 1]
        dat = data[y - h:y + h + 1, x - h:x + h + 1]
        ok = np.isfinite(num) & np.isfinite(dat)
        if not ok.any():
            continue
        flux = float(np.sum(dat[ok] - sky[i]))
        if flux > 0:
            out[i] = float(np.sum(num[ok])) / flux
    return out


def _git_sha():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def star_record(stars, nn_arcsec, halfwidth, model_ratio, resid_fraction,
                dates, git_sha=None):
    """The per-star record as a JSON-ready dict.

    ``resid_fraction`` is ``sum(resid[box]) / sum(data[box] - sky)`` over the
    same box and sky as the model ratio, kept so that a move of
    `test_residual_contains_no_stars` to an integrated statistic can be
    calibrated from this file without another pass over the mosaics.
    """
    def _num(x, nd):
        return None if not np.isfinite(x) else round(float(x), nd)

    stars = SkyCoord(stars)
    rows = [dict(index=i, ra=round(float(c.ra.deg), 6),
                 dec=round(float(c.dec.deg), 6),
                 nn_arcsec=_num(nn, 2), halfwidth_px=int(h),
                 model_ratio=_num(mr, 3), resid_fraction=_num(rf, 3))
            for i, (c, nn, h, mr, rf) in enumerate(
                zip(stars, nn_arcsec, halfwidth, model_ratio, resid_fraction))]
    return dict(product_dates=dict(dates), git_sha=git_sha, stars=rows)


def record_diff(new, committed_path=RECORD_PATH):
    """Unified diff of the regenerated record against the committed copy.

    Returns ``""`` when they agree, and a message when there is no committed
    copy or it cannot be parsed.  Never raises on a bad file: the diff is a
    report, and nothing asserts on it.  ``git_sha`` is left out of the
    comparison, since it differs after every commit.
    """
    def _dump(rec):
        rec = {k: v for k, v in rec.items() if k != "git_sha"}
        return json.dumps(rec, indent=1, sort_keys=True).splitlines(keepends=True)

    try:
        with open(committed_path) as fh:
            old = json.load(fh)
    except FileNotFoundError:
        return f"no committed record at {committed_path}\n"
    except (OSError, ValueError) as exc:
        return f"committed record {committed_path} is unreadable: {exc}\n"
    return "".join(difflib.unified_diff(_dump(old), _dump(new),
                                        "committed", "regenerated"))


def write_record(rec, path=RECORD_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")


def _integrated_products(data, resid, model, dw, stars, dates):
    """Boxes, sky, ratios and the record for the curated stars."""
    pix = float(np.mean(proj_plane_pixel_scales(dw))) * 3600.0
    nn = nearest_neighbour_arcsec(stars)
    half = star_halfwidth_px(nn, pix)
    sky = annulus_sky(data, dw, stars, half)
    ratio = integrated_ratio(model, data, dw, stars, half, sky)
    rfrac = integrated_ratio(resid, data, dw, stars, half, sky)
    return dict(nn_arcsec=nn, halfwidth=half, sky=sky, int_ratio=ratio,
                int_resid=rfrac,
                record=star_record(stars, nn, half, ratio, rfrac, dates,
                                   _git_sha()))


#: Ceilings for the OVER-subtraction direction (issue #266 item 4).  Both are
#: ratios to the star's own data peak, and both are deliberately loose against
#: the current products.  Measured over the 39 curated stars in the sickle F480M
#: ``resbgsub_m5`` products of 2026-08-21:
#:
#:     model/data peak      max  1.46   (4 stars over 1.2, 3 over 1.3, 1 over 1.4)
#:     resid core / data    min -0.47   (4 stars below -0.3, 3 below -0.4)
#:
#: so the tighter pair suggested on the issue (1.5 and -0.5) clears today's worst
#: star by 3% and 5%, which is inside the run-to-run movement this file's own
#: history records -- a permanently-red assertion one re-reduction later.  These
#: sit where the claim stops depending on the metric's known bias instead: the
#: model mosaic is sharper than the drizzled data mosaic, so a peak ratio
#: over-reads by roughly 1.1-1.3x, and nothing in that explains a model at twice
#: the star's peak or a crater deeper than the whole star.  What they pin is that
#: the direction is TESTED, which it was not; tightening them wants the
#: integrated metric the issue asks for, calibrated on its own.
MODEL_PEAK_CEILING = 2.0
RESID_CORE_FLOOR = -1.0


def over_rendered(d, m, ceiling=MODEL_PEAK_CEILING):
    """Indices where the model peak exceeds ``ceiling`` x the data peak."""
    d, m = np.asarray(d, float), np.asarray(m, float)
    ok = np.isfinite(d) & (d > 0) & np.isfinite(m)
    return np.flatnonzero(ok & (m > ceiling * d))


def over_subtracted(d, rmin, floor=RESID_CORE_FLOOR):
    """Indices where the residual core digs below ``floor`` x the data peak."""
    d, rmin = np.asarray(d, float), np.asarray(rmin, float)
    ok = np.isfinite(d) & (d > 0) & np.isfinite(rmin)
    return np.flatnonzero(ok & (rmin < floor * d))


#: The EXACT products the fixture opens.  The skip predicate must be keyed on
#: these and not on anything looser: it used to check
#: ``*mergedcat_residual_i2d.fits`` while the fixture globbed
#: ``*_m7_*mergedcat_residual_i2d.fits``, so on a field that had reached m5 but
#: not m7 the predicate found 4 products, declined to skip, and the fixture then
#: opened ``None`` -- three ERRORS on main since 2026-07-05 (issue #266).
#: sickle F480M is exactly that field: it carries m2/m3/m4/resbgsub_m5 residuals
#: and no m7, because it is parked at m6 (issue #285).
#: The stage the content checks run against.  It was ``_m7_``.  sickle F480M
#: has 400 ``*_m7_*`` files, including 48 per-frame
#: ``m7_daophot_basic_mergedcat_residual.fits``; what it has none of is the m7
#: **i2d** (``*_m7_*i2d.fits`` -> 0), which is what these globs want, and it is
#: not scheduled -- the field's newest ungrouped i2d products are
#: ``resbgsub_m5``.  Keying on ``_m7_`` therefore made this a permanent green
#: skip on the analysis machine, where the docstring says it should run "as a
#: hard regression".  A test that can never run is worse than one that fails.
#:
#: ``resbgsub_m5`` also removes the cross-run hazard issue #266 records: the
#: ``_m7_`` glob matched both ``resbgsub_m7`` and ``resbgsub_group_m7``, so
#: ``_latest`` could take the numerator from one run and the denominator from
#: another.  There is no ``group`` variant at m5, so each glob matches exactly
#: one file.
STAGE = "resbgsub_m5"

_REQUIRED = {
    "data i2d": f"{SICKLE}/F480M/pipeline/*-f480m-nrcb_data_i2d.fits",
    f"{STAGE} residual i2d":
        f"{SICKLE}/F480M/pipeline/*_{STAGE}_*mergedcat_residual_i2d.fits",
    f"{STAGE} model i2d":
        f"{SICKLE}/F480M/pipeline/*_{STAGE}_*mergedcat_model_i2d.fits",
}


def _missing_products():
    missing = [] if os.path.exists(REG) else [f"region file {REG}"]
    missing += [f"{what} ({pat})" for what, pat in sorted(_REQUIRED.items())
                if not glob.glob(pat)]
    return missing


pytestmark = pytest.mark.skipif(
    bool(_missing_products()),
    reason="sickle F480M QA products / region file not present: "
           + "; ".join(_missing_products()))


@pytest.fixture(scope="module")
def f480m():
    stars = _read_points(REG)
    data_path = _latest(_REQUIRED["data i2d"])
    resid_path = _latest(_REQUIRED[f"{STAGE} residual i2d"])
    model_path = _latest(_REQUIRED[f"{STAGE} model i2d"])
    # The three globs are resolved FIRST so this check cannot mask which
    # products the fixture asks for (test_residual_policy_meta.py records them).
    mismatch = cross_run_reason(_product_date(data_path),
                                _product_date(resid_path),
                                _product_date(model_path))
    if mismatch:
        pytest.skip(mismatch)
    data, dw = _img(data_path, "data i2d")
    resid, rw = _img(resid_path, f"{STAGE} residual i2d")
    model, mw = _img(model_path, f"{STAGE} model i2d")
    dates = dict(data=_product_date(data_path), residual=_product_date(resid_path),
                 model=_product_date(model_path))
    return dict(stars=stars,
                d=_peaks(data, dw, stars), r=_peaks(resid, rw, stars),
                m=_peaks(model, mw, stars), rmin=_troughs(resid, rw, stars),
                model=model, resid=resid,
                **_integrated_products(data, resid, model, dw, stars, dates))


@pytest.mark.xfail(strict=True, reason=(
    "issue #266 satstar defect, now REPRODUCED rather than skipped: 5 of 39 "
    "curated bright stars remain in the sickle F480M resbgsub_m5 residual at "
    ">30% of their data peak (median resid/data 0.21).  strict=True, so this "
    "FAILS once the model is fixed -- delete the marker then."))
def test_residual_contains_no_stars(f480m):
    """At every curated bright star the residual peak must be a small fraction
    of the data peak -- the star must be SUBTRACTED, leaving only background."""
    d, r = f480m['d'], f480m['r']
    ok = np.isfinite(d) & (d > 0) & np.isfinite(r)
    frac = r[ok] / d[ok]
    n_bad = int(np.sum(frac > 0.3))
    assert n_bad == 0, (
        f"{n_bad}/{int(ok.sum())} curated bright stars remain in the F480M "
        f"residual at >30% of their data peak (median resid/data="
        f"{np.median(frac):.2f}); the residual must contain background only.")


def test_model_contains_the_stars(f480m):
    """The MODEL i2d must contain every curated bright star (saturated ones too)."""
    d, m = f480m['d'], f480m['m']
    ok = np.isfinite(d) & (d > 0) & np.isfinite(m)
    frac = m[ok] / d[ok]
    n_missing = int(np.sum(frac < 0.2))
    assert n_missing == 0, (
        f"{n_missing}/{int(ok.sum())} curated bright stars are missing/weak in "
        f"the F480M model i2d (peak <20% of data); the model must contain all "
        f"stars, saturated and unsaturated.")


def test_model_not_overrendered(f480m):
    """The other direction of `test_model_contains_the_stars`.

    That test fails a model peak BELOW 20% of the data peak and says nothing
    about one above it, so a star rendered at several times its own brightness
    passes cleanly -- one did, at 3.3x, in the census on issue #266.  An
    over-rendered model is subtracted from the data, so it is the same defect as
    a missing one seen from the other side.
    """
    bad = over_rendered(f480m['d'], f480m['m'])
    assert bad.size == 0, (
        f"{bad.size} curated bright star(s) render at more than "
        f"{MODEL_PEAK_CEILING}x their data peak in the F480M model i2d "
        f"(worst {np.nanmax(np.asarray(f480m['m'])[bad] / np.asarray(f480m['d'])[bad]):.2f}x); "
        f"the model must carry the star's flux, not more.")


def test_residual_not_oversubtracted(f480m):
    """The other direction of `test_residual_contains_no_stars`.

    That test reads the residual's MAXIMUM, so it fails a star left behind and
    passes a star gouged out: a crater and a clean subtraction both have a small
    positive maximum.  A hole biases the sky estimate the same way a leftover
    star does, and the residual-bg feedback loop reads that sky.
    """
    bad = over_subtracted(f480m['d'], f480m['rmin'])
    assert bad.size == 0, (
        f"{bad.size} curated bright star(s) leave a residual core below "
        f"{RESID_CORE_FLOOR}x their data peak "
        f"(worst {np.nanmin(np.asarray(f480m['rmin'])[bad] / np.asarray(f480m['d'])[bad]):.2f}x); "
        f"the residual must contain background only, in both directions.")


def test_model_background_not_negative(f480m):
    """MODEL i2d is stars-on-zero: faint pixels ~0, never a negative pedestal."""
    m = f480m['model']
    fin = m[np.isfinite(m)]
    faint = fin[np.abs(fin) < 5]
    assert np.median(faint) > -0.5, (
        f"F480M model i2d has a negative background pedestal "
        f"(faint-pixel median {np.median(faint):.2f}); model must have no bg.")
    assert np.mean(fin < -0.5) < 0.05, (
        f"{np.mean(fin < -0.5):.1%} of F480M model pixels are < -0.5; "
        f"the model must be stars on a zero background.")


def test_integrated_model_ratio_in_band(f480m):
    """The model carries the stars' flux: ``sum(model) / sum(data - sky)`` over
    each star's own box, asserted on the ENSEMBLE (issue #695).

    The isolated stars (box at its cap) are the clean sample; stars in close
    pairs get a smaller box and a larger aperture-loss bias, so they appear in
    the record and in the gross-bound test below, not in this median.
    """
    ratio, half = f480m['int_ratio'], f480m['halfwidth']
    sample = ratio[(half == BOX_HALFWIDTH_CAP_PX) & np.isfinite(ratio)]
    assert sample.size >= INTEGRATED_MIN_STARS, (
        f"only {sample.size} isolated, in-frame curated stars have an "
        f"integrated ratio; need {INTEGRATED_MIN_STARS}")
    lo, hi = INTEGRATED_MEDIAN_BAND
    assert lo <= np.median(sample) <= hi, (
        f"median integrated model/data ratio {np.median(sample):.2f} over "
        f"{sample.size} isolated stars is outside [{lo}, {hi}] "
        f"(2026-08-21 products: 0.87)")


def test_integrated_model_ratio_gross_outliers_are_rare(f480m):
    """No more than a small fraction of the stars sit outside the gross bounds."""
    ratio = f480m['int_ratio']
    ratio = ratio[np.isfinite(ratio)]
    assert ratio.size >= INTEGRATED_MIN_STARS
    lo, hi = INTEGRATED_GROSS_BOUNDS
    frac = float(np.mean((ratio < lo) | (ratio > hi)))
    assert frac <= INTEGRATED_GROSS_FRACTION_MAX, (
        f"{frac:.0%} of {ratio.size} curated stars have an integrated "
        f"model/data ratio outside [{lo}, {hi}] "
        f"(limit {INTEGRATED_GROSS_FRACTION_MAX:.0%}; 2026-08-21 products: 0%)")


def test_per_star_record_diff(f480m, capsys):
    """Print the per-star record as a diff against the committed copy.

    Nothing here asserts on the diff: the record documents how each star moved
    between product generations, and a legitimate re-reduction must not turn the
    suite red.  Read it with ``pytest -s``.  ``RESIDUAL_RECORD_UPDATE=1``
    rewrites the committed copy.  A cross-generation set of products never
    reaches this test, because the fixture skips it first.
    """
    rec = f480m['record']
    if os.environ.get(RECORD_UPDATE_ENV):
        write_record(rec)
        print(f"wrote {RECORD_PATH}")
        return
    diff = record_diff(rec)
    with capsys.disabled():
        print("\nper-star residual record vs committed copy:")
        print(diff if diff else "  (no differences)")


# NB pytest ORs skip conditions: a function-level skipif(False) CANNOT cancel a
# module-level pytestmark.  These two must therefore live outside the module
# mark, which is why _REQUIRED and the mark are module-level and these tests
# take no fixture -- see test_residual_policy_meta.py.
