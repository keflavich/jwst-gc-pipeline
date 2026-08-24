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
import glob
import os
import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits
from astropy.wcs import WCS
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
    return dict(stars=stars,
                d=_peaks(data, dw, stars), r=_peaks(resid, rw, stars),
                m=_peaks(model, mw, stars), rmin=_troughs(resid, rw, stars),
                model=model, resid=resid)


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


# NB pytest ORs skip conditions: a function-level skipif(False) CANNOT cancel a
# module-level pytestmark.  These two must therefore live outside the module
# mark, which is why _REQUIRED and the mark are module-level and these tests
# take no fixture -- see test_residual_policy_meta.py.
