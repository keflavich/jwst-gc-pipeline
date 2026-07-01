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
import glob
import os
import numpy as np
import pytest

pytest.importorskip("astropy")
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

SICKLE = "/orange/adamginsburg/jwst/sickle"
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


def _img(path):
    h = fits.open(path)
    sci = h['SCI'] if 'SCI' in [x.name for x in h] else h[0]
    return sci.data, WCS(sci.header)


def _peaks(arr, w, stars, box=3):
    out = []
    ny, nx = arr.shape
    xs, ys = w.world_to_pixel(stars)
    for x, y in zip(np.atleast_1d(xs), np.atleast_1d(ys)):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < nx and 0 <= yi < ny):
            out.append(np.nan); continue
        sub = arr[max(0, yi - box):yi + box + 1, max(0, xi - box):xi + box + 1]
        out.append(float(np.nanmax(sub)) if np.isfinite(sub).any() else np.nan)
    return np.array(out)


pytestmark = pytest.mark.skipif(
    not os.path.exists(REG)
    or not glob.glob(f"{SICKLE}/F480M/pipeline/*mergedcat_residual_i2d.fits"),
    reason="sickle F480M QA products / region file not present")


@pytest.fixture(scope="module")
def f480m():
    stars = _read_points(REG)
    data, dw = _img(_latest(f"{SICKLE}/F480M/pipeline/*-f480m-nrcb_data_i2d.fits"))
    resid, rw = _img(_latest(f"{SICKLE}/F480M/pipeline/*_m7_*mergedcat_residual_i2d.fits"))
    model, mw = _img(_latest(f"{SICKLE}/F480M/pipeline/*_m7_*mergedcat_model_i2d.fits"))
    return dict(stars=stars,
                d=_peaks(data, dw, stars), r=_peaks(resid, rw, stars),
                m=_peaks(model, mw, stars),
                model=model, resid=resid)


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
