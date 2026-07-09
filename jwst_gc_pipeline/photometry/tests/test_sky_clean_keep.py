"""Regression: the SKY-CLEAN vetting keep tier (_filter_extended_emission).

Where the deep-i2d local emission floor at a source is consistent with the
field's dark-sky reference, "extended emission turned into a star" is
physically impossible, so a blend-degraded real star (high qfit, high
prominence, decent S/N) must be KEPT with qfit ignored.  Where the local
floor is elevated (real nebulosity) the tier must be INERT so the
conservative emission gates stand unchanged -- this per-source switch is the
depth-vs-purity contract (Brick F182M zero-background clump: the qfit gate
deleted 7/15 detected real stars, all prominence 16-48 on clean sky).

Guards pinned here: satstar proximity (spike knots on dark sky are prominent
+ bad-qfit, exactly the tier's admit pattern) and the prominence floor.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission

RNG = np.random.default_rng(42)
NOISE = 1.0
EMISSION = 25.0          # >> 2 dark-sky sigma: unambiguous nebulosity
AMP = 30.0               # star peak: prominence ~ AMP / annulus-MAD >> 5
NY = NX = 400
X_CLEAN, X_EMIT = 300, 60   # emission occupies x < 150 only


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.03 / 3600, 0.03 / 3600]
    return w


def _image():
    data = RNG.normal(0.0, NOISE, (NY, NX))
    data[:, :150] += EMISSION
    return data


def _add_star(data, x, y, amp=AMP, sig=1.5):
    yy, xx = np.mgrid[0:NY, 0:NX]
    data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _cat(w, rows):
    """rows: list of dicts with x, y, qfit, and optional is_saturated."""
    sc = w.pixel_to_world(np.array([r['x'] for r in rows]),
                          np.array([r['y'] for r in rows]))
    return Table({
        'skycoord': sc,
        'qfit': np.array([r['qfit'] for r in rows]),
        'flags': np.zeros(len(rows)),
        'local_bkg': np.zeros(len(rows)),
        'flux': np.full(len(rows), 100.0),
        'flux_err': np.full(len(rows), 10.0),   # S/N 10 > sky_clean_snr_min
        'group_size': np.ones(len(rows)),
        'is_saturated': np.array([bool(r.get('is_saturated', False))
                                  for r in rows]),
    })


def _run(rows, add_star_at=None, **kw):
    w = _wcs()
    data = _image()
    for r in (add_star_at or []):
        _add_star(data, r[0], r[1])
    cat = _cat(w, rows)
    return _filter_extended_emission(cat, data_i2d_image=data, ww_i2d=w,
                                     label='test', **kw)


def test_blend_degraded_star_on_clean_sky_kept():
    """qfit 0.8 (blend-degraded) star on emission-free sky: prominent + decent
    S/N -> the sky-clean tier must keep it (it is dropped without the tier)."""
    rows = [{'x': X_CLEAN, 'y': 200.0, 'qfit': 0.8}]
    kept = _run(rows, add_star_at=[(X_CLEAN, 200.0)])
    assert len(kept) == 1, "sky-clean tier must keep the blend-degraded star"
    assert bool(kept['sky_clean'][0])
    dropped = _run(rows, add_star_at=[(X_CLEAN, 200.0)], sky_clean_keep=False)
    assert len(dropped) == 0, ("without the tier the qfit gate drops it -- "
                               "the test no longer discriminates")


def test_same_qfit_source_on_emission_not_kept():
    """The SAME bad-qfit source sitting ON the elevated emission floor must
    NOT be admitted: the tier is inert where local emission is real."""
    rows = [{'x': X_EMIT, 'y': 200.0, 'qfit': 0.8}]
    kept = _run(rows, add_star_at=[(X_EMIT, 200.0)])
    assert len(kept) == 0, ("tier admitted a bad-qfit source on emission -- "
                            "the per-source emission switch is broken")


def test_low_prominence_clean_sky_bump_not_kept():
    """Clean sky but NO underlying peak (prominence ~ noise): not admitted."""
    rows = [{'x': X_CLEAN, 'y': 120.0, 'qfit': 0.8}]
    kept = _run(rows)          # no star injected at the position
    assert len(kept) == 0


def test_satstar_proximity_guard():
    """A prominent bad-qfit candidate near a saturated star (spike-knot
    pattern) must not be admitted via the sky-clean tier."""
    rows = [{'x': X_CLEAN, 'y': 200.0, 'qfit': 0.8},
            {'x': X_CLEAN + 20, 'y': 200.0, 'qfit': 0.05,
             'is_saturated': True}]     # 20 px * 0.03" = 0.6" < 2" guard
    kept = _run(rows, add_star_at=[(X_CLEAN, 200.0), (X_CLEAN + 20, 200.0)])
    names = np.asarray(kept['qfit'])
    assert not np.any(np.isclose(names, 0.8)), (
        "sky-clean tier admitted a candidate within the satstar guard radius")


def test_good_qfit_star_unaffected():
    """Baseline: a qfit-confident star is kept with and without the tier."""
    rows = [{'x': X_CLEAN, 'y': 200.0, 'qfit': 0.05}]
    assert len(_run(rows, add_star_at=[(X_CLEAN, 200.0)])) == 1
    assert len(_run(rows, add_star_at=[(X_CLEAN, 200.0)],
                    sky_clean_keep=False)) == 1
