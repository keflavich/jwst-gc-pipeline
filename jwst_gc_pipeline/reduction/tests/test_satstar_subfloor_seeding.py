"""Regression: sub-floor suppression-strip seeding + degenerate-pair flatness.

Core suppression is continuous in well fill, not a step at the severity
floor.  Stars peaking at ~0.4-1.0x the floor carry up to ~0.4 mag of
unflagged core suppression with ZERO DQ-SATURATED pixels (Brick F410M
12.2<m<13.3: the near-degenerate F405N-F410M color plunges -0.10 -> -0.49
exactly below the flagging floor; F182M analog below F187N~14.7).  Guards:

1. find_saturated_stars seeds DQ-clean components peaking in
   [SATSTAR_SUBFLOOR_SEED_FRAC * floor, floor) with an amplitude-derived
   mask (>=2 px, <=50 px so extended emission cannot seed).
2. degenerate_pair_flatness/assert_degenerate_pair_flatness: the released
   catalog's near-degenerate colors must be flat with magnitude.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    find_saturated_stars)
from jwst_gc_pipeline.photometry.saturation_continuity import (
    degenerate_pair_flatness, assert_degenerate_pair_flatness,
    DEGENERATE_PAIRS)

SATBIT = 2


def _fitsdata(data, satmask=None):
    ny, nx = data.shape
    dq = np.zeros((ny, nx), dtype=np.uint32)
    if satmask is not None:
        dq[satmask] = SATBIT
    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=data.astype('float32'), name='SCI'),
        fits.ImageHDU(data=dq, name='DQ'),
        fits.ImageHDU(data=np.ones((ny, nx), dtype='float32'),
                      name='VAR_POISSON'),
    ])


def _blob(shape, x, y, r=2):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


FLOOR = 4000.0


def test_subfloor_star_is_seeded():
    """DQ-clean star peaking at 0.6x floor: seeded with an amplitude mask."""
    data = np.full((100, 100), 5.0)
    data[_blob(data.shape, 50, 50, r=2)] = 0.6 * FLOOR
    sat, src, coms = find_saturated_stars(_fitsdata(data),
                                          severity_floor=FLOOR)
    assert len(coms) == 1, "sub-floor suppression-strip star must be seeded"
    # amplitude-derived mask: bright px + 2-px shoulder dilation, not ~0 px
    assert sat.sum() >= 13


def test_faint_star_below_frac_not_seeded():
    """Peak at 0.2x floor (below the 0.35 default fraction): not seeded."""
    data = np.full((100, 100), 5.0)
    data[_blob(data.shape, 50, 50, r=2)] = 0.2 * FLOOR
    sat, src, coms = find_saturated_stars(_fitsdata(data),
                                          severity_floor=FLOOR)
    assert len(coms) == 0


def test_single_hot_pixel_not_seeded():
    data = np.full((100, 100), 5.0)
    data[50, 50] = 0.7 * FLOOR
    sat, src, coms = find_saturated_stars(_fitsdata(data),
                                          severity_floor=FLOOR)
    assert len(coms) == 0, "single hot pixel must not seed"


def test_extended_emission_not_seeded():
    """A >50-px region above frac*floor (bright emission): not seeded."""
    data = np.full((200, 200), 5.0)
    data[_blob(data.shape, 100, 100, r=8)] = 0.5 * FLOOR
    sat, src, coms = find_saturated_stars(_fitsdata(data),
                                          severity_floor=FLOOR)
    assert len(coms) == 0, "extended emission must not seed"


def test_gate_off_when_floor_zero():
    data = np.full((100, 100), 5.0)
    data[_blob(data.shape, 50, 50, r=2)] = 3000.0
    sat, src, coms = find_saturated_stars(_fitsdata(data), severity_floor=0.0)
    assert len(coms) == 0


def test_subfloor_frac_env_override(monkeypatch):
    monkeypatch.setenv('SATSTAR_SUBFLOOR_SEED_FRAC', '0')
    data = np.full((100, 100), 5.0)
    data[_blob(data.shape, 50, 50, r=2)] = 0.6 * FLOOR
    sat, src, coms = find_saturated_stars(_fitsdata(data),
                                          severity_floor=FLOOR)
    assert len(coms) == 0, 'SATSTAR_SUBFLOOR_SEED_FRAC=0 must disable'


# --------------------------- flatness metric ---------------------------

def _synthetic_pair(strip_offset=0.0, n=20000, seed=0):
    """Flat intrinsic color -0.10 + optional suppression strip at the
    bright end of band B (mag 12.2-13.3 in an 12-18 catalog)."""
    rng = np.random.default_rng(seed)
    mB = rng.uniform(12.0, 18.0, n)
    color = np.full(n, -0.10) + rng.normal(0, 0.05, n)
    strip = (mB > 12.2) & (mB < 13.3)
    color[strip] += strip_offset
    return Table({'mag_vega_f405n': mB + color, 'mag_vega_f410m': mB})


def test_flat_pair_passes():
    cat = _synthetic_pair(strip_offset=0.0)
    r = degenerate_pair_flatness(cat, 'f405n', 'f410m')
    assert np.isfinite(r['metric']) and r['metric'] < 0.05
    assert_degenerate_pair_flatness(cat, pairs=[('f405n', 'f410m')])


def test_suppression_strip_fails():
    cat = _synthetic_pair(strip_offset=-0.35)
    r = degenerate_pair_flatness(cat, 'f405n', 'f410m')
    assert r['metric'] > 0.25
    assert r['worst_bin']['magB_lo'] >= 12.0
    assert r['worst_bin']['magB_lo'] <= 13.3
    with pytest.raises(AssertionError, match='drift'):
        assert_degenerate_pair_flatness(cat, pairs=[('f405n', 'f410m')])


def test_default_pairs_defined():
    assert ('f405n', 'f410m') in DEGENERATE_PAIRS
    assert ('f182m', 'f187n') in DEGENERATE_PAIRS


def test_missing_bands_no_crash():
    cat = Table({'mag_vega_f090w': np.linspace(12, 18, 500)})
    r = degenerate_pair_flatness(cat, 'f405n', 'f410m')
    assert not np.isfinite(r['metric'])
    assert_degenerate_pair_flatness(cat)  # nan metrics -> no failure
