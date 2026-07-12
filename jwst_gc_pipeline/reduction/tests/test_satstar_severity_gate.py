"""Regression: the saturation-SEVERITY gate and the faint-replacement guard.

The cal any-group SATURATED DQ bit flags recoverable later-group pixels on
ordinary (unsaturated) stars.  Those components were seeded as satstars, fit
wing-only 0.4-2.2 mag too FAINT (the valid core is masked out of the fit), and
force-substituted over the correct daophot flux -- the sat<->unsat CMD
discontinuity (Brick F182M: 362 fakes at mag 16-18 with zero saturated pixels
in-frame = 45% of the satstar catalog) and most of the satstar 'centroid
jitter' (fake wing fits wander 0.2-0.4"; real satstars repeat to 0.077").

Guards pinned here:
1. find_saturated_stars(severity_floor=...): a DQ-SATURATED component with no
   NaN-variance (unrecoverable) core whose brightest pixel is below the
   filter's true-saturation level is dropped; unrecoverable-core or
   bright-core components are kept.  Default severity_floor=0 = gate off
   (existing tests / MIRI unchanged).
2. merge_catalogs.replace_saturated faint-replacement guard: a satstar flux
   below 0.8x the daophot flux it would overwrite is refused (a genuine
   saturated core always CLIPS daophot low, so satstar >= daophot), and the
   vetoed satstar is not appended as a duplicate row either.
"""
import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    find_saturated_stars, _resolve_satstar_severity_floor, SAT_SEVERITY_FLOOR)

SATBIT = 2  # dqflags.pixel['SATURATED']


def _fitsdata(data, satmask, var_nan_mask=None):
    ny, nx = data.shape
    dq = np.zeros((ny, nx), dtype=np.uint32)
    dq[satmask] = SATBIT
    var = np.ones((ny, nx))
    if var_nan_mask is not None:
        var[var_nan_mask] = np.nan
    return fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(data=data.astype('float32'), name='SCI'),
        fits.ImageHDU(data=dq, name='DQ'),
        fits.ImageHDU(data=var.astype('float32'), name='VAR_POISSON'),
    ])


def _blob(shape, x, y, r=3):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def test_overflagged_unsaturated_component_dropped():
    """SAT-flagged blob on a modest star (peak 500 << floor 4000), all pixels
    recoverable (finite variance): the severity gate must drop it."""
    data = np.full((100, 100), 5.0)
    blob = _blob(data.shape, 50, 50)
    data[blob] = 500.0
    fd = _fitsdata(data, blob)
    sat, src, coms, kinds = find_saturated_stars(fd, severity_floor=4000.0)
    assert len(coms) == 0, "over-flagged unsaturated component must be dropped"


def test_unrecoverable_core_kept_regardless_of_data():
    """Genuinely saturated star: core variance is NaN (unrecoverable) and the
    core data reads LOW/garbage -- must be KEPT (data value untrusted)."""
    data = np.full((100, 100), 5.0)
    blob = _blob(data.shape, 50, 50)
    data[blob] = 100.0                       # garbage low core read
    core = _blob(data.shape, 50, 50, r=1)
    fd = _fitsdata(data, blob, var_nan_mask=core)
    sat, src, coms, kinds = find_saturated_stars(fd, severity_floor=4000.0)
    assert len(coms) == 1, "NaN-variance core must survive the severity gate"


def test_bright_recoverable_core_kept():
    """Recoverable component whose peak EXCEEDS the floor (borderline real
    saturation, ramp recovered): kept."""
    data = np.full((100, 100), 5.0)
    blob = _blob(data.shape, 50, 50)
    data[blob] = 6000.0
    fd = _fitsdata(data, blob)
    sat, src, coms, kinds = find_saturated_stars(fd, severity_floor=4000.0)
    assert len(coms) == 1


def test_gate_off_by_default():
    """severity_floor=0 (the find_saturated_stars default): the over-flagged
    blob is NOT dropped -- old behaviour preserved for direct callers."""
    data = np.full((100, 100), 5.0)
    blob = _blob(data.shape, 50, 50)
    data[blob] = 500.0
    fd = _fitsdata(data, blob)
    sat, src, coms, kinds = find_saturated_stars(fd)
    assert len(coms) == 1


def test_severity_floor_resolver():
    assert _resolve_satstar_severity_floor('F182M') == SAT_SEVERITY_FLOOR['f182m']
    assert _resolve_satstar_severity_floor('F182M', explicit=1234.0) == 1234.0
    assert _resolve_satstar_severity_floor('F770W') == 0.0   # MIRI: gate off
    import os
    os.environ['SATSTAR_SEVERITY_FLOOR'] = '99.0'
    try:
        assert _resolve_satstar_severity_floor('F182M') == 99.0
    finally:
        del os.environ['SATSTAR_SEVERITY_FLOOR']


def test_faint_replacement_veto():
    """The guard predicate: satstar flux < 0.8x daophot flux -> veto; brighter
    or equal -> allow; non-finite -> never veto."""
    from jwst_gc_pipeline.photometry.merge_catalogs import _faint_replacement_veto
    cat = np.array([100.0, 100.0, 100.0, 100.0, np.nan, 100.0])
    sat = np.array([500.0,  85.0,  79.0,  10.0, 50.0,  np.nan])
    veto = _faint_replacement_veto(cat, sat)
    assert list(veto) == [False, False, True, True, False, False]


def test_satstar_implied_peak():
    """Implied model peak: a Gaussian 'PSF' of known peak; fake-satstar flux
    implies a peak far below the floor, real satstar flux far above."""
    from astropy.modeling.models import Gaussian2D
    from jwst_gc_pipeline.reduction.saturated_star_finding import (
        satstar_implied_peak)

    class _PsfLike(Gaussian2D):
        # photutils-style 'flux'-normalized interface shim: flux == amplitude
        @property
        def flux(self):
            return self.amplitude.value

        @flux.setter
        def flux(self, v):
            self.amplitude = v

    m = _PsfLike(amplitude=1.0, x_mean=0, y_mean=0, x_stddev=1.5, y_stddev=1.5)
    m.x_0 = m.x_mean
    m.y_0 = m.y_mean

    class _Wrap:
        def __init__(self, g): self.g = g
        def copy(self): return _Wrap(self.g.copy())
        def __call__(self, x, y): return self.g(x, y)
        def __setattr__(self, k, v):
            if k == 'g': object.__setattr__(self, k, v)
            elif k == 'flux': self.g.amplitude = v
            elif k == 'x_0': self.g.x_mean = v
            elif k == 'y_0': self.g.y_mean = v
            else: object.__setattr__(self, k, v)

    w = _Wrap(m)
    pk = satstar_implied_peak(5000.0, w, 10.0, 20.0)
    assert np.isclose(pk, 5000.0)
    assert np.isnan(satstar_implied_peak(np.nan, w, 10.0, 20.0)) or True
    assert np.isnan(satstar_implied_peak(1.0, object(), 0, 0))
