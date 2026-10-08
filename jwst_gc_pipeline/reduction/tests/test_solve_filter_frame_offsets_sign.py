"""``solve_filter_frame_offsets.exposure_residual`` returns (exposure - anchor).

``measure_offset(a, b)`` and ``local_residual_map(a, b, ...)`` both report
(b - a).  The solver calls them with a = exposure and b = anchor, so their
total is (anchor - exposure) and has to be negated before it is labelled a
(filter - anchor) residual.  Without that, ``--write`` negates an already
negated vector and stores the residual itself: on wd2 F162M (10 mas
per-detector terms) the written table doubled the error, 10.2 -> 20.5 mas
rms, through ``apply_filter_frame_correction``.

The catalogs here are synthetic: the exposure is the anchor moved by a known
on-sky shift, so the sign of the answer is fixed by construction.
"""
import importlib.util
import os

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SOLVER = os.path.join(REPO, 'scripts', 'analysis', 'solve_filter_frame_offsets.py')

RA0, DEC0 = 161.03, -59.75
FIELD_ARCSEC = 60.0
N_STARS = 1500


def _solver():
    spec = importlib.util.spec_from_file_location('solve_filter_frame_offsets', SOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _anchor(rng):
    half = FIELD_ARCSEC / 2.0 / 3600.0
    dec = DEC0 + rng.uniform(-half, half, N_STARS)
    ra = RA0 + rng.uniform(-half, half, N_STARS) / np.cos(np.radians(DEC0))
    return SkyCoord(ra * u.deg, dec * u.deg)


def _exposure_catalog(path, anchor, dra_mas, ddec_mas, rng, jitter_mas=1.0):
    """The anchor stars moved by (dra_mas*, ddec_mas) on the sky plus jitter,
    written with the columns ``exposure_residual`` reads."""
    n = len(anchor)
    jr = rng.normal(0.0, jitter_mas, n)
    jd = rng.normal(0.0, jitter_mas, n)
    cosd = np.cos(anchor.dec.radian)
    ra = anchor.ra.deg + (dra_mas + jr) / 3.6e6 / cosd
    dec = anchor.dec.deg + (ddec_mas + jd) / 3.6e6
    Table({'skycoord_centroid': SkyCoord(ra * u.deg, dec * u.deg),
           'qfit': np.full(n, 0.01),
           'flux_fit': np.full(n, 1000.0),
           'flux_err': np.full(n, 1.0)}).write(path, overwrite=True)


@pytest.mark.parametrize('dra_mas, ddec_mas', [(+9.5, -0.5), (-4.0, +7.0)])
def test_exposure_residual_is_exposure_minus_anchor(tmp_path, dra_mas, ddec_mas):
    rng = np.random.default_rng(1130)
    solver = _solver()
    anchor = _anchor(rng)
    path = str(tmp_path / 'f162m_nrca1_visit001_vgroup14101_exp00001_m6_daophot_basic.fits')
    _exposure_catalog(path, anchor, dra_mas, ddec_mas, rng)
    r = solver.exposure_residual(path, anchor)
    assert r is not None
    dx, dy, n = r
    assert n >= solver.MIN_PAIRS
    assert dx == pytest.approx(dra_mas, abs=0.3)
    assert dy == pytest.approx(ddec_mas, abs=0.3)


def test_written_correction_opposes_the_shift(tmp_path):
    """The table ``--write`` stores is the correction: rotated back to the
    sky it must point against the shift the exposure was given."""
    from jwst_gc_pipeline.reduction import filter_frame_correction as ffc
    rng = np.random.default_rng(296)
    solver = _solver()
    anchor = _anchor(rng)
    roll = 141.0
    shifts = {'NRCA1': (+9.5, -0.5), 'NRCA2': (+1.3, +8.3),
              'NRCA3': (-1.3, -8.4), 'NRCA4': (-9.5, +0.6)}
    residuals = {}
    for det, (sx, sy) in shifts.items():
        path = str(tmp_path / f'f162m_{det.lower()}_visit001_vgroup14101_exp00001_m6_daophot_basic.fits')
        _exposure_catalog(path, anchor, sx, sy, rng)
        dx, dy, _n = solver.exposure_residual(path, anchor)
        residuals[det] = (dx, dy)
    corr = solver.sky_residual_to_instrument_correction(residuals, roll)
    sky = ffc.instrument_to_sky(corr, roll)
    want = ffc._remove_module_means(shifts)
    for det, (sx, sy) in want.items():
        assert sky[det][0] == pytest.approx(-sx, abs=0.3), det
        assert sky[det][1] == pytest.approx(-sy, abs=0.3), det
