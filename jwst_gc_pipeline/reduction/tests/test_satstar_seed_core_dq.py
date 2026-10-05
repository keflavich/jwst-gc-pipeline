"""Outlier and bad-pixel NaN variance must not steer the satstar seed (#1098).

``_refine_coms_by_data`` re-centres a seed on the largest NaN-``VAR_POISSON``
sub-cluster of its SATURATED component, taken as the star's genuinely saturated
core (added for the 2526 cloud-c filament).  Outlier detection and the static
bad-pixel mask leave NaN variance too.  On the wd2 dolphot benchmark:

* F187N nrcb1 (57,756), dolphot 14.24: the star's own core has finite variance;
  two OUTLIER pairs (DQ 19) on the component's edge tie for largest and the
  first moved the seed 2.4 px.  With KEEP_FINITE on, the fit ended on the
  1.5 FWHM position bound, 0.46 mag faint.
* F250M nrcalong (1603,803), dolphot 13.18: a WARM / HOT / TELEGRAPH clump
  (DQ 4099, 1067011, 4247555) moved the seed 3.6 px and the fit ended on the
  bound, 0.95 mag faint.

``seed_saturation_core`` (``SATSTAR_SEED_CORE_DQ``, default on) leaves those
pixels out of the core that ``get_saturated_stars`` hands the refinement.
"""
import inspect
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

ON = {}
OFF = {'SATSTAR_SEED_CORE_DQ': '0'}

FRAME = 'jw02221001001_07101_00001_nrca2_destreak_o001_crf.fits'
RAMP = 'jw02221001001_07101_00001_nrca2_ramp.fits'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)


def _disk(shape, cy, cx, r):
    yy, xx = np.indices(shape)
    return np.hypot(yy - cy, xx - cx) <= r


def _seed(sat, unrecoverable, dq, env):
    """The refined seed of the one SATURATED component, as get_saturated_stars
    computes it."""
    sources = sat.astype(int)
    com = tuple(float(v) for v in np.argwhere(sat).mean(axis=0))
    core = ssf.seed_saturation_core(unrecoverable, dq, env=env)
    (cy, cx), = ssf._refine_coms_by_data([com], np.zeros(sat.shape), sources,
                                         unrecoverable=core)
    return cy, cx


def _f187n_like():
    """A weakly saturated star: a 45-pixel SATURATED disk at (20, 20) whose own
    core has finite variance, with two 2-pixel OUTLIER clumps (DQ 19,
    NaN variance) on its edge."""
    sat = _disk((40, 40), 20, 20, 3.8)
    dq = np.where(sat, 3, 0)
    unrec = np.zeros(sat.shape, bool)
    for y, x in [(17, 20), (17, 21), (23, 19), (23, 20)]:
        unrec[y, x] = True
        dq[y, x] = 19
    return sat, unrec, dq


def test_outlier_clumps_no_longer_move_the_seed():
    sat, unrec, dq = _f187n_like()
    assert int(sat.sum()) == 45
    cy, cx = _seed(sat, unrec, dq, OFF)
    assert np.hypot(cy - 20, cx - 20) > 2       # old: onto the first clump
    assert (cy, cx) == pytest.approx((17.0, 20.5))
    cy, cx = _seed(sat, unrec, dq, ON)
    assert np.hypot(cy - 20, cx - 20) < 0.5     # eroded core: on the star


@pytest.mark.parametrize('dq_clump', [4099, 1067011, 4247555])
def test_bad_pixel_clump_no_longer_moves_the_seed(dq_clump):
    """The three F250M nrcalong DQ values (WARM / RC / HOT / NO_LIN_CORR /
    TELEGRAPH / UNRELIABLE_BIAS with SATURATED | DO_NOT_USE)."""
    sat = _disk((40, 40), 20, 20, 3.8)
    dq = np.where(sat, 3, 0)
    unrec = np.zeros(sat.shape, bool)
    for y, x in [(22, 17), (22, 18), (23, 18)]:
        unrec[y, x] = True
        dq[y, x] = dq_clump
    assert np.hypot(*np.subtract(_seed(sat, unrec, dq, OFF), (20, 20))) > 2.5
    assert np.hypot(*np.subtract(_seed(sat, unrec, dq, ON), (20, 20))) < 0.5


def _cloud_c_like(hot_in_core=False):
    """A 3x3 genuinely saturated core (DQ 3, NaN variance) at (10, 10) inside a
    large SATURATED component of finite emission spanning 2..37."""
    sat = np.zeros((40, 40), bool)
    sat[2:38, 2:38] = True
    dq = np.where(sat, 3, 0)
    unrec = np.zeros(sat.shape, bool)
    unrec[9:12, 9:12] = True
    if hot_in_core:
        dq[10, 11] = 2051          # HOT | SATURATED | DO_NOT_USE
    return sat, unrec, dq


@pytest.mark.parametrize('env', [ON, OFF])
def test_a_genuine_core_still_wins(env):
    """The cloud-c case the refinement was written for is unchanged."""
    sat, unrec, dq = _cloud_c_like()
    assert _seed(sat, unrec, dq, env) == pytest.approx((10.0, 10.0))


def test_a_hot_pixel_inside_a_genuine_core_keeps_the_seed_on_it():
    sat, unrec, dq = _cloud_c_like(hot_in_core=True)
    cy, cx = _seed(sat, unrec, dq, ON)
    assert np.hypot(cy - 10, cx - 10) < 0.2
    assert np.hypot(cy - 20, cx - 20) > 10      # not the emission blob


@pytest.mark.parametrize('dq, kept', [
    (3, True),                  # DO_NOT_USE | SATURATED
    (7, True),                  # + JUMP_DET
    (3 | 65536, True),          # + NONLINEAR
    (19, False),                # + OUTLIER
    (3 | 1024, False),          # + DEAD
    (3 | 2048, False),          # + HOT
    (3 | 4096, False),          # + WARM
    (3 | 16384, False),         # + RC
    (3 | 32768, False),         # + TELEGRAPH
    (3 | 262144, False),        # + NO_FLAT_FIELD
    (3 | 1048576, False),       # + NO_LIN_CORR
    (3 | 2 ** 31, False),       # + REFERENCE_PIXEL
])
def test_which_dq_bits_leave_the_core(dq, kept):
    unrec = np.ones((1, 1), bool)
    dqa = np.full((1, 1), dq, dtype=np.uint32)
    assert bool(ssf.seed_saturation_core(unrec, dqa, env=ON)[0, 0]) is kept
    assert bool(ssf.seed_saturation_core(unrec, dqa, env=OFF)[0, 0])


def test_no_dq_leaves_the_core_unchanged():
    unrec = np.array([[True, False]])
    assert ssf.seed_saturation_core(unrec, None, env=ON) is unrec


def test_get_saturated_stars_hands_the_filtered_core_to_both_seed_paths():
    """The refined seed and the size-gated lock both read the filtered core;
    the flag image keeps every NaN-variance pixel."""
    src = inspect.getsource(ssf.get_saturated_stars)
    assert src.count('unrecoverable=_seed_core') == 2
    assert 'unrecoverable=_unrecoverable' not in src
    assert 'flag_img[saturated & _unrecoverable] |= 2' in src


# --------------------------------------------------------------------------
# the cache key
# --------------------------------------------------------------------------

def _frame(tmp_path, with_ramp):
    fn = tmp_path / FRAME
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
                      {'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCA2',
                       'FILTER': 'F182M'})),
                  fits.ImageHDU(data=np.zeros((8, 8), 'float32'), name='SCI'),
                  fits.ImageHDU(data=np.zeros((8, 8), 'int32'), name='DQ')]
                 ).writeto(fn, overwrite=True)
    if with_ramp:
        fits.PrimaryHDU().writeto(tmp_path / RAMP, overwrite=True)
    return str(fn)


def test_signature_carries_sq_with_or_without_a_ramp(tmp_path, monkeypatch):
    (tmp_path / 'noramp').mkdir()
    fn0 = _frame(tmp_path / 'noramp', with_ramp=False)
    fn1 = _frame(tmp_path, with_ramp=True)
    assert ssf.satstar_fit_switch_signature(fn0) == 'sq'
    assert ssf.satstar_fit_switch_signature(fn1) == 'zfg1.3b_sq'
    monkeypatch.setenv('SATSTAR_SEED_CORE_DQ', '0')
    assert ssf.satstar_fit_switch_signature(fn0) == ''
    assert ssf.satstar_fit_switch_signature(fn1) == 'zfg1.3b'


def test_a_frame_without_a_ramp_is_refit_once(tmp_path, monkeypatch):
    """An older catalog (no SATFITSW) was seeded without the filter, so it is
    refit once, then reused."""
    fn = _frame(tmp_path, with_ramp=False)
    t = Table({'flux': [1.0, 2.0]})
    t.meta['SATRECOV'] = 'off'
    t.write(tmp_path / FRAME.replace('.fits', '_satstar_catalog.fits'))
    calls = {'n': 0}

    def _stub(filename, overwrite=True, file_suffix='', recovery_signature=None,
              fit_switch_signature=None, **kw):
        calls['n'] += 1
        out = Table({'flux': [1.0, 2.0]})
        out.meta['SATRECOV'] = recovery_signature
        out.meta['SATFITSW'] = fit_switch_signature
        out.write(filename.replace('.fits',
                                   f'{file_suffix}_satstar_catalog.fits'),
                  overwrite=True)
    monkeypatch.setattr(CL, 'remove_saturated_stars', _stub)
    for expected in (1, 1):
        CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                        recovery_signature='off')
        assert calls['n'] == expected
