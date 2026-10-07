"""The previous-phase (hysteresis) keep of the merged-catalog vetting.

Each phase re-vets its merged catalog from scratch, so a star near a keep
threshold (the bright-isolated keep's qfit < 0.4 cap, qfit <= qfit_max,
flags == 1) can be vetted at phase K and dropped at K+1 after a small refit
change, leaving the whole star in the K+1 residual.  The keep re-admits a
source within ``hysteresis_match_arcsec`` of the previous phase's vetted
catalog while qfit < ``hysteresis_qfit_max`` and S/N >= ``hysteresis_snr_min``.
"""
import numpy as np
import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

RA0, DEC0 = 266.5, -28.8


def _mk(qfit, snr, n=None):
    """Sources no other branch keeps: qfit above qfit_max, flags 0, in a
    group (no bright-isolated keep), no data image (no peak / prominence)."""
    qfit = np.atleast_1d(np.asarray(qfit, float))
    snr = np.broadcast_to(np.asarray(snr, float), qfit.shape)
    n = len(qfit)
    return Table({
        'id': np.arange(n),
        'skycoord': SkyCoord((RA0 + np.arange(n) * 0.001) * u.deg,
                             np.full(n, DEC0) * u.deg),
        'qfit': qfit,
        'flux': 10.0 * snr,
        'flux_err': np.full(n, 10.0),
        'flags': np.zeros(n, int),
        'group_size': np.full(n, 2),
        'local_bkg': np.zeros(n),
    })


def _kept(t, prev, **kw):
    kw.setdefault('hysteresis_qfit_max', 0.6)
    kw.setdefault('hysteresis_snr_min', 10.0)
    kw.setdefault('hysteresis_match_arcsec', 0.036)
    out = _filter_extended_emission(t, sky_clean_keep=False,
                                    prev_vetted_skycoord=prev, **kw)
    return set(np.asarray(out['id']).tolist())


def _prev_at(t, offset_arcsec=0.01):
    return SkyCoord(t['skycoord']).spherical_offsets_by(
        offset_arcsec * u.arcsec, 0 * u.arcsec)


def test_previously_vetted_star_is_kept():
    t = _mk([0.45, 0.45, 0.65, 0.45], [30, 30, 30, 5])
    prev = _prev_at(t)
    # without the keep nothing passes
    assert _kept(t, prev, hysteresis_qfit_max=0.0) == set()
    assert _kept(t, None) == set()
    # qfit 0.45 / S/N 30 kept; qfit 0.65 and S/N 5 are not
    assert _kept(t, prev) == {0, 1}


def test_only_matched_rows_are_kept():
    t = _mk([0.45, 0.45], [30, 30])
    prev = _prev_at(t[:1])                       # only row 0 was vetted
    assert _kept(t, prev) == {0}
    # previous-phase source 0.05" away, outside the 0.036" match radius
    assert _kept(t, _prev_at(t, 0.05)) == set()


def test_nan_coordinates_and_qfit_are_safe():
    t = _mk([0.45, np.nan], [30, 30])
    prev = SkyCoord([RA0, np.nan] * u.deg, [DEC0, np.nan] * u.deg)
    assert _kept(t, prev) == {0}


def test_overshoot_still_drops():
    t = _mk([0.45, 0.45], [30, 30])
    t['model_overshoot'] = [True, False]
    assert _kept(t, _prev_at(t)) == {1}


def test_miri_path_ignores_the_keep():
    t = _mk([0.45], [30])
    # min_prominence > 0 is the MIRI prominence-only path
    assert _kept(t, _prev_at(t), min_prominence=5.0) == set()


def test_pipeline_defaults_and_cli():
    assert MANUAL_DEFAULTS['manual_ext_hysteresis_qfit_max'] == 0.6
    assert MANUAL_DEFAULTS['manual_ext_hysteresis_snr_min'] == 10.0
    import inspect
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as CL
    src = inspect.getsource(CL)
    assert '--manual-ext-hysteresis-qfit-max' in src
    assert '--manual-ext-hysteresis-snr-min' in src


@pytest.mark.parametrize('phase,prev', [('m3', ('m2', False)), ('m4', ('m3', False)),
                                        ('m5', ('m4', False)), ('m6', ('m5', True)),
                                        ('m7', ('m6', True))])
def test_previous_vetted_phase_map(phase, prev):
    # the same previous-phase catalogs the m3..m7 seeds read (vetted_prev)
    assert C._PREV_VETTED_PHASE[phase] == prev
    assert 'm12' not in C._PREV_VETTED_PHASE
