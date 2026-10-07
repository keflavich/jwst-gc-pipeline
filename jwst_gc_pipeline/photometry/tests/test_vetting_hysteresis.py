"""The previous-phase (hysteresis) keep of the merged-catalog vetting.

Each phase re-vets its merged catalog from scratch, so a star near a keep
threshold (the bright-isolated keep's qfit < 0.4 cap, qfit <= qfit_max,
flags == 1) can be vetted at phase K and dropped at K+1 after a small refit
change, leaving the whole star in the K+1 residual.  The keep re-admits a
source within ``hysteresis_match_arcsec`` of the previous phase's vetted
catalog while qfit < ``hysteresis_qfit_max`` and S/N >= ``hysteresis_snr_min``,
except within ``hysteresis_satstar_guard_arcsec`` of a saturated star.
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


def _with_satstar(sep_arcsec, masked=False):
    """Row 0 a saturated star, rows 1.. sources ``sep_arcsec`` east of it;
    every row passes the keep's qfit and S/N gates."""
    sep = np.atleast_1d(np.asarray(sep_arcsec, float))
    t = _mk(np.full(len(sep) + 1, 0.45), 30)
    sat = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    t['skycoord'] = SkyCoord([sat] + [sat.spherical_offsets_by(d * u.arcsec, 0 * u.arcsec)
                                      for d in sep])
    t['is_saturated'] = np.r_[True, np.zeros(len(sep), bool)]
    if masked:
        from astropy.table import MaskedColumn
        t['is_saturated'] = MaskedColumn(t['is_saturated'],
                                         mask=np.r_[False, np.ones(len(sep), bool)])
    return t


@pytest.mark.parametrize('masked', [False, True])
def test_satstar_guard_refuses_nearby_sources(masked, capsys):
    t = _with_satstar([0.3, 0.6, 1.0], masked=masked)
    prev = _prev_at(t)
    assert _kept(t, prev) == {0, 1, 2, 3}                       # no guard
    capsys.readouterr()
    # sources within 0.612" (4.5 FWHM at F405N) lose the keep.  The saturated
    # row is not guarded (and the model==catalog invariant keeps it anyway),
    # so the log counts the two neighbours only.
    assert _kept(t, prev, hysteresis_satstar_guard_arcsec=0.612) == {0, 3}
    assert '2 refused within 0.612"' in capsys.readouterr().out
    assert _kept(t, prev, hysteresis_satstar_guard_arcsec=0.2) == {0, 1, 2, 3}


def test_satstar_guard_without_saturated_rows_is_inert():
    t = _mk([0.45, 0.45], [30, 30])
    assert _kept(t, _prev_at(t), hysteresis_satstar_guard_arcsec=5.0) == {0, 1}
    t['is_saturated'] = np.zeros(2, bool)
    assert _kept(t, _prev_at(t), hysteresis_satstar_guard_arcsec=5.0) == {0, 1}


def test_hysteresis_radii_from_the_fwhm_table(tmp_path):
    from jwst_gc_pipeline.reduction.fwhm import fwhm_table_path
    packaged = Table.read(fwhm_table_path(None, 'NIRCAM'))
    fw = float(packaged[np.char.upper(np.asarray(packaged['Filter']).astype(str))
                        == 'F405N']['PSF FWHM (arcsec)'][0])
    # case-insensitive band match; packaged table when basepath has none
    for filt in ('F405N', 'f405n'):
        r, g = C._hysteresis_radii(filt, str(tmp_path), 4.5)
        assert r == pytest.approx(0.5 * fw)
        assert g == pytest.approx(4.5 * fw)
    assert C._hysteresis_radii('F405N', None, 0.0)[1] == 0.0
    # a band missing from the table turns the keep off instead of raising
    assert C._hysteresis_radii('F999X', None, 4.5) is None
    # the field tree's reduction/fwhm_table.ecsv takes precedence
    (tmp_path / 'reduction').mkdir()
    Table({'Filter': ['F405N'], 'PSF FWHM (arcsec)': [0.2]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv', format='ascii.ecsv')
    r, g = C._hysteresis_radii('F405N', str(tmp_path), 4.5)
    assert (r, g) == (pytest.approx(0.1), pytest.approx(0.9))


def test_pipeline_defaults_and_cli():
    assert MANUAL_DEFAULTS['manual_ext_hysteresis_qfit_max'] == 0.6
    assert MANUAL_DEFAULTS['manual_ext_hysteresis_snr_min'] == 10.0
    assert MANUAL_DEFAULTS['manual_ext_hysteresis_satstar_guard_fwhm'] == 4.5
    import inspect
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as CL
    src = inspect.getsource(CL)
    assert '--manual-ext-hysteresis-qfit-max' in src
    assert '--manual-ext-hysteresis-snr-min' in src
    assert '--manual-ext-hysteresis-satstar-guard-fwhm' in src


@pytest.mark.parametrize('phase,prev', [('m3', ('m2', False)), ('m4', ('m3', False)),
                                        ('m5', ('m4', False)), ('m6', ('m5', True)),
                                        ('m7', ('m6', True))])
def test_previous_vetted_phase_map(phase, prev):
    # the same previous-phase catalogs the m3..m7 seeds read (vetted_prev)
    assert C._PREV_VETTED_PHASE[phase] == prev
    assert 'm12' not in C._PREV_VETTED_PHASE
