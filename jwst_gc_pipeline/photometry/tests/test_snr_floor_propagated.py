"""The vetting S/N floor uses the uncertainty of the merged flux.

A merged catalog's flux is the mean over nmatch per-frame fits, while its
flux_err is the weighted mean of the per-frame errors (one frame's
uncertainty).  flux_err_prop = 1/sqrt(sum 1/sigma_i^2) is the uncertainty of
the merged flux, ~flux_err/sqrt(nmatch).  With ``snr_floor_propagated`` the
floor is applied to flux / flux_err_prop.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS


def _mk(flux_err_prop, n=3):
    """star_like (flags=1, qfit 0.5) sources with per-frame S/N 4."""
    t = Table({
        'id': np.arange(n),
        'skycoord': SkyCoord((266.5 + np.arange(n) * 0.001) * u.deg,
                             np.full(n, -28.8) * u.deg),
        'qfit': np.full(n, 0.5),
        'flux': np.full(n, 40.0),
        'flux_err': np.full(n, 10.0),
        'flags': np.ones(n, int),
        'local_bkg': np.zeros(n),
    })
    if flux_err_prop is not None:
        t['flux_err_prop'] = np.asarray(flux_err_prop, float)
    return t


def _kept(t, on):
    out = _filter_extended_emission(t, local_snr_min=5.0, snr_floor_propagated=on,
                                    sky_clean_keep=False)
    return set(np.asarray(out['id']).tolist())


def test_floor_on_merged_flux_uncertainty():
    # merged-flux S/N: 16 (16 frames), 8, 4
    t = _mk([2.5, 5.0, 10.0])
    assert _kept(t, False) == set()          # per-frame S/N 4 < 5 for all
    assert _kept(t, True) == {0, 1}


def test_missing_or_bad_flux_err_prop_keeps_per_frame_floor():
    t = _mk([np.nan, 0.0, 2.5])
    assert _kept(t, True) == {2}
    t = _mk(None)
    t['flux_err'] = [10.0, 5.0, 10.0]        # per-frame S/N 4, 8, 4
    assert _kept(t, True) == {1}


def test_qfit_confident_exemption_unchanged():
    t = _mk([10.0, 10.0, 10.0])
    t['qfit'] = [0.1, 0.5, 0.5]
    assert _kept(t, True) == {0}


def test_pipeline_default_on():
    assert MANUAL_DEFAULTS['manual_ext_snr_floor_propagated'] is True


def _sky_clean_case(flux_err):
    """A qfit 0.8 (blend-degraded) star on emission-free sky, which only the
    sky-clean tier can keep; flux_err_prop puts its merged S/N at 10."""
    from astropy.wcs import WCS
    rng = np.random.default_rng(3)
    ny = nx = 300
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.03 / 3600, 0.03 / 3600]
    data = rng.normal(0.0, 1.0, (ny, nx))
    data[:, :100] += 25.0                     # emission strip, x < 100
    yy, xx = np.mgrid[0:ny, 0:nx]
    data += 30.0 * np.exp(-((xx - 220) ** 2 + (yy - 150) ** 2) / (2 * 1.5 ** 2))
    t = Table({
        'id': [0],
        'skycoord': w.pixel_to_world(np.array([220.0]), np.array([150.0])),
        'qfit': [0.8], 'flags': [0], 'local_bkg': [0.0], 'group_size': [1],
        'is_saturated': [False],
        'flux': [20.0], 'flux_err': [float(flux_err)], 'flux_err_prop': [2.0],
    })
    out = _filter_extended_emission(t, data_i2d_image=data, ww_i2d=w, label='test',
                                    local_snr_min=5.0, snr_floor_propagated=True,
                                    sky_clean_keep=True, sky_clean_snr_min=3.0)
    return len(out)


def test_sky_clean_floor_stays_per_frame():
    # per-frame S/N 2 < 3: the tier must not admit it on the merged S/N of 10
    assert _sky_clean_case(flux_err=10.0) == 0
    # per-frame S/N 4 >= 3: the tier keeps it (the case discriminates)
    assert _sky_clean_case(flux_err=5.0) == 1


def test_bright_isolated_keep_stays_per_frame():
    # qfit 0.3 (above qfit_max 0.2, below qfit_high_keep_max 0.4), isolated,
    # flags 0: only the bright-isolated keep (S/N >= 20) can admit it.  Its
    # merged S/N is 40 in every row; the per-frame S/N decides.
    t = _mk([1.0, 1.0], n=2)
    t['flags'] = [0, 0]
    t['qfit'] = [0.3, 0.3]
    t['group_size'] = [1, 1]
    t['flux_err'] = [4.0, 1.6]               # per-frame S/N 10, 25
    t['flux_err_prop'] = [1.0, 1.0]
    out = _filter_extended_emission(t, local_snr_min=5.0, snr_floor_propagated=True,
                                    sky_clean_keep=False, snr_high_keep=20.0,
                                    qfit_high_keep_max=0.4)
    assert set(np.asarray(out['id']).tolist()) == {1}
