"""Regression: the RECOVER tier of the extended-emission vetting.

Recovers real, neighbour-blended stars that the strict qfit<=qfit_max cut deletes
(Arches F212N: ~24k real stars at qfit 0.2-0.5, S/N ~15) WITHOUT re-admitting
diffraction spikes / emission knots and WITHOUT breaking the "cataloged =>
subtracted" invariant (recovered stars enter star_like => vetted => modelled =>
subtracted).

Gates (ALL required):
  (a) qfit in (qfit_max, qfit_recover_max]
  (b) S/N >= local_snr_min
  (c) NOT within recover_satstar_guard_arcsec of a catalog saturated star
  (d) SLOPED PROMINENCE gate: log10(prominence) >= intercept + slope*qfit on the
      data_i2d -- the rise above the local annulus, with the floor rising with
      qfit.  This is THE star-vs-emission discriminator: qfit 0.2-0.5 means
      crowded real star in a star field but emission knot on bright nebulosity
      (pillar_head: real star prominence 7.4 vs ridge knots <=2.4).

DEFAULT (qfit_recover_max<=qfit_max) is a NO-OP -> byte-identical to before.
The (a)-(c) gates are unit-tested with recover_prom_gate=False (prominence gate
disabled, no data_i2d needed); the sloped prominence gate (d) is tested with a
synthetic i2d.
"""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import wcs
import astropy.units as u

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission


def _mk(qfit, flux, flux_err, is_sat=None, ra0=266.5, dec0=-28.8, dra=0.001,
        nmatch=None, std_ra=None, std_dec=None):
    n = len(qfit)
    ras = ra0 + np.arange(n) * dra
    t = Table({
        'skycoord': SkyCoord(ras * u.deg, np.full(n, dec0) * u.deg),
        'qfit': np.asarray(qfit, float),
        'flux': np.asarray(flux, float),
        'flux_err': np.asarray(flux_err, float),
        'flags': np.zeros(n, int),
        'local_bkg': np.zeros(n),
    })
    if is_sat is not None:
        t['is_saturated'] = np.asarray(is_sat, bool)
    if nmatch is not None:
        t['nmatch'] = np.asarray(nmatch, int)
    if std_ra is not None:
        t['std_ra'] = np.asarray(std_ra, float)
        t['std_dec'] = np.asarray(std_dec, float)
    return t


# a faint blended star: qfit 0.40 (fails qfit<=0.2 and the recover prominence
# gate), low snr, but detected in 6 exposures -> real by multi-frame confirmation.
def _faint_multiframe(nmatch=6, std=1e-6):
    return _mk(qfit=[0.40], flux=[60.0], flux_err=[12.0], nmatch=[nmatch],
               std_ra=[std], std_dec=[std])


def test_nmatch_confirm_default_off_drops_faint_multiframe():
    vet = _filter_extended_emission(_faint_multiframe(), qfit_max=0.2,
                                    local_snr_min=5.0, label='t')
    assert len(vet) == 0   # nmatch_confirm defaults to 0 -> keep is unchanged


def test_nmatch_confirm_admits_faint_multiframe_star():
    vet = _filter_extended_emission(_faint_multiframe(), qfit_max=0.2,
                                    nmatch_confirm=3, local_snr_min=5.0, label='t')
    assert len(vet) == 1


def test_nmatch_confirm_respects_qfit_ceiling():
    t = _mk(qfit=[0.70], flux=[60.0], flux_err=[12.0], nmatch=[6],
            std_ra=[1e-6], std_dec=[1e-6])            # qfit 0.70 > 0.6 ceiling
    vet = _filter_extended_emission(t, qfit_max=0.2, nmatch_confirm=3,
                                    nmatch_confirm_qfit_max=0.6, local_snr_min=5.0, label='t')
    assert len(vet) == 0


def test_nmatch_confirm_requires_enough_frames():
    t = _faint_multiframe(nmatch=2)                    # only 2 frames < 3
    vet = _filter_extended_emission(t, qfit_max=0.2, nmatch_confirm=3,
                                    local_snr_min=5.0, label='t')
    assert len(vet) == 0


def test_nmatch_confirm_position_guard_rejects_wandering():
    # nmatch=6 but centroid scatter ~100 mas (2.8e-5 deg) -> emission-like, rejected
    t = _mk(qfit=[0.40], flux=[60.0], flux_err=[12.0], nmatch=[6],
            std_ra=[2.8e-5], std_dec=[2.8e-5])
    vet = _filter_extended_emission(t, qfit_max=0.2, nmatch_confirm=3,
                                    nmatch_confirm_maxpos_mas=20.0, local_snr_min=5.0, label='t')
    assert len(vet) == 0
    # same source with tight ~3 mas scatter -> kept
    t2 = _mk(qfit=[0.40], flux=[60.0], flux_err=[12.0], nmatch=[6],
             std_ra=[8e-7], std_dec=[8e-7])
    vet2 = _filter_extended_emission(t2, qfit_max=0.2, nmatch_confirm=3,
                                     nmatch_confirm_maxpos_mas=20.0, local_snr_min=5.0, label='t')
    assert len(vet2) == 1


BLENDED_REAL = dict(qfit=[0.30], flux=[150.0], flux_err=[10.0])  # S/N 15


# ----- gates (a)-(c): prominence disabled (=0), no data_i2d -----
def test_recover_default_is_noop_drops_blended_star():
    vet = _filter_extended_emission(_mk(**BLENDED_REAL), qfit_max=0.2,
                                    local_snr_min=5.0, label='t')
    assert len(vet) == 0


def test_recover_admits_blended_real_star():
    vet = _filter_extended_emission(_mk(**BLENDED_REAL), qfit_max=0.2,
                                    qfit_recover_max=0.5, recover_prom_gate=False,
                                    local_snr_min=5.0, label='t')
    assert len(vet) == 1


def test_recover_rejects_emission_knot_above_ceiling():
    t = _mk(qfit=[0.60], flux=[3000.0], flux_err=[100.0])  # S/N 30, qfit>0.5
    vet = _filter_extended_emission(t, qfit_max=0.2, qfit_recover_max=0.5,
                                    recover_prom_gate=False, local_snr_min=5.0, label='t')
    assert len(vet) == 0


def test_recover_requires_snr_floor():
    t = _mk(qfit=[0.30], flux=[40.0], flux_err=[20.0])  # S/N 2
    vet = _filter_extended_emission(t, qfit_max=0.2, qfit_recover_max=0.5,
                                    recover_prom_gate=False, local_snr_min=5.0, label='t')
    assert len(vet) == 0


def test_recover_rejects_near_satstar_spike():
    t = _mk(qfit=[0.01, 0.30], flux=[1e6, 150.0], flux_err=[1e3, 10.0],
            is_sat=[True, False], dra=1.0 / 3600.0)
    vet = _filter_extended_emission(t, qfit_max=0.2, qfit_recover_max=0.5,
                                    recover_prom_gate=False, local_snr_min=5.0,
                                    recover_satstar_guard_arcsec=2.0, label='t')
    assert len(vet) == 1   # satstar kept (qfit-confident); spike source rejected


def test_recover_admits_same_source_far_from_satstar():
    t = _mk(qfit=[0.01, 0.30], flux=[1e6, 150.0], flux_err=[1e3, 10.0],
            is_sat=[True, False], dra=5.0 / 3600.0)
    vet = _filter_extended_emission(t, qfit_max=0.2, qfit_recover_max=0.5,
                                    recover_prom_gate=False, local_snr_min=5.0,
                                    recover_satstar_guard_arcsec=2.0, label='t')
    assert len(vet) == 2


# ----- gate (d): prominence separates real star from emission ridge -----
def _i2d_with(ra, dec, kind):
    """Build a small data_i2d + WCS centered on (ra,dec).
    'star'  = sharp PSF-like peak on a flat ~0 background -> HIGH prominence.
    'emission' = uniformly bright field (a broad emission region) with NO distinct
    core at the source -- the daofind peak is just emission, so the core barely
    rises above the equally-bright annulus -> LOW prominence."""
    ny = nx = 61
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [31, 31]; w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    yo, xo = np.mgrid[0:ny, 0:nx]; rr = np.hypot(xo - 30, yo - 30)
    rng = np.random.default_rng(0)
    if kind == 'star':
        img = 100 * np.exp(-rr**2 / (2 * 1.3**2)) + rng.normal(0, 0.5, (ny, nx))
    else:  # broad bright emission, no peak at center -> core ~ annulus -> low prom
        img = 80.0 + rng.normal(0, 0.5, (ny, nx))
    return img, w


def test_prominence_gate_admits_real_star_on_flat_bg():
    ra, dec = 266.5, -28.8
    t = _mk(qfit=[0.30], flux=[150.0], flux_err=[10.0], ra0=ra, dec0=dec)
    img, w = _i2d_with(ra, dec, 'star')
    vet = _filter_extended_emission(t, data_i2d_image=img, ww_i2d=w,
                                    qfit_max=0.2, qfit_recover_max=0.5,
                                    local_snr_min=5.0,
                                    label='t')
    assert len(vet) == 1, "prominent point source must be recovered"


def test_prominence_and_peaksb_columns_persisted():
    # prominence + peak_sb are written to the catalog for ALL sources when a
    # data_i2d is supplied (universally-useful quality metrics), independent of
    # the recover tier.
    ra, dec = 266.5, -28.8
    t = _mk(qfit=[0.05], flux=[150.0], flux_err=[10.0], ra0=ra, dec0=dec)  # confident
    img, w = _i2d_with(ra, dec, 'star')
    vet = _filter_extended_emission(t, data_i2d_image=img, ww_i2d=w,
                                    qfit_max=0.2, local_snr_min=5.0, label='t')
    assert 'prominence' in vet.colnames and 'peak_sb' in vet.colnames
    assert np.isfinite(vet['prominence'][0]) and vet['prominence'][0] > 5
    assert np.isfinite(vet['peak_sb'][0])


def test_prominence_gate_rejects_emission_ridge_knot():
    ra, dec = 266.5, -28.8
    t = _mk(qfit=[0.30], flux=[150.0], flux_err=[10.0], ra0=ra, dec0=dec)
    img, w = _i2d_with(ra, dec, 'ridge')
    vet = _filter_extended_emission(t, data_i2d_image=img, ww_i2d=w,
                                    qfit_max=0.2, qfit_recover_max=0.5,
                                    local_snr_min=5.0,
                                    label='t')
    assert len(vet) == 0, "low-prominence emission ridge knot must NOT be recovered"
