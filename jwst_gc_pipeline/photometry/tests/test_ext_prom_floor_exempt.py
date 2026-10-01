"""Tight, bright fits are exempt from the extended-emission prominence floor.

On a nebular field the 4-10 px annulus MAD that normalizes prominence is set
by emission structure, so a real bright star can read prominence 2-3 and the
hard floor (3.0) deletes it.  With ``ext_prom_exempt_qfit`` > 0 a source with
qfit <= ext_prom_exempt_qfit AND merged S/N >= ext_prom_exempt_snr passes at
prominence >= ext_prom_exempt_prom_min.
"""
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS


def _run(exempt_qfit):
    """0: prominent star; 1: prom 2-3, tight + bright; 2: prom 2-3, qfit 0.3;
    3: prom 2-3, merged S/N 10; 4: prom < 2, tight + bright."""
    ny = nx = 200
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 1.0, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    pos = [(40, 40), (80, 40), (120, 40), (160, 40), (40, 120)]
    for (x, y), amp in zip(pos, [12.0, 2.0, 2.0, 2.0, 0.6]):
        data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.0)
    xy = np.array(pos, float)
    t = Table({'id': np.arange(5), 'skycoord': w.pixel_to_world(xy[:, 0], xy[:, 1]),
               'qfit': [0.15, 0.15, 0.3, 0.15, 0.15],
               'flux': np.full(5, 300.0), 'flux_err': np.full(5, 30.0),
               'flux_err_prop': [7.5, 7.5, 7.5, 30.0, 7.5],
               'flags': np.zeros(5), 'local_bkg': np.zeros(5),
               'group_size': np.ones(5)})
    out = _filter_extended_emission(t, data_i2d_image=data, ww_i2d=w,
                                    ext_prom_min=3.0, ext_prom_exempt_qfit=exempt_qfit,
                                    ext_prom_exempt_snr=30.0,
                                    ext_prom_exempt_prom_min=2.0,
                                    snr_floor_propagated=True, sky_clean_keep=False)
    return set(np.asarray(out['id']).tolist()), np.asarray(t['prominence'])


def test_tight_bright_low_prominence_star_kept():
    kept, prom = _run(0.0)
    assert prom[0] > 5
    assert np.all((prom[1:4] >= 2) & (prom[1:4] < 3)) and prom[4] < 2
    assert kept == {0}                  # floor alone: only the prominent star
    kept, _ = _run(0.2)
    assert kept == {0, 1}               # 2: qfit 0.3, 3: S/N 10, 4: prom < 2


def test_exemption_inert_without_floor():
    # the exemption only relaxes the floor; with the floor off (star-dominated
    # fields) nothing changes
    ny = nx = 60
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    data = np.random.default_rng(0).normal(0.0, 1.0, (ny, nx))
    t = Table({'id': np.arange(1), 'skycoord': w.pixel_to_world([30.0], [30.0]),
               'qfit': [0.15], 'flux': [300.0], 'flux_err': [30.0],
               'flux_err_prop': [10.0], 'flags': [0], 'local_bkg': [0.0],
               'group_size': [1]})
    for q in (0.0, 0.2):
        out = _filter_extended_emission(t, data_i2d_image=data, ww_i2d=w, ext_prom_min=0.0,
                                        ext_prom_exempt_qfit=q, snr_floor_propagated=True,
                                        sky_clean_keep=False)
        assert len(out) == 1


def test_pipeline_defaults():
    assert MANUAL_DEFAULTS['manual_ext_prom_exempt_qfit'] == 0.2
    assert MANUAL_DEFAULTS['manual_ext_prom_exempt_snr'] == 30.0
    assert MANUAL_DEFAULTS['manual_ext_prom_exempt_prom_min'] == 2.0
