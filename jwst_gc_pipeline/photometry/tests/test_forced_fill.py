"""Unit test for m8 forced cross-band fill (forced_fill.forced_fill_band).

Deterministic: a mock PSF + WCS + injected sources, no pipeline plumbing.
Verifies that force-fitting at the reference position of a masked (phantom)
non-detection recovers the injected flux, calibrates to Jy / Vega from the
band's own detections, clears the mask above the SNR threshold, and leaves
already-detected rows untouched.
"""
import numpy as np
from types import SimpleNamespace
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import forced_fill as ff

SIG = 1.5


class _MockPSF:
    grid_xypos = [(0, 0)]
    oversampling = 1
    data = np.zeros((1, 5, 5))

    def evaluate(self, x, y, flux, x0, y0):
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * SIG ** 2))
        return flux * g / (2 * np.pi * SIG ** 2)


def _build():
    rng = np.random.default_rng(0)
    ww = WCS(naxis=2)
    ww.wcs.crpix = [50, 50]
    ww.wcs.crval = [266.4, -29.0]
    ww.wcs.cdelt = [-0.017 / 3600, 0.017 / 3600]
    ww.wcs.ctype = ['RA---TAN', 'DEC--TAN']

    NY = NX = 120
    cjy, zp, atrue = 3.0e-8, 1.5e-7, 500.0
    inj = [(35.0, 60.0), (80.0, 25.0)]
    image = rng.normal(0, 1.0, (NY, NX)).astype('float32')
    psf = _MockPSF()
    yy, xx = np.mgrid[0:NY, 0:NX]
    for px, py in inj:
        image += psf.evaluate(xx, yy, atrue, px, py)

    ndet = 25
    dflux = rng.uniform(200, 5000, ndet)
    dfjy = dflux * cjy
    dmagv = -2.5 * np.log10(dfjy / zp)
    dra, ddec = ww.all_pix2world(rng.uniform(5, 115, ndet), rng.uniform(5, 115, ndet), 0)
    pra, pdec = ww.all_pix2world([p[0] for p in inj], [p[1] for p in inj], 0)

    n = ndet + 2
    tbl = Table()
    tbl['skycoord_ref'] = SkyCoord(np.r_[dra, pra] * u.deg, np.r_[ddec, pdec] * u.deg)
    tbl['flux_f405n'] = np.r_[dflux, [np.nan, np.nan]]
    tbl['flux_jy_f405n'] = np.r_[dfjy, [np.nan, np.nan]]
    tbl['mag_ab_f405n'] = np.r_[-2.5 * np.log10(dfjy) + 8.90, [np.nan, np.nan]]
    tbl['mag_vega_f405n'] = np.r_[dmagv, [np.nan, np.nan]]
    tbl['emag_ab_f405n'] = np.r_[np.full(ndet, 0.02), [np.nan, np.nan]]
    tbl['mask_f405n'] = np.array([False] * ndet + [True, True])
    tbl['is_saturated_f405n'] = np.zeros(n, bool)
    tbl['near_saturated_f405n_f405n'] = np.zeros(n, bool)

    def prepare_frame(**kw):
        return SimpleNamespace(nan_replaced_data=image,
                               err=np.ones((NY, NX), 'float32'),
                               mask=np.zeros((NY, NX), bool), ww=ww, dao_psf_model=psf)

    return tbl, prepare_frame, ndet, cjy, zp, atrue


def test_forced_fill_recovers_phantom():
    tbl, prepare_frame, ndet, cjy, zp, atrue = _build()
    dflux0 = np.array(tbl['flux_f405n'][:ndet])

    nrec = ff.forced_fill_band(tbl, 'f405n', ['frame1'], prepare_frame=prepare_frame,
                               frame_arg_builder=lambda fn: {}, nsigma=3.0,
                               fit_shape=(5, 5), verbose=False)

    assert nrec == 2
    truemagv = -2.5 * np.log10(atrue * cjy / zp)
    for i in (ndet, ndet + 1):
        assert tbl['forced_filled_f405n'][i]
        assert abs(tbl['flux_f405n'][i] - atrue) < 25          # flux recovered
        assert abs(tbl['flux_jy_f405n'][i] - atrue * cjy) < 25 * cjy
        assert abs(tbl['mag_vega_f405n'][i] - truemagv) < 0.1   # Vega calibrated
        assert not tbl['mask_f405n'][i]                         # mask cleared
        assert tbl['forced_snr_f405n'][i] > 3
    # already-detected rows untouched
    assert np.allclose(tbl['flux_f405n'][:ndet], dflux0)


if __name__ == '__main__':
    test_forced_fill_recovers_phantom()
    print("PASS")
