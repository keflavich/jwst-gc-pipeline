"""``frame_wcs`` must transform through the GWCS, not the SIP approximation."""
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

from jwst_gc_pipeline.frame_wcs import (FrameWCS, MissingGwcsWarning,
                                        SlicedFrameWCSWarning, frame_wcs)
from jwst_gc_pipeline.reduction.fits_wcs_sync import sip_header_from_gwcs
from jwst_gc_pipeline.reduction.tests.test_fits_wcs_sync import _distorted_gwcs

SHAPE = (512, 512)


def _invertible_distorted_gwcs():
    """:func:`_distorted_gwcs` with an explicit analytic inverse.

    A real JWST GWCS ships an inverse distortion polynomial, so ``invert()``
    is analytic.  The synthetic fixture's ``Polynomial2D`` pair has none, and
    gwcs 1.0.3's ``numerical_inverse`` fallback crashes on a projection
    instance -- unrelated to what is under test here.  Fit the inverse pair on
    a grid (residual ~1e-6 px) and attach it, so the fixture behaves like a
    real frame.
    """
    gwcs = pytest.importorskip('gwcs')
    from gwcs import coordinate_frames as cf
    from astropy import coordinates as coord
    from astropy.modeling import models
    from astropy.modeling.fitting import LinearLSQFitter

    ny, nx = SHAPE
    amplitude = 3e-4
    xpoly = models.Polynomial2D(degree=5, c0_0=0.0, c1_0=1.0, c0_1=0.0,
                                c2_0=1e-6, c1_1=-2e-6,
                                c5_0=amplitude / nx ** 4)
    ypoly = models.Polynomial2D(degree=5, c0_0=0.0, c1_0=0.0, c0_1=1.0,
                                c0_2=1.5e-6, c1_1=1e-6,
                                c0_5=amplitude / ny ** 4)
    shift = models.Shift(-nx / 2.0) & models.Shift(-ny / 2.0)
    dist = shift | models.Mapping((0, 1, 0, 1)) | (xpoly & ypoly)

    yy, xx = np.mgrid[0:ny:16, 0:nx:16]
    xs, ys = xx.ravel().astype(float), yy.ravel().astype(float)
    xd, yd = dist(xs, ys)

    fitter = LinearLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        invx = fitter(models.Polynomial2D(degree=6), xd, yd, xs)
        invy = fitter(models.Polynomial2D(degree=6), xd, yd, ys)
    dist.inverse = models.Mapping((0, 1, 0, 1)) | (invx & invy)
    assert np.max(np.abs(np.asarray(dist.inverse(xd, yd)) -
                         np.asarray([xs, ys]))) < 1e-3

    from astropy import units as u
    scale = models.Scale(0.031 / 3600) & models.Scale(0.031 / 3600)
    tan = models.Pix2Sky_TAN()
    n2c = models.RotateNative2Celestial(266.5, -28.8, 180.0)
    detector = cf.Frame2D(name='detector', axes_order=(0, 1),
                          unit=(u.pix, u.pix))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
                            unit=(u.deg, u.deg))
    g = gwcs.WCS([(detector, dist | scale | tan | n2c), (sky, None)])
    g.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))
    return g


@pytest.fixture
def frame():
    """A GWCS plus a DELIBERATELY loose SIP header of it, as FrameWCS would hold."""
    g = _invertible_distorted_gwcs()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loose = fits.Header(g.to_fits_sip())     # gwcs's 0.25 px default
    from astropy import wcs as awcs
    return g, FrameWCS(g, awcs.WCS(loose, relax=True), filename='synthetic.fits')


def _xy():
    x = np.array([5.0, 128.0, 256.0, 400.0, 505.0])
    y = np.array([7.0, 200.0, 256.0, 300.0, 500.0])
    return x, y


def test_transforms_use_the_gwcs_not_the_sip_header(frame):
    """The whole point. If this ever passes trivially, the SIP header is not loose."""
    g, ww = frame
    x, y = _xy()
    ra_g, dec_g = g(x, y)

    ra_f, dec_f = ww.all_pix2world(x, y, 0)
    assert np.allclose(ra_f, ra_g) and np.allclose(dec_f, dec_g)

    ra_s, dec_s = ww.fits_wcs.all_pix2world(x, y, 0)
    sep = np.hypot((ra_s - ra_g) * np.cos(np.radians(dec_g)), dec_s - dec_g) * 3.6e6
    assert sep.max() > 0.5, ("the SIP fixture is not loose enough to prove "
                             "anything; max disagreement %g mas" % sep.max())

    sc = ww.pixel_to_world(x, y)
    assert np.allclose(sc.ra.deg, ra_g) and np.allclose(sc.dec.deg, dec_g)


def test_inverse_is_exact_and_returns_nan_off_footprint(frame):
    """Off-footprint, SIP either raises NoConvergence (the #187 m8 abort) or --
    with quiet=True -- returns finite garbage silently. The GWCS returns NaN,
    which the callers' in-bounds tests already drop."""
    g, ww = frame
    x, y = _xy()
    ra, dec = g(x, y)

    xb, yb = ww.all_world2pix(ra, dec, 0)
    assert np.max(np.abs(xb - x)) < 1e-3 and np.max(np.abs(yb - y)) < 1e-3

    # far off-footprint: NaN, never an exception
    far_x, far_y = ww.all_world2pix(np.array([10.0]), np.array([40.0]), 0)
    assert np.isnan(far_x).all() and np.isnan(far_y).all()


def test_astropy_calling_conventions(frame):
    """Call sites pass (x, y, origin), ((N,2), origin), and origin=1."""
    g, ww = frame
    x, y = _xy()
    ra, dec = g(x, y)

    arr = ww.all_pix2world(np.column_stack([x, y]), 0)
    assert arr.shape == (len(x), 2)
    assert np.allclose(arr[:, 0], ra)

    back = ww.all_world2pix(np.column_stack([ra, dec]), 0)
    assert back.shape == (len(x), 2)
    assert np.allclose(back[:, 0], x, atol=1e-3)

    x1, y1 = ww.all_pix2world(x + 1, y + 1, 1)     # 1-based origin
    assert np.allclose(x1, ra)

    xk, yk = ww.all_world2pix(ra, dec, origin=0)
    assert np.allclose(xk, x, atol=1e-3)


def test_sip_only_inverse_kwargs_are_accepted_and_ignored(frame):
    """`quiet`/`tolerance`/`maxiter` are astropy SIP-solver knobs; an exact
    inverse has no use for them but call sites still pass them."""
    _, ww = frame
    x, y = _xy()
    ra, dec = ww.all_pix2world(x, y, 0)
    a = ww.all_world2pix(ra, dec, 0, quiet=True, tolerance=1e-4, maxiter=20)
    b = ww.all_world2pix(ra, dec, 0)
    assert np.allclose(a[0], b[0], equal_nan=True)


def test_world_to_pixel_skycoord(frame):
    g, ww = frame
    x, y = _xy()
    sc = SkyCoord(*g(x, y), unit=u.deg)
    xb, yb = ww.world_to_pixel(sc)
    assert np.max(np.abs(xb - x)) < 1e-3


def test_non_transform_attributes_delegate_to_the_fits_wcs(frame):
    """`.wcs.crval`, pixel scales, to_header must keep working."""
    _, ww = frame
    assert len(ww.wcs.crval) == 2
    assert ww.proj_plane_pixel_area() > 0
    assert 'CTYPE1' in ww.to_header(relax=True)


def test_footprint_comes_from_the_gwcs_like_the_transforms(frame):
    """A FrameWCS whose transforms and whose footprint came from DIFFERENT
    representations would disagree by the SIP residual -- harmless for
    footprint gating, but it looks exactly like a bug to whoever compares a
    footprint corner against a transformed corner."""
    g, ww = frame
    ny, nx = SHAPE
    corners = ww.calc_footprint(axes=(nx, ny))
    assert corners.shape == (4, 2)

    # every footprint corner must equal pixel_to_world of that same corner
    xs = np.array([0.0, 0.0, nx - 1.0, nx - 1.0])
    ys = np.array([0.0, ny - 1.0, ny - 1.0, 0.0])
    ra_t, dec_t = g(xs, ys)
    sep = np.hypot((corners[:, 0] - ra_t) * np.cos(np.radians(dec_t)),
                   corners[:, 1] - dec_t) * 3.6e6
    assert sep.max() < 1e-6, sep

    # and it must actually differ from the SIP footprint, or the test is vacuous
    sip_corners = ww.fits_wcs.calc_footprint(axes=(nx, ny))
    dsip = np.hypot((corners[:, 0] - sip_corners[:, 0]) * np.cos(np.radians(dec_t)),
                    corners[:, 1] - sip_corners[:, 1]) * 3.6e6
    assert dsip.max() > 0.1, dsip


def test_slicing_warns_before_degrading_to_sip(frame):
    """Slicing silently returned a SIP WCS; 'display only' was enforced by
    convention rather than by the code."""
    from astropy import wcs as awcs
    _, ww = frame
    with pytest.warns(SlicedFrameWCSWarning):
        sliced = ww[0:10, 0:10]
    assert isinstance(sliced, awcs.WCS) and not isinstance(sliced, FrameWCS)


def test_wcs_provenance_cards_distinguish_gwcs_from_sip(tmp_path, monkeypatch):
    """A catalog must be self-identifying as GWCS- or SIP-built; before this
    change the only discriminator was the build date."""
    import jwst_gc_pipeline.frame_wcs as fw

    monkeypatch.setattr(fw, '_RESOLUTION_TALLY', {'gwcs': 0, 'sip': 0})
    assert dict((k, v) for k, v, _ in fw.wcs_provenance_cards())['WCSSRC'] == 'NONE'

    fw._RESOLUTION_TALLY['gwcs'] = 12
    cards = dict((k, v) for k, v, _ in fw.wcs_provenance_cards())
    assert cards['WCSSRC'] == 'GWCS' and cards['WCSNGW'] == 12 and cards['WCSNSIP'] == 0

    fw._RESOLUTION_TALLY['sip'] = 3
    cards = dict((k, v) for k, v, _ in fw.wcs_provenance_cards())
    assert cards['WCSSRC'] == 'MIXED' and cards['WCSNSIP'] == 3

    fw._RESOLUTION_TALLY['gwcs'] = 0
    assert dict((k, v) for k, v, _ in fw.wcs_provenance_cards())['WCSSRC'] == 'FITS-SIP'


def test_missing_gwcs_falls_back_with_a_warning(tmp_path):
    g = _distorted_gwcs()
    hdr = sip_header_from_gwcs(g)
    path = tmp_path / 'no_gwcs.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros(SHAPE, dtype='f4'), header=hdr,
                                name='SCI')]).writeto(path)

    from astropy import wcs as awcs
    with pytest.warns(MissingGwcsWarning):
        ww = frame_wcs(str(path))
    assert isinstance(ww, awcs.WCS) and not isinstance(ww, FrameWCS)

    with pytest.raises(ValueError, match='no GWCS'):
        frame_wcs(str(path), require_gwcs=True)


def test_survives_pickle_and_deepcopy_as_a_framewcs(frame):
    """The pipeline hands frame WCSes to multiprocessing / dask workers.

    Without explicit state handling, ``__getattr__`` answered pickle's
    ``__reduce_ex__`` lookup from the wrapped FITS WCS: pickling recursed
    forever, and ``deepcopy`` silently returned a plain ``astropy.wcs.WCS`` --
    so every worker would quietly fall back to SIP while the parent used the
    GWCS. Both must round-trip as a FrameWCS that still transforms via the GWCS.
    """
    import copy
    import pickle

    g, ww = frame
    x, y = _xy()
    ra_g, dec_g = g(x, y)

    for clone in (pickle.loads(pickle.dumps(ww)), copy.deepcopy(ww), copy.copy(ww)):
        assert isinstance(clone, FrameWCS), type(clone)
        ra, dec = clone.all_pix2world(x, y, 0)
        assert np.allclose(ra, ra_g) and np.allclose(dec, dec_g)


def test_frame_wcs_passes_through_an_existing_wcs(frame):
    """Call sites may hand in either a path or an already-built WCS."""
    _, ww = frame
    assert frame_wcs(ww) is ww
    assert frame_wcs(ww.fits_wcs) is ww.fits_wcs


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
