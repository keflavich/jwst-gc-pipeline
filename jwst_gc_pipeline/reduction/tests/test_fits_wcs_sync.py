"""The FITS/SIP header written for a frame must reproduce its GWCS.

Regression for the 2026-07-29 finding: every ``header.update(ww.to_fits()[0])``
in the reduction used gwcs's ``max_pix_error=0.25`` px default and produced a
degree-3 SIP fit that disagreed with the GWCS by up to 5.5 mas (NIRCam SW),
6.6 mas (LW) and 8.0 mas (MIRI) -- larger than the pipeline's own 2 mas
per-exposure and 5 mas cross-filter astrometric tolerances -- and merged over
the delivered degree-4 fit, orphaning its high-order coefficients.
"""
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.modeling import models

from jwst_gc_pipeline.reduction.fits_wcs_sync import (
    FitsGwcsMismatchError, fits_gwcs_discrepancy_mas, sip_header_from_gwcs,
    strip_sip_keywords, sync_header_to_gwcs)

SHAPE = (512, 512)


def _distorted_gwcs(shape=SHAPE, amplitude=3e-4):
    """A gwcs with a genuinely non-linear (quintic) distortion term.

    ``amplitude`` is chosen so a degree-3 SIP fit CANNOT represent it: the
    5th-order term is what a low-degree fit truncates, which is exactly the
    real-frame situation (a degree-3 fit of a SIAF polynomial).
    """
    gwcs = pytest.importorskip('gwcs')
    from gwcs import coordinate_frames as cf
    from astropy import coordinates as coord
    from astropy import units as u

    ny, nx = shape
    # polynomial distortion in x and y, including a 5th-order cross term
    xpoly = models.Polynomial2D(degree=5, c0_0=0.0, c1_0=1.0, c0_1=0.0,
                                c2_0=1e-6, c1_1=-2e-6,
                                c5_0=amplitude / nx ** 4)
    ypoly = models.Polynomial2D(degree=5, c0_0=0.0, c1_0=0.0, c0_1=1.0,
                                c0_2=1.5e-6, c1_1=1e-6,
                                c0_5=amplitude / ny ** 4)
    shift = models.Shift(-nx / 2.0) & models.Shift(-ny / 2.0)
    distortion = shift | models.Mapping((0, 1, 0, 1)) | (xpoly & ypoly)
    scale = models.Scale(1.0 / 3600 * 0.031) & models.Scale(1.0 / 3600 * 0.031)
    tan = models.Pix2Sky_TAN()
    n2c = models.RotateNative2Celestial(266.5, -28.8, 180.0)

    detector = cf.Frame2D(name='detector', axes_order=(0, 1),
                          unit=(u.pix, u.pix))
    sky = cf.CelestialFrame(reference_frame=coord.ICRS(), name='icrs',
                            unit=(u.deg, u.deg))
    w = gwcs.WCS([(detector, distortion | scale | tan | n2c), (sky, None)])
    w.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))
    return w


def test_default_to_fits_is_too_loose_and_the_tight_fit_fixes_it():
    """The exact defect: gwcs's 0.25 px default is mas-scale wrong; 0.01 px is not."""
    g = _distorted_gwcs()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loose = fits.Header(g.to_fits_sip())          # gwcs default, 0.25 px
    loose_max, _ = fits_gwcs_discrepancy_mas(loose, g, SHAPE)

    tight = sip_header_from_gwcs(g)                   # this module, 0.01 px
    tight_max, _ = fits_gwcs_discrepancy_mas(tight, g, SHAPE)

    # the whole point: the default is materially worse than the tight fit
    assert tight_max < loose_max / 5.0, (loose_max, tight_max)
    # and the tight fit is inside the pipeline's tightest astrometric tolerance
    assert tight_max < 0.5, tight_max


def test_sync_header_to_gwcs_verifies_and_reports():
    g = _distorted_gwcs()
    hdr = fits.Header()
    max_mas, med_mas = sync_header_to_gwcs(hdr, g, SHAPE)
    assert max_mas < 0.5 and med_mas <= max_mas
    assert hdr['CTYPE1'] == 'RA---TAN-SIP'
    # independently re-measure what was written
    remeasured, _ = fits_gwcs_discrepancy_mas(hdr, g, SHAPE)
    assert remeasured == pytest.approx(max_mas, rel=1e-6)


def test_sync_header_to_gwcs_raises_when_it_cannot_meet_tolerance():
    """A verification that cannot fail is not a verification."""
    g = _distorted_gwcs()
    with pytest.raises(FitsGwcsMismatchError, match='disagrees with'):
        sync_header_to_gwcs(fits.Header(), g, SHAPE, tol_mas=1e-9)


def test_stale_high_order_sip_coefficients_are_stripped():
    """``Header.update`` merges: a degree-3 fit over a degree-4 header leaves
    orphan A_0_4/A_4_0/... cards that disagree with the written A_ORDER.
    Observed on every ``_destreak.fits`` in the archive."""
    g = _distorted_gwcs()
    hdr = fits.Header()
    # pre-existing higher-degree fit
    hdr['A_ORDER'] = 4
    hdr['B_ORDER'] = 4
    for k in ('A_4_0', 'A_0_4', 'A_2_2', 'B_4_0', 'B_0_4', 'B_2_2',
              'AP_4_0', 'BP_0_4'):
        hdr[k] = 1.234e-9

    sync_header_to_gwcs(hdr, g, SHAPE)

    order = int(hdr['A_ORDER'])
    orphans = [k for k in hdr
               if k.startswith(('A_', 'B_', 'AP_', 'BP_'))
               and '_' in k[2:] and not k.endswith('ORDER')
               and sum(int(p) for p in k.split('_')[-2:]) > order]
    assert not orphans, f"orphan SIP terms above A_ORDER={order}: {orphans}"


def test_mixed_cd_and_pc_cdelt_cannot_survive():
    """A header must not end up with BOTH linear representations: astropy then
    silently ignores CDELT, so the header reads differently from what was
    written.  (Same defect class as #181, at the fitting level.)"""
    g = _distorted_gwcs()
    hdr = fits.Header()
    for k, v in (('PC1_1', 1.0), ('PC1_2', 0.0), ('PC2_1', 0.0), ('PC2_2', 1.0),
                 ('CDELT1', -8.6e-6), ('CDELT2', 8.6e-6),
                 ('CD1_1', -8.6e-6), ('CD2_2', 8.6e-6)):
        hdr[k] = v

    sync_header_to_gwcs(hdr, g, SHAPE)

    has_cd = any(k.startswith('CD1_') or k.startswith('CD2_') for k in hdr)
    has_pc = any(k.startswith('PC') for k in hdr) or 'CDELT1' in hdr
    assert has_cd != has_pc, (
        f"header carries both linear representations: "
        f"{sorted(k for k in hdr if k.startswith(('CD', 'PC')))}")


def test_strip_sip_keywords_leaves_the_linear_wcs_alone():
    hdr = fits.Header({'CRVAL1': 266.5, 'CRPIX1': 100.0, 'CD1_1': -1e-5,
                       'A_ORDER': 3, 'A_2_0': 1e-7, 'BP_1_1': 2e-7})
    strip_sip_keywords(hdr)
    assert 'A_ORDER' not in hdr and 'A_2_0' not in hdr and 'BP_1_1' not in hdr
    assert hdr['CRVAL1'] == 266.5 and hdr['CD1_1'] == -1e-5


def test_discrepancy_ignores_ctype_sip_suffix_loss():
    """A header whose CTYPE lost '-SIP' still carries A_*/B_*; the comparison
    must read it with relax=True or it silently reports the distortion as an
    error (or, worse, as agreement)."""
    g = _distorted_gwcs()
    hdr = sip_header_from_gwcs(g)
    stripped = hdr.copy()
    stripped['CTYPE1'] = 'RA---TAN'
    stripped['CTYPE2'] = 'DEC--TAN'
    a, _ = fits_gwcs_discrepancy_mas(hdr, g, SHAPE)
    b, _ = fits_gwcs_discrepancy_mas(stripped, g, SHAPE)
    assert b == pytest.approx(a, rel=1e-6)


@pytest.mark.parametrize('bad', [np.nan])
def test_no_finite_samples_raises(bad):
    g = _distorted_gwcs()
    g.bounding_box = ((-1e6, -1e6 + 1), (-1e6, -1e6 + 1))
    hdr = sip_header_from_gwcs(g, max_pix_error=0.25)
    with pytest.raises(FitsGwcsMismatchError, match='no finite samples'):
        fits_gwcs_discrepancy_mas(hdr, g, SHAPE)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
