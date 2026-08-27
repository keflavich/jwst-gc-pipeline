"""The in-driver alignment check must report a real separation on every
instrument.

Three drivers carried three copies of ``check_wcs``.  MIRI's and NIRISS's
still evaluated the GWCS at the hardcoded NIRCam detector centre
``(1024, 1024)``.  A MIRI imaging array is ``(1024, 1032)`` and its GWCS
bounding box stops at ``x1 = 1023.5``, so the call returns ``NaN`` -- measured
on a production frame::

    /orange/adamginsburg/jwst/w51/F770W/pipeline/
        jw06151002001_02103_00001_mirimage_cal.fits
    data.shape (ny, nx) = (1024, 1032)
    pix(1024,1024) ->  (nan, nan)
    array center   ->  (290.94196378, 14.51179787)

The separations that check prints are the only in-driver evidence that
``fix_alignment`` moved the frame by the amount the offsets table asked for.
On MIRI they were all ``nan``, so a frame that got the wrong shift, no shift,
or a SIP header disagreeing with its GWCS printed what a correct one printed.
"""
import numpy as np
import pytest

astropy_modeling = pytest.importorskip('astropy.modeling')
gwcs = pytest.importorskip('gwcs')

from astropy import units as u
from astropy.coordinates import ICRS
from astropy.modeling import models
from gwcs import coordinate_frames as cf


def _gwcs_for(shape):
    """A minimal TAN GWCS whose bounding box is exactly ``shape``."""
    ny, nx = shape
    det = cf.Frame2D(name='detector', axes_order=(0, 1), unit=(u.pix, u.pix))
    sky = cf.CelestialFrame(reference_frame=ICRS(), name='icrs')
    transform = ((models.Shift(-nx / 2.0) & models.Shift(-ny / 2.0))
                 | models.Scale(1e-5) & models.Scale(1e-5)
                 | models.Pix2Sky_TAN()
                 | models.RotateNative2Celestial(266.4, -28.9, 180.0))
    w = gwcs.wcs.WCS([(det, transform), (sky, None)])
    w.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))
    return w


@pytest.mark.parametrize('shape,label', [
    ((1024, 1032), 'MIRI imaging'),
    ((2048, 2048), 'NIRCam full array'),
    ((256, 256), 'a subarray'),
])
def test_the_array_centre_is_on_the_detector(shape, label):
    """What the shared ``check_wcs`` evaluates.  ``data.shape`` puts the point
    inside the bounding box for every instrument and every subarray."""
    ny, nx = shape
    w = _gwcs_for(shape)
    coord = w.pixel_to_world(nx / 2.0, ny / 2.0)
    assert np.isfinite(coord.ra.deg), f'{label}: array centre is off-array'
    assert np.isfinite(coord.dec.deg), label


def test_the_hardcoded_nircam_centre_is_off_a_MIRI_array():
    """The defect itself: the point the MIRI and NIRISS copies evaluated.

    ``NaN`` is not a loud failure -- it prints in place of a separation and
    reads like a clean check.
    """
    w = _gwcs_for((1024, 1032))
    coord = w.pixel_to_world(1024, 1024)
    assert np.isnan(coord.ra.deg), (
        'if this ever becomes finite the regression this test pins is gone, '
        'but so is the reason the shared helper uses data.shape')


def test_all_three_drivers_import_the_shared_check():
    """No driver may grow a private copy back."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ('PipelineMIRI.py', 'PipelineRerunNIRISS.py',
                 'PipelineRerunNIRCAM-LONG.py'):
        text = (root / name).read_text()
        assert 'from jwst_gc_pipeline.reduction.wcs_check import check_wcs' \
            in text, f'{name} does not use the shared check'
        assert '\ndef check_wcs(' not in text, (
            f'{name} defines its own check_wcs again; the three copies '
            f'diverged once already')
        assert 'pixel_to_world(1024, 1024)' not in text, name
