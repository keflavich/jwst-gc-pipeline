"""STDGDC loader/evaluator tests.

The spot-value fixtures are hand-copied from the COMMENT cards embedded in
``STDGDC_NRCB1_F212N.fits`` HDU1/2/3 ("X-FORWARD DISTORTION MAPPING",
"Y-FORWARD DISTORTION MAPPING", "PIXEL-AREA CORRECTION (MAGNITUDES)"): the 5x5
grid at 1-based X, Y in {1, 512, 1024, 1536, 2048}.  They pin the [y, x] map
orientation, the 1-based value convention, and the index origin.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.astrometry_gdc.stdgdc import (
    STDGDC, GDCFileNotFoundError, resolve_gdc_file, detector_filter_from_header)

REAL_FILE = ('/orange/adamginsburg/jwst/distortion/jayander_stdgdc/NIRCam/'
             'SWC/F212N/STDGDC_NRCB1_F212N.fits')

needs_library = pytest.mark.skipif(
    not __import__('os').path.isfile(REAL_FILE),
    reason='STDGDC library mirror not available on this host')

# 1-based grid positions of the embedded spot tables.
SPOT_POS = (1, 512, 1024, 1536, 2048)

# XGC forward mapping [row = Y index, col = X index], from HDU1 COMMENTs
# (rows here ordered Y=1 .. Y=2048, i.e. the header table flipped).
XGC_TABLE = np.array([
    [-0.955,  512.423, 1026.608, 1540.846, 2055.423],   # Y=0001
    [0.435,   512.274, 1024.929, 1537.628, 2050.689],   # Y=0512
    [2.264,   512.670, 1023.873, 1535.135, 2046.674],   # Y=1024
    [4.498,   513.568, 1023.448, 1533.376, 2043.619],   # Y=1536
    [7.131,   514.976, 1023.624, 1532.322, 2041.383],   # Y=2048
])

# YGC forward mapping, from HDU2 COMMENTs.
YGC_TABLE = np.array([
    [-4.862,   -4.719,   -3.369,   -0.680,    3.297],   # Y=0001
    [509.966,  509.649,  510.687,  513.112,  516.911],  # Y=0512
    [1024.199, 1023.440, 1024.143, 1026.293, 1029.913], # Y=1024
    [1537.115, 1535.917, 1536.270, 1538.157, 1541.636], # Y=1536
    [2048.964, 2047.290, 2047.291, 2048.975, 2052.288], # Y=2048
])

# MGC pixel-area correction (magnitudes), from HDU3 COMMENTs.
MGC_TABLE = np.array([
    [-0.0159, -0.0136, -0.0132, -0.0134, -0.0130],      # Y=0001
    [-0.0089, -0.0068, -0.0060, -0.0060, -0.0067],      # Y=0512
    [-0.0020, -0.0008, -0.0001,  0.0002,  0.0001],      # Y=1024
    [0.0030,   0.0046,  0.0056,  0.0058,  0.0045],      # Y=1536
    [0.0069,   0.0095,  0.0098,  0.0102,  0.0093],      # Y=2048
])


@pytest.fixture(scope='module')
def gdc():
    return STDGDC.load('NRCB1', 'F212N', version='v1')


@needs_library
def test_spot_values_forward(gdc):
    """(a) forward() reproduces the COMMENT-card spot tables.

    API is 0-based in/out: raw 1-based (X, Y) -> input (X-1, Y-1); the table
    value is a 1-based corrected position -> expect value - 1.
    """
    for j, Y in enumerate(SPOT_POS):
        for i, X in enumerate(SPOT_POS):
            xc, yc = gdc.forward(X - 1, Y - 1)
            # table printed to 3 decimals -> 5e-4 rounding + 1e-5 quantisation
            assert xc == pytest.approx(XGC_TABLE[j, i] - 1.0, abs=1e-3), (X, Y)
            assert yc == pytest.approx(YGC_TABLE[j, i] - 1.0, abs=1e-3), (X, Y)


@needs_library
def test_spot_values_pixel_area_mag(gdc):
    for j, Y in enumerate(SPOT_POS):
        for i, X in enumerate(SPOT_POS):
            m = gdc.pixel_area_mag(X - 1, Y - 1)
            assert m == pytest.approx(MGC_TABLE[j, i], abs=2e-4), (X, Y)


@needs_library
def test_grid_node_identity(gdc):
    """Bilinear interpolation is exact at integer nodes: forward at 0-based
    (i, j) must equal xgc[j, i] - 1 to machine precision.  Any off-by-one or
    transposed indexing breaks this immediately."""
    rng = np.random.default_rng(7)
    ii = rng.integers(0, 2048, 50)
    jj = rng.integers(0, 2048, 50)
    xc, yc = gdc.forward(ii, jj)
    np.testing.assert_allclose(xc, gdc.xgc[jj, ii] - 1.0, atol=1e-9)
    np.testing.assert_allclose(yc, gdc.ygc[jj, ii] - 1.0, atol=1e-9)


@needs_library
def test_forward_reverse_roundtrip(gdc):
    """(b) forward -> reverse round-trip < 1e-3 pix on a random sample."""
    rng = np.random.default_rng(42)
    x = rng.uniform(5.0, 2042.0, 500)
    y = rng.uniform(5.0, 2042.0, 500)
    xc, yc = gdc.forward(x, y)
    xr, yr = gdc.reverse(xc, yc)
    assert np.max(np.abs(xr - x)) < 1e-3
    assert np.max(np.abs(yr - y)) < 1e-3


@needs_library
def test_indexing_shift_sanity(gdc):
    """(d) shifting the input by 1 pixel shifts the output by ~1 pixel."""
    rng = np.random.default_rng(3)
    x = rng.uniform(100.0, 1900.0, 200)
    y = rng.uniform(100.0, 1900.0, 200)
    xc0, yc0 = gdc.forward(x, y)
    xc1, _ = gdc.forward(x + 1.0, y)
    _, yc1 = gdc.forward(x, y + 1.0)
    np.testing.assert_allclose(xc1 - xc0, 1.0, atol=0.05)
    np.testing.assert_allclose(yc1 - yc0, 1.0, atol=0.05)


@needs_library
def test_out_of_bounds_is_nan(gdc):
    xc, yc = gdc.forward([-10.0, 3000.0], [100.0, 100.0])
    assert np.all(np.isnan(xc)) and np.all(np.isnan(yc))


@needs_library
def test_resolve_versions_and_cache():
    p1 = resolve_gdc_file('NRCB1', 'F212N', version='v1')
    p2 = resolve_gdc_file('NRCB1', 'F212N', version='v2')
    pa = resolve_gdc_file('NRCB1', 'F212N', version='auto')
    assert p1.endswith('STDGDC_NRCB1_F212N.fits')
    assert '/v2/' in p2 and p2.endswith('_2.fits')
    assert pa == p2  # auto prefers v2 when present (peppar behaviour)
    # F210M uses the FILT_DET naming order; the glob must still find it
    p3 = resolve_gdc_file('NRCA3', 'F210M')
    assert p3.endswith('STDGDC_F210M_NRCA3.fits')
    # LW alias mapping
    p4 = resolve_gdc_file('NRCBLONG', 'F277W')
    assert p4.endswith('STDGDC_NRCBL_F277W.fits')
    with pytest.raises(GDCFileNotFoundError):
        resolve_gdc_file('NRCBLONG', 'F405N')
    # caching: same object back
    a = STDGDC.load('NRCB1', 'F212N', version='v1')
    b = STDGDC.load('NRCB1', 'F212N', version='v1')
    assert a is b


@needs_library
def test_missing_filter_and_fallback():
    with pytest.raises(GDCFileNotFoundError):
        STDGDC.load('NRCB1', 'F187N')
    with pytest.warns(UserWarning, match='FALLING BACK'):
        g = STDGDC.load('NRCB1', 'F187N', fallback_filter='F212N')
    assert g.meta['filter_fallback'] == 'F187N->F212N'
    assert g.meta['filter'] == 'F212N'


def test_detector_filter_from_header():
    assert detector_filter_from_header(
        {'DETECTOR': 'NRCB1', 'FILTER': 'F212N', 'PUPIL': 'CLEAR'}) == ('NRCB1', 'F212N')
    assert detector_filter_from_header(
        {'DETECTOR': 'NRCA2', 'FILTER': 'CLEAR', 'PUPIL': 'F164N'}) == ('NRCA2', 'F164N')
