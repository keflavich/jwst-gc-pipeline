"""Sky-clean tier, LOCAL reference: the annulus floor is compared to the 5th
percentile of the source's ~3" tile in i2d-ERR units, OR-ed with the global
dark-sky test.

The global reference (5th percentile of the whole mosaic) is the darkest
lane of the field, so outside a dark cloud every source reads "on emission",
including a smooth bright plateau that cannot be turned into a star any more
than dark sky can.  What can is PSF-scale structure, and the local test still
rejects it.
"""
import numpy as np
from astropy.table import Table
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from jwst_gc_pipeline.photometry.cataloging import (_filter_extended_emission,
                                                   _tile_percentile_at)
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

NY = NX = 400
PIX_AS = 0.063
X_DARK, X_PLATEAU, X_STRUCT = 330, 200, 60   # dark | smooth plateau | structured


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-PIX_AS / 3600, PIX_AS / 3600]
    return w


def _image(seed=7):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, (NY, NX))
    data[:, :270] += 25.0                       # elevated emission level, x < 270
    struct = gaussian_filter(rng.normal(0, 1, (NY, NX)), 1.5)
    struct *= 6.0 / np.std(struct)              # PSF-scale structure, x < 130
    data[:, :130] += struct[:, :130]
    return data


def _add_star(data, x, y, amp=30.0, sig=1.0):
    yy, xx = np.mgrid[0:NY, 0:NX]
    data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _run(xs, local_as, err=1.0, **kw):
    w = _wcs()
    data = _image()
    for x in xs:
        _add_star(data, x, 200.0)
    sc = w.pixel_to_world(np.asarray(xs, float), np.full(len(xs), 200.0))
    cat = Table({'skycoord': sc, 'qfit': np.full(len(xs), 0.8),
                 'flags': np.zeros(len(xs)), 'local_bkg': np.zeros(len(xs)),
                 'flux': np.full(len(xs), 100.0), 'flux_err': np.full(len(xs), 10.0),
                 'group_size': np.ones(len(xs)), 'id': np.arange(len(xs))})
    out = _filter_extended_emission(cat, data_i2d_image=data, ww_i2d=w,
                                    err_i2d_image=None if err is None else np.full(data.shape, err),
                                    sky_clean_local_arcsec=local_as, label='test', **kw)
    return set(np.asarray(out['id']).tolist()), cat


def test_global_only_keeps_dark_sky_only():
    kept, _ = _run([X_DARK, X_PLATEAU, X_STRUCT], local_as=0.0)
    assert kept == {0}


def test_local_reference_admits_smooth_plateau_not_structure():
    kept, cat = _run([X_DARK, X_PLATEAU, X_STRUCT], local_as=3.0)
    assert kept == {0, 1}
    s = np.asarray(cat['local_structure_snr'])
    assert s[1] < 2.0 < s[2]


def test_local_reference_uses_err_unit():
    # an ERR 10x too small makes the plateau's pure-noise floor-p5 gap ~10 ERR
    kept, _ = _run([X_DARK, X_PLATEAU], local_as=3.0, err=0.1)
    assert kept == {0}


def test_local_threshold_is_its_own_option():
    # the global test in dark-sky sigma and the local one in ERR are set separately
    kept, _ = _run([X_DARK, X_PLATEAU], local_as=3.0, sky_clean_max_sky_snr=-100.0)
    assert kept == {0, 1}
    kept, _ = _run([X_DARK, X_PLATEAU], local_as=3.0, sky_clean_local_max_err=-100.0)
    assert kept == {0}


def test_no_err_plane_logs_and_writes_nan_column(capsys):
    kept, cat = _run([X_DARK, X_PLATEAU, X_STRUCT], local_as=3.0, err=None)
    assert kept == {0}
    assert 'no i2d ERR plane' in capsys.readouterr().out
    assert np.all(np.isnan(np.asarray(cat['local_structure_snr'])))


def test_tile_percentile_interpolates():
    img = np.zeros((128, 128))
    img[:, 64:] = 10.0
    v = _tile_percentile_at(img, np.array([10.0, 120.0, 63.5]), np.array([64.0] * 3), 32, 5)
    assert v[0] == 0 and v[1] == 10 and 0 < v[2] < 10
    img[:64, :64] = np.nan
    assert np.isnan(_tile_percentile_at(img, np.array([5.0]), np.array([5.0]), 32, 5)[0])


def test_pipeline_default_on():
    assert MANUAL_DEFAULTS['manual_sky_clean_local_arcsec'] == 3.0
    assert MANUAL_DEFAULTS['manual_sky_clean_local_max_err'] == 2.0
