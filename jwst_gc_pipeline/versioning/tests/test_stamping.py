"""Tests for the stage-write provenance stamping (sidecar + mirrored keys)."""
import numpy as np
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.versioning import stamping, fingerprint as fp, prov_sidecar


def _i2d(path, data=None, wcs=True, meta=None):
    data = np.arange(9).reshape(3, 3) if data is None else data
    phdu = fits.PrimaryHDU()
    for k, v in (meta or {}).items():
        phdu.header[k] = v
    sci = fits.ImageHDU(np.asarray(data, dtype='float32'), name='SCI')
    if wcs:
        sci.header['CRVAL1'] = 266.4
        sci.header['CRPIX1'] = 100.0
        sci.header['CD1_1'] = -8e-6
        sci.header['CTYPE1'] = 'RA---TAN'
    fits.HDUList([phdu, sci]).writeto(path, overwrite=True)
    return path


def test_stamp_product_writes_sidecar_and_mirrors_keys(tmp_path):
    p = _i2d(str(tmp_path / 'x_i2d.fits'),
             meta={'CAL_VER': '1.14.0', 'CRDS_CTX': 'jwst_1253.pmap'})
    record, sc_path = stamping.stamp_product(p, 'imaging')
    # sidecar exists and round-trips
    assert sc_path == prov_sidecar.sidecar_path(p)
    read = prov_sidecar.read_sidecar(p)
    assert read['stage'] == 'imaging'
    assert read['outputs'] == record['outputs']
    # env auto-filled from header
    assert read['inputs']['env']['jwst_version'] == '1.14.0'
    assert read['inputs']['env']['crds_context'] == 'jwst_1253.pmap'
    # output facets equal an independent recompute
    assert record['outputs'] == fp.facet_hashes(p)
    # mirrored keys present in the primary header
    with fits.open(p) as hdul:
        h = hdul[0].header
        assert h['GCSTAGE'] == 'imaging'
        assert h['GCDATAH'] == record['outputs']['data'][:16]
        assert h['GCWCSH'] == record['outputs']['wcs'][:16]


def test_stamp_product_code_hash_recorded(tmp_path):
    p = _i2d(str(tmp_path / 'y_i2d.fits'))
    record, _ = stamping.stamp_product(p, 'imaging')
    # code_hash resolves against the real repo for a known stage
    assert record['inputs']['code'] is not None
    assert len(record['inputs']['code']) == 64


def test_stamp_catalog_excludes_reprojected_cols(tmp_path):
    p = str(tmp_path / 'cat_m6_dao_basic.fits')
    Table({'x_fit': [1.0, 2.0], 'ra': [10.0, 11.0], 'dec': [-1.0, -2.0],
           'flux_fit': [100.0, 200.0]}).write(p, overwrite=True)
    record, _ = stamping.stamp_catalog(p, 'm6')
    assert record['stage'] == 'm6'
    assert record['outputs']['wcs'] is None
    # data facet is invariant to an RA/Dec change (reproject-only path)
    Table({'x_fit': [1.0, 2.0], 'ra': [99.0, 98.0], 'dec': [5.0, 6.0],
           'flux_fit': [100.0, 200.0]}).write(p, overwrite=True)
    record2, _ = stamping.stamp_catalog(p, 'm6')
    assert record2['outputs']['data'] == record['outputs']['data']
    # but a flux change does move it
    Table({'x_fit': [1.0, 2.0], 'ra': [99.0, 98.0], 'dec': [5.0, 6.0],
           'flux_fit': [100.0, 999.0]}).write(p, overwrite=True)
    record3, _ = stamping.stamp_catalog(p, 'm6')
    assert record3['outputs']['data'] != record['outputs']['data']


def test_try_stamp_missing_file_returns_none_no_raise(tmp_path):
    import pytest
    with pytest.warns(UserWarning):
        assert stamping.try_stamp_product(str(tmp_path / 'nope.fits'), 'imaging') is None
    with pytest.warns(UserWarning):
        assert stamping.try_stamp_catalog(str(tmp_path / 'nope.fits'), 'm3') is None


def test_stamp_upstream_and_params_recorded(tmp_path):
    p = _i2d(str(tmp_path / 'z_i2d.fits'))
    up = {'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}}
    record, _ = stamping.stamp_product(p, 'm12', params='PHASH', upstream=up)
    assert record['inputs']['params'] == 'PHASH'
    assert record['inputs']['upstream'] == up
