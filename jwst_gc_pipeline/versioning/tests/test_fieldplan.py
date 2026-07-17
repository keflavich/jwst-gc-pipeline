"""Tests for the on-disk field planner (`rerun plan --field`)."""
import numpy as np
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.versioning import (fieldplan, stamping, fingerprint as fp,
                                         prov_sidecar, rerun)


def _i2d(path, data=None, meta=None):
    data = np.arange(16).reshape(4, 4) if data is None else data
    phdu = fits.PrimaryHDU()
    for k, v in (meta or {}).items():
        phdu.header[k] = v
    sci = fits.ImageHDU(np.asarray(data, dtype='float32'), name='SCI')
    sci.header['CRVAL1'] = 266.4
    sci.header['CRPIX1'] = 100.0
    sci.header['CD1_1'] = -8e-6
    sci.header['CTYPE1'] = 'RA---TAN'
    fits.HDUList([phdu, sci]).writeto(path, overwrite=True)
    return path


def _cat(path, flux=(100.0, 200.0)):
    Table({'x_fit': [1.0, 2.0], 'ra': [10.0, 11.0], 'dec': [-1.0, -2.0],
           'flux_fit': list(flux)}).write(path, overwrite=True)
    return path


def _field(tmp_path):
    """A minimal stamped field: imaging i2d + m12 + m3 catalogs, upstream wired."""
    d = tmp_path
    i2d = _i2d(str(d / 'img_i2d.fits'))
    stamping.stamp_product(i2d, 'imaging')
    img_facets = prov_sidecar.read_sidecar(i2d)['outputs']

    m12 = _cat(str(d / 'f200w_nrca_m12.fits'))
    stamping.stamp_catalog(m12, 'm12', upstream={'imaging': img_facets})
    m12_facets = prov_sidecar.read_sidecar(m12)['outputs']

    m3 = _cat(str(d / 'f200w_nrca_m3.fits'))
    stamping.stamp_catalog(m3, 'm3',
                           upstream={'imaging': img_facets, 'm12': m12_facets})
    return d


def test_all_skip_when_nothing_changed(tmp_path):
    _field(tmp_path)
    decisions, products = fieldplan.plan_field(str(tmp_path), use_live_env=False)
    assert set(products) == {'imaging', 'm12', 'm3'}
    by = {x.stage: x for x in decisions}
    assert all(by[s].verdict == rerun.SKIP for s in ('imaging', 'm12', 'm3'))


def test_code_drift_forces_refit(tmp_path):
    _field(tmp_path)
    # Corrupt the recorded code hash of m3 so the live code_hash differs.
    m3 = str(tmp_path / 'f200w_nrca_m3.fits')
    rec = prov_sidecar.read_sidecar(m3)
    rec['inputs']['code'] = 'STALECODE'
    prov_sidecar.write_sidecar(m3, rec)
    decisions, _ = fieldplan.plan_field(str(tmp_path), use_live_env=False)
    by = {x.stage: x for x in decisions}
    assert by['m3'].verdict == rerun.REFIT


def test_live_env_bump_triggers_re_reduce(tmp_path):
    # imaging recorded env carries an OLD jwst version (from the header); the live
    # env differs -> a re-reduction is pending.
    i2d = _i2d(str(tmp_path / 'img_i2d.fits'), meta={'CAL_VER': '0.0-ancient'})
    stamping.stamp_product(i2d, 'imaging')
    decisions, _ = fieldplan.plan_field(str(tmp_path), use_live_env=True)
    by = {x.stage: x for x in decisions}
    # only asserts a re-reduce IF a live jwst version is resolvable
    if fieldplan.live_env().get('jwst_version'):
        assert by['imaging'].verdict == rerun.RE_REDUCE
        assert by['imaging'].conditional


def test_wcs_only_reseed_blocks_frozen_stage(tmp_path):
    i2d = _i2d(str(tmp_path / 'img_i2d.fits'))
    stamping.stamp_product(i2d, 'imaging')
    img = prov_sidecar.read_sidecar(i2d)['outputs']
    # m4 recorded upstream: same DATA as current imaging, but a STALE wcs hash ->
    # a WCS-only change vs the recomputed current facets.
    m4 = _cat(str(tmp_path / 'f200w_nrca_m4.fits'))
    stamping.stamp_catalog(m4, 'm4', upstream={
        'imaging': {'data': img['data'], 'wcs': 'STALEWCS', 'meta': img['meta']}})
    decisions, _ = fieldplan.plan_field(str(tmp_path), use_live_env=False,
                                        wcs_change_mode='reseed')
    by = {x.stage: x for x in decisions}
    assert by['m4'].verdict == rerun.BLOCKED


def test_unreadable_image_parent_returns_none_not_table(tmp_path):
    import pytest
    # A known image (is_catalog=False) that can't be read must NOT fall through
    # to the table path (which would mis-type it); it returns None + warns.
    bad = str(tmp_path / 'broken_i2d.fits')
    with open(bad, 'wb') as fh:
        fh.write(b'not a fits file')
    with pytest.warns(UserWarning):
        assert fieldplan._current_facets(bad, is_catalog=False) is None
    # a valid image still yields facets
    good = _i2d(str(tmp_path / 'ok_i2d.fits'))
    f = fieldplan._current_facets(good, is_catalog=False)
    assert f and f['data']


def test_wcs_only_posthoc_is_reproject(tmp_path):
    i2d = _i2d(str(tmp_path / 'img_i2d.fits'))
    stamping.stamp_product(i2d, 'imaging')
    img = prov_sidecar.read_sidecar(i2d)['outputs']
    m4 = _cat(str(tmp_path / 'f200w_nrca_m4.fits'))
    stamping.stamp_catalog(m4, 'm4', upstream={
        'imaging': {'data': img['data'], 'wcs': 'STALEWCS', 'meta': img['meta']}})
    decisions, _ = fieldplan.plan_field(str(tmp_path), use_live_env=False,
                                        wcs_change_mode='posthoc')
    by = {x.stage: x for x in decisions}
    assert by['m4'].verdict == rerun.REPROJECT
