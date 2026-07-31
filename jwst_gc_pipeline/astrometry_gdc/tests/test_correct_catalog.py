"""correct_catalog column/metadata behaviour (synthetic frame + GDC)."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.astrometry_gdc import correct_catalog as cc
from .test_gdc_wcs import N, synthetic_gdc, synthetic_wcs


class _StubLoader:
    @staticmethod
    def load(detector, filtername, root=None, version='auto',
             fallback_filter=None, method='linear'):
        gdc = synthetic_gdc()
        gdc.meta.update(gdc_file='STDGDC_STUB.fits', detector=detector,
                        filter=str(filtername).upper(),
                        version_requested=version)
        return gdc


@pytest.fixture()
def patched(monkeypatch):
    header = fits.Header({'DETECTOR': 'NRCB1', 'FILTER': 'F212N',
                          'PUPIL': 'CLEAR'})
    monkeypatch.setattr(cc, 'STDGDC', _StubLoader)
    monkeypatch.setattr(cc, 'load_frame_wcs',
                        lambda cal_file, prefer_gwcs=True: (synthetic_wcs(), header))


def _catalog():
    rng = np.random.default_rng(1)
    return Table({'x_fit': rng.uniform(5, N - 6, 40),
                  'y_fit': rng.uniform(5, N - 6, 40),
                  'flux_fit': rng.uniform(1, 100, 40)})


def test_adds_columns_and_meta(patched):
    cat = _catalog()
    cat, sol = cc.add_gdc_skycoords(cat, 'fake_cal.fits')
    assert cc.GDC_RA_COL in cat.colnames and cc.GDC_DEC_COL in cat.colnames
    assert np.all(np.isfinite(cat[cc.GDC_RA_COL]))
    assert cat.meta['GDCFILE'] == 'STDGDC_STUB.fits'
    assert cat.meta['GDCXYCOL'] == 'x_fit,y_fit'
    assert 'GDCAFFX' in cat.meta and 'GDCRMS' in cat.meta


def test_refuses_to_overwrite_gdc_columns(patched):
    cat = _catalog()
    cat, _ = cc.add_gdc_skycoords(cat, 'fake_cal.fits')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        cc.add_gdc_skycoords(cat, 'fake_cal.fits')


def test_existing_skycoord_columns_untouched(patched):
    cat = _catalog()
    cat['skycoord_ra'] = np.full(len(cat), 266.5)
    before = np.array(cat['skycoord_ra'])
    cat, _ = cc.add_gdc_skycoords(cat, 'fake_cal.fits')
    np.testing.assert_array_equal(np.array(cat['skycoord_ra']), before)


def test_column_fallback_order(patched):
    cat = _catalog()
    cat.rename_column('x_fit', 'xcentroid')
    cat.rename_column('y_fit', 'ycentroid')
    cat, _ = cc.add_gdc_skycoords(cat, 'fake_cal.fits')
    assert cat.meta['GDCXYCOL'] == 'xcentroid,ycentroid'


def test_cli_writes_sibling_file(patched, tmp_path):
    cat = _catalog()
    catpath = tmp_path / 'jw01182_m1_cat.fits'
    cat.write(catpath)
    out = cc.main(['--catalog', str(catpath), '--cal', 'fake_cal.fits'])
    assert out == str(tmp_path / 'jw01182_m1_cat_gdc.fits')
    result = Table.read(out)
    assert cc.GDC_RA_COL in result.colnames
    # original untouched
    orig = Table.read(catpath)
    assert cc.GDC_RA_COL not in orig.colnames


def test_cli_inplace(patched, tmp_path):
    cat = _catalog()
    catpath = tmp_path / 'jw01182_m1_cat.fits'
    cat.write(catpath)
    out = cc.main(['--catalog', str(catpath), '--cal', 'fake_cal.fits',
                   '--inplace-cols'])
    assert out == str(catpath)
    assert cc.GDC_RA_COL in Table.read(catpath).colnames
