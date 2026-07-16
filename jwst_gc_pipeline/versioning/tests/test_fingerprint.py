"""Tests for the per-facet content fingerprints."""
import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.versioning import fingerprint as fp


def _sci_hdulist(data, wcs=True, extra_meta=None):
    """A minimal SCI-extension HDUList with optional WCS + extra header cards."""
    phdu = fits.PrimaryHDU()
    if extra_meta:
        for k, v in extra_meta.items():
            phdu.header[k] = v
    sci = fits.ImageHDU(data=np.asarray(data, dtype='float32'), name='SCI')
    if wcs:
        sci.header['CRVAL1'] = 266.4
        sci.header['CRVAL2'] = -29.0
        sci.header['CRPIX1'] = 100.0
        sci.header['CRPIX2'] = 100.0
        sci.header['CD1_1'] = -8.0e-6
        sci.header['CD2_2'] = 8.0e-6
        sci.header['CTYPE1'] = 'RA---TAN'
        sci.header['CTYPE2'] = 'DEC--TAN'
    return fits.HDUList([phdu, sci])


def test_data_hash_is_header_independent():
    a = _sci_hdulist(np.arange(9).reshape(3, 3))
    b = _sci_hdulist(np.arange(9).reshape(3, 3))
    # Change only a header card in b.
    b['SCI'].header['CRVAL1'] = 12.3456
    b[0].header['SOMEKEY'] = 'different'
    assert fp.data_hash(a) == fp.data_hash(b)


def test_data_hash_detects_pixel_change():
    a = _sci_hdulist(np.arange(9).reshape(3, 3))
    b = _sci_hdulist(np.arange(9).reshape(3, 3))
    b['SCI'].data[1, 1] += 1
    assert fp.data_hash(a) != fp.data_hash(b)


def test_data_hash_detects_dtype_change():
    a = _sci_hdulist(np.ones((3, 3)))
    b = fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(np.ones((3, 3), dtype='float64'), name='SCI')])
    assert fp.data_hash(a) != fp.data_hash(b)


def test_wcs_hash_detects_raoffset_change():
    a = _sci_hdulist(np.zeros((2, 2)))
    b = _sci_hdulist(np.zeros((2, 2)))
    a['SCI'].header['RAOFFSET'] = 0.0
    b['SCI'].header['RAOFFSET'] = 0.001
    assert fp.wcs_hash(a['SCI'].header) != fp.wcs_hash(b['SCI'].header)


def test_wcs_hash_ignores_comment_and_order():
    h1 = _sci_hdulist(np.zeros((2, 2)))['SCI'].header
    h2 = fits.Header()
    # Same WCS values, inserted in a different order, with a comment.
    for k in ('CTYPE2', 'CD2_2', 'CTYPE1', 'CD1_1', 'CRPIX2', 'CRPIX1',
              'CRVAL2', 'CRVAL1'):
        h2[k] = h1[k]
    h2.comments['CRVAL1'] = 'a comment that must not matter'
    assert fp.wcs_hash(h1) == fp.wcs_hash(h2)


def test_meta_hash_excludes_wcs_and_volatile():
    h = fits.Header()
    h['TELESCOP'] = 'JWST'
    h['CRVAL1'] = 266.4          # WCS -> excluded
    h['GCPIPEV'] = 'abc-dirty'   # volatile provenance -> excluded
    h['GCTAG'] = '2026-07-16_PR1'
    base = fp.meta_hash(h)
    # Changing a WCS or volatile card leaves meta_hash unchanged.
    h['CRVAL1'] = 0.0
    h['GCPIPEV'] = 'def'
    assert fp.meta_hash(h) == base
    # Changing a real meta card DOES change it.
    h['TELESCOP'] = 'HST'
    assert fp.meta_hash(h) != base


def test_facet_hashes_structure(tmp_path):
    p = str(tmp_path / 'x_i2d.fits')
    _sci_hdulist(np.arange(4).reshape(2, 2),
                 extra_meta={'TELESCOP': 'JWST'}).writeto(p)
    f = fp.facet_hashes(p)
    assert set(f) == {'data', 'wcs', 'meta'}
    assert all(isinstance(v, str) and len(v) == 64 for v in f.values())


def test_table_hash_excludes_reprojected_cols():
    from astropy.table import Table
    t = Table({'x_fit': [1.0, 2.0], 'ra': [10.0, 11.0], 'dec': [-1.0, -2.0]})
    t2 = Table({'x_fit': [1.0, 2.0], 'ra': [99.0, 98.0], 'dec': [5.0, 6.0]})
    # RA/Dec differ but are excluded -> hash matches on the invariant x_fit.
    assert (fp.table_hash(t, exclude_cols=('ra', 'dec'))
            == fp.table_hash(t2, exclude_cols=('ra', 'dec')))
    # Without excluding, they differ.
    assert fp.table_hash(t) != fp.table_hash(t2)


def test_params_hash_ignores_volatile():
    class Opts:
        pass
    a, b = Opts(), Opts()
    for o in (a, b):
        o.threshold = 5.0
        o.filternames = ['F200W']
    a.ncores = 8
    b.ncores = 16          # volatile -> ignored
    a.job_name = 'brick-a'
    b.job_name = 'brick-b'  # volatile -> ignored
    assert fp.params_hash(a) == fp.params_hash(b)
    b.threshold = 4.0       # real param -> changes hash
    assert fp.params_hash(a) != fp.params_hash(b)


def test_code_hash_unknown_stage():
    with pytest.raises(KeyError):
        fp.code_hash('nonesuch', repo_dir='/tmp')


def test_facet_meta_detects_extension_header_change(tmp_path):
    # A non-WCS metadata change in the SCI (extension) header must move meta.
    a = str(tmp_path / 'a_i2d.fits')
    b = str(tmp_path / 'b_i2d.fits')
    ha = _sci_hdulist(np.zeros((2, 2)))
    hb = _sci_hdulist(np.zeros((2, 2)))
    ha['SCI'].header['BUNIT'] = 'MJy/sr'
    hb['SCI'].header['BUNIT'] = 'DN/s'
    ha.writeto(a)
    hb.writeto(b)
    fa, fb = fp.facet_hashes(a), fp.facet_hashes(b)
    assert fa['data'] == fb['data']   # pixels identical
    assert fa['wcs'] == fb['wcs']     # WCS identical
    assert fa['meta'] != fb['meta']   # extension-header meta caught


def test_blob_fallback_replicates_git_object_id(tmp_path):
    import hashlib
    import subprocess
    content = b'print(1)\n'
    p = tmp_path / 'x.py'
    p.write_bytes(content)
    repo = str(tmp_path)
    subprocess.check_call(['git', '-C', repo, 'init', '-q'],
                          stdout=subprocess.DEVNULL)
    git_id = subprocess.check_output(
        ['git', '-C', repo, 'hash-object', str(p)], text=True).strip()
    # The fallback framing used in fingerprint._blob_hash must match git exactly.
    h = hashlib.sha1()
    h.update(b'blob %d\0' % len(content))
    h.update(content)
    assert h.hexdigest() == git_id
