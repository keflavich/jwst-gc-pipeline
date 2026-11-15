"""``load_or_make_satstar_catalog`` prints the position QA on every return path (#951).

The QA helper is unit-tested in ``test_satstar_position_qa.py``; these pin the
two call sites in ``crowdsource_catalogs_long``: the cache-hit path (inside
``_on_current_frame``, after the WCS refresh) and the fresh-build return.
The QA is log-only, so each test also checks that the returned table is the
one the path would have returned without it.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL

FRAME = 'jw02221001001_07101_00001_nrca2_destreak_o001_crf.fits'


@pytest.fixture
def frame(tmp_path):
    hdr = fits.Header({'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCA2',
                       'FILTER': 'F182M'})
    fits.HDUList([fits.PrimaryHDU(header=hdr),
                  fits.ImageHDU(data=np.zeros((8, 8), dtype='float32'), name='SCI'),
                  fits.ImageHDU(data=np.zeros((8, 8), dtype='int32'), name='DQ')]
                 ).writeto(tmp_path / FRAME)
    return str(tmp_path / FRAME)


def _catalog(path, marker):
    Table({'x_fit': np.zeros(2), 'y_fit': np.zeros(2),
           'flux_fit': np.ones(2), 'marker': np.full(2, marker)}).write(
               path, overwrite=True)


@pytest.fixture
def qa_calls(monkeypatch):
    """Record QA calls; make the refresh stamp the table so the order shows."""
    calls = []

    def _qa(tbl, frame_path=None, label=''):
        calls.append({'marker': int(tbl['marker'][0]), 'frame_path': frame_path,
                      'label': label})
        return {}

    def _refresh(tbl, frame_path=None, catalog_path=None, **kw):
        tbl = tbl.copy()
        tbl['marker'] = tbl['marker'] + 100
        return tbl, (np.nan, np.nan)

    monkeypatch.setattr(CL, 'log_satstar_position_qa', _qa)
    monkeypatch.setattr(CL, 'refresh_satstar_skycoords', _refresh)
    monkeypatch.setattr(CL, 'finalize_satstar_columns', lambda t: t)
    return calls


def _boom(*a, **kw):
    raise AssertionError('remove_saturated_stars ran on a cache hit')


def test_plain_cache_hit_logs_after_the_wcs_refresh(frame, qa_calls, monkeypatch):
    cat = frame.replace('.fits', '_satstar_catalog.fits')
    _catalog(cat, 1)
    monkeypatch.setattr(CL, 'remove_saturated_stars', _boom)
    out = CL.load_or_make_satstar_catalog(frame, 'unused', overwrite=False)
    assert len(qa_calls) == 1
    assert qa_calls[0]['label'] == cat.rsplit('/', 1)[1]
    assert qa_calls[0]['frame_path'] == frame
    assert qa_calls[0]['marker'] == 101, "QA ran before the anchor was refreshed"
    assert int(out['marker'][0]) == 101


def test_extended_cache_hit_logs_with_the_extended_label(frame, qa_calls, monkeypatch):
    cat = frame.replace('.fits', '_extended_satstar_catalog.fits')
    _catalog(cat, 2)
    monkeypatch.setattr(CL, 'remove_saturated_stars', _boom)
    CL.load_or_make_satstar_catalog(frame, 'unused', overwrite=False)
    assert [c['label'] for c in qa_calls] == [cat.rsplit('/', 1)[1]]


def test_fresh_build_return_logs_the_written_catalog(frame, qa_calls, monkeypatch):
    cat = frame.replace('.fits', '_satstar_catalog.fits')

    def _fit(filename, overwrite=True, file_suffix='', **kw):
        _catalog(cat, 3)
    monkeypatch.setattr(CL, 'remove_saturated_stars', _fit)
    out = CL.load_or_make_satstar_catalog(frame, 'unused', overwrite=True)
    assert len(qa_calls) == 1
    assert qa_calls[0]['label'] == cat.rsplit('/', 1)[1]
    assert qa_calls[0]['frame_path'] == frame
    assert qa_calls[0]['marker'] == 3        # fresh fit: no refresh, no shift
    assert int(out['marker'][0]) == 3


def test_no_catalog_after_the_fit_means_no_qa_line(frame, qa_calls, monkeypatch):
    monkeypatch.setattr(CL, 'remove_saturated_stars', lambda *a, **kw: None)
    assert CL.load_or_make_satstar_catalog(frame, 'unused', overwrite=True) is None
    assert qa_calls == []
