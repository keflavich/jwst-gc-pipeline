"""Tests for CMZ-wide catalog assembly."""
import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.cmz import catalog_assembly as CA


def _field(ra, dec, f212=None, f405=None, meta_tag=None):
    n = len(ra)
    t = Table()
    t['ra'] = np.asarray(ra, float)
    t['dec'] = np.asarray(dec, float)
    t['F212N_flux_jy'] = np.asarray(f212 if f212 is not None else [1.0] * n, float)
    if f405 is not None:
        t['F405N_flux_jy'] = np.asarray(f405, float)
    t['qfit'] = np.full(n, 0.1)
    if meta_tag:
        t.meta['GCTAG'] = meta_tag
    return t


def test_load_stamps_provenance_and_tag(tmp_path):
    p = str(tmp_path / 'brick.fits')
    _field([266.4, 266.5], [-28.9, -28.8], meta_tag='2026-07-17_PR120').write(p)
    t = CA.load_field_catalog(p, 'brick', program='2221', obsid='001')
    assert set(CA.PROV_COLS) <= set(t.colnames)
    assert list(t['cmz_field']) == ['brick', 'brick']
    assert t['cmz_program'][0] == '2221'
    assert t['cmz_src_tag'][0] == '2026-07-17_PR120'   # read from meta GCTAG


def test_assemble_vstack_and_coverage():
    ta = _field([266.40], [-28.90], f212=[1.0], f405=[2.0])
    ta['cmz_field'] = ['brick']
    tb = _field([266.60], [-28.70], f212=[3.0])  # no F405 column
    tb['cmz_field'] = ['sgrc']
    out = CA.assemble([ta, tb], dedup_radius_arcsec=0.2)
    assert len(out) == 2
    # outer join created F405N column; sgrc row masked there
    assert 'F405N_flux_jy' in out.colnames
    # coverage: brick row has 2 bands, sgrc row has 1
    cov = {f: n for f, n in zip(out['cmz_field'], out['cmz_n_bands'])}
    assert cov['brick'] == 2 and cov['sgrc'] == 1


def test_cross_field_dedup_keeps_better_coverage_and_records_also_in():
    # same star seen in two overlapping fields; brick has 2 bands, sgrc has 1
    brick = _field([266.5000], [-28.8000], f212=[1.0], f405=[2.0])
    brick['cmz_field'] = ['brick']
    sgrc = _field([266.50001], [-28.80000], f212=[1.1])  # ~0.03" away
    sgrc['cmz_field'] = ['sgrc']
    out = CA.assemble([brick, sgrc], dedup_radius_arcsec=0.2)
    assert len(out) == 1                    # duplicate collapsed
    assert out['cmz_field'][0] == 'brick'   # kept the higher-coverage detection
    assert out['cmz_also_in'][0] == 'sgrc'  # provenance of the dropped detection kept


def test_non_overlap_sources_have_empty_also_in():
    a = _field([266.40], [-28.90], f212=[1.0]); a['cmz_field'] = ['brick']
    b = _field([267.00], [-28.50], f212=[2.0]); b['cmz_field'] = ['sgrc']
    out = CA.assemble([a, b], dedup_radius_arcsec=0.2)
    assert list(out['cmz_also_in']) == ['', '']


def test_same_field_close_pair_not_deduped():
    # two close sources in the SAME field are a real blend -> both kept
    t = _field([266.5000, 266.50001], [-28.8000, -28.80000], f212=[1.0, 1.1])
    t['cmz_field'] = ['brick', 'brick']
    out = CA.assemble([t], dedup_radius_arcsec=0.2)
    assert len(out) == 2


def test_write_outputs_fits_ecsv_roundtrip(tmp_path):
    t = _field([266.4, 266.5], [-28.9, -28.8], f212=[1.0, 2.0])
    t['cmz_field'] = ['brick', 'brick']
    out = CA.assemble([t])
    stem = str(tmp_path / 'cmz_cat')
    written = CA.write_outputs(out, stem, formats=('fits', 'ecsv'))
    assert len(written) == 2
    back = Table.read(stem + '.fits')
    assert len(back) == 2 and 'F212N_flux_jy' in back.colnames


def test_write_parquet_when_pyarrow_present(tmp_path):
    import pytest
    pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    t = _field([266.4, 266.5], [-28.9, -28.8], f212=[1.0, 2.0])
    t['cmz_field'] = ['brick', 'brick']
    out = CA.assemble([t])
    stem = str(tmp_path / 'cmz_cat')
    written = CA.write_outputs(out, stem, formats=('parquet',))
    assert written == [stem + '.parquet']
    tbl = pq.read_table(stem + '.parquet')   # read via pyarrow (no pandas dep)
    assert tbl.num_rows == 2 and 'F212N_flux_jy' in tbl.column_names


def test_missing_coords_raises(tmp_path):
    import pytest
    p = str(tmp_path / 'bad.fits')
    Table({'flux': [1.0]}).write(p)
    with pytest.raises(KeyError):
        CA.load_field_catalog(p, 'x')
