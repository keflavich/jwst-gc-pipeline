"""Tests for CMZ-wide catalog assembly."""
import numpy as np
from astropy import units as u
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


# ---------------------------------------------------------------------------
# Vectorized _dedup_cross_field: equivalence against the ORIGINAL pure-Python
# union-find + np.unique/np.where algorithm it replaced (O(N^2), correct but
# unusable at 10678 scale).  Kept here, frozen, purely as a reference oracle.
# ---------------------------------------------------------------------------
def _dedup_cross_field_reference(table, radius_arcsec, coverage_cols,
                                 field_col='cmz_field'):
    """The pre-vectorization algorithm (do not use in production)."""
    n = len(table)
    keep = np.ones(n, dtype=bool)
    also_in = np.array([''] * n, dtype=object)
    if n < 2 or radius_arcsec <= 0:
        return keep, also_in
    sc = CA._skycoord(table)
    fields = np.asarray(table[field_col]).astype(str)
    cover = CA._coverage_count(table, coverage_cols)
    i1, i2, _, _ = sc.search_around_sky(sc, radius_arcsec * u.arcsec)

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    linked = False
    for a, b in zip(np.asarray(i1), np.asarray(i2)):
        if a < b and fields[a] != fields[b]:
            ra_, rb_ = find(int(a)), find(int(b))
            if ra_ != rb_:
                parent[max(ra_, rb_)] = min(ra_, rb_)
                linked = True
    if not linked:
        return keep, also_in
    roots = np.array([find(i) for i in range(n)])
    for root in np.unique(roots):
        members = np.where(roots == root)[0]
        if members.size < 2:
            continue
        best = members[np.lexsort((members, -cover[members]))][0]
        other = sorted({str(fields[m]) for m in members if m != best})
        also_in[best] = ','.join(other)
        for m in members:
            if m != best:
                keep[m] = False
    return keep, also_in


def _random_multi_group_table(rng, n, n_groups, cluster_scale_arcsec):
    """Random point field split into ``n_groups`` named groups, with enough
    spatial clustering (points drawn in small clumps) that cross-group
    'duplicates' within ``radius_arcsec`` actually occur."""
    ra0, dec0 = 266.5, -29.0
    n_clumps = max(1, n // 4)
    clump_ra = ra0 + rng.uniform(-0.01, 0.01, n_clumps)
    clump_dec = dec0 + rng.uniform(-0.01, 0.01, n_clumps)
    clump_idx = rng.integers(0, n_clumps, n)
    jitter = cluster_scale_arcsec / 3600.0
    ra = clump_ra[clump_idx] + rng.uniform(-jitter, jitter, n)
    dec = clump_dec[clump_idx] + rng.uniform(-jitter, jitter, n)
    t = Table()
    t['ra'] = ra
    t['dec'] = dec
    t['F212N_flux_jy'] = rng.uniform(0.5, 2.0, n)
    has_f405 = rng.random(n) > 0.4
    f405 = np.where(has_f405, rng.uniform(0.5, 2.0, n), np.nan)
    t['F405N_flux_jy'] = f405
    groups = rng.integers(0, n_groups, n)
    t['cmz_field'] = np.array([f'field{g}' for g in groups])
    return t


def test_vectorized_dedup_matches_reference_algorithm_random():
    cov = ['F212N_flux_jy', 'F405N_flux_jy']
    for seed in range(15):
        rng = np.random.default_rng(seed)
        t = _random_multi_group_table(rng, n=80, n_groups=4,
                                      cluster_scale_arcsec=0.15)
        keep_new, also_new = CA._dedup_cross_field(t, 0.2, cov)
        keep_ref, also_ref = _dedup_cross_field_reference(t, 0.2, cov)
        assert np.array_equal(keep_new, keep_ref), f'seed={seed} keep mismatch'
        assert list(also_new) == list(also_ref), f'seed={seed} also_in mismatch'


def test_vectorized_dedup_custom_field_col():
    # same scenario as test_cross_field_dedup_... but grouped by a different
    # column name entirely (10678 use case: group by TILE/obsid, not 'cmz_field')
    brick = _field([266.5000], [-28.8000], f212=[1.0], f405=[2.0])
    brick['obsid'] = ['040']
    sgrc = _field([266.50001], [-28.80000], f212=[1.1])
    sgrc['obsid'] = ['041']
    from astropy.table import vstack
    t = vstack([brick, sgrc], join_type='outer', metadata_conflicts='silent')
    keep, also_in = CA._dedup_cross_field(
        t, 0.2, ['F212N_flux_jy', 'F405N_flux_jy'], field_col='obsid')
    assert list(keep) == [True, False]
    assert also_in[0] == '041'


def test_vectorized_dedup_within_group_pairs_never_merge():
    # two close sources in the SAME tile must never merge, however many OTHER
    # tiles' sources are also nearby (tests the vectorized cross-filter, not
    # just the n=2 case the old test covers).
    rng = np.random.default_rng(0)
    ra = [266.50000, 266.50001] + list(266.5 + rng.uniform(-0.01, 0.01, 30))
    dec = [-28.80000, -28.80000] + list(-28.8 + rng.uniform(-0.01, 0.01, 30))
    t = _field(ra, dec, f212=[1.0] * 32)
    t['obsid'] = ['tileA', 'tileA'] + list(
        rng.choice(['tileB', 'tileC'], 30))
    keep, _ = CA._dedup_cross_field(t, 0.2, ['F212N_flux_jy'], field_col='obsid')
    assert keep[0] and keep[1]   # the same-tile close pair both survive


def test_vectorized_dedup_custom_rank_cols():
    # winner rule overridden to prefer the LARGER 'priority' value (encoded as
    # ascending == preferred via a precomputed negated column), not coverage.
    t = _field([266.50000, 266.50001], [-28.80000, -28.80000], f212=[1.0, 1.0])
    t['obsid'] = ['tileA', 'tileB']
    t['neg_priority'] = [-1.0, -5.0]   # row 1 has the "better" (larger) priority
    keep, also_in = CA._dedup_cross_field(
        t, 0.2, ['F212N_flux_jy'], field_col='obsid', rank_cols=('neg_priority',))
    assert list(keep) == [False, True]
    assert also_in[1] == 'tileA'
