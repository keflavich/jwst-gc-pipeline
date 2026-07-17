"""Tests for upstream-facet resolution + pooling."""
from jwst_gc_pipeline.versioning import upstream as U, prov_sidecar


def test_pool_facets_order_independent():
    a = {'data': 'da', 'wcs': 'wa', 'meta': 'ma'}
    b = {'data': 'db', 'wcs': 'wb', 'meta': 'mb'}
    assert U.pool_facets([a, b]) == U.pool_facets([b, a])


def test_pool_facets_change_sensitive():
    a = {'data': 'da', 'wcs': 'wa', 'meta': 'ma'}
    b = {'data': 'db', 'wcs': 'wb', 'meta': 'mb'}
    base = U.pool_facets([a, b])
    b2 = dict(b, data='dX')
    assert U.pool_facets([a, b2])['data'] != base['data']
    # a facet unaffected by an unrelated member change stays put
    assert U.pool_facets([a, b2])['wcs'] == base['wcs']


def test_pool_facets_none_tolerant():
    a = {'data': 'da', 'wcs': None, 'meta': 'ma'}
    b = {'data': 'db', 'wcs': None, 'meta': 'mb'}
    p = U.pool_facets([a, b])
    assert p['wcs'] is None and p['data'] is not None


def _write_rec(path, stage, facets):
    prov_sidecar.write_sidecar(path, prov_sidecar.build_record(stage, 't', facets))


def test_upstream_single_and_pooled(tmp_path):
    # single parent
    p_m3 = str(tmp_path / 'm3.fits')
    _write_rec(p_m3, 'm3', {'data': 'D3', 'wcs': 'W', 'meta': 'M'})
    up = U.upstream_from_sidecars({'m3': p_m3})
    assert up == {'m3': {'data': 'D3', 'wcs': 'W', 'meta': 'M'}}

    # pooled parent (two filters' m6)
    p_a = str(tmp_path / 'm6_a.fits')
    p_b = str(tmp_path / 'm6_b.fits')
    _write_rec(p_a, 'm6', {'data': 'Da', 'wcs': 'W', 'meta': 'M'})
    _write_rec(p_b, 'm6', {'data': 'Db', 'wcs': 'W', 'meta': 'M'})
    up = U.upstream_from_sidecars({'m6': [p_a, p_b]})
    assert set(up) == {'m6'}
    assert up['m6']['data'] == U.pool_facets(
        [{'data': 'Da', 'wcs': 'W', 'meta': 'M'},
         {'data': 'Db', 'wcs': 'W', 'meta': 'M'}])['data']


def test_upstream_omits_missing_sidecar(tmp_path):
    # a parent whose sidecar does not exist is omitted, not errored
    up = U.upstream_from_sidecars({'m5': str(tmp_path / 'nope.fits')})
    assert up == {}


def test_stage_parents_cover_all_stages():
    from jwst_gc_pipeline.versioning import rerun
    assert set(U.STAGE_PARENTS) == set(rerun.STAGES)
