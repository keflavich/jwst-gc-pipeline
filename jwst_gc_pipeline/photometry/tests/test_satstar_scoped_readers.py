"""Readers of the consolidated satstar cache follow the observation scope (#925).

Once the merge writes ``{filt}_oNNN_consolidated_satstar_catalog.fits`` for a
scoped observation, the program-wide ``{filt}_consolidated_satstar_catalog.fits``
left on disk from earlier runs is stale.  A reader still on the old name gets
that stale pool instead of an error, so each reader is pinned here.
"""
import importlib.util
import os

import pytest

from jwst_gc_pipeline.monitoring import scan
from jwst_gc_pipeline.photometry import aperture_photometry as ap
from jwst_gc_pipeline.photometry.merge_catalogs import (
    consolidated_satstar_cache_path, satstar_obs_scope)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def _fresh_listing():
    scan.clear_cache()
    yield
    scan.clear_cache()


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')


def _load_purge():
    spec = importlib.util.spec_from_file_location(
        'purge_satstar_caches',
        os.path.join(REPO, 'scripts', 'reduction', 'purge_satstar_caches.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('proposal,obsid', [('10678', '132'), ('2221', '001'),
                                            ('10678', '*'), ('10678', None)])
def test_scan_scope_matches_merge(proposal, obsid):
    want = '' if obsid in (None, '*') else satstar_obs_scope(proposal, obsid)
    assert scan._satstar_scope(proposal, obsid) == want


def test_scan_counts_scoped_cache_not_pooled(tmp_path):
    cat = tmp_path / 'catalogs'
    _touch(cat / 'f480m_consolidated_satstar_catalog.fits')
    rows = scan._catalog_stages(str(tmp_path), 'F480M', False, '10678', '132')
    assert rows['satstar']['n'] == 0
    _touch(cat / 'f480m_o132_consolidated_satstar_catalog.fits')
    scan.clear_cache()
    rows = scan._catalog_stages(str(tmp_path), 'F480M', False, '10678', '132')
    assert rows['satstar']['n'] == 1


def test_scan_unscoped_field_unchanged(tmp_path):
    _touch(tmp_path / 'catalogs' / 'f182m_consolidated_satstar_catalog.fits')
    rows = scan._catalog_stages(str(tmp_path), 'F182M', False, '2221', '001')
    assert rows['satstar']['n'] == 1


def test_purge_matches_scoped_caches(tmp_path):
    purge = _load_purge()
    cat = tmp_path / 'gc-treasury' / 'catalogs'
    for name in ('f480m_consolidated_satstar_catalog.fits',
                 'f480m_o132_consolidated_satstar_catalog.fits',
                 'f480m_o127_consolidated_satstar_catalog.fits',
                 'f212n_o132_consolidated_satstar_catalog.fits'):
        _touch(cat / name)
    (tmp_path / 'gc-treasury' / 'F480M' / 'pipeline').mkdir(parents=True)
    n = purge.purge(str(tmp_path), 'gc-treasury', ['F480M'], execute=True)
    assert n == 3
    left = sorted(p.name for p in cat.iterdir())
    assert 'f212n_o132_consolidated_satstar_catalog.fits' in left
    assert not [p for p in left if p.startswith('f480m')
                and not p.endswith(purge.SUFFIX)]


def test_aperture_cli_uses_scoped_cache(tmp_path):
    want = consolidated_satstar_cache_path(str(tmp_path), 'F480M', '_o132')
    assert ap._default_satstar_cache(str(tmp_path), 'F480M', 'o132') == want
    assert ap._default_satstar_cache(str(tmp_path), 'F480M', '132') == want


def test_aperture_cli_refuses_pool_beside_scoped(tmp_path):
    cat = tmp_path / 'catalogs'
    _touch(cat / 'f480m_consolidated_satstar_catalog.fits')
    _touch(cat / 'f480m_o132_consolidated_satstar_catalog.fits')
    with pytest.raises(ValueError, match='--obs'):
        ap._default_satstar_cache(str(tmp_path), 'F480M')


def test_aperture_cli_unscoped_field_unchanged(tmp_path):
    _touch(tmp_path / 'catalogs' / 'f182m_consolidated_satstar_catalog.fits')
    assert ap._default_satstar_cache(str(tmp_path), 'F182M') == (
        consolidated_satstar_cache_path(str(tmp_path), 'F182M'))


def _write_rej(path, ra_deg):
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u
    t = Table()
    t['skycoord_fit'] = SkyCoord([ra_deg, ra_deg] * u.deg, [0.0, 0.0] * u.deg)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.write(str(path), overwrite=True)


def test_rejected_loader_scopes_to_observation(tmp_path):
    """The gate-rejected channel is obs-scoped like the accepted one (#931
    follow-up): a scoped merge reads only its own observation's rejected files,
    not sibling observations' on the shared tree."""
    from astropy.coordinates import SkyCoord
    from jwst_gc_pipeline.photometry import merge_catalogs as MC
    assert MC.satstar_obs_scope('10678', '132')          # guards the premise
    pipe = tmp_path / 'F480M' / 'pipeline'
    _write_rej(pipe / 'jw10678132001_x_m8_satstar_rejected.fits', 10.0)
    _write_rej(pipe / 'jw10678127001_x_m8_satstar_rejected.fits', 20.0)
    out = MC.load_rejected_satstar_catalog('F480M', basepath=str(tmp_path),
                                           proposal_id='10678', field='132')
    assert out is not None and len(out) == 2
    assert {round(r) for r in SkyCoord(out['skycoord_fit']).ra.deg} == {10}


def test_rejected_loader_unscoped_pools_all(tmp_path):
    """Without a scope (single-observation field, or omitted) every rejected
    file for the band is pooled -- unchanged behaviour."""
    from jwst_gc_pipeline.photometry import merge_catalogs as MC
    pipe = tmp_path / 'F480M' / 'pipeline'
    _write_rej(pipe / 'jw10678132001_x_m8_satstar_rejected.fits', 10.0)
    _write_rej(pipe / 'jw10678127001_x_m8_satstar_rejected.fits', 20.0)
    both = MC.load_rejected_satstar_catalog('F480M', basepath=str(tmp_path))
    assert both is not None and len(both) == 4
