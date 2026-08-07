"""A cached CRDS reference must be read from disk, never re-fetched.

The reduction drivers used to `asdf.open` the tweakreg pars reference straight
off https://jwst-crds.stsci.edu/unchecked_get/..., which has no retry (it goes
through fsspec/aiohttp) and made every reduce depend on STScI being reachable at
that moment.  A single 504 killed 4 of 8 sgrc filters twice over on 2026-08-07
while the reference sat checksum-correct in the cache (issue #327).
"""
import os

import pytest

from jwst_gc_pipeline.reduction import crds_cache


def _fake_asdf(monkeypatch):
    opened = []
    monkeypatch.setattr(crds_cache.asdf, 'open',
                        lambda path, *a, **k: opened.append(path) or path)
    return opened


def test_cached_reference_is_opened_from_disk(tmp_path, monkeypatch):
    ref = tmp_path / 'references' / 'jwst' / 'nircam'
    ref.mkdir(parents=True)
    (ref / 'jwst_nircam_pars-tweakregstep_0048.asdf').write_bytes(b'x')
    opened = _fake_asdf(monkeypatch)

    crds_cache.open_crds_reference(str(tmp_path), 'nircam',
                                   'jwst_nircam_pars-tweakregstep_0048.asdf')

    assert len(opened) == 1
    assert not opened[0].startswith('http'), (
        f'a cached reference must not be fetched over the network: {opened[0]}')
    assert opened[0] == str(ref / 'jwst_nircam_pars-tweakregstep_0048.asdf')


def test_uncached_reference_falls_back_to_the_server(tmp_path, monkeypatch):
    opened = _fake_asdf(monkeypatch)

    crds_cache.open_crds_reference(str(tmp_path), 'nircam', 'missing_0000.asdf')

    assert opened == [f'{crds_cache.CRDS_REFERENCE_URL}/missing_0000.asdf']


@pytest.mark.parametrize('instrument', ['nircam', 'miri', 'niriss'])
def test_cache_path_matches_the_crds_layout(instrument):
    assert crds_cache.cached_reference_path('/c', instrument, 'r.asdf') == \
        os.path.join('/c', 'references', 'jwst', instrument, 'r.asdf')


def test_no_driver_fetches_a_reference_directly():
    """Grep-guard: the drivers must go through open_crds_reference()."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(crds_cache.__file__)))
    drivers = ['reduction/PipelineRerunNIRCAM-LONG.py',
               'reduction/PipelineMIRI.py',
               'reduction/PipelineRerunNIRISS.py']
    for rel in drivers:
        with open(os.path.join(here, rel)) as fh:
            text = fh.read()
        assert 'unchecked_get' not in text, (
            f'{rel} fetches a CRDS reference directly; use '
            'crds_cache.open_crds_reference so the local cache is honoured')
