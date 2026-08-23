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


class _Flaky:
    """``asdf.open`` that raises ``exc`` for the first ``n_failures`` calls."""

    def __init__(self, n_failures, exc):
        self.n_failures = n_failures
        self.exc = exc
        self.calls = []

    def __call__(self, path, *a, **k):
        self.calls.append(path)
        if len(self.calls) <= self.n_failures:
            raise self.exc
        return path


def test_a_transient_fetch_failure_is_retried(monkeypatch):
    """The 504 that killed 4 of 8 sgrc filters must not kill the reduce.

    fsspec converts the gateway timeout into ``FileNotFoundError``, so that is
    the exception the retry has to survive.
    """
    flaky = _Flaky(2, FileNotFoundError('504 dressed as a missing file'))
    monkeypatch.setattr(crds_cache.asdf, 'open', flaky)
    slept = []

    got = crds_cache.fetch_crds_reference(
        'https://example.invalid/r.asdf', sleep=slept.append)

    assert got == 'https://example.invalid/r.asdf'
    assert len(flaky.calls) == 3, (
        f'expected two failures then a success, got {len(flaky.calls)} attempts')
    assert slept == [5.0, 10.0], (
        f'the delay must grow between attempts, got {slept}')


def test_the_uncached_driver_path_retries_too(tmp_path, monkeypatch):
    """The retry has to be on the path the DRIVERS take, not only the helper.

    Measured 2026-08-23: MIRI F770W resolves to
    jwst_miri_pars-tweakregstep_0020.asdf, which no CRDS tree a field points at
    holds, so this fallback is live rather than hypothetical.
    """
    flaky = _Flaky(1, FileNotFoundError('504'))
    monkeypatch.setattr(crds_cache.asdf, 'open', flaky)
    monkeypatch.setattr(crds_cache.time, 'sleep', lambda s: None)

    got = crds_cache.open_crds_reference(
        str(tmp_path), 'miri', 'jwst_miri_pars-tweakregstep_0020.asdf')

    assert len(flaky.calls) == 2
    assert got.endswith('jwst_miri_pars-tweakregstep_0020.asdf')
    assert got.startswith('http')


def test_retries_are_bounded_and_the_last_error_is_re_raised(monkeypatch):
    """A reference that genuinely is not there must still fail, with its own error."""
    flaky = _Flaky(99, FileNotFoundError('really absent'))
    monkeypatch.setattr(crds_cache.asdf, 'open', flaky)

    with pytest.raises(FileNotFoundError, match='really absent'):
        crds_cache.fetch_crds_reference('https://example.invalid/r.asdf',
                                        sleep=lambda s: None)

    assert len(flaky.calls) == crds_cache.DEFAULT_FETCH_RETRIES


def test_the_attempt_count_comes_from_the_env_var_the_code_reads(monkeypatch):
    flaky = _Flaky(99, FileNotFoundError('504'))
    monkeypatch.setattr(crds_cache.asdf, 'open', flaky)
    monkeypatch.setenv(crds_cache.CRDS_FETCH_RETRIES_ENV, '2')
    monkeypatch.setenv(crds_cache.CRDS_FETCH_DELAY_ENV, '0')

    with pytest.raises(FileNotFoundError):
        crds_cache.fetch_crds_reference('https://example.invalid/r.asdf',
                                        sleep=lambda s: None)

    assert len(flaky.calls) == 2


def test_the_crds_client_knobs_do_not_govern_this_path(monkeypatch):
    """#315's knobs govern ``crds.client``; setting them here changed nothing.

    Pinning that keeps anyone from re-diagnosing the sgrc outage as "the retry
    count was too low" when the two settings are unrelated.
    """
    flaky = _Flaky(99, FileNotFoundError('504'))
    monkeypatch.setattr(crds_cache.asdf, 'open', flaky)
    monkeypatch.setenv('CRDS_CLIENT_RETRY_COUNT', '99')
    monkeypatch.delenv(crds_cache.CRDS_FETCH_RETRIES_ENV, raising=False)

    with pytest.raises(FileNotFoundError):
        crds_cache.fetch_crds_reference('https://example.invalid/r.asdf',
                                        sleep=lambda s: None)

    assert len(flaky.calls) == crds_cache.DEFAULT_FETCH_RETRIES


def test_an_aiohttp_error_is_retryable_even_though_it_is_not_an_oserror():
    aiohttp = pytest.importorskip('aiohttp')
    assert not issubclass(aiohttp.ClientError, OSError)
    assert aiohttp.ClientError in crds_cache._retryable_exceptions()


def test_a_malformed_retry_setting_is_reported_not_ignored(monkeypatch):
    monkeypatch.setenv(crds_cache.CRDS_FETCH_RETRIES_ENV, 'lots')
    with pytest.raises(ValueError, match=crds_cache.CRDS_FETCH_RETRIES_ENV):
        crds_cache.fetch_crds_reference('https://example.invalid/r.asdf',
                                        sleep=lambda s: None)


def test_a_cached_reference_is_opened_once_with_no_retry(tmp_path, monkeypatch):
    """The cache hit must not pick up the retry ladder: a corrupt local file is
    not a transient outage, and retrying it would sleep 30s before reporting."""
    ref = tmp_path / 'references' / 'jwst' / 'miri'
    ref.mkdir(parents=True)
    (ref / 'r.asdf').write_bytes(b'x')
    flaky = _Flaky(99, OSError('corrupt local file'))
    monkeypatch.setattr(crds_cache.asdf, 'open', flaky)

    with pytest.raises(OSError, match='corrupt local file'):
        crds_cache.open_crds_reference(str(tmp_path), 'miri', 'r.asdf')

    assert len(flaky.calls) == 1


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
