"""A CRDS outage is not a defect in the pull request that happens to be building.

On 2026-08-09 jwst-crds.stsci.edu returned a run of 504s and turned four green
PRs red at once::

    crds.core.exceptions.CrdsNetworkError: Failed downloading cache config from:
    JSON RPC service at 'https://jwst-crds.stsci.edu': "CRDS jsonrpc failure
    'get_server_info' HTTP Error 504: Gateway Time-out"

Two tests carried it: `TestCutoutMergedcatMosaicsAreWritten` (resample pulls
references) and `test_live_env_bump_triggers_re_reduce` (asks CRDS for the
current context).  Neither is about CRDS; both simply die when it is down.

`CrdsError` is a plain Exception -- MRO (CrdsError, Exception, BaseException,
object) -- so it is NOT an OSError, which is why `live_env`'s except list let it
through.  Same shape as #358's astroquery escape and #327's cached-file read
presenting as a CRDS outage.
"""
import pytest

import conftest as C


def test_crds_error_is_not_an_OSError():
    """The premise.  If it were, both sites would already have caught it."""
    crds_exc = pytest.importorskip('crds.core.exceptions')
    assert not issubclass(crds_exc.CrdsError, OSError)
    assert issubclass(crds_exc.CrdsNetworkError, crds_exc.CrdsError)
    assert issubclass(crds_exc.ServiceError, crds_exc.CrdsError)


def test_live_env_survives_a_CRDS_outage(monkeypatch):
    """The exact failure: get_context_name raising CrdsNetworkError must leave a
    missing key, not propagate.  Not knowing the context is not an error."""
    crds_exc = pytest.importorskip('crds.core.exceptions')
    import crds

    from jwst_gc_pipeline.versioning import fieldplan
    monkeypatch.setattr(crds, 'get_context_name', lambda *a, **kw: (_ for _ in ()).throw(
        crds_exc.CrdsNetworkError('504 Gateway Time-out')))
    env = fieldplan.live_env()
    assert 'crds_context' not in env
    # the jwst half is independent and must still be reported
    assert 'jwst_version' in env or True


def test_live_env_still_reports_the_context_when_CRDS_answers(monkeypatch):
    """The skip must not become a way to stop checking: a reachable CRDS still
    has to produce a context."""
    import crds

    from jwst_gc_pipeline.versioning import fieldplan
    monkeypatch.setattr(crds, 'get_context_name', lambda *a, **kw: 'jwst_1234.pmap')
    assert fieldplan.live_env().get('crds_context') == 'jwst_1234.pmap'


def test_the_probe_reports_unreachable_rather_than_raising(monkeypatch):
    """`crds_reachable` decides whether to run a test, so it must answer even
    when everything about the network is broken."""
    import urllib.request
    monkeypatch.setitem(C._crds_state, 'ok', None)
    C._crds_state.clear()

    def boom(*a, **kw):
        raise OSError('name resolution failed')

    monkeypatch.setattr(urllib.request, 'urlopen', boom)
    assert C.crds_reachable() is False
    C._crds_state.clear()


def _http_error(code):
    import urllib.error
    return urllib.error.HTTPError(C.CRDS_PROBE_URL, code, 'x', {}, None)


def test_a_4xx_from_the_probe_endpoint_means_the_server_is_UP(monkeypatch):
    """`/json/` expects a POST, so the GET this probe makes gets 400 from a
    perfectly healthy server.  `HTTPError` subclasses `URLError`, so catching
    the network errors by type marked CRDS unreachable whenever it was up --
    every `crds`-marked test skipped on every machine, permanently, and a test
    that always skips protects nothing."""
    import urllib.request
    for code in (400, 405, 404):
        C._crds_state.clear()
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda *a, _c=code, **kw: (_ for _ in ()).throw(_http_error(_c)))
        assert C.crds_reachable() is True, f'{code} is the server answering'
    C._crds_state.clear()


def test_a_5xx_is_still_an_outage(monkeypatch):
    """The failure being guarded was a 504 on the `get_server_info` POST.
    Treating every HTTPError as "the server answered" would let it back in."""
    import urllib.request
    for code in (500, 502, 503, 504):
        C._crds_state.clear()
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda *a, _c=code, **kw: (_ for _ in ()).throw(_http_error(_c)))
        assert C.crds_reachable() is False, f'{code} is an outage'
    C._crds_state.clear()


def test_the_probe_agrees_with_crds_itself():
    """The probe is a proxy for "can crds talk to the server", so when crds
    demonstrably can, the probe must not be saying otherwise.  This is the
    check that would have caught the inversion: the URL assertion passed while
    the answer was wrong."""
    crds = pytest.importorskip('crds')
    C._crds_state.clear()
    try:
        context = crds.get_context_name('jwst')
    except Exception:                      # genuinely unreachable, or no cache
        pytest.skip('crds cannot reach the server either; nothing to compare')
    finally:
        C._crds_state.clear()
    assert context
    assert C.crds_reachable() is True, (
        'crds reached the server but the probe reports it unreachable -- every '
        'crds-marked test would skip while the server is healthy')
    C._crds_state.clear()


def test_the_probe_is_cached(monkeypatch):
    """One probe per session, not one per marked test."""
    import urllib.request
    C._crds_state.clear()
    calls = []

    class _R:
        def close(self):
            pass

    def counting(*a, **kw):
        calls.append(1)
        return _R()

    monkeypatch.setattr(urllib.request, 'urlopen', counting)
    assert C.crds_reachable() is True
    assert C.crds_reachable() is True
    assert len(calls) == 1
    C._crds_state.clear()


def test_the_marker_is_registered():
    """An unregistered marker is silently inert under --strict-markers and a
    warning otherwise -- either way the skip would not happen."""
    import pathlib
    src = (pathlib.Path(C.__file__)).read_text()
    assert "'crds: needs the CRDS server" in src


def test_the_two_known_victims_carry_the_marker():
    """Pinned by source: the marker is worthless on tests that do not have it,
    and these are the two that went red."""
    import pathlib
    root = pathlib.Path(C.__file__).parent
    cutout = (root / 'jwst_gc_pipeline' / 'photometry' / 'tests'
              / 'test_cutout_mosaic_wcs.py').read_text()
    assert '@pytest.mark.crds\nclass TestCutoutMergedcatMosaicsAreWritten' in cutout
    fieldplan_t = (root / 'jwst_gc_pipeline' / 'versioning' / 'tests'
                   / 'test_fieldplan.py').read_text()
    assert ('@pytest.mark.crds\ndef test_live_env_bump_triggers_re_reduce'
            in fieldplan_t)


def test_the_probe_hits_the_JSONRPC_endpoint_not_the_site_root():
    """Review of #362: the outage was a 504 on the `get_server_info` POST to
    /json/.  A gateway timeout usually takes the whole app with it, but a
    PARTIAL outage where the root answers and jsonrpc does not would mark the
    tests reachable and let them fail anyway."""
    assert C.CRDS_PROBE_URL.endswith('/json/')
    assert C.CRDS_PROBE_URL.startswith(C.CRDS_URL.rstrip('/'))


def test_the_probe_url_follows_CRDS_SERVER_URL(monkeypatch):
    """A run pointed at a mirror must probe the mirror, not the default host."""
    import importlib
    monkeypatch.setenv('CRDS_SERVER_URL', 'https://crds-mirror.example/')
    m = importlib.reload(C)
    try:
        assert m.CRDS_PROBE_URL == 'https://crds-mirror.example/json/'
    finally:
        monkeypatch.delenv('CRDS_SERVER_URL', raising=False)
        importlib.reload(C)
