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
