"""A missing local data file is not a failed download, and must not cost an hour.

gc2211 o050's m12 finalize (38895067) died after 1 h 21 m -- astrometry
checkpoint already PASSED -- with::

    Failed to download PSF after 11 attempts; last error: ValueError: File
    wss_target_phase_fp6.fits, not found under
    /orange/adamginsburg/repos/webbpsf/data/.

Nothing was downloaded and nothing could have been: the file is local stpsf
data, absent from the tree STPSF_PATH points at and present in its sibling
(#346).  Eleven retries with backoff went to a network that was never involved,
and the reported cause was wrong -- the same shape as #327.
"""
import os

import pytest

from jwst_gc_pipeline.photometry import psf_preflight as P

#: The exact message stpsf raised, from the o050 log.
REAL = ("File wss_target_phase_fp6.fits, not found under "
        "/orange/adamginsburg/repos/webbpsf/data/.")


def test_the_o050_error_is_recognised_as_a_missing_LOCAL_file():
    got = P.missing_local_data(ValueError(REAL))
    assert got == ('wss_target_phase_fp6.fits',
                   '/orange/adamginsburg/repos/webbpsf/data/')


@pytest.mark.parametrize('exc', [
    ConnectionError('Connection reset by peer'),
    TimeoutError('Read timed out'),
    ValueError('Filter F277W is not valid for NIRCam'),
    RuntimeError('MAST returned 503'),
    ValueError('detector NRCALONG not found'),
])
def test_a_real_failure_is_NOT_claimed_as_missing_local_data(exc):
    """The classifier decides whether to abandon the retry loop, so a false
    positive would turn a recoverable network blip into a hard stop."""
    assert P.missing_local_data(exc) is None
    assert P.missing_local_data_message(exc) is None


def test_a_generic_not_found_does_not_qualify():
    """Anchored on BOTH halves -- the file and the root it searched -- so
    'not found' from somewhere else is not swept in."""
    assert P.missing_local_data(ValueError('source not found in catalog')) is None


def test_the_message_says_it_is_not_a_download(monkeypatch):
    monkeypatch.setenv('STPSF_PATH', '/orange/adamginsburg/repos/webbpsf/data/')
    msg = P.missing_local_data_message(ValueError(REAL))
    assert 'MISSING LOCAL FILE' in msg
    assert 'not a failed download' in msg
    assert 'retrying cannot fix it' in msg
    assert 'wss_target_phase_fp6.fits' in msg
    assert '/orange/adamginsburg/repos/webbpsf/data' in msg


def test_the_message_points_at_the_SIBLING_tree_that_has_the_file(tmp_path):
    """The whole cost of #346 was reading the wrong cause: the file was sitting
    in the sibling tree, byte-identical, one cp away."""
    root = tmp_path / 'webbpsf' / 'data'
    (root / 'NIRCam' / 'OPD').mkdir(parents=True)
    sib = tmp_path / 'webbpsf' / 'stpsf-data' / 'NIRCam' / 'OPD'
    sib.mkdir(parents=True)
    (sib / 'wss_target_phase_fp6.fits').write_bytes(b'x')
    msg = P.missing_local_data_message(
        ValueError(f'File wss_target_phase_fp6.fits, not found under {root}.'))
    assert 'present in a sibling tree' in msg
    assert str(sib / 'wss_target_phase_fp6.fits') in msg


def test_no_sibling_copy_means_no_misleading_suggestion(tmp_path):
    root = tmp_path / 'webbpsf' / 'data'
    root.mkdir(parents=True)
    (tmp_path / 'webbpsf' / 'stpsf-data').mkdir(parents=True)
    msg = P.missing_local_data_message(
        ValueError(f'File nowhere.fits, not found under {root}.'))
    assert 'MISSING LOCAL FILE' in msg
    assert 'sibling tree' not in msg


def test_a_root_that_does_not_exist_is_survivable():
    msg = P.missing_local_data_message(
        ValueError('File x.fits, not found under /no/such/tree/.'))
    assert msg is not None


# ---------------------------------------------------------------------------
# preflight_psf_data
# ---------------------------------------------------------------------------

class _Inst:
    def __init__(self, exc=None):
        self.exc = exc
        self.loaded = None

    def load_wss_opd_by_date(self, when):
        self.loaded = when
        if self.exc is not None:
            raise self.exc


class _Mod:
    def __init__(self, inst):
        self._inst = inst

    def NIRCam(self):
        return self._inst

    def MIRI(self):
        return self._inst

    def NIRISS(self):
        return self._inst


def _install(monkeypatch, inst):
    import sys
    monkeypatch.setitem(sys.modules, 'stpsf', _Mod(inst))


def test_the_preflight_loads_the_OPD_for_the_RUNS_date(monkeypatch):
    """Which OPD file is needed depends on obsdate, so there is no static
    manifest to check -- doing the lookup is the check."""
    inst = _Inst()
    _install(monkeypatch, inst)
    assert P.preflight_psf_data('NIRCAM', 'F277W', '2022-09-01', verbose=False)
    assert inst.loaded == '2022-09-01T00:00:00'


def test_the_preflight_RAISES_on_the_o050_failure(monkeypatch):
    _install(monkeypatch, _Inst(ValueError(REAL)))
    with pytest.raises(P.PSFDataMissingError) as ex:
        P.preflight_psf_data('NIRCAM', 'F277W', '2022-09-01', verbose=False)
    assert 'wss_target_phase_fp6.fits' in str(ex.value)
    assert 'F277W' in str(ex.value)


@pytest.mark.parametrize('exc', [
    ValueError('some other stpsf complaint'),
    OSError('transient NFS hiccup'),
    KeyError('OPD'),
])
def test_the_preflight_DECLINES_rather_than_failing_a_survivable_run(monkeypatch,
                                                                    exc):
    """A preflight that stops a run for a reason the run itself would have
    survived is worse than no preflight: say nothing, let the real PSF build
    report whatever it finds."""
    _install(monkeypatch, _Inst(exc))
    assert P.preflight_psf_data('NIRCAM', 'F277W', '2022-09-01',
                                verbose=False) is False


def test_an_unknown_instrument_declines(monkeypatch):
    _install(monkeypatch, _Inst())
    assert P.preflight_psf_data('NIRSPEC', 'G140M', '2022-09-01',
                                verbose=False) is False


def test_no_obsdate_declines(monkeypatch):
    _install(monkeypatch, _Inst())
    assert P.preflight_psf_data('NIRCAM', 'F277W', None, verbose=False) is False


def test_stpsf_not_importable_declines(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_psf(name, *a, **kw):
        if name in ('stpsf', 'webbpsf'):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', no_psf)
    assert P.preflight_psf_data('NIRCAM', 'F277W', '2022-09-01',
                                verbose=False) is False


def test_the_retry_loop_asks_the_classifier():
    """Pinned by source: the point of the change is that the generic
    `except Exception` no longer retries a file that is not on this
    filesystem."""
    import inspect

    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as C
    src = inspect.getsource(C.get_psf_model)
    assert 'missing_local_data_message' in src
    assert 'PSFDataMissingError' in src
    # It must come before the GENERIC handler's give-up (`rindex`, not
    # `index`: the first occurrence is the network branch above, which this
    # deliberately does not touch -- a ReadTimeout still retries).
    assert (src.index('missing_local_data_message')
            < src.rindex('Failed to download PSF after'))
    assert src.count('Failed to download PSF after') == 2, (
        'the network retry branch must survive unchanged')


# ---------------------------------------------------------------------------
# The preflight makes a MAST query and downloads an OPD, so it is exposed to
# every failure the network is.  `requests` errors are covered by OSError
# (RequestException derives from IOError); astroquery's RemoteServiceError is
# NOT -- its MRO is (Exception, BaseException, object) -- so a MAST outage
# ESCAPED and killed build_mergedcat_residuals outright, where the same failure
# inside get_psf_model is caught by the retry loop and retried.
# ---------------------------------------------------------------------------

class _RemoteServiceError(Exception):
    """astroquery's shape: a plain Exception, not an OSError."""


def test_astroquerys_error_really_is_not_an_OSError():
    """The premise, pinned -- this is why OSError was not enough."""
    assert not issubclass(_RemoteServiceError, OSError)
    try:
        from astroquery.exceptions import RemoteServiceError
        assert not issubclass(RemoteServiceError, OSError)
    except ImportError:
        pass


def test_astroquerys_error_is_on_the_decline_list():
    try:
        from astroquery.exceptions import RemoteServiceError
    except ImportError:
        pytest.skip('astroquery not installed')
    assert issubclass(RemoteServiceError, P._decline_on())


@pytest.mark.parametrize('exc', [
    RuntimeError('stpsf internals'),
    IndexError('empty OPD table'),
    ImportError('lazy import failed mid-call'),
])
def test_a_MAST_side_failure_DECLINES_instead_of_escaping(monkeypatch, exc):
    """Each of these escaped before.  A preflight that kills a run the run
    itself would have survived is worse than no preflight."""
    _install(monkeypatch, _Inst(exc))
    assert P.preflight_psf_data('NIRCAM', 'F277W', '2022-09-01',
                                verbose=False) is False


def test_a_real_astroquery_error_DECLINES(monkeypatch):
    try:
        from astroquery.exceptions import RemoteServiceError
    except ImportError:
        pytest.skip('astroquery not installed')
    _install(monkeypatch, _Inst(RemoteServiceError('MAST 503')))
    assert P.preflight_psf_data('NIRCAM', 'F277W', '2022-09-01',
                                verbose=False) is False


def test_requests_errors_are_covered_by_OSError():
    """Not by accident -- documented, so nobody removes OSError from the list."""
    try:
        import requests
    except ImportError:
        pytest.skip('requests not installed')
    assert issubclass(requests.exceptions.RequestException, OSError)
    assert issubclass(requests.exceptions.ConnectionError, P._decline_on())
    assert issubclass(requests.exceptions.ReadTimeout, P._decline_on())


def test_the_missing_FILE_error_still_RAISES_through_the_wider_net(monkeypatch):
    """Widening the decline list must not swallow the one condition this exists
    to report."""
    _install(monkeypatch, _Inst(ValueError(REAL)))
    with pytest.raises(P.PSFDataMissingError):
        P.preflight_psf_data('NIRCAM', 'F277W', '2022-09-01', verbose=False)
