"""SVO filter lookups fall back to the vendored JWST list when SVO is down.

The 10678 o067 latest-versions run lost its m4 finalize to a 60 s connect
timeout on svo2.cab.inta-csic.es after its photometry had finished.  These
tests never touch the network: astroquery's ``SvoFps.get_filter_list`` is
replaced with a fake that either answers or raises.
"""
import io
import os

import numpy as np
import pytest
import requests

pytest.importorskip('astroquery.svo_fps')

from astropy.io.votable import parse_single_table  # noqa: E402
from astroquery.exceptions import TimeoutError as AstroqueryTimeoutError  # noqa: E402
from astroquery.svo_fps import SvoFps as astroquery_svofps  # noqa: E402

from jwst_gc_pipeline import svo_filters  # noqa: E402


class _Calls:
    def __init__(self, result=None, error=None, body=None):
        self.result, self.error, self.body, self.kwargs = result, error, body, []

    def __call__(self, facility, *, instrument=None, **kwargs):
        self.kwargs.append(dict(kwargs, facility=facility, instrument=instrument))
        if self.body is not None:
            # what astroquery does with the response body
            return parse_single_table(io.BytesIO(self.body)).to_table()
        if self.error is not None:
            raise self.error
        return self.result


_HTML_PAGE = b'<html><body>SVO is down for maintenance</body></html>'


@pytest.fixture
def shim(monkeypatch):
    monkeypatch.delenv(svo_filters.OFFLINE_ENV, raising=False)
    return svo_filters._SvoFpsWithFallback()


def test_vendored_list_carries_the_treasury_zero_points():
    table = svo_filters.vendored_filter_list()
    table.add_index('filterID')
    for fid in ('JWST/NIRCam.F212N', 'JWST/NIRCam.F480M', 'JWST/MIRI.F770W'):
        zp = float(table.loc[fid]['ZeroPoint'])
        assert np.isfinite(zp) and zp > 0
    assert str(table['ZeroPoint'].unit) == 'Jy'
    assert str(table['WavelengthEff'].unit) == 'Angstrom'


def test_connect_timeout_falls_back_once_per_process(shim, monkeypatch, capsys):
    fake = _Calls(error=requests.exceptions.ConnectTimeout('svo down'))
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', fake)
    first = shim.get_filter_list('JWST')
    second = shim.get_filter_list('JWST', instrument='NIRCam')
    assert len(fake.kwargs) == 1           # the second call did not wait again
    assert len(first) == len(svo_filters.vendored_filter_list())
    assert set(second['Instrument']) == {'NIRCam'}
    out = capsys.readouterr().out
    assert 'ConnectTimeout' in out and svo_filters.VENDORED_FETCHED in out


@pytest.mark.parametrize('error', [
    requests.exceptions.ReadTimeout('slow'),
    requests.exceptions.HTTPError('503'),
    AstroqueryTimeoutError('timed out'),
    IndexError('No filter found'),
])
def test_other_unreachable_errors_fall_back(shim, monkeypatch, error):
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', _Calls(error=error))
    assert 'JWST/NIRCam.F212N' in list(shim.get_filter_list('JWST')['filterID'])


@pytest.mark.parametrize('body', [_HTML_PAGE, b'', b'Service unavailable'])
def test_unparseable_response_falls_back(shim, monkeypatch, capsys, body):
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', _Calls(body=body))
    assert 'JWST/NIRCam.F212N' in list(shim.get_filter_list('JWST')['filterID'])
    assert 'not a filter table' in capsys.readouterr().out


def test_unparseable_response_is_not_sticky(shim, monkeypatch):
    fake = _Calls(body=_HTML_PAGE)
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', fake)
    shim.get_filter_list('JWST')
    sentinel = object()
    fake.body, fake.result = None, sentinel
    assert shim.get_filter_list('JWST') is sentinel
    assert len(fake.kwargs) == 2


def test_valueerror_outside_the_votable_parser_raises(shim, monkeypatch):
    """A bad argument is a caller bug, not an SVO outage."""
    err = ValueError('Timeout value connect was -1, but it must be an int, '
                     'float or None.')
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', _Calls(error=err))
    with pytest.raises(ValueError, match='Timeout value'):
        shim.get_filter_list('JWST', timeout=-1)


def test_live_answer_wins_and_gets_a_timeout(shim, monkeypatch):
    sentinel = object()
    fake = _Calls(result=sentinel)
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', fake)
    assert shim.get_filter_list('JWST', instrument='MIRI') is sentinel
    assert fake.kwargs[0]['timeout'] == svo_filters.DEFAULT_TIMEOUT_S
    assert fake.kwargs[0]['instrument'] == 'MIRI'


def test_instrument_match_ignores_case(shim, monkeypatch):
    monkeypatch.setenv(svo_filters.OFFLINE_ENV, '1')
    nirspec = shim.get_filter_list('JWST', instrument='NIRSpec')
    assert len(nirspec) > 0 and {str(v).upper() for v in nirspec['Instrument']} == {'NIRSPEC'}


def test_offline_env_skips_the_network(shim, monkeypatch):
    fake = _Calls(result=object())
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', fake)
    monkeypatch.setenv(svo_filters.OFFLINE_ENV, '1')
    shim.get_filter_list('JWST')
    assert fake.kwargs == []


def test_non_jwst_facility_still_raises(shim, monkeypatch):
    err = requests.exceptions.ConnectTimeout('svo down')
    monkeypatch.setattr(astroquery_svofps, 'get_filter_list', _Calls(error=err))
    with pytest.raises(requests.exceptions.ConnectTimeout):
        shim.get_filter_list('2MASS')


def test_stamp_fetched_date_rewrites_one_line(tmp_path):
    src = tmp_path / 'svo_filters_copy.py'
    with open(svo_filters.__file__, encoding='utf-8') as fh:
        src.write_text(fh.read(), encoding='utf-8')
    svo_filters._stamp_fetched_date('2031-01-02', source=str(src))
    lines = [ln for ln in src.read_text(encoding='utf-8').splitlines()
             if ln.startswith('VENDORED_FETCHED = ')]
    assert lines == ["VENDORED_FETCHED = '2031-01-02'"]


def test_other_attributes_are_astroquerys(shim):
    assert shim.get_transmission_data == astroquery_svofps.get_transmission_data


def test_no_module_imports_astroquery_svofps_directly():
    """Every caller goes through the fallback shim.

    Walks the package directory (not ``git ls-files``) so a new, untracked
    module is checked too.
    """
    pkg = os.path.dirname(svo_filters.__file__)
    offenders = []
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d not in ('tests', '__pycache__')]
        for name in files:
            path = os.path.join(root, name)
            if not name.endswith('.py') or path == svo_filters.__file__:
                continue
            with open(path, encoding='utf-8') as fh:
                for lineno, line in enumerate(fh, 1):
                    if 'astroquery.svo_fps' in line and 'import' in line:
                        offenders.append(f'{os.path.relpath(path, pkg)}:{lineno}')
    assert not offenders, ('import SvoFps from jwst_gc_pipeline.svo_filters, '
                           f'not astroquery: {offenders}')
