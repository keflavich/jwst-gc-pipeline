"""Repo-level pytest configuration.

Defines two markers for tests that need something the machine may not have:

``localdata``
    reads the survey products on the local cluster filesystem (/blue|/orange),
    skipped wherever that data is absent (CI, laptops).

``crds``
    needs the CRDS server at ``jwst-crds.stsci.edu``.  Skipped when it cannot be
    reached, because a CRDS outage is not a defect in the pull request that
    happens to be building at the time.  On 2026-08-09 a run of 504 Gateway
    Time-outs turned four green PRs red at once::

        crds.core.exceptions.CrdsNetworkError: Failed downloading cache config
        from: JSON RPC service at 'https://jwst-crds.stsci.edu': "CRDS jsonrpc
        failure 'get_server_info' HTTP Error 504: Gateway Time-out"

    That is the same class of problem as #327 and #358: an outage arriving as a
    failure that names something else.
"""
import os
import socket
import urllib.error
import urllib.request

import pytest

LOCAL_DATA_ROOT = '/blue/adamginsburg/adamginsburg/jwst'

#: Cheap reachability probe.  Deliberately NOT `crds.get_context_name`: that
#: pulls the whole cache config and raises the fatal error being avoided.
#:
#: Probe the JSONRPC endpoint, not the site root.  The outage was a 504 on the
#: `get_server_info` POST to /json/, and while a gateway timeout usually takes
#: the whole app with it, a PARTIAL outage where the root answers and jsonrpc
#: does not would mark the tests reachable and let them fail anyway -- which is
#: the exact failure this is here to prevent.  A GET to /json/ is not the POST
#: crds makes, so this is still a proxy; it is just a much closer one.
CRDS_URL = os.environ.get('CRDS_SERVER_URL', 'https://jwst-crds.stsci.edu')
CRDS_PROBE_URL = CRDS_URL.rstrip('/') + '/json/'
CRDS_PROBE_TIMEOUT = 10

_crds_state = {}


def crds_reachable():
    """Is the CRDS server answering?  Probed once per session.

    Any failure means "not reachable" -- the point is to decide whether to run
    a test, and every way this can fail is a reason not to.  Errors are named
    rather than blanket-caught: the network ones (`URLError` covers
    `HTTPError`), the DNS/socket ones, and CRDS's own, whose `CrdsError` is a
    plain `Exception` and so is NOT covered by `OSError`.
    """
    if 'ok' in _crds_state:
        return _crds_state['ok']
    errors = [urllib.error.URLError, socket.timeout, socket.gaierror, OSError,
              ValueError]
    try:
        from crds.core.exceptions import CrdsError
        errors.append(CrdsError)
    except ImportError:
        pass
    try:
        urllib.request.urlopen(CRDS_PROBE_URL,
                               timeout=CRDS_PROBE_TIMEOUT).close()
        _crds_state['ok'] = True
    except tuple(errors):
        _crds_state['ok'] = False
    return _crds_state['ok']


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'localdata: needs the survey data tree on local disk (skipped on CI)')
    config.addinivalue_line(
        'markers',
        'crds: needs the CRDS server (skipped when it cannot be reached)')


def pytest_collection_modifyitems(config, items):
    if not os.path.isdir(LOCAL_DATA_ROOT):
        skip = pytest.mark.skip(reason=f'local survey data not available '
                                       f'({LOCAL_DATA_ROOT})')
        for item in items:
            if 'localdata' in item.keywords:
                item.add_marker(skip)

    if any('crds' in item.keywords for item in items) and not crds_reachable():
        skip = pytest.mark.skip(
            reason=f'CRDS server unreachable ({CRDS_PROBE_URL}); a CRDS '
                   f'outage is not a defect in this change')
        for item in items:
            if 'crds' in item.keywords:
                item.add_marker(skip)
