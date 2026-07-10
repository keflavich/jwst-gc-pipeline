"""Repo-level pytest configuration.

Defines the ``localdata`` marker: tests that read the survey products on the
local cluster filesystem (/blue|/orange) carry it and are skipped wherever
that data is absent (CI, laptops), so the suite is runnable anywhere.
"""
import os

import pytest

LOCAL_DATA_ROOT = '/blue/adamginsburg/adamginsburg/jwst'


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'localdata: needs the survey data tree on local disk (skipped on CI)')


def pytest_collection_modifyitems(config, items):
    if os.path.isdir(LOCAL_DATA_ROOT):
        return
    skip = pytest.mark.skip(reason=f'local survey data not available '
                                   f'({LOCAL_DATA_ROOT})')
    for item in items:
        if 'localdata' in item.keywords:
            item.add_marker(skip)
