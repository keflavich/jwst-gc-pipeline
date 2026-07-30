"""The PSF-data path must come from the environment when the user set one.

`saturated_star_finding` raises unless STPSF_PATH is set, while importing
`reduction.filtering` used to overwrite it with a HiPerGator path -- so setting
it correctly off HiPerGator got you a directory that does not exist.
"""
import importlib
import os

import pytest


@pytest.mark.parametrize('var', ['STPSF_PATH', 'WEBBPSF_PATH'])
def test_an_existing_setting_is_not_overwritten(monkeypatch, tmp_path, var):
    monkeypatch.setenv(var, str(tmp_path))
    import jwst_gc_pipeline.reduction.filtering as F
    importlib.reload(F)
    assert os.environ[var] == str(tmp_path)


def test_a_default_is_still_supplied_when_unset(monkeypatch):
    monkeypatch.delenv('STPSF_PATH', raising=False)
    monkeypatch.delenv('WEBBPSF_PATH', raising=False)
    import jwst_gc_pipeline.reduction.filtering as F
    importlib.reload(F)
    assert os.environ.get('STPSF_PATH') or os.environ.get('WEBBPSF_PATH')
