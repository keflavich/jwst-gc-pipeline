"""A pre-obs-token catalog may only be quarantined if something replaced it.

The duplicate that emptied every gc2211 m2 record (#350):

    f150w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits       June
    f150w_nrca1_o028_visit001_vgroup02201_exp00001_m2_daophot_basic.fits  August

Same (filter, detector, visit, vgroup, exposure) identity, so the visit
consensus ingests it twice and refuses to build.

The tool renames the untokened one out of the glob, but ONLY where a tokened
file exists for that identity -- a field that simply predates the token, with no
replacement, must be left alone.  That condition is what makes it safe to run
without checking dates by hand first.
"""
import importlib.util
import os

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reduction')


def _load():
    spec = importlib.util.spec_from_file_location(
        'quarantine_pre_obstoken_catalogs',
        os.path.join(SCRIPTS, 'quarantine_pre_obstoken_catalogs.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


UNT = 'f150w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
TOK = 'f150w_nrca1_o028_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'


def _mk(tmp_path, *names):
    for n in names:
        (tmp_path / n).write_bytes(b'x')
    return str(tmp_path)


def test_the_gc2211_pair_is_the_same_identity():
    m = _load()
    assert m.identity(UNT) == m.identity(TOK)
    assert not m.has_obs_token(UNT)
    assert m.has_obs_token(TOK)


def test_an_untokened_file_with_a_tokened_twin_is_planned(tmp_path):
    m = _load()
    d = _mk(tmp_path, UNT, TOK)
    plan = m.plan_for_dir(d)
    assert [p[0] for p in plan] == [UNT]
    assert plan[0][1] == [TOK]


def test_an_untokened_file_with_NO_twin_is_LEFT_ALONE(tmp_path):
    """The safety property.  A field that predates the token and was never
    re-run has no replacement, and deleting its only catalog would destroy the
    data rather than de-duplicate it."""
    m = _load()
    d = _mk(tmp_path, UNT)
    assert m.plan_for_dir(d) == []


def test_a_tokened_file_is_never_planned(tmp_path):
    m = _load()
    d = _mk(tmp_path, TOK)
    assert m.plan_for_dir(d) == []


def test_a_twin_for_a_DIFFERENT_exposure_does_not_count(tmp_path):
    """Identity is the whole (visit, vgroup, exposure, detector) tuple; a
    tokened file for exposure 2 must not license removing exposure 1."""
    m = _load()
    other = TOK.replace('exp00001', 'exp00002')
    d = _mk(tmp_path, UNT, other)
    assert m.plan_for_dir(d) == []


def test_a_twin_on_a_DIFFERENT_detector_does_not_count(tmp_path):
    m = _load()
    other = TOK.replace('nrca1', 'nrca2')
    d = _mk(tmp_path, UNT, other)
    assert m.plan_for_dir(d) == []


def test_several_observations_all_count_as_replacements(tmp_path):
    """F200W's untokened frame is superseded by o023, o046 AND o049 -- the
    identity exists once per observation, and any of them is a replacement."""
    m = _load()
    u = 'f200w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
    toks = [u.replace('nrca1_', f'nrca1_o{o}_') for o in ('023', '046', '049')]
    d = _mk(tmp_path, u, *toks)
    plan = m.plan_for_dir(d)
    assert len(plan) == 1
    assert sorted(plan[0][1]) == sorted(toks)


def test_execute_renames_and_a_dry_run_does_not(tmp_path):
    m = _load()
    d = _mk(tmp_path, UNT, TOK)
    m.main(['--field', 'x'])           # wrong field: no dirs, changes nothing
    assert os.path.exists(os.path.join(d, UNT))
    for old, _ in m.plan_for_dir(d):
        os.rename(os.path.join(d, old), os.path.join(d, old + m.SUFFIX))
    assert not os.path.exists(os.path.join(d, UNT))
    assert os.path.exists(os.path.join(d, UNT + m.SUFFIX))
    assert os.path.exists(os.path.join(d, TOK)), 'the replacement must survive'


def test_an_already_quarantined_file_is_not_replanned(tmp_path):
    """Running twice must be a no-op, not a double suffix."""
    m = _load()
    d = _mk(tmp_path, UNT + m.SUFFIX, TOK)
    assert m.plan_for_dir(d) == []


def test_a_non_catalog_name_is_ignored(tmp_path):
    m = _load()
    d = _mk(tmp_path, 'jw02211023001_02201_00001_nrca1_cal.fits', TOK)
    assert m.plan_for_dir(d) == []
