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

import pytest  # noqa: F401

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


def test_an_untokened_file_with_a_tokened_twin_is_planned(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'source_token', lambda p: 'o028')
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


def test_only_the_files_OWN_observation_counts_as_its_replacement(tmp_path,
                                                                  monkeypatch):
    """This test used to assert the opposite -- "any of them is a replacement" --
    which is the assumption the review flagged.  The identity is shared ACROSS
    observations (that collision is why the token exists), so o023's catalog
    does not license removing o046's only copy.  It happened to be harmless on
    disk today only because every gc2211 observation had been re-reduced."""
    m = _load()
    monkeypatch.setattr(m, 'source_token', lambda p: 'o046')
    u = 'f200w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
    toks = [u.replace('nrca1_', f'nrca1_o{o}_') for o in ('023', '046', '049')]
    d = _mk(tmp_path, u, *toks)
    plan = m.plan_for_dir(d)
    assert len(plan) == 1
    assert plan[0][1] == [toks[1]], 'only the o046 twin may justify it' 


def test_an_already_quarantined_file_is_not_replanned(tmp_path):
    """Running twice must be a no-op, not a double suffix."""
    m = _load()
    d = _mk(tmp_path, UNT + m.SUFFIX, TOK)
    assert m.plan_for_dir(d) == []


def test_a_non_catalog_name_is_ignored(tmp_path):
    m = _load()
    d = _mk(tmp_path, 'jw02211023001_02201_00001_nrca1_cal.fits', TOK)
    assert m.plan_for_dir(d) == []


# ---------------------------------------------------------------------------
# main() FOR REAL.  The first version of these called main() once against a
# NONEXISTENT field -- so field_dirs returned [], main returned 2, and the
# assertion held no matter what the dry-run branch did -- and hand-rolled the
# rename instead of passing --execute.  `if args.execute:` -> `if True:` left
# the suite green.  BASE is redirected so everything below plan_for_dir runs.
# ---------------------------------------------------------------------------

def _field(tmp_path, *names, filt='F150W'):
    d = tmp_path / 'gc2211' / filt
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_bytes(b'x')
    return d


def test_the_DRY_RUN_really_renames_nothing(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    monkeypatch.setattr(m, 'source_token', lambda p: 'o028')
    d = _field(tmp_path, UNT, TOK)
    assert m.main(['--field', 'gc2211']) == 0
    assert (d / UNT).exists(), 'the default must not touch anything'
    assert not (d / (UNT + m.SUFFIX)).exists()


def test_EXECUTE_really_renames(tmp_path, monkeypatch, capsys):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    monkeypatch.setattr(m, 'source_token', lambda p: 'o028')
    d = _field(tmp_path, UNT, TOK)
    assert m.main(['--field', 'gc2211', '--execute']) == 0
    assert not (d / UNT).exists()
    assert (d / (UNT + m.SUFFIX)).exists()
    assert (d / TOK).exists(), 'the replacement must survive'
    assert 'renamed 1' in capsys.readouterr().out


def test_a_second_run_does_not_DESTROY_the_first_quarantine(tmp_path,
                                                            monkeypatch, capsys):
    """os.rename replaces the target silently, so re-running over an existing
    quarantine would delete the file the first run preserved -- the opposite of
    reversible."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    monkeypatch.setattr(m, 'source_token', lambda p: 'o028')
    d = _field(tmp_path, TOK)
    (d / (UNT + m.SUFFIX)).write_bytes(b'ORIGINAL-QUARANTINED')
    (d / UNT).write_bytes(b'NEW-UNTOKENED')
    assert m.main(['--field', 'gc2211', '--execute']) == 0
    assert (d / (UNT + m.SUFFIX)).read_bytes() == b'ORIGINAL-QUARANTINED'
    assert (d / UNT).exists(), 'the un-renamed file stays put'
    assert 'SKIP' in capsys.readouterr().out


def test_the_twin_must_carry_the_files_OWN_token(tmp_path, monkeypatch):
    """The identity is shared ACROSS observations -- that collision is why the
    token exists -- so `some tokened file shares the identity` is satisfiable by
    a different observation's catalog.  An observation not yet re-reduced would
    have its only catalog renamed away."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    monkeypatch.setattr(m, 'source_token', lambda p: 'o046')   # file is o046's
    d = _field(tmp_path, UNT, TOK)                              # twin is o028's
    assert m.plan_for_dir(str(d)) == []


def test_an_unreadable_header_DECLINES_rather_than_assumes(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'source_token', lambda p: None)
    d = _field(tmp_path, UNT, TOK)
    assert m.plan_for_dir(str(d)) == []


def test_the_j_token_family_is_recognised():
    """ngc6334 shares a directory, filter list and obsid between proposals 6778
    and 7213, so its disambiguator is the PROPOSAL.  Matching only `o\\d{3}` made
    its 1680 tokened F200W catalogs invisible and printed a confident zero."""
    m = _load()
    j = 'f200w_nrca1_j6778_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
    u = 'f200w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
    assert m.token_of(j) == 'j6778'
    assert m.identity(j) == m.identity(u)


def test_MIRI_four_digit_filters_are_visible(tmp_path):
    """`f1000w_mirim_...` has FOUR digits, and field_dirs matched only three --
    consistent with each other, so MIRI read as "nothing superseded" rather than
    "not examined"."""
    m = _load()
    u = 'f1000w_mirim_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
    t = 'f1000w_mirim_o002_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
    assert m.identity(u) is not None
    assert m.identity(u) == m.identity(t)
    (tmp_path / 'gc2211' / 'F1000W').mkdir(parents=True)
    m.BASE = str(tmp_path)
    assert any(d.endswith('F1000W') for d in m.field_dirs('gc2211'))


def test_an_already_quarantined_name_is_not_itself_a_candidate(tmp_path):
    m = _load()
    assert m.identity(UNT + m.SUFFIX) is None
