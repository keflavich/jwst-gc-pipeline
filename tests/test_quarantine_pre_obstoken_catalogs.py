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


# ---------------------------------------------------------------------------
# MERGED catalogs (catalogs/), where the token sits in the MODULE slot.
#
# m4 is the case: 1979/002 and 003 share the m4/ tree, so 59 untokened merged
# catalogs sat in catalogs/ that no post-token reader spells.  The pre-token
# file pooled BOTH observations, so it cannot be re-tokened -- only renamed out
# of the way, and only when something newer has actually replaced it.
# ---------------------------------------------------------------------------

MUNT = 'f150w2_nrca_indivexp_merged_m2_dao_basic.fits'
MTOK = 'f150w2_nrca_o003_indivexp_merged_m2_dao_basic.fits'


def _mkm(tmp_path, *specs):
    """``specs`` are ``(name, mtime)``; the catalogs/ dir is returned."""
    d = tmp_path / 'catalogs'
    d.mkdir(parents=True, exist_ok=True)
    for name, mtime in specs:
        (d / name).write_bytes(b'x')
        os.utime(d / name, (mtime, mtime))
    return str(d)


def test_a_merged_name_parses_with_the_token_in_the_MODULE_slot():
    m = _load()
    assert m.merged_identity(MUNT) == m.merged_identity(MTOK)
    assert m.merged_token_of(MUNT) is None
    assert m.merged_token_of(MTOK) == 'o003'


def test_module_named_merged_is_not_confused_with_indivexp_merged():
    """``f150w2_merged_indivexp_merged_m2_...`` is module=merged, token=absent.

    A pattern that anchors on the literal ``merged`` in ``indivexp_merged``
    parses the module as the token and every ``_merged`` product silently drops
    out of the plan.
    """
    m = _load()
    u = 'f150w2_merged_indivexp_merged_m2_dao_basic.fits'
    t = 'f150w2_merged_o003_indivexp_merged_m2_dao_basic.fits'
    assert m.merged_identity(u) is not None
    assert m.merged_identity(u) == m.merged_identity(t)
    assert m.merged_token_of(u) is None
    assert m.merged_token_of(t) == 'o003'


def test_a_superseded_merged_catalog_is_planned(tmp_path):
    m = _load()
    d = _mkm(tmp_path, (MUNT, 1000), (MTOK, 2000))
    plan, orphans = m.plan_for_merged_dir(d)
    assert [p[0] for p in plan] == [MUNT]
    assert plan[0][1] == [MTOK]
    assert orphans == []


def test_a_merged_catalog_with_NO_replacement_is_REPORTED_not_renamed(tmp_path):
    """The safety property, and the state m4 is actually in: its m3-m6 F322W2
    chain is unreachable under the new token and nothing has re-written it yet.
    Renaming those would take the field from a stale catalog to none at all."""
    m = _load()
    d = _mkm(tmp_path, (MUNT, 1000))
    plan, orphans = m.plan_for_merged_dir(d)
    assert plan == []
    assert orphans == [MUNT]


def test_a_replacement_at_a_DIFFERENT_STAGE_does_not_count(tmp_path):
    """An ``_o003`` m2 catalog does not license removing an untokened m5 one."""
    m = _load()
    u5 = 'f150w2_nrca_indivexp_merged_resbgsub_m5_dao_basic.fits'
    d = _mkm(tmp_path, (u5, 1000), (MTOK, 2000))
    plan, orphans = m.plan_for_merged_dir(d)
    assert plan == []
    assert orphans == [u5]


def test_a_replacement_for_a_DIFFERENT_VARIANT_does_not_count(tmp_path):
    """``_vetted`` / ``_allcols`` / ``_i2dseed`` are separate products; m4 has
    36 untokened merged catalogs whose only tokened sibling is a different
    variant, and treating those as replacements would delete all 36."""
    m = _load()
    u = 'f150w2_nrca_indivexp_merged_m2_dao_basic_allcols.fits'
    d = _mkm(tmp_path, (u, 1000), (MTOK, 2000))
    plan, orphans = m.plan_for_merged_dir(d)
    assert plan == []
    assert orphans == [u]


def test_a_replacement_on_a_DIFFERENT_MODULE_does_not_count(tmp_path):
    m = _load()
    d = _mkm(tmp_path, (MUNT, 1000),
             (MTOK.replace('_nrca_', '_nrcb_'), 2000))
    plan, orphans = m.plan_for_merged_dir(d)
    assert plan == []
    assert orphans == [MUNT]


def test_an_OLDER_tokened_file_does_not_supersede_a_FRESH_untokened_one(tmp_path):
    """The rollback case.  Someone re-runs an old checkout, which writes an
    untokened merge NEWER than the tokened one beside it; that file is the
    current product, and renaming it away because an older tokened name exists
    destroys the run that just finished."""
    m = _load()
    d = _mkm(tmp_path, (MUNT, 5000), (MTOK, 1000))
    plan, orphans = m.plan_for_merged_dir(d)
    assert plan == []
    assert orphans == [MUNT]


def test_the_merged_pass_never_re_tokens(tmp_path, monkeypatch, capsys):
    """The pre-token merge pooled both observations, so ownership is not
    recoverable and the tool must not invent one.  The HDU-1 observation is
    printed with that caveat and never used as a guard: a file whose header
    names 002 is still superseded by an 003 twin, because what is being asserted
    is staleness, not ownership."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    monkeypatch.setattr(m, 'source_observation', lambda p: ('1979', '002'))
    (tmp_path / 'm4').mkdir()
    d = _mkm(tmp_path / 'm4', (MUNT, 1000), (MTOK, 2000))
    assert m.main(['--field', 'm4', '--execute']) == 0
    out = capsys.readouterr().out
    assert 'NOT attributable' in out
    assert not os.path.exists(os.path.join(d, MUNT))
    assert os.path.exists(os.path.join(d, MUNT + m.SUFFIX))
    assert os.path.exists(os.path.join(d, MTOK)), 'the replacement must survive'
    # the tool renamed; it did not rename INTO a tokened name
    assert not os.path.exists(os.path.join(
        d, 'f150w2_nrca_o002_indivexp_merged_m2_dao_basic.fits'))


def test_the_prov_sidecar_moves_with_its_catalog(tmp_path, monkeypatch):
    """A provenance record left pointing at a renamed catalog reads as a corrupt
    record rather than an absent one."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    monkeypatch.setattr(m, 'source_observation', lambda p: None)
    (tmp_path / 'm4').mkdir()
    d = _mkm(tmp_path / 'm4', (MUNT, 1000), (MTOK, 2000))
    prov = os.path.join(d, MUNT + '.prov.json')
    with open(prov, 'w') as fh:
        fh.write('{}')
    assert m.main(['--field', 'm4', '--execute']) == 0
    assert not os.path.exists(prov)
    assert os.path.exists(prov + m.SUFFIX)


def test_no_merged_leaves_the_catalogs_dir_alone(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    (tmp_path / 'm4' / 'F150W2').mkdir(parents=True)
    d = _mkm(tmp_path / 'm4', (MUNT, 1000), (MTOK, 2000))
    assert m.main(['--field', 'm4', '--no-merged', '--execute']) == 0
    assert os.path.exists(os.path.join(d, MUNT))


def test_the_merged_pass_runs_when_there_are_no_per_filter_dirs(tmp_path,
                                                                monkeypatch):
    """``field_dirs`` returning [] used to be an immediate rc=2, which would
    have reported "nothing to do" for a tree whose merged catalogs were all
    unreachable."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    monkeypatch.setattr(m, 'source_observation', lambda p: None)
    (tmp_path / 'm4').mkdir()
    _mkm(tmp_path / 'm4', (MUNT, 1000), (MTOK, 2000))
    assert m.main(['--field', 'm4']) == 0


def test_a_per_frame_name_is_not_matched_by_the_merged_pattern():
    """The two families must not cross: a per-frame name reaching the merged
    plan would be judged by mtime instead of by its own observation token."""
    m = _load()
    assert m.merged_identity(UNT) is None
    assert m.identity(MUNT) is None
