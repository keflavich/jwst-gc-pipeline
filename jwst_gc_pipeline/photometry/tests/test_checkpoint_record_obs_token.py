"""Checkpoint records must not collide across observations (issue #281).

cloudef 2092 obs 002 and 005 share `cloudef/astrometry_checkpoints/`, so an
untokened `checkpoint_m2_F360M_latest.json` written by one run REPLACES the
other's -- and every frozen-stage reader then compares one observation's
exposures against the other's baseline.
"""
import json
import os

import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    _m2_exposure_baseline, _m2_record_path, _record_name)


def test_record_name_carries_the_token():
    assert _record_name("m2", "F360M") == "checkpoint_m2_F360M"
    assert _record_name("m2", "F360M", "_o002") == "checkpoint_m2_F360M_o002"
    assert _record_name("m2", "F360M", "_o005") == "checkpoint_m2_F360M_o005"
    assert _record_name("m2", None, "_o002") == "checkpoint_m2_all_o002"


def test_two_observations_do_not_share_a_record_name():
    a = _record_name("m2", "F360M", "_o002")
    b = _record_name("m2", "F360M", "_o005")
    assert a != b


def _write(tmp_path, name, dra):
    rec = dict(stage="m2", filtername="F360M", visits=[dict(
        visit="1", filtername="F360M", exposures=[dict(
            key=["1", 1, "nrcblong", "F360M", "02101"], misaligned=True,
            dra=dra, ddec=0.0, ok=True)])])
    (tmp_path / f"{name}_latest.json").write_text(json.dumps(rec))


def test_reader_prefers_its_own_observation(tmp_path):
    _write(tmp_path, "checkpoint_m2_F360M_o002", 1.0)
    _write(tmp_path, "checkpoint_m2_F360M_o005", 9.0)
    a = _m2_exposure_baseline(str(tmp_path), "F360M", "1", "_o002")
    b = _m2_exposure_baseline(str(tmp_path), "F360M", "1", "_o005")
    assert a[("1", 1, "nrcblong", "F360M", "02101")][0] == 1.0
    assert b[("1", 1, "nrcblong", "F360M", "02101")][0] == 9.0


def test_untokened_legacy_record_is_refused_when_it_could_be_another_obs(
        tmp_path, capsys):
    """Fail-CLOSED.  The field cannot be determined from a bare tmp_path, and
    an untokened record body carries no observation identity, so this must
    refuse rather than read.  A missing baseline is reported as unverified; a
    wrong baseline is a silent wrong answer.

    (The permissive fallback this test originally asserted was the hazard #281
    describes, with a warning attached -- see the review on PR #306.)
    """
    _write(tmp_path, "checkpoint_m2_F360M", 2.0)
    assert _m2_exposure_baseline(str(tmp_path), "F360M", "1", "_o002") == {}
    assert "REFUSING" in capsys.readouterr().out


def test_tokened_record_wins_over_the_legacy_one(tmp_path, capsys):
    _write(tmp_path, "checkpoint_m2_F360M", 2.0)
    _write(tmp_path, "checkpoint_m2_F360M_o002", 7.0)
    base = _m2_exposure_baseline(str(tmp_path), "F360M", "1", "_o002")
    assert base[("1", 1, "nrcblong", "F360M", "02101")][0] == 7.0
    assert "falling back" not in capsys.readouterr().out


def test_no_token_behaves_exactly_as_before(tmp_path, capsys):
    _write(tmp_path, "checkpoint_m2_F360M", 3.0)
    base = _m2_exposure_baseline(str(tmp_path), "F360M", "1")
    assert base[("1", 1, "nrcblong", "F360M", "02101")][0] == 3.0
    assert "falling back" not in capsys.readouterr().out


def test_missing_record_is_still_no_baseline(tmp_path):
    assert _m2_record_path(str(tmp_path), "F360M", "_o002") is None


def test_ambiguous_filter_refuses_the_untokened_record(tmp_path, monkeypatch, capsys):
    """An untokened record body carries NO observation identity (`visit` is "1"
    for both jw02092002001 and jw02092005001), so on a filter more than one
    observation images, falling back IS the hazard, not a degraded read."""
    monkeypatch.setattr(
        "jwst_gc_pipeline.monitoring.scan.shared_filters",
        lambda target, instrument="nircam": {"F360M"})
    monkeypatch.setattr(
        "jwst_gc_pipeline.fields.BY_NAME", {"cloudef": object()})
    d = tmp_path / "cloudef" / "astrometry_checkpoints"
    d.mkdir(parents=True)
    _write(d, "checkpoint_m2_F360M", 2.0)
    assert _m2_record_path(str(d), "F360M", "_o002") is None
    assert "REFUSING the untokened m2 record" in capsys.readouterr().out


def test_unambiguous_filter_still_reads_the_untokened_record(tmp_path, monkeypatch, capsys):
    """Brick's two observations use disjoint filter sets, so its untokened
    records are this run's own and must still be readable."""
    monkeypatch.setattr(
        "jwst_gc_pipeline.monitoring.scan.shared_filters",
        lambda target, instrument="nircam": set())
    monkeypatch.setattr(
        "jwst_gc_pipeline.fields.BY_NAME", {"brick": object()})
    d = tmp_path / "brick" / "astrometry_checkpoints"
    d.mkdir(parents=True)
    _write(d, "checkpoint_m2_F212N", 2.0)
    assert _m2_record_path(str(d), "F212N", "_o001") is not None
    assert "unambiguous" in capsys.readouterr().out


def test_unknown_field_fails_closed(tmp_path, capsys):
    """Reading the wrong observation's baseline is a silent wrong answer;
    refusing is a loud unverified."""
    d = tmp_path / "astrometry_checkpoints"
    d.mkdir(parents=True)
    _write(d, "checkpoint_m2_F360M", 2.0)
    assert _m2_record_path(str(d), "F360M", "_o002") is None


def test_apply_script_refuses_to_union_two_observations(tmp_path):
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "apply_m2", pathlib.Path(__file__).resolve().parents[3]
        / "scripts" / "reduction" / "apply_m2_checkpoint_corrections.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _write(tmp_path, "checkpoint_m2_F360M_o002", 1.0)
    _write(tmp_path, "checkpoint_m2_F360M_o005", 9.0)
    with pytest.raises(SystemExit, match="more than one observation"):
        mod.load_corrections(str(tmp_path))
    # with a token it reads exactly one
    assert mod.load_corrections(str(tmp_path), obs_token="_o002") is not None


@pytest.mark.parametrize("path,filt,ambiguous", [
    # the field NEAREST the record dir wins: first-in-dict-order read this as
    # 'brick' and, brick having no shared filters, called it unambiguous
    ("/blue/x/jwst/brick/scratch/cloudef/astrometry_checkpoints", "F360M", True),
    # longest-match does not fix it either -- 'arches' and 'sickle' tie
    ("/home/arches/runs/sickle/astrometry_checkpoints", "F187N", True),
    ("/orange/adamginsburg/jwst/brick/astrometry_checkpoints", "F212N", False),
    # sgrb2 registers nircam ['001'] but miri ['001','002','998'], so the
    # nircam default returned False for all 14 of its genuinely shared filters
    ("/orange/adamginsburg/jwst/sgrb2/astrometry_checkpoints", "F360M", True),
    ("/tmp/nowhere/astrometry_checkpoints", "F360M", True),      # fail-closed
])
def test_field_detection_and_ambiguity(path, filt, ambiguous):
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        _filter_is_obs_ambiguous)
    assert _filter_is_obs_ambiguous(path, filt) is ambiguous


def test_the_applier_exposes_the_flag_its_error_names():
    """The refusal told the operator to pass --obs-token, which did not exist:
    once tokened records appear, the sanctioned recovery tool always exits
    non-zero on advice that cannot be followed."""
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    src = (root / "scripts" / "reduction"
           / "apply_m2_checkpoint_corrections.py").read_text()
    assert '"--obs-token"' in src
    assert "load_corrections(args.records_dir, args.obs_token)" in src
    assert "load_exposure_universe(args.records_dir," in src


@pytest.mark.parametrize("name,token", [
    ("checkpoint_m2_F360M_o002_latest.json", "_o002"),
    ("checkpoint_m2_F360M_o002-998_latest.json", "_o002-998"),   # sgrb2
    ("checkpoint_m2_F360M_o001-002_latest.json", "_o001-002"),   # sickle
    ("checkpoint_m2_F360M_j7213_latest.json", "_j7213"),
    ("checkpoint_m2_F360M_latest.json", None),
])
def test_joint_obsids_are_recognised(name, token):
    """Registered obsids include joint forms; a bare o\\d{3} missed them, so the
    union went unrefused and scan.py's keys collided back to last-wins."""
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "apply_m2", root / "scripts" / "reduction"
        / "apply_m2_checkpoint_corrections.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod._TOKEN_RE.search(name)
    assert (m.group(1) if m else None) == token
