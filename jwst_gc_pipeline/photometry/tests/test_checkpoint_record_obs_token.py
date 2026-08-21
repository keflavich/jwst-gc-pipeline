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


@pytest.fixture(autouse=True)
def _enforce_at_the_stage(monkeypatch):
    """These tests assert on the EXCEPTION, which is the detection they are for.

    `ASTROM_CHECKPOINT_ENFORCE` now defaults to `release`, so a frozen-stage
    failure is recorded and the chain continues.  What is detected is unchanged;
    only where the stop happens moved.  See test_checkpoint_enforcement.py.
    """
    monkeypatch.setenv('ASTROM_CHECKPOINT_ENFORCE', 'stage')

def test_record_name_carries_the_token():
    assert _record_name("m2", "F360M") == "checkpoint_m2_F360M"
    assert _record_name("m2", "F360M", "_o002") == "checkpoint_m2_F360M_o002"
    assert _record_name("m2", "F360M", "_o005") == "checkpoint_m2_F360M_o005"
    assert _record_name("m2", None, "_o002") == "checkpoint_m2_all_o002"


def test_two_observations_do_not_share_a_record_name():
    a = _record_name("m2", "F360M", "_o002")
    b = _record_name("m2", "F360M", "_o005")
    assert a != b


def _write(tmp_path, name, dra, corrections=True):
    rec = dict(stage="m2", filtername="F360M", visits=[dict(
        visit="1", filtername="F360M", exposures=[dict(
            key=["1", 1, "nrcblong", "F360M", "02101"], misaligned=True,
            dra=dra, ddec=0.0, ok=True)])])
    if corrections:
        # The applier reads THIS list.  Without it every `load_corrections`
        # assertion is vacuous -- an empty directory also returns ([], []).
        rec["corrections"] = [dict(visit="1", exposure=1, module="nrcblong",
                                   filtername="F360M", vgroup="02101",
                                   dra_onsky_mas=dra, ddec_onsky_mas=0.0)]
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
    # With a token it reads exactly ONE record's corrections.  `is not None`
    # was vacuous -- an empty directory returns ([], []) -- which is exactly
    # why the "applier ignores obs_token" mutant survived.
    records, corrections = mod.load_corrections(str(tmp_path),
                                                obs_token="_o002")
    assert len(records) == 1, records
    assert corrections and all("_o002" in c["_record"] for c in corrections), \
        corrections


def test_apply_script_refuses_legacy_BESIDE_tokened(tmp_path):
    """The state this change creates on every field at its FIRST tokened run,
    and it was not refused: the token census counted tokened names only, so
    legacy + one tokened gave len(tokens) == 1.  On a real cloudef F360M pair
    that summed two corrections onto each of 3 of 3 targeted rows."""
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "apply_m2", pathlib.Path(__file__).resolve().parents[3]
        / "scripts" / "reduction" / "apply_m2_checkpoint_corrections.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _write(tmp_path, "checkpoint_m2_F360M", 1.0)            # legacy
    _write(tmp_path, "checkpoint_m2_F360M_o002", 9.0)       # this run's
    with pytest.raises(SystemExit, match="more than one observation"):
        mod.load_corrections(str(tmp_path))
    with pytest.raises(SystemExit, match="more than one observation"):
        mod.load_exposure_universe(str(tmp_path))
    records, _ = mod.load_corrections(str(tmp_path), obs_token="_o002")
    assert len(records) == 1, records


@pytest.mark.parametrize("path,filt,ambiguous", [
    # the field NEAREST the record dir wins: first-in-dict-order read this as
    # 'brick' and, brick having no shared filters, called it unambiguous
    ("/blue/x/jwst/brick/scratch/cloudef/astrometry_checkpoints", "F360M", False),
    # longest-match does not fix it either -- 'arches' and 'sickle' tie.  The
    # path resolves to sickle (nearest component naming a field); what is then
    # asked is whether the FILTER is shared across that field's observations.
    #
    # F187N: NOT ambiguous.  sickle images it in exactly one observation (007),
    # so an untokened F187N record can only belong to that one.  This asserted
    # True while 001 and 002 were listed under `nircam` -- the bug this change
    # fixes -- so the old answer came from two MIRI observations pretending to
    # carry NIRCam filters.
    ("/home/arches/runs/sickle/astrometry_checkpoints", "F187N", False),
    # ...and a GENUINE ambiguity on the same path shape, so the field-resolution
    # case this row was written for is still exercised: F770W IS imaged by both
    # of sickle's MIRI observations (001 and 002), so an untokened record cannot
    # say which.
    ("/home/arches/runs/sickle/astrometry_checkpoints", "F770W", True),
    ("/orange/adamginsburg/jwst/brick/astrometry_checkpoints", "F212N", False),
    # sgrb2 registers nircam ['001'] but miri ['001','002','998'] -- and ONE
    # `filters` list shared by both.  Asking for the nircam default returned
    # False for its genuinely shared MIRI bands; unioning the two instruments
    # then swung to the opposite error and made all 14 filters ambiguous,
    # including 11 NIRCam-only ones, refusing 10 records whose NIRCam side has
    # exactly one observation.  Scoped by band wavelength now (MIRI imaging
    # starts at F560W), so each side gets its own answer:
    ("/orange/adamginsburg/jwst/sgrb2/astrometry_checkpoints", "F360M", False),
    ("/orange/adamginsburg/jwst/sgrb2/astrometry_checkpoints", "F212N", False),
    ("/orange/adamginsburg/jwst/sgrb2/astrometry_checkpoints", "F770W", True),
    ("/orange/adamginsburg/jwst/sgrb2/astrometry_checkpoints", "F1280W", True),
    # ... and the fields the refusal exists for are unchanged.  cloudef USED to
    # be one of them and is now the counter-example: obs 005 moved to
    # `cloudef_controlfield` (its own tree), so each side has a single NIRCam
    # obsid and neither can be ambiguous.  Both are asserted -- a True on either
    # means the two observations are sharing a tree again.
    ("/orange/adamginsburg/jwst/cloudef/astrometry_checkpoints", "F360M", False),
    ("/orange/adamginsburg/jwst/cloudef_controlfield/astrometry_checkpoints",
     "F360M", False),
    ("/orange/adamginsburg/jwst/gc2211/astrometry_checkpoints", "F200W", True),
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


# ----------------------------------------------------------------- the WRITER

def test_the_writer_puts_the_token_in_the_RECORD_filename(tmp_path):
    """The headline behaviour, and until now nothing asserted it.

    Every test above reads records this file wrote by hand, so reverting
    `_record_name(stage, filtername, obs_token)` at the writer to
    `_record_name(stage, filtername)` left all of them green -- the mutant
    survived its own PR.  Exactly one test in the repo passed `obs_token` to
    `run_visit_checkpoint`, and it asserted the CONSENSUS CATALOG name, which
    is a different code path.
    """
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_visit_checkpoint)
    from .test_astrometry_checkpoint import _two_visit_tables

    run_visit_checkpoint(_two_visit_tables(), "m2", filtername="F360M",
                         basepath=str(tmp_path), record_dir=str(tmp_path),
                         context="test", obs_token="_o002")
    assert os.path.exists(tmp_path / "checkpoint_m2_F360M_o002_latest.json")
    assert not os.path.exists(tmp_path / "checkpoint_m2_F360M_latest.json")


def test_two_observations_write_two_records(tmp_path):
    """The collision itself: without the token the second run REPLACES the
    first, and cloudef's o002 and o005 both write here."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_visit_checkpoint)
    from .test_astrometry_checkpoint import _two_visit_tables

    for tok in ("_o002", "_o005"):
        run_visit_checkpoint(_two_visit_tables(), "m2", filtername="F360M",
                             basepath=str(tmp_path), record_dir=str(tmp_path),
                             context="test", obs_token=tok)
    assert os.path.exists(tmp_path / "checkpoint_m2_F360M_o002_latest.json")
    assert os.path.exists(tmp_path / "checkpoint_m2_F360M_o005_latest.json")


# ------------------------------------------------------- the `_all` fallback

def _ambiguous(monkeypatch, value=True):
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    monkeypatch.setattr(ac, "_filter_is_obs_ambiguous", lambda *a, **k: value)


def test_the_all_fallback_is_gated_by_the_same_refusal(tmp_path, monkeypatch, capsys):
    """The refusal lived only in the per-filter loop.  `run_astrometry_checkpoint.py`'s
    `--filter` defaults to None, so one filterless invocation creates an
    untokened `_all` record that every observation would then read as its own --
    the per-filter gate closed while the `_all` door stood open."""
    _ambiguous(monkeypatch, True)
    (tmp_path / "checkpoint_m2_all_latest.json").write_text("{}")
    assert _m2_record_path(str(tmp_path), "F360M", "_o002") is None
    assert "REFUSING" in capsys.readouterr().out


def test_an_unambiguous_filter_still_reads_the_legacy_all_record(tmp_path, monkeypatch):
    _ambiguous(monkeypatch, False)
    p = tmp_path / "checkpoint_m2_all_latest.json"
    p.write_text("{}")
    assert _m2_record_path(str(tmp_path), "F360M", "_o002") == str(p)


def test_a_None_filtername_cannot_be_tested_and_fails_closed(tmp_path, monkeypatch):
    """Ambiguity is a property of a FILTER.  With no filter to ask about there
    is no answer, so the untokened `_all` record is refused rather than
    guessed at."""
    _ambiguous(monkeypatch, False)      # would say "unambiguous" if consulted
    (tmp_path / "checkpoint_m2_all_latest.json").write_text("{}")
    assert _m2_record_path(str(tmp_path), None, "_o002") is None


def test_refusing_the_per_filter_legacy_record_does_not_hide_our_own_all_record(
        tmp_path, monkeypatch, capsys):
    """`return None` at the refusal made a legacy untokened per-filter file on
    disk shadow a perfectly good TOKENED mixed-filter record -- failing closed
    against the wrong file.  cloudef's untokened per-filter records are on disk
    now, so this is one filterless run from live."""
    _ambiguous(monkeypatch, True)
    (tmp_path / "checkpoint_m2_F360M_latest.json").write_text("{}")   # legacy
    ours = tmp_path / "checkpoint_m2_all_o002_latest.json"
    ours.write_text("{}")                                            # ours
    got = _m2_record_path(str(tmp_path), "F360M", "_o002")
    assert got == str(ours), got
    assert "REFUSING" in capsys.readouterr().out


def test_the_tokened_all_record_still_wins_over_the_legacy_one(tmp_path, monkeypatch):
    _ambiguous(monkeypatch, True)
    ours = tmp_path / "checkpoint_m2_all_o002_latest.json"
    ours.write_text("{}")
    (tmp_path / "checkpoint_m2_all_latest.json").write_text("{}")
    assert _m2_record_path(str(tmp_path), "F360M", "_o002") == str(ours)


# ------------------------------------------------------------------ scan.py

def _record(path, filt, date="2026-08-01T00:00:00"):
    path.write_text(json.dumps(
        {"stage": "m2", "filtername": filt, "date": date, "visits": [],
         "corrections": [], "failures": [], "passed": True}))


def test_scan_keys_two_observations_apart_and_keeps_the_bare_filter(tmp_path):
    from jwst_gc_pipeline.monitoring import scan

    ck = tmp_path / "astrometry_checkpoints"
    ck.mkdir()
    _record(ck / "checkpoint_m2_F360M_o002_latest.json", "F360M")
    _record(ck / "checkpoint_m2_F360M_o005_latest.json", "F360M")
    got = scan.astrometry_checkpoints(str(tmp_path), filters=("F360M",))
    assert sorted(got) == ["F360M_o002", "F360M_o005"]
    # the KEY carries the observation; the payload keeps the bare filter, or
    # every consumer that looks a filter up elsewhere silently stops matching
    assert {v["filter"] for v in got.values()} == {"F360M"}
    assert {v["obs_token"] for v in got.values()} == {"_o002", "_o005"}


def test_a_tokened_record_is_attributable_even_on_a_shared_filter(tmp_path):
    """`attributable` was still computed from the bare filter, so a correctly
    tokened record was reported as "cannot be attributed to this observation --
    it describes whichever one last ran m2", which is no longer true.  It
    downgrades a real failure to a warning."""
    from jwst_gc_pipeline.monitoring import scan

    ck = tmp_path / "astrometry_checkpoints"
    ck.mkdir()
    _record(ck / "checkpoint_m2_F360M_o002_latest.json", "F360M")
    _record(ck / "checkpoint_m2_F360M_latest.json", "F360M")
    got = scan.astrometry_checkpoints(str(tmp_path), filters=("F360M",),
                                      ambiguous_filters=("F360M",))
    assert got["F360M_o002"]["attributable"] is True
    assert got["F360M"]["attributable"] is False


def test_the_supersede_suppression_survives_the_new_key():
    """`run['per_filter']` is keyed on BARE filter names, so looking up the
    astrom dict's key never matched and a corrected field went permanently
    red."""
    from jwst_gc_pipeline.monitoring import checks

    astrom = {"F360M_o002": {"path": "/x/checkpoint_m2_F360M_o002_latest.json",
                             "filter": "F360M", "obs_token": "_o002",
                             "mtime": 1000, "attributable": True,
                             "n_misaligned": 4, "n_exposures": 8,
                             "misaligned_exposures": [], "all_exposures": []}}
    run = {"proposal": "2092", "obsid": "002", "astrometry": astrom,
           "per_filter": {"F360M": {"reduced": {"mtime": 2000}}}}
    out = checks.check_astrometry(run)
    hit = [v for v in out if "misaligned" in v["name"]]
    assert hit, out
    assert hit[0]["severity"] == "warn", hit[0]
    assert "predates the frames" in hit[0]["summary"] + (hit[0].get("detail") or "")


def test_the_crossfilter_record_carries_the_token_too(tmp_path):
    """brick's 1182 and 2221 m7 runs write into one `astrometry_checkpoints/`,
    and the untokened name meant 2221's verdict replaced 1182's.  Write-only,
    so what it costs is the audit trail rather than a wrong correction."""
    import inspect
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        run_crossfilter_checkpoint)
    assert "obs_token" in inspect.signature(run_crossfilter_checkpoint).parameters
    src = inspect.getsource(run_crossfilter_checkpoint)
    assert 'f"checkpoint_m7_crossfilter{obs_token}"' in src


# ------------------------------------------ the READER call sites, one by one

@pytest.mark.parametrize("reader", [
    "_m2_exposure_baseline", "_m2_skipped_exposures",
    "_m2_reference_tie_baseline",
])
def test_every_reader_passes_the_token_through(tmp_path, reader, monkeypatch):
    """Dropping `obs_token` at a reader CALL SITE does not merely read the
    wrong record -- the refusal is guarded by `if obs_token and not _tok:`, so
    it turns the entire refusal OFF and silently restores the read-side #281
    bug.  Three mutants survived the whole suite:

        M3_exposure_baseline_call_drops_token   SURVIVED
        M4_skipped_call_drops_token             SURVIVED
        M5_reftie_call_drops_token              SURVIVED

    Each reader is exercised here against a directory holding ONLY the
    untokened record, on an ambiguous filter: with the token it must refuse
    (None / empty), without it it would read the legacy record.
    """
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    monkeypatch.setattr(ac, "_filter_is_obs_ambiguous", lambda *a, **k: True)
    # a record rich enough that ALL THREE readers find something in it
    rec = dict(stage="m2", filtername="F360M", visits=[dict(
        visit="1", filtername="F360M",
        exposures=[dict(key=["1", 1, "nrcblong", "F360M", "02101"],
                        misaligned=True, dra=11.0, ddec=0.0, ok=True)],
        consensus=dict(skipped=[["1", 2, "nrcblong", "F360M", "02101"]]),
        reference_tie=dict(apply_ok=True, dra_mas=7.0, ddec_mas=-3.0))])
    (tmp_path / "checkpoint_m2_F360M_latest.json").write_text(json.dumps(rec))
    fn = getattr(ac, reader)
    permitted = fn(str(tmp_path), "F360M", "1", "")
    refused = fn(str(tmp_path), "F360M", "1", "_o002")
    # the untokened record IS readable, so the un-tokened call finds data ...
    assert permitted, (reader, permitted)
    # ... and the tokened call must refuse it rather than read another
    # observation's numbers
    assert refused != permitted, (reader, refused, permitted)


def test_the_refusal_message_does_not_claim_the_solution_moved(tmp_path, monkeypatch):
    """A refused untokened record and a genuinely absent one both reach the
    caller as None, and the caller then emitted the frozen-stage MOVEMENT
    failure.  Nothing was measured to have moved -- there was nothing to
    compare against.  Still blocking; just not a false statement about the
    data."""
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    monkeypatch.setattr(ac, "_filter_is_obs_ambiguous", lambda *a, **k: True)
    _write(tmp_path, "checkpoint_m2_F360M", 11.0)
    why = ac._m2_refusal_reason(str(tmp_path), "F360M", "_o002")
    assert why and "REFUSED" in why, why
    # ... and a genuinely absent baseline is NOT reported as a refusal
    assert ac._m2_refusal_reason(str(tmp_path), "F480M", "_o002") is None
    # ... nor is an unambiguous filter's legacy record
    monkeypatch.setattr(ac, "_filter_is_obs_ambiguous", lambda *a, **k: False)
    assert ac._m2_refusal_reason(str(tmp_path), "F360M", "_o002") is None


def test_the_m7_crossfilter_record_is_wired_at_the_PRODUCTION_call(tmp_path, monkeypatch):
    """`run_crossfilter_checkpoint` grew `obs_token` and the CLI wired it, but
    `cataloging._run_crossfilter_astrom_checkpoint` passed neither a
    `record_dir` nor a token -- so the pipeline path wrote no record at all,
    and the previous test string-matched the CALLEE's source rather than
    asserting at the call site, which is why it could not see that.
    """
    import inspect
    from jwst_gc_pipeline.photometry import cataloging
    sig = inspect.signature(cataloging._run_crossfilter_astrom_checkpoint)
    assert "record_dir" in sig.parameters and "obs_token" in sig.parameters

    seen = {}

    def _spy(catalogs, refcat=None, basepath=None, context="",
             record_dir=None, obs_token="", **kw):
        seen.update(record_dir=record_dir, obs_token=obs_token)
        return dict(passed=True)

    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    monkeypatch.setattr(ac, "run_crossfilter_checkpoint", _spy)
    from astropy.table import Table
    paths = {}
    for filt in ("F212N", "F410M"):
        fn = tmp_path / f"{filt}.fits"
        Table({"skycoord_ra": [266.5], "skycoord_dec": [-28.7]}).write(fn)
        paths[filt] = str(fn)
    cataloging._run_crossfilter_astrom_checkpoint(
        paths, str(tmp_path), str(tmp_path), {"refcat": None},
        context="test", record_dir=str(tmp_path / "recs"),
        obs_token="_o002")
    assert seen.get("record_dir", "").endswith("recs"), seen
    assert seen.get("obs_token") == "_o002", seen


def test_the_frozen_stage_says_MISSING_BASELINE_not_MOVED(tmp_path, monkeypatch):
    """The fourth member of the reader-call-site set, and the one that was
    missing.

    `_m2_refusal_reason` returns None for an empty token by its first line, so
    dropping the token at its ONE production call site turns the whole
    MISSING-BASELINE path off and the caller falls back to the frozen-stage
    movement text -- the false statement that ask existed to remove.  The three
    tests covering it called the helper directly, which is the same shape as
    the mutants they were written to kill.  This drives the frozen stage.
    """
    import jwst_gc_pipeline.photometry.astrometry_checkpoint as ac
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        AstrometryRegressionError, run_visit_checkpoint)
    from .test_visit_consensus import _exposure_table, _field

    monkeypatch.setattr(ac, "_filter_is_obs_ambiguous", lambda *a, **k: True)
    # one exposure MISaligned, which is what reaches the no-baseline branch
    ra, dec = _field(n=400)
    tables = [_exposure_table(ra, dec, exposure=e, filtername="F360M",
                              module=f"nrcb{e}") for e in (1, 2, 3)]
    tables.append(_exposure_table(ra, dec, exposure=4, filtername="F360M",
                                  module="nrcb4", dra_mas=40.0))
    # an untokened m2 record exists, describing exposures this run does not
    # have -- so every exposure of the frozen stage lands with no baseline
    (tmp_path / "checkpoint_m2_F360M_latest.json").write_text(json.dumps(
        dict(stage="m2", filtername="F360M", visits=[dict(
            visit="1", filtername="F360M",
            exposures=[dict(key=["1", 99, "nrcalong", "F360M", "09901"],
                            dra=0.0, ddec=0.0, ok=True)])])))
    with pytest.raises(AstrometryRegressionError) as exc:
        run_visit_checkpoint(tables, "m4", filtername="F360M",
                             basepath=str(tmp_path), record_dir=str(tmp_path),
                             context="test", obs_token="_o002")
    msg = str(exc.value)
    assert "MISSING BASELINE" in msg, msg
    assert "REFUSED" in msg, msg
    # ... and it must NOT claim a movement nobody measured
    assert "MOVED" not in msg, msg
