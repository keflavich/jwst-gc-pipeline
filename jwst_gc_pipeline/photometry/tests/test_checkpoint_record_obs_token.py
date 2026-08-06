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


def test_untokened_legacy_record_is_still_found(tmp_path, capsys):
    """Every record on disk today predates the token; failing to find a
    baseline at a frozen stage fails closed and stops a healthy field."""
    _write(tmp_path, "checkpoint_m2_F360M", 2.0)
    base = _m2_exposure_baseline(str(tmp_path), "F360M", "1", "_o002")
    assert base[("1", 1, "nrcblong", "F360M", "02101")][0] == 2.0
    out = capsys.readouterr().out
    assert "falling back to the untokened" in out
    assert "#281" in out


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
