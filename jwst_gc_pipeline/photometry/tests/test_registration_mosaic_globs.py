"""The registration failsafe picks a merged mosaic per (field, filter, module).
A permissive ``jw*-o*`` glob would silently ``g[0]`` when a shared target dir
holds a stray mosaic from ANOTHER observation/program (the same contamination
that fed the overlap gate stale 2221 o002 = cloudc crf).  The picker must:
name-validate candidates, and REFUSE (fail closed) on a cross-observation
ambiguity rather than silently resolve it."""
import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "registration_failsafes",
    REPO_ROOT / "scripts" / "release" / "registration_failsafes.py")
rf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rf)


def _touch(pipeline, name):
    (pipeline / name).write_bytes(b"x")


def _pipeline(tmp_path, field, filt):
    p = tmp_path / field / filt / "pipeline"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_mosaic_re_parses_and_rejects():
    m = rf._MOSAIC_RE.match(
        "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")
    assert (m.group("prop"), m.group("obs"), m.group("filt"),
            m.group("module")) == ("02221", "001", "f405n", "merged")
    # a crf is not a mosaic; a stray-program name still parses its own obs
    assert rf._MOSAIC_RE.match(
        "jw02221001001_03101_00001_nrcalong_destreak_o001_crf.fits") is None
    assert rf._MOSAIC_RE.match("random.fits") is None


def test_mosaic_picks_single_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    pdir = _pipeline(tmp_path, "brick", "F405N")
    _touch(pdir, "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")
    got = rf.mosaic("brick", "F405N", "merged")
    assert got is not None and got.endswith(
        "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")


def test_mosaic_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _pipeline(tmp_path, "brick", "F405N")
    assert rf.mosaic("brick", "F405N", "merged") is None


def test_mosaic_refuses_cross_observation_ambiguity(tmp_path, monkeypatch):
    """A stray o002 (cloudc) mosaic misfiled next to the o001 release mosaic
    must RAISE, not silently pick one."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    pdir = _pipeline(tmp_path, "brick", "F405N")
    _touch(pdir, "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")
    _touch(pdir, "jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits")
    with pytest.raises(rf.AmbiguousMosaicError):
        rf.mosaic("brick", "F405N", "merged")


def test_field_bands_validates_names_and_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _touch(_pipeline(tmp_path, "brick", "F405N"),
           "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")
    _touch(_pipeline(tmp_path, "brick", "F200W"),
           "jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits")
    # a mosaic misfiled under the WRONG filter dir must not be reported
    _touch(_pipeline(tmp_path, "brick", "F410M"),
           "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")
    # a malformed / non-mosaic name must be ignored
    _touch(_pipeline(tmp_path, "brick", "F466N"), "notamosaic_i2d.fits")
    assert rf.field_bands("brick") == ["F200W", "F405N"]
