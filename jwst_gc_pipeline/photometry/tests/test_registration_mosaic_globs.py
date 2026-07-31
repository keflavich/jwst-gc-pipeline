"""The registration failsafe picks a merged mosaic per (field, filter, module).
A permissive ``jw*-o*`` glob + ``g[0]`` on an UNSORTED list picked a
non-deterministic file when >1 matched.  The picker must: name-validate
candidates (covering the wide-double bands F150W2/F322W2), pick DETERMINISTICALLY
(sorted), and honour an optional release ``observations`` scope so a misfiled
stray from another observation is excluded -- WITHOUT refusing on a legitimate
multi-observation layout (gc2211) or multi-proposal layout (ngc6334)."""
import importlib.util
import pathlib


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


def test_mosaic_re_parses_wide_double_bands():
    """F150W2 / F322W2 end in '2' -- the old [wmn] class dropped them, which
    fails OPEN (band dropped -> <2 bands -> gate warns instead of checking)."""
    for name, filt in (
            ("jw01979-o002_t001_nircam_clear-f150w2-merged_i2d.fits", "f150w2"),
            ("jw01234-o001_t001_nircam_clear-f322w2-merged_i2d.fits", "f322w2")):
        m = rf._MOSAIC_RE.match(name)
        assert m is not None and m.group("filt") == filt


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


def test_mosaic_deterministic_pick_multi_observation(tmp_path, monkeypatch):
    """>1 observation in one filter dir is a NORMAL layout (gc2211 is multi-obs;
    ngc6334 F200W is two proposals).  The picker must be deterministic (sorted),
    NOT raise -- raising propagated a traceback that stage_release mapped to a
    false 'misregistered' claim about never-measured data."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    pdir = _pipeline(tmp_path, "gc2211", "F200W")
    for o in ("023", "046", "049"):
        _touch(pdir, f"jw02211-o{o}_t001_nircam_clear-f200w-merged_i2d.fits")
    got = rf.mosaic("gc2211", "F200W", "merged")
    assert got is not None and got.endswith(
        "jw02211-o023_t001_nircam_clear-f200w-merged_i2d.fits")  # sorted-first
    # ngc6334 F200W: two proposals, both o001 -> proposal disambiguates the key,
    # still a deterministic pick (no raise)
    p2 = _pipeline(tmp_path, "ngc6334", "F200W")
    _touch(p2, "jw07213-o001_t001_nircam_clear-f200w-merged_i2d.fits")
    _touch(p2, "jw06778-o001_t001_nircam_clear-f200w-merged_i2d.fits")
    assert rf.mosaic("ngc6334", "F200W", "merged").endswith(
        "jw06778-o001_t001_nircam_clear-f200w-merged_i2d.fits")


def test_mosaic_scope_excludes_stray_observation(tmp_path, monkeypatch):
    """With a release scope, a stray o002 misfiled next to the o001 release
    mosaic is excluded; the in-scope o001 is picked."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    pdir = _pipeline(tmp_path, "brick", "F405N")
    _touch(pdir, "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")
    _touch(pdir, "jw02221-o002_t001_nircam_clear-f405n-merged_i2d.fits")
    got = rf.mosaic("brick", "F405N", "merged", observations={"02221-001"})
    assert got.endswith("jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits")
    # out-of-scope-only -> None (not a wrong pick)
    assert rf.mosaic("brick", "F405N", "merged",
                     observations={"09999-999"}) is None


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
