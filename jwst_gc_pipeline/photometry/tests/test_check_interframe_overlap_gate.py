"""The release-gate wrapper (scripts/release/check_interframe_overlap.py) must
FAIL CLOSED: a glob that matches nothing, or frames that yield no detections,
is a could-not-verify (exit 2), never a silent PASS.  MIRI (no ``_destreak``
token in its crf names) must be covered, not silently excluded."""
import importlib.util
import pathlib
import sys

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    REPO_ROOT / "scripts" / "release" / "check_interframe_overlap.py")
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["check_interframe_overlap"] = gate
_SPEC.loader.exec_module(gate)


def _coords(n=500, ra0=266.54, dec0=-28.70, seed=0):
    rng = np.random.default_rng(seed)
    return SkyCoord((ra0 + (rng.random(n) - 0.5) * 0.02) * u.deg,
                    (dec0 + (rng.random(n) - 0.5) * 0.02) * u.deg)


def test_zero_frames_is_could_not_verify_not_pass(monkeypatch):
    monkeypatch.setattr(gate, "build_groups", lambda field, filt, observations=None: ({}, {}, 0))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True


def test_frames_but_no_detections_is_could_not_verify(monkeypatch):
    monkeypatch.setattr(gate, "build_groups", lambda field, filt, observations=None: ({}, {}, 12))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True


def test_single_group_with_frames_is_a_genuine_pass(monkeypatch):
    monkeypatch.setattr(gate, "build_groups",
                        lambda field, filt, observations=None: (
                            {"v001:nrca": _coords()}, {"v001:nrca": 500}, 4))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is True
    assert not r.get("could_not_verify")


def test_main_exit_2_on_could_not_verify(monkeypatch):
    monkeypatch.setattr(gate, "check_filter",
                        lambda field, filt, refcat=None, verbose=True, observations=None: dict(
                            field=field, filt=filt, PASS=False,
                            could_not_verify=True, note="no crf frames matched"))
    rc = gate.main(["--field", "x", "--filter", "F200W"])
    assert rc == 2


def test_main_exit_1_on_measured_fail_beats_noverify(monkeypatch):
    results = iter([
        dict(field="x", filt="F200W", PASS=False, n_fail=3),
        dict(field="x", filt="F212N", PASS=False, could_not_verify=True),
    ])
    monkeypatch.setattr(gate, "check_filter",
                        lambda field, filt, refcat=None, verbose=True, observations=None: next(results))
    monkeypatch.setattr(gate, "field_filters", lambda field: ["F200W", "F212N"])
    rc = gate.main(["--field", "x", "--scan"])
    assert rc == 1


def test_group_key_covers_miri_and_nircam():
    assert gate._group_key(
        "jw01182004001_04101_00001_nrca3_destreak_o004_crf.fits") == "004001:nrca"
    assert gate._group_key(
        "jw01182004002_02101_00002_nrcblong_destreak_o004_crf.fits") == "004002:nrcb"
    # MIRI crf carry NO _destreak token; must group as mirimage, not 'det?'
    assert gate._group_key(
        "jw02221001001_03201_00001_mirimage_o001_crf.fits") == "001001:mirimage"


def _same_footprint_groups(off_arcsec=0.0, n=4000, seed=61):
    rng = np.random.default_rng(seed)
    ra = 266.54 + (rng.random(n) - 0.5) * 0.02
    dec = -28.70 + (rng.random(n) - 0.5) * 0.02
    cosd = np.cos(np.radians(dec))
    g1 = SkyCoord(ra * u.deg, dec * u.deg)
    g2 = SkyCoord((ra + off_arcsec / 3600.0 / cosd) * u.deg, dec * u.deg)
    return {"v001:nrcb": g1, "v002:nrcb": g2}


def test_driver_catches_gross_offset_reference_free(monkeypatch):
    """PR review should-fix: a >grid-margin (brick-1182 v001 ~20" class)
    visit-vs-visit offset empties the mutual-coverage cells (fine layer blind)
    -- the pooled SWEPT layer must still FAIL it with NO reference catalog."""
    groups = _same_footprint_groups(off_arcsec=20.0)
    monkeypatch.setattr(gate, "build_groups",
                        lambda field, filt, observations=None: (
                            groups, {k: len(v) for k, v in groups.items()}, 8))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert not r.get("could_not_verify")   # it was MEASURED (gross), not skipped
    assert r["n_fail"] == 1


def test_driver_zero_offset_passes(monkeypatch):
    groups = _same_footprint_groups(off_arcsec=0.0)
    monkeypatch.setattr(gate, "build_groups",
                        lambda field, filt, observations=None: (
                            groups, {k: len(v) for k, v in groups.items()}, 8))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is True
    assert not r.get("could_not_verify")


def test_driver_unmeasurable_pair_is_could_not_verify_without_refcat(monkeypatch):
    """Neither layer can measure two unrelated populations sharing a footprint
    -> could-not-verify (exit-2 path), NEVER a silent pass."""
    rng = np.random.default_rng(71)
    n = 3000
    ra1 = 266.54 + (rng.random(n) - 0.5) * 0.02
    dec1 = -28.70 + (rng.random(n) - 0.5) * 0.02
    ra2 = 266.54 + (rng.random(n) - 0.5) * 0.02
    dec2 = -28.70 + (rng.random(n) - 0.5) * 0.02
    groups = {"a": SkyCoord(ra1 * u.deg, dec1 * u.deg),
              "b": SkyCoord(ra2 * u.deg, dec2 * u.deg)}
    monkeypatch.setattr(gate, "build_groups",
                        lambda field, filt, observations=None: (groups, {"a": n, "b": n}, 8))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True
    rc = gate.main(["--field", "x", "--filter", "F200W"])
    assert rc == 2


def test_parse_crf_extracts_proposal_observation_module():
    """The precise crf parser yields (proposal, observation, visit, module) and
    a proposal-aware obs_key -- so 1182-o004 and 2221-o001 never collide, and a
    permissive `o*` wildcard is not needed to enumerate frames."""
    p = gate._parse_crf(
        "jw02221001001_03101_00002_nrcalong_destreak_o001_crf.fits")
    assert (p["prop"], p["obs"], p["visit"], p["module"], p["obs_key"]) == (
        "02221", "001", "001", "nrca", "02221-001")
    # MIRI: no _destreak lineage token
    p = gate._parse_crf("jw02221001001_03201_00001_mirimage_o001_crf.fits")
    assert p["module"] == "mirimage" and p["obs_key"] == "02221-001"
    # non-crf / malformed names are rejected (fail-closed, not mis-parsed)
    assert gate._parse_crf("jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits") is None
    assert gate._parse_crf("random_file.fits") is None
    # leading obs and trailing _oOOO_ must AGREE, else parse failure
    assert gate._parse_crf(
        "jw02221001001_03101_00002_nrcalong_destreak_o002_crf.fits") is None


def test_mosaic_obs_keys_handles_combined_and_instruments():
    assert gate._mosaic_obs_keys(
        "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits") == {"02221-001"}
    # combined multi-observation MIRI association -> BOTH observations
    assert gate._mosaic_obs_keys(
        "jw05365-o002-998_t001_miri_clear-f770w-mirimage_data_i2d.fits") == {
            "05365-002", "05365-998"}


def _write_brick_f405n(tmp_path):
    pdir = tmp_path / "brick" / "F405N" / "pipeline"
    pdir.mkdir(parents=True)
    for n in ("jw02221001001_03101_00001_nrcblong_destreak_o001_crf.fits",
              "jw02221001001_03101_00002_nrcalong_destreak_o001_crf.fits",
              "jw02221002001_03101_00001_nrcblong_destreak_o002_crf.fits"):
        (pdir / n).write_bytes(b"x")
    # the released brick F405N mosaic is o001 only
    (pdir / "jw02221-o001_t001_nircam_clear-f405n-merged_i2d.fits").write_bytes(b"x")
    return pdir


def test_observation_scoping_filters_stray_programs(monkeypatch, tmp_path):
    """A shared target dir carries stray crf from other programs/observations
    (the brick dir holds 2221 o002 = cloudc); scoping must exclude them.  The
    DEFAULT derives the released observations from THIS filter directory's
    mosaics (per-directory), so the stray o002 is dropped with no argument."""
    _write_brick_f405n(tmp_path)
    monkeypatch.setattr(gate, "BASE", str(tmp_path))
    monkeypatch.setattr(gate, "_detect", lambda path: None)  # count frames only
    # NO_SCOPE = scoping disabled -> every well-formed crf
    _, _, nframes_all = gate.build_groups("brick", "F405N",
                                          observations=gate.NO_SCOPE)
    # default (None) -> per-directory derivation from the o001 mosaic present
    _, _, nframes_default = gate.build_groups("brick", "F405N")
    # explicit proposal-aware scope
    _, _, nframes_scoped = gate.build_groups("brick", "F405N",
                                             observations={"02221-001"})
    assert nframes_all == 3
    assert nframes_default == 2
    assert nframes_scoped == 2


def test_scope_is_restrict_only_no_readmit_across_instruments(monkeypatch, tmp_path):
    """A broad field-level scope that also lists a MIRI observation sharing the
    stray's proposal-obs key (brick MIRI F2550W and the cloudc NIRCam strays are
    both 2221 o002) must NOT re-admit the stray: the per-directory derivation is
    intersected, never expanded."""
    _write_brick_f405n(tmp_path)
    monkeypatch.setattr(gate, "BASE", str(tmp_path))
    monkeypatch.setattr(gate, "_detect", lambda path: None)
    _, _, nframes = gate.build_groups(
        "brick", "F405N", observations={"02221-001", "01182-004", "02221-002"})
    assert nframes == 2   # o002 NOT re-admitted despite being in the passed set


def test_underivable_scope_warns_and_disables(monkeypatch, tmp_path, capsys):
    """No mosaic on disk -> derivation empty -> scoping OFF but LOUD (a silent
    green-because-nothing-derived is the banned false-agreement class)."""
    pdir = tmp_path / "brick" / "F405N" / "pipeline"
    pdir.mkdir(parents=True)
    (pdir / "jw02221001001_03101_00001_nrcalong_destreak_o001_crf.fits").write_bytes(b"x")
    (pdir / "jw02221002001_03101_00001_nrcblong_destreak_o002_crf.fits").write_bytes(b"x")
    monkeypatch.setattr(gate, "BASE", str(tmp_path))
    monkeypatch.setattr(gate, "_detect", lambda path: None)
    _, _, nframes = gate.build_groups("brick", "F405N")   # no mosaic -> no scope
    assert nframes == 2   # both accepted (scoping disabled)
    assert "scoping DISABLED" in capsys.readouterr().out
