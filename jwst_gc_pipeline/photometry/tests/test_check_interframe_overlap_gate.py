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
    monkeypatch.setattr(gate, "build_groups", lambda field, filt, visits=None: ({}, {}, 0))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True


def test_frames_but_no_detections_is_could_not_verify(monkeypatch):
    monkeypatch.setattr(gate, "build_groups", lambda field, filt, visits=None: ({}, {}, 12))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True


def test_single_group_with_frames_is_a_genuine_pass(monkeypatch):
    monkeypatch.setattr(gate, "build_groups",
                        lambda field, filt, visits=None: (
                            {"v001:nrca": _coords()}, {"v001:nrca": 500}, 4))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is True
    assert not r.get("could_not_verify")


def test_main_exit_2_on_could_not_verify(monkeypatch):
    monkeypatch.setattr(gate, "check_filter",
                        lambda field, filt, refcat=None, verbose=True, visits=None: dict(
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
                        lambda field, filt, refcat=None, verbose=True, visits=None: next(results))
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
                        lambda field, filt, visits=None: (
                            groups, {k: len(v) for k, v in groups.items()}, 8))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert not r.get("could_not_verify")   # it was MEASURED (gross), not skipped
    assert r["n_fail"] == 1


def test_driver_zero_offset_passes(monkeypatch):
    groups = _same_footprint_groups(off_arcsec=0.0)
    monkeypatch.setattr(gate, "build_groups",
                        lambda field, filt, visits=None: (
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
                        lambda field, filt, visits=None: (groups, {"a": n, "b": n}, 8))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True
    rc = gate.main(["--field", "x", "--filter", "F200W"])
    assert rc == 2


def test_visits_scoping_filters_stray_programs(monkeypatch, tmp_path):
    """A shared target dir carries stray crf from other programs (brick dir
    holds 2221 o002 = cloudc); --visits must exclude them from the verdict."""
    import os
    pdir = tmp_path / "brick" / "F405N" / "pipeline"
    pdir.mkdir(parents=True)
    names = ["jw02221001001_03101_00001_nrcblong_destreak_o001_crf.fits",
             "jw02221001001_03101_00002_nrcalong_destreak_o001_crf.fits",
             "jw02221002001_03101_00001_nrcblong_destreak_o002_crf.fits"]
    for n in names:
        (pdir / n).write_bytes(b"x")
    monkeypatch.setattr(gate, "BASE", str(tmp_path))
    monkeypatch.setattr(gate, "_detect", lambda path: None)  # count frames only
    _, _, nframes_all = gate.build_groups("brick", "F405N")
    _, _, nframes_scoped = gate.build_groups("brick", "F405N",
                                             visits={"001001"})
    assert nframes_all == 3
    assert nframes_scoped == 2
