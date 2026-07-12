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
    monkeypatch.setattr(gate, "build_groups", lambda field, filt: ({}, {}, 0))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True


def test_frames_but_no_detections_is_could_not_verify(monkeypatch):
    monkeypatch.setattr(gate, "build_groups", lambda field, filt: ({}, {}, 12))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is False
    assert r["could_not_verify"] is True


def test_single_group_with_frames_is_a_genuine_pass(monkeypatch):
    monkeypatch.setattr(gate, "build_groups",
                        lambda field, filt: ({"v001:nrca": _coords()},
                                             {"v001:nrca": 500}, 4))
    r = gate.check_filter("x", "F200W", verbose=False)
    assert r["PASS"] is True
    assert not r.get("could_not_verify")


def test_main_exit_2_on_could_not_verify(monkeypatch):
    monkeypatch.setattr(gate, "check_filter",
                        lambda field, filt, refcat=None, verbose=True: dict(
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
                        lambda field, filt, refcat=None, verbose=True: next(results))
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
