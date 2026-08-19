"""The absolute-frame gate (scripts/release/stage_release.py::check_catalog_on_frame)
must measure the shipped catalog's bulk offset vs the Gaia-tied refcat by the
SANCTIONED same-star method and must EXCLUDE saturated / replaced-saturated
sources.

Regression: including saturated stars + using the raw histogram made brick
F187N (narrow Pa-alpha, hardest-saturating stars, worst centroid bias) read 68
mas -- a FALSE off-frame -- while its clean same-star tie is ~1 mas.  The gate
must (a) drop saturated sources, and (b) still FAIL a genuinely off-frame
(deprecated-frame ~hundreds-of-mas) catalog."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

_SPEC = importlib.util.spec_from_file_location(
    "stage_release",
    Path(__file__).resolve().parents[3] / "scripts" / "release" / "stage_release.py")
stage_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_release)

RA0, DEC0 = 266.54, -28.70


def _refcat(tmp_path, n=6000, seed=3):
    rng = np.random.default_rng(seed)
    ra = RA0 + (rng.random(n) - 0.5) * 0.03
    dec = DEC0 + (rng.random(n) - 0.5) * 0.03
    t = Table({"ra": ra, "dec": dec})
    t["skycoord"] = SkyCoord(ra * u.deg, dec * u.deg)
    p = tmp_path / "refcat.fits"
    t.write(p)
    return SkyCoord(ra * u.deg, dec * u.deg), str(p)


def _catalog(tmp_path, ref, shift_mas=0.0, sat_shift_mas=0.0, sat_frac=0.15, seed=4):
    """A catalog built FROM the refcat positions: unsaturated stars sit on-frame
    (+ tiny jitter + optional real ``shift_mas``); a ``sat_frac`` subset is flagged
    saturated and carries a large RA ``sat_shift_mas`` centroid bias."""
    rng = np.random.default_rng(seed)
    n = len(ref)
    cosd = np.cos(np.radians(DEC0))
    dra = np.full(n, shift_mas) + rng.normal(0, 3.0, n)
    ddec = rng.normal(0, 3.0, n)
    sat = rng.random(n) < sat_frac
    dra[sat] += sat_shift_mas
    ra = ref.ra.deg + dra / 3.6e6 / cosd
    dec = ref.dec.deg + ddec / 3.6e6
    t = Table()
    t["skycoord"] = SkyCoord(ra * u.deg, dec * u.deg)
    t["is_saturated"] = sat
    t["replaced_saturated"] = sat
    src = tmp_path / "cat.fits"
    t.write(src)
    return [{"kind": "catalog_per_filter_vetted", "filter": "F187N",
             "observation": None, "src": str(src)}]


def _use_refcat(monkeypatch, refpath):
    monkeypatch.setattr(stage_release, "FRAME_REFCAT", {"brick": refpath})


def test_on_frame_catalog_passes(monkeypatch, tmp_path):
    ref, refpath = _refcat(tmp_path)
    _use_refcat(monkeypatch, refpath)
    items = _catalog(tmp_path, ref, shift_mas=0.0, sat_shift_mas=0.0)
    assert stage_release.check_catalog_on_frame(items, "brick") == []


def test_saturated_bias_does_not_fail_a_clean_frame(monkeypatch, tmp_path):
    """The F187N regression: unsaturated stars are on-frame but a saturated
    subset carries a +60 mas RA centroid bias.  Excluding saturated -> PASS.
    (Confirm the bias WOULD matter: its magnitude far exceeds the 15 mas tol.)"""
    ref, refpath = _refcat(tmp_path)
    _use_refcat(monkeypatch, refpath)
    items = _catalog(tmp_path, ref, shift_mas=0.0, sat_shift_mas=60.0, sat_frac=0.2)
    assert stage_release.check_catalog_on_frame(items, "brick") == []


def test_genuine_off_frame_still_fails(monkeypatch, tmp_path):
    """A deprecated-frame catalog: ALL sources rigidly shifted ~700 mas (the
    brick-1182 v001 class).  The sweep detects it, refinement refuses, and the
    gate FAILS -- saturated exclusion must not defeat the gross-shift catch."""
    ref, refpath = _refcat(tmp_path)
    _use_refcat(monkeypatch, refpath)
    items = _catalog(tmp_path, ref, shift_mas=700.0, sat_shift_mas=0.0)
    fails = stage_release.check_catalog_on_frame(items, "brick")
    assert len(fails) == 1
    assert fails[0][1] > stage_release.FRAME_TOL_MAS


def test_no_refcat_mapped_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(stage_release, "FRAME_REFCAT", {})
    ref, refpath = _refcat(tmp_path)
    items = _catalog(tmp_path, ref)
    assert stage_release.check_catalog_on_frame(items, "brick") is None


@pytest.mark.localdata
def test_every_mapped_refcat_is_on_disk_and_belongs_to_its_field():
    """A `FRAME_REFCAT` entry pointing at a path that does not exist disables the
    gate for that field SILENTLY: `check_catalog_on_frame` returns `None` on
    `not os.path.exists(refpath)`, which the caller reports as "cannot enforce".
    So a typo, a moved catalog or a copy-paste of a neighbouring field's path
    turns a mapped field back into an unmapped one with the map still claiming
    otherwise.  Both halves are checked here because the second failure mode --
    a path under the WRONG field's directory -- would gate a field against
    another field's sky and read as a gross off-frame."""
    # `localdata`: the on-disk half can only be checked where the survey tree
    # exists, and CI has no /orange.  The path-belongs-to-its-field half needs
    # no data and is checked unconditionally below, so a CI run still catches
    # the copy-paste that would gate a field against another field's sky.
    for field, path in stage_release.FRAME_REFCAT.items():
        assert Path(path).is_file(), f"{field}: refcat missing at {path}"


def test_every_mapped_refcat_path_belongs_to_its_own_field():
    """No data needed: a neighbour's path copy-pasted into a field's entry would
    gate that field against another field's sky and read as a gross off-frame."""
    for field, path in stage_release.FRAME_REFCAT.items():
        assert f"/{field}/" in path, f"{field}: refcat path is not under {field}/"
