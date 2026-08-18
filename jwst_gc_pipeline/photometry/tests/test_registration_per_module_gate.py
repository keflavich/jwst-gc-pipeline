"""The cross-band registration gate must adapt to a field's module geometry.

Two modules that image ADJACENT, non-overlapping sky (arches, quintuplet, and
sickle's single-module case) never produce a merged mosaic, because there is
nothing to merge.  The gate enumerated bands by their merged mosaic only, so
those fields yielded zero bands, reported ``need >=2 bands for cross-band``, and
``main`` returned 0 -- staging proceeded past a gate that had never run.  Fields
whose modules DO overlap hit a quieter version of the same hole: a band that was
drizzled per-module but never merged (cloudc F182M, sgrc F115W/F162M, cloudef
F162M/F210M, sgrb2 F150W as of 2026-08-03) was simply absent from the scan.

So: gate PER MODULE when the modules are disjoint or only one was used, gate the
merged mosaic when they overlap, and make "could not verify" block rather than
pass.
"""
import importlib.util
import pathlib

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "registration_failsafes",
    REPO_ROOT / "scripts" / "release" / "registration_failsafes.py")
rf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rf)


def _pipeline(tmp_path, field, filt):
    p = tmp_path / field / filt / "pipeline"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mosaic_file(path, ra_c, dec_c, npix=256, scale=1.0 / 3600.0):
    """A tiny plain-TAN i2d-shaped mosaic centred on (ra_c, dec_c), all-valid."""
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2, npix / 2]
    w.wcs.crval = [ra_c, dec_c]
    w.wcs.cdelt = [-scale, scale]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = w.to_header()
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.ones((npix, npix), dtype="f4"), hdr)]
                 ).writeto(path, overwrite=True)
    return str(path)


def _name(prop, obs, filt, module):
    return f"jw{prop}-o{obs}_t001_nircam_clear-{filt}-{module}_i2d.fits"


# ---------------------------------------------------------------------------
# inventory: bands are found by ANY module product, not just merged
# ---------------------------------------------------------------------------

def test_inventory_finds_bands_with_no_merged_mosaic(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    a = _pipeline(tmp_path, "arches", "F212N")
    (a / _name("02045", "001", "f212n", "nrca")).write_bytes(b"x")
    (a / _name("02045", "001", "f212n", "nrcb")).write_bytes(b"x")
    b = _pipeline(tmp_path, "arches", "F323N")
    (b / _name("02045", "001", "f323n", "nrca")).write_bytes(b"x")

    assert rf.field_bands("arches") == []          # the old, merged-only view
    inv = rf.field_band_mosaics("arches")
    assert sorted(inv) == ["F212N", "F323N"]
    assert sorted(inv["F212N"]) == ["nrca", "nrcb"]


def test_inventory_honours_observation_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    p = _pipeline(tmp_path, "brick", "F405N")
    (p / _name("02221", "001", "f405n", "merged")).write_bytes(b"x")
    (p / _name("02221", "002", "f405n", "nrca")).write_bytes(b"x")   # stray obs
    inv = rf.field_band_mosaics("brick", observations={"02221-001"})
    assert sorted(inv["F405N"]) == ["merged"]


def test_module_family_ignores_the_long_suffix():
    """arches names its LW mosaics ``-nrca``, not ``-nrcalong``; both are the
    same piece of sky and must land in the same family."""
    assert rf.module_family("nrca") == rf.module_family("nrcalong") == "a"
    assert rf.module_family("nrcb") == rf.module_family("nrcblong") == "b"


# ---------------------------------------------------------------------------
# geometry: real shared DATA, not touching bounding boxes
# ---------------------------------------------------------------------------

def test_modules_overlap_detects_shared_data(tmp_path):
    pa = _mosaic_file(tmp_path / "a.fits", 266.0, -28.0)
    pb = _mosaic_file(tmp_path / "b.fits", 266.0, -28.0)   # same sky
    r = rf.modules_overlap(pa, pb)
    assert r["overlaps"] and r["n_both"] > rf.MIN_OVERLAP_SAMPLES


def test_modules_overlap_false_for_adjacent_footprints(tmp_path):
    """The arches/quintuplet case: two modules side by side, no shared pixel."""
    pa = _mosaic_file(tmp_path / "a.fits", 266.0, -28.0)
    pb = _mosaic_file(tmp_path / "b.fits", 266.1, -28.0)   # ~6' away
    r = rf.modules_overlap(pa, pb)
    assert not r["overlaps"] and r["n_both"] == 0


def test_geometry_single_module(tmp_path, monkeypatch):
    """sickle: nrcb only -- no seam exists, so a per-module gate is complete."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F187N", "F335M"):
        p = _pipeline(tmp_path, "sickle", filt)
        _mosaic_file(p / _name("01182", "004", filt.lower(), "nrcb"), 266.0, -28.0)
    g = rf.field_module_geometry("sickle")
    assert g["mode"] == "single-module" and g["families"] == ["b"]


def test_geometry_disjoint_and_overlapping(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    p = _pipeline(tmp_path, "arches", "F212N")
    _mosaic_file(p / _name("02045", "001", "f212n", "nrca"), 266.0, -28.0)
    _mosaic_file(p / _name("02045", "001", "f212n", "nrcb"), 266.1, -28.0)
    assert rf.field_module_geometry("arches")["mode"] == "disjoint"

    q = _pipeline(tmp_path, "cloudc", "F182M")
    _mosaic_file(q / _name("02221", "002", "f182m", "nrca"), 266.5, -28.5)
    _mosaic_file(q / _name("02221", "002", "f182m", "nrcb"), 266.5, -28.5)
    assert rf.field_module_geometry("cloudc")["mode"] == "overlapping"


# ---------------------------------------------------------------------------
# scan_field: which mosaics get gated, and the tri-state verdict
# ---------------------------------------------------------------------------

def _stub_checks(monkeypatch, passing=True):
    """Replace the expensive detection/matching with a deterministic verdict,
    recording which mosaics were actually gated.  Detections stay real SkyCoords
    so the truth pooling in ``_scan_view`` runs for real; only the per-cell
    verdict is stubbed."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    seen = []

    def _detect(path, thr=30.0):
        seen.append(pathlib.Path(path).name)
        n = 20
        return (SkyCoord(np.linspace(266.0, 266.01, n) * u.deg,
                         np.linspace(-28.0, -27.99, n) * u.deg),
                np.ones(n))

    def _per_cell(det, flux, truth, label, bright_pct=None,
                  fail_min_ratio=None):
        return dict(label=label, PASS=passing, n_fail=0 if passing else 3)

    monkeypatch.setattr(rf, "detect", _detect)
    monkeypatch.setattr(rf, "per_cell", _per_cell)
    monkeypatch.setattr(rf, "catalog_sc", lambda field, filt: None)
    return seen


def test_disjoint_field_gates_each_module_separately(tmp_path, monkeypatch):
    """arches: no merged mosaic anywhere.  Both modules must be gated on their
    own, and the field must reach a real verdict rather than 'need >=2 bands'."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt, ra in (("F212N", 266.0), ("F187N", 266.0)):
        p = _pipeline(tmp_path, "arches", filt)
        _mosaic_file(p / _name("02045", "001", filt.lower(), "nrca"), ra, -28.0)
        _mosaic_file(p / _name("02045", "001", filt.lower(), "nrcb"), ra + 0.1, -28.0)
    seen = _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("arches", verbose=False, images_only=True)
    assert res["geometry"] == "disjoint"
    assert sorted(res["views"]) == ["module-a", "module-b"]
    assert res["PASS"] is True
    assert any("nrca" in n for n in seen) and any("nrcb" in n for n in seen)


def test_disjoint_field_fails_when_one_module_fails(tmp_path, monkeypatch):
    """Accepting the modules separately means EVERY module must pass on its own."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F212N", "F187N"):
        p = _pipeline(tmp_path, "arches", filt)
        _mosaic_file(p / _name("02045", "001", filt.lower(), "nrca"), 266.0, -28.0)
        _mosaic_file(p / _name("02045", "001", filt.lower(), "nrcb"), 266.1, -28.0)
    _stub_checks(monkeypatch, passing=False)
    res = rf.scan_field("arches", verbose=False, images_only=True)
    assert res["PASS"] is False


def test_overlapping_field_missing_merged_band_is_unverified(tmp_path, monkeypatch):
    """cloudc-shaped: modules overlap, most bands merged, one band per-module
    only.  That band's inter-module seam is not covered, so the field is
    UNVERIFIED (None), not PASS."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F187N", "F212N"):
        p = _pipeline(tmp_path, "cloudc", filt)
        _mosaic_file(p / _name("02221", "002", filt.lower(), "merged"), 266.5, -28.5)
        _mosaic_file(p / _name("02221", "002", filt.lower(), "nrca"), 266.5, -28.5)
        _mosaic_file(p / _name("02221", "002", filt.lower(), "nrcb"), 266.5, -28.5)
    p = _pipeline(tmp_path, "cloudc", "F182M")
    _mosaic_file(p / _name("02221", "002", "f182m", "nrca"), 266.5, -28.5)
    _mosaic_file(p / _name("02221", "002", "f182m", "nrcb"), 266.5, -28.5)

    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("cloudc", verbose=False, images_only=True)
    assert res["geometry"] == "overlapping"
    assert res["PASS"] is None
    assert any("F182M" in u and "no merged mosaic" in u
               for u in res["unresolved"])


def test_overlapping_field_all_merged_passes(tmp_path, monkeypatch):
    """The unchanged happy path: every band merged -> gate the merged mosaics."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F187N", "F212N"):
        p = _pipeline(tmp_path, "brick", filt)
        _mosaic_file(p / _name("02221", "001", filt.lower(), "merged"), 266.5, -28.5)
        _mosaic_file(p / _name("02221", "001", filt.lower(), "nrca"), 266.5, -28.5)
        _mosaic_file(p / _name("02221", "001", filt.lower(), "nrcb"), 266.5, -28.5)
    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("brick", verbose=False, images_only=True)
    assert res["geometry"] == "overlapping"
    assert "merged" in res["views"]
    assert res["PASS"] is True


def test_merged_only_field_is_gated_and_labelled(tmp_path, monkeypatch):
    """brick keeps no per-module mosaics, so the geometry cannot be measured.
    That is 'merged-only', not 'unknown': the merged mosaic is all there is and
    gating it is both the only option and the right one."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F187N", "F212N"):
        p = _pipeline(tmp_path, "brick", filt)
        _mosaic_file(p / _name("02221", "001", filt.lower(), "merged"), 266.5, -28.5)
    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("brick", verbose=False, images_only=True)
    assert res["geometry"] == "merged-only"
    assert list(res["views"]) == ["merged"]
    assert res["PASS"] is True


def test_no_mosaics_is_unverified_not_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    res = rf.scan_field("nowhere", verbose=False, images_only=True)
    assert res["PASS"] is None and res["error"]


# ---------------------------------------------------------------------------
# exit code: PASS=None blocks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("passed,expect", [(True, 0), (False, 1), (None, 2)])
def test_main_exit_code_is_tristate(monkeypatch, passed, expect):
    """`PASS: null` used to return 0 -- a green gate that never ran.  Ambiguous
    means we have not explicitly passed, so it blocks (exit 2)."""
    monkeypatch.setattr(
        rf, "scan_field",
        lambda field, **kw: dict(field=field, PASS=passed, geometry="disjoint",
                                 unresolved=["F182M: no merged mosaic"]
                                 if passed is None else []))
    assert rf.main(["--field", "arches", "--scan"]) == expect


# ---------------------------------------------------------------------------
# review findings 1, 2 and 4 -- which must land together (see the sickle test)
# ---------------------------------------------------------------------------

def test_disjoint_field_still_gates_its_merged_mosaic(tmp_path, monkeypatch):
    """m92/gc2211/sgra shape: modules disjoint, but merged mosaics exist and SHIP.

    Gating only per-module opened zero of them. A merged drizzle that places
    module B at the wrong offset, or writes a wrong output WCS, is invisible in
    the per-module views -- and it is the merged product that ships."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt, ra in (("F090W", 266.0), ("F150W", 266.0)):
        p = _pipeline(tmp_path, "m92", filt)
        _mosaic_file(p / _name("01234", "001", filt.lower(), "nrca"), ra, -28.0)
        _mosaic_file(p / _name("01234", "001", filt.lower(), "nrcb"), ra + 0.1, -28.0)
        _mosaic_file(p / _name("01234", "001", filt.lower(), "merged"), ra, -28.0)
    seen = _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("m92", verbose=False, images_only=True)
    assert res["geometry"] == "disjoint"
    assert "merged" in res["views"], res["views"]
    assert any("merged" in n for n in seen), seen


def test_single_module_field_with_one_merged_mosaic_still_passes(tmp_path, monkeypatch):
    """sickle: five bands on one module, and ONE of them also has a merged mosaic.

    A one-band merged view cannot be cross-band-checked against anything, and
    admitting it is not neutral -- it lands in `unresolved` and the field verdict
    becomes None, which BLOCKS.  No re-reduction short of producing four more
    merged mosaics could clear that, so a correct field would be permanently
    unstageable.  Drop the view instead; the module view still gates every band.
    """
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F187N", "F210M", "F335M", "F470N", "F480M"):
        p = _pipeline(tmp_path, "sickle", filt)
        _mosaic_file(p / _name("03958", "007", filt.lower(), "nrcb"), 266.57, -28.80)
    # exactly one band also kept a merged product from an older generation
    p = _pipeline(tmp_path, "sickle", "F210M")
    _mosaic_file(p / _name("03958", "007", "f210m", "merged"), 266.57, -28.80)

    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("sickle", verbose=False, images_only=True)
    assert "merged" not in res["views"], res["views"]
    assert res["PASS"] is True, res
    assert not res.get("unchecked"), res.get("unchecked")


def test_two_band_merged_view_on_a_disjoint_field_is_still_gated(tmp_path, monkeypatch):
    """The drop is at <2 bands, not at 'single module' -- two merged mosaics are
    cross-checkable and must still be gated."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F090W", "F150W"):
        p = _pipeline(tmp_path, "sgra", filt)
        _mosaic_file(p / _name("01939", "001", filt.lower(), "nrcb"), 266.4, -29.0)
        _mosaic_file(p / _name("01939", "001", filt.lower(), "merged"), 266.4, -29.0)
    seen = _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("sgra", verbose=False, images_only=True)
    assert "merged" in res["views"], res["views"]
    assert any("merged" in n for n in seen), seen


def test_disjoint_field_without_merged_is_unchanged(tmp_path, monkeypatch):
    """arches/quintuplet: no merged mosaic exists, so the merged view is empty
    and the disjoint branch behaves exactly as before."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F212N", "F187N"):
        p = _pipeline(tmp_path, "arches", filt)
        _mosaic_file(p / _name("02045", "001", filt.lower(), "nrca"), 266.0, -28.0)
        _mosaic_file(p / _name("02045", "001", filt.lower(), "nrcb"), 266.1, -28.0)
    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("arches", verbose=False, images_only=True)
    assert sorted(res["views"]) == ["module-a", "module-b"]


def test_sole_band_with_passing_own_catalog_is_not_blocked(tmp_path, monkeypatch):
    """Finding 2. A field with one band per channel (a property of its observing
    program, not a defect) whose own-catalog check RAN and PASSED must not exit
    2 -- no re-reduction can ever give it a second SW or LW band, so a gate it
    cannot pass only teaches people to use the override."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F212N", "F405N"):          # one SW, one LW
        p = _pipeline(tmp_path, "sgra", filt)
        _mosaic_file(p / _name("04147", "001", filt.lower(), "merged"), 266.5, -28.5)
    _stub_checks(monkeypatch, passing=True)   # catalog_sc stubbed to None below
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    monkeypatch.setattr(rf, "catalog_sc",
                        lambda field, filt: SkyCoord([266.5] * u.deg, [-28.5] * u.deg))
    res = rf.scan_field("sgra", verbose=False, images_only=False)
    assert res["PASS"] is True, res.get("unresolved")


def test_errored_check_is_not_counted_as_a_pass(tmp_path, monkeypatch):
    """Finding 4. `per_cell` returns dict(error=...) with no PASS key for "too
    few pairs"; `.get("PASS", True)` read that as a pass. Reachable on gc2211,
    whose SW view pools mosaics 13.5 arcmin apart."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F150W", "F200W"):
        p = _pipeline(tmp_path, "gc2211", filt)
        _mosaic_file(p / _name("02211", "023", filt.lower(), "merged"), 266.5, -28.5)
    _stub_checks(monkeypatch, passing=True)
    monkeypatch.setattr(rf, "per_cell",
                        lambda *a, **k: dict(label="x", error="too few pairs"))
    res = rf.scan_field("gc2211", verbose=False, images_only=True)
    assert res["PASS"] is None, res
    assert any("could not be evaluated" in u for u in res["unresolved"])


def test_single_module_one_merged_band_is_dropped_not_blocking(tmp_path, monkeypatch):
    """The reason findings 1 and 4 must land TOGETHER -- sickle.

    sickle is nrcb-only with ONE merged mosaic across its bands. Fix 1 gives it a
    merged view containing a single band, whose cross-band truth ("all OTHER
    bands") is empty. Fix 1 alone -> that view passes vacuously, reintroducing the
    silent pass one layer up. Fix 4 alone -> it returns something falsy, the view
    lands in `unresolved`, and sickle BLOCKS on a band that is not defective, only
    unverifiable -- which no re-reduction could clear.

    Neither is right, because there is no "not covered, and that is fine" verdict
    to classify it INTO: `unresolved` blocks and so does `ungated`.  So the
    distinction is made when the view is BUILT -- a <2-band merged view is not a
    view and is dropped, with a printed line so the drop is not silent.
    """
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    for filt in ("F187N", "F335M", "F470N"):
        p = _pipeline(tmp_path, "sickle", filt)
        _mosaic_file(p / _name("03958", "007", filt.lower(), "nrcb"), 266.5, -28.8)
    p = _pipeline(tmp_path, "sickle", "F210M")
    _mosaic_file(p / _name("03958", "007", "f210m", "nrcb"), 266.5, -28.8)
    _mosaic_file(p / _name("03958", "007", "f210m", "merged"), 266.5, -28.8)

    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("sickle", verbose=False, images_only=True)
    assert res["geometry"] == "single-module"
    assert "merged" not in res["views"], res["views"]
    # every band is still gated, on the module view
    assert "module-b" in res["views"]
    assert res["views"]["module-b"]["bands"] == ["F187N", "F210M", "F335M", "F470N"]
    assert res["PASS"] is True, res
    assert not res["unresolved"], res["unresolved"]
    assert res["views"]["module-b"]["PASS"] is True


# ---------------------------------------------------------------------------
# a band with no same-channel partner: unavailable, not unresolved
#
# The cross-band check pools a band against the field's OTHER bands IN THE SAME
# CHANNEL (`_channel`; SW-vs-LW is excluded deliberately, it yields chance
# pairs).  A field observed in one SW and one LW filter therefore has no partner
# for either band and never can -- it is a property of the observing program,
# not of this release.  With `--images-only` also removing the own-catalog leg,
# arches and quintuplet (F212N + F323N) came back with every band ungated and a
# field verdict of None, which BLOCKS.  A gate a correct field cannot pass under
# any circumstances is a standing instruction to reach for
# --allow-registration-fail, which is the habit this script exists to prevent.
# ---------------------------------------------------------------------------

def _two_channel_field(tmp_path, field="quintuplet", prop="02045", obs="003",
                       filters=("F212N", "F323N")):
    for filt in filters:
        p = _pipeline(tmp_path, field, filt)
        _mosaic_file(p / _name(prop, obs, filt.lower(), "nrca"), 266.0, -28.0)
        _mosaic_file(p / _name(prop, obs, filt.lower(), "nrcb"), 266.1, -28.0)


def test_sole_band_in_its_channel_does_not_block(tmp_path, monkeypatch):
    """quintuplet/arches: F212N is the only SW band, F323N the only LW band."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _two_channel_field(tmp_path)
    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("quintuplet", verbose=False, images_only=True)
    assert res["PASS"] is True, res
    assert not res["unresolved"], res["unresolved"]
    # ...and it is still SAID, on every view, so the summary never implies a
    # check ran that could not have
    assert len(res["unavailable"]) == 4, res["unavailable"]
    assert all("sole" in u for u in res["unavailable"])


def test_sole_band_still_fails_when_a_check_fails(tmp_path, monkeypatch):
    """Not blocking on 'no partner' must not stop a band that IS checked and
    fails from failing.  Here the own-catalog leg runs (images_only=False) and
    returns FAIL."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _two_channel_field(tmp_path)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    _stub_checks(monkeypatch, passing=False)
    monkeypatch.setattr(rf, "catalog_sc", lambda field, filt: SkyCoord(
        np.linspace(266.0, 266.01, 20) * u.deg,
        np.linspace(-28.0, -27.99, 20) * u.deg))
    res = rf.scan_field("quintuplet", verbose=False, images_only=False)
    assert res["PASS"] is False, res


def test_partner_that_exists_but_yields_nothing_still_blocks(tmp_path,
                                                             monkeypatch):
    """The exemption is for "no partner CAN exist", never for "the partner did
    not work".  A same-channel sibling whose mosaic will not open is a defect in
    this release and is fixable, so it must keep blocking -- otherwise the
    exemption would swallow the unreadable-mosaic case it sits next to."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    # two SW bands, so F212N HAS a partner...
    _two_channel_field(tmp_path, filters=("F212N", "F187N"))
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    def _detect(path, thr=30.0):
        if "f187n" in pathlib.Path(path).name:
            return None, None          # ...whose mosaic yields no detections
        n = 20
        return (SkyCoord(np.linspace(266.0, 266.01, n) * u.deg,
                         np.linspace(-28.0, -27.99, n) * u.deg), np.ones(n))

    monkeypatch.setattr(rf, "detect", _detect)
    monkeypatch.setattr(rf, "per_cell",
                        lambda *a, **k: dict(label="x", PASS=True, n_fail=0))
    monkeypatch.setattr(rf, "catalog_sc", lambda field, filt: None)
    res = rf.scan_field("quintuplet", verbose=False, images_only=True)
    assert res["PASS"] is None, res
    assert res["unresolved"], res
    assert not res["unavailable"], res["unavailable"]


def test_two_bands_per_channel_needs_no_exemption(tmp_path, monkeypatch):
    """m92-shaped: 2 SW + 2 LW.  Every band has a partner, so nothing is
    exempted and the pass is a fully-checked one."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _two_channel_field(tmp_path, field="m92", prop="01334", obs="001",
                       filters=("F090W", "F150W", "F277W", "F444W"))
    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("m92", verbose=False, images_only=True)
    assert res["PASS"] is True, res
    assert not res["unavailable"], res["unavailable"]
    assert not res["unresolved"], res["unresolved"]


def test_a_pass_records_whether_anything_was_actually_graded(tmp_path,
                                                             monkeypatch):
    """`PASS: True` on an empty report and `PASS: True` on graded checks are
    different facts, and a caller could not tell them apart from the return
    value.  The one-SW-one-LW field under `--images-only` is the first: no
    cross-band partner can exist and the own-catalog leg is switched off, so
    every band lands in `unavailable` and nothing is measured.  It stays a pass
    -- a gate a correct field can never satisfy is a standing instruction to
    reach for the override -- but it says so."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _two_channel_field(tmp_path)
    _stub_checks(monkeypatch, passing=True)
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    res = rf.scan_field("quintuplet", verbose=False, images_only=True)
    assert res["PASS"] is True and res["n_graded"] == 0
    assert res["evidence"] == "none", res

    # ...and the same field on the full path, where own-catalog supplies the
    # evidence, is a pass of the other kind.
    monkeypatch.setattr(rf, "catalog_sc", lambda field, filt: SkyCoord(
        np.linspace(266.0, 266.01, 20) * u.deg,
        np.linspace(-28.0, -27.99, 20) * u.deg))
    res = rf.scan_field("quintuplet", verbose=False, images_only=False)
    assert res["PASS"] is True and res["n_graded"] > 0
    assert res["evidence"] == "graded", res


def test_the_no_view_shortcut_reports_no_evidence_too(tmp_path, monkeypatch):
    """The other PASS-with-nothing-graded exit -- a field whose mosaics never
    form a 2-band view -- must not read as a checked pass either."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _two_channel_field(tmp_path, filters=("F212N",))
    _stub_checks(monkeypatch, passing=True)
    res = rf.scan_field("quintuplet", verbose=False, images_only=True)
    assert res["PASS"] is True
    assert res["evidence"] == "none" and res["n_graded"] == 0, res


def test_a_check_that_errored_is_not_counted_as_graded(tmp_path, monkeypatch):
    """`n_graded` counts checks that produced a VERDICT. A check that returned
    `dict(error=...)` -- too few pairs, missing detections -- is the silent-pass
    hole this script exists to close, so counting it as evidence would put the
    hole back one level up: the field would report evidence for a pass that
    nothing graded."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _two_channel_field(tmp_path, filters=("F212N", "F187N"))
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    def _detect(path, thr=30.0):
        n = 20
        return (SkyCoord(np.linspace(266.0, 266.01, n) * u.deg,
                         np.linspace(-28.0, -27.99, n) * u.deg), np.ones(n))

    monkeypatch.setattr(rf, "detect", _detect)
    monkeypatch.setattr(rf, "per_cell",
                        lambda *a, **k: dict(label="x", error="too few pairs"))
    monkeypatch.setattr(rf, "catalog_sc", lambda field, filt: None)
    res = rf.scan_field("quintuplet", verbose=False, images_only=True)
    assert res["n_graded"] == 0 and res["evidence"] == "none", res
    assert res["unresolved"], res          # errored checks still block
