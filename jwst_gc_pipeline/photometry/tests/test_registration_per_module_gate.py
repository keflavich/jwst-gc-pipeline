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
