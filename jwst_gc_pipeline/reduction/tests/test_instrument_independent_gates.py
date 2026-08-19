"""A MIRI verdict must not gate NIRCam products, or the reverse.

NIRCam and MIRI are independent observations of the same sky -- different
detectors, different exposures, usually a different program entirely (cloudc's
MIRI is 2221-o001 and 2526-o021 against its NIRCam 2221-o002).  A MIRI band that
fails or cannot be verified says nothing about the NIRCam mosaics, so refusing
the field on it withholds good data for a reason that does not apply to it.

cloudc was refused exactly that way on 2026-08-19 -- over a MIRI band its
release did not even ship.

The failing instrument is WITHHELD and recorded; the rest of the release goes
out.  Nothing is loosened: a NIRCam failure still refuses the field.
"""
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "stage_release",
    Path(__file__).resolve().parents[3] / "scripts" / "release" / "stage_release.py")
sr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sr)


def _item(filt, category="image"):
    return {"category": category, "kind": "science", "filter": filt,
            "src": f"/x/{filt}.fits", "dest": f"images/{filt}.fits",
            "observation": None, "iteration": None, "size_bytes": 1}


def test_miri_and_nircam_filters_are_told_apart():
    assert sr._instrument_of(_item("F770W")) == "miri"
    assert sr._instrument_of(_item("F2550W")) == "miri"
    assert sr._instrument_of(_item("F405N")) == "nircam"
    assert sr._instrument_of(_item("F480M")) == "nircam"


def test_a_filterless_product_belongs_to_the_nircam_side():
    """A merged catalog or the offsets table has no filter; it is built from the
    NIRCam bands, so withholding MIRI must not take it with them."""
    assert sr._instrument_of({"category": "catalog", "filter": None}) == "nircam"
    assert sr._instrument_of({"category": "astrometry"}) == "nircam"


@pytest.mark.parametrize("rc,expect", [(1, "FAILED"), (2, "could not run")])
def test_a_failing_miri_band_withholds_miri_and_ships_nircam(rc, expect, capsys):
    """Both failure modes -- a measured misregistration and a gate that could
    not run. Either way the NIRCam release goes out."""
    items = [_item("F405N"), _item("F212N"), _item("F770W")]
    kept, withheld, refusal = sr.gate_by_instrument(
        "cloudc", items, lambda instrument: rc if instrument == "miri" else 0)
    assert refusal is None, "a MIRI verdict must not refuse the field"
    assert [it["filter"] for it in kept] == ["F405N", "F212N"]
    assert expect in withheld["miri"]
    assert "WITHHOLDING MIRI" in capsys.readouterr().out


def test_a_failing_nircam_band_still_refuses_the_field():
    """Nothing is loosened. The NIRCam mosaics are the main deliverable, and a
    release without them is not a release."""
    items = [_item("F405N"), _item("F770W")]
    kept, withheld, refusal = sr.gate_by_instrument(
        "cloudc", items, lambda instrument: 1 if instrument == "nircam" else 0)
    assert refusal and "REFUSING TO STAGE" in refusal
    assert withheld == {}


def test_a_clean_field_is_untouched():
    items = [_item("F405N"), _item("F770W")]
    kept, withheld, refusal = sr.gate_by_instrument(
        "cloudc", items, lambda instrument: 0)
    assert kept == items and withheld == {} and refusal is None


def test_withholding_everything_is_a_refusal_not_an_empty_release():
    """A MIRI-only field whose MIRI is withheld has nothing left; staging an
    empty tree would publish a release that says the field was checked."""
    items = [_item("F770W")]
    kept, withheld, refusal = sr.gate_by_instrument(
        "cloudc", items, lambda instrument: 2)
    assert refusal and "leaves nothing to ship" in refusal


def test_the_gate_runs_once_per_instrument_present():
    """Not once per band: the gate scans an instrument's bands itself."""
    calls = []
    items = [_item("F405N"), _item("F212N"), _item("F770W"), _item("F2550W")]
    sr.gate_by_instrument("cloudc", items,
                          lambda instrument: calls.append(instrument) or 0)
    assert sorted(calls) == ["miri", "nircam"]


def test_the_manifest_records_what_was_withheld(tmp_path, monkeypatch):
    """"This field has no MIRI" and "this field's MIRI was withheld" are
    different facts, and the file list alone cannot tell them apart."""
    import json
    src = tmp_path / "a_i2d.fits"
    from astropy.io import fits
    fits.PrimaryHDU().writeto(src)
    items = [{"category": "image", "kind": "science", "src": str(src),
              "dest": "images/F405N/a_i2d.fits", "filter": "F405N",
              "observation": None, "iteration": None,
              "size_bytes": src.stat().st_size}]
    root = tmp_path / "releases"
    monkeypatch.setattr(sr, "GLOBUS_COLLECTION_ROOT", root)
    field_dir = sr.stage(items, "cloudc", "v9", root, "copy", False,
                         withheld_instruments={"miri": "could not run (rc=2)"})
    manifest = json.loads((field_dir / "MANIFEST.json").read_text())
    assert manifest["withheld_instruments"] == {"miri": "could not run (rc=2)"}


def test_a_release_with_nothing_withheld_says_so_explicitly(tmp_path,
                                                            monkeypatch):
    import json
    src = tmp_path / "a_i2d.fits"
    from astropy.io import fits
    fits.PrimaryHDU().writeto(src)
    items = [{"category": "image", "kind": "science", "src": str(src),
              "dest": "images/F405N/a_i2d.fits", "filter": "F405N",
              "observation": None, "iteration": None,
              "size_bytes": src.stat().st_size}]
    root = tmp_path / "releases"
    monkeypatch.setattr(sr, "GLOBUS_COLLECTION_ROOT", root)
    field_dir = sr.stage(items, "cloudc", "v9", root, "copy", False)
    manifest = json.loads((field_dir / "MANIFEST.json").read_text())
    assert manifest["withheld_instruments"] == {}
