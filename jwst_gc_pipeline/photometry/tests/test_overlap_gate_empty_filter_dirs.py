"""An empty filter directory is not a band that failed verification.

``check_interframe_overlap.field_filters`` enumerated ``<field>/<FILT>/pipeline/``
directories and kept every name starting with "F". A directory created and never
populated therefore counted as a band; ``check_filter`` reported "NO crf frames
matched -- cannot verify", and the gate's (correct) fail-closed rule took the
whole field to exit 2.

w51 (2026-08-03) has four such leftovers -- F115W, F200W, F212N, F356W, each
containing zero files -- and they alone blocked a field whose eleven real NIRCam
bands all pass 0 FAIL / 0 could-not-verify.

The fail-closed behaviour must survive for the case it was written for: a
directory that HOLDS products but whose crf glob matches nothing is a real
mismatch (wrong suffix, half-finished reduction) and still has to block.
"""
import importlib.util
import os
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    REPO_ROOT / "scripts" / "release" / "check_interframe_overlap.py")
cio = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cio)


def _pipeline(tmp_path, field, filt):
    p = tmp_path / field / filt / "pipeline"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _declare(monkeypatch, *filts):
    """Pin the fields.yaml declared-filter set the gate sees, so the on-disk
    enumeration logic is tested independently of the live registry."""
    declared = {f.upper() for f in filts}
    monkeypatch.setattr(cio.fields, "declared_filters", lambda field: declared)


def test_empty_filter_dir_is_not_a_band(tmp_path, monkeypatch):
    """Real bands kept, undeclared empty leftovers dropped."""
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch, "F140M", "F150W")
    for filt in ("F140M", "F150W"):
        p = _pipeline(tmp_path, "w51", filt)
        (p / f"jw06151001001_03109_00001_nrca1_align_o001_crf.fits").write_bytes(b"x")
    for filt in ("F115W", "F200W", "F212N", "F356W"):
        _pipeline(tmp_path, "w51", filt)          # created, never populated, undeclared

    assert cio.field_filters("w51") == ["F140M", "F150W"]


def test_directory_with_products_but_no_crf_is_still_reported(tmp_path, monkeypatch):
    """Fail-closed preserved: products present, crf glob matches nothing.

    That is a real mismatch -- a wrong suffix or a half-finished reduction -- and
    must still reach ``check_filter`` so the field blocks, rather than being
    silently skipped as if the band did not exist."""
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch)                         # declared set irrelevant here
    p = _pipeline(tmp_path, "w51", "F444W")
    (p / "jw06151001001_t001_nircam_clear-f444w-merged_i2d.fits").write_bytes(b"x")

    assert cio.field_filters("w51") == ["F444W"]


def test_half_finished_reduction_no_fits_is_still_reported(tmp_path, monkeypatch):
    """The case where ``os.listdir`` and ``*.fits`` diverge: a reduction that
    wrote its association files and died before any FITS.  It HAS files and zero
    ``.fits`` -- a real half-finished mismatch that must still block, not be
    dropped as an empty leftover."""
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch)
    p = _pipeline(tmp_path, "w51", "F250M")
    (p / "jw06151001001_03109_00001_nrca1_asn.json").write_bytes(b"{}")

    assert cio.field_filters("w51") == ["F250M"]


def test_declared_band_with_empty_dir_still_blocks(tmp_path, monkeypatch):
    """A DECLARED band whose directory is empty is a reduction that produced
    nothing -- it must reach check_filter and block, not be dropped as a
    leftover.  An UNDECLARED empty dir is skipped."""
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch, "F140M")
    _pipeline(tmp_path, "w51", "F140M")      # declared, empty -> must block
    _pipeline(tmp_path, "w51", "F115W")      # undeclared, empty -> skipped

    assert cio.field_filters("w51") == ["F140M"]


def test_declared_band_with_no_directory_at_all_blocks(tmp_path, monkeypatch):
    """A band the registry declares but the archive never reduced (no pipeline
    directory at all) must still be reported at release time -- the same
    declared-but-nothing-there class, one step out.  Undeclared missing dirs are
    simply not bands."""
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch, "F140M", "F999M")
    p = _pipeline(tmp_path, "w51", "F140M")  # declared, reduced
    (p / "jw_x_crf.fits").write_bytes(b"x")
    # F999M declared but has NO directory; F250M (undeclared) also absent

    assert cio.field_filters("w51") == ["F140M", "F999M"]


def test_unreadable_dir_is_reported_not_skipped(tmp_path, monkeypatch):
    """glob proves the dir existed a moment ago; if os.listdir then raises
    (removed / re-permissioned mid-scan) we cannot determine emptiness, so the
    band is reported (fail-closed), never silently skipped."""
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch)
    _pipeline(tmp_path, "w51", "F115W")      # undeclared; would normally be skippable

    real_listdir = os.listdir

    def boom(path):
        if "F115W" in str(path):
            raise OSError("permission denied")
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", boom)
    assert cio.field_filters("w51") == ["F115W"]


def test_non_filter_directories_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch)
    p = _pipeline(tmp_path, "w51", "F140M")
    (p / "jw_x_crf.fits").write_bytes(b"x")
    q = _pipeline(tmp_path, "w51", "scratch")
    (q / "jw_y_crf.fits").write_bytes(b"x")

    assert cio.field_filters("w51") == ["F140M"]
