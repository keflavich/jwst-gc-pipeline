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
    monkeypatch.setattr(cio.fields, "declared_filters",
                        lambda field, *a, **kw: declared)


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


def test_niriss_only_band_is_not_a_missing_nircam_directory(tmp_path, monkeypatch):
    """``field_filters`` enumerates the NIRCam/MIRI ``{FILTER}/pipeline/`` tree.

    A band declared for NIRISS alone reduces to a different layout, so it can
    NEVER have a directory here -- reporting it as "declared, never reduced"
    would block a correct field with no reduction able to clear it.  sgrc is the
    live case: it declares F158M/F200W/F356W for NIRISS only.  A gate a correct
    field cannot pass is a gate that teaches people to use the override.
    """
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    p = _pipeline(tmp_path, "sgrc", "F212N")
    (p / "jw04147012001_03109_00001_nrca1_destreak_o012_crf.fits").write_bytes(b"x")

    # The real registry, not a pinned set: this is a claim about fields.yaml.
    got = set(cio.field_filters("sgrc"))
    niriss_only = (cio.fields.declared_filters("sgrc", instruments=("niriss",))
                   - cio.fields.declared_filters("sgrc"))
    assert {"F158M", "F200W", "F356W"} <= niriss_only, (
        "sgrc's NIRISS-only declaration changed; update this test")
    assert not (got & niriss_only), (
        f"NIRISS-only bands reported against the NIRCam tree: {got & niriss_only}")


def test_declared_filters_default_excludes_niriss():
    """The union is the defect; the default must be NIRCam+MIRI only."""
    nircam_miri = cio.fields.declared_filters("sgrc")
    niriss = cio.fields.declared_filters("sgrc", instruments=("niriss",))
    both = cio.fields.declared_filters("sgrc",
                                       instruments=("nircam", "miri", "niriss"))
    assert niriss - nircam_miri, "sgrc no longer has a NIRISS-only band"
    assert both == nircam_miri | niriss
    # F480M is declared for both, so it must survive the narrowing.
    assert "F480M" in nircam_miri and "F480M" in niriss


def test_on_disk_names_keep_their_directory_casing(tmp_path, monkeypatch):
    """The returned name is a PATH COMPONENT, not just a label.

    ``check_filter`` globs ``{BASE}/{field}/{filt}/pipeline/...`` with it, so
    normalising a lower-case directory to upper case would make that glob match
    nothing and the field would block with "no crf frames" -- a correct field
    failing a gate no reduction can clear, which is the same anti-pattern as the
    NIRISS union above.  A declared band matches case-insensitively regardless.
    """
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch, "F140M")
    p = _pipeline(tmp_path, "w51", "f140m")      # lower-case on disk, declared
    (p / "jw_x_crf.fits").write_bytes(b"x")

    out = cio.field_filters("w51")
    assert out == ["f140m"], out                 # usable as a path component
    # and it was recognised as the declared band, so it is NOT also appended as
    # "declared with no directory at all".
    assert "F140M" not in out


def test_non_filter_directories_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "BASE", str(tmp_path))
    _declare(monkeypatch)
    p = _pipeline(tmp_path, "w51", "F140M")
    (p / "jw_x_crf.fits").write_bytes(b"x")
    q = _pipeline(tmp_path, "w51", "scratch")
    (q / "jw_y_crf.fits").write_bytes(b"x")

    assert cio.field_filters("w51") == ["F140M"]
