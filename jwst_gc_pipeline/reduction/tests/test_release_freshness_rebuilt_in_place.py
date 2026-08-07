"""Presence is not freshness (issue: the #308 review).

`source_state` resolved supersession as "the source is gone and a quarantined
twin is in its place", which catches only the RENAME.  A mosaic rebuilt IN
PLACE under the same name is a perfectly good file, and the guard called it
live.

Measured across the whole release tree: 115 live entries, 52 of them with a
source whose size no longer matched what was staged.  On cloudc -- the field
the guard is named for -- five of six sources were rebuilt the same morning,
each to a different size, and every one read `live` while the page kept
serving July's bytes.
"""
import importlib.util
import json
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "release_freshness",
    Path(__file__).resolve().parents[3] / "scripts" / "release"
    / "release_freshness.py")
rf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rf)


def _src(tmp_path, name, nbytes):
    p = tmp_path / name
    p.write_bytes(b"\0" * nbytes)
    return str(p)


def test_a_source_rebuilt_in_place_is_REBUILT(tmp_path):
    """The cloudc case: same path, different bytes.

    REBUILT, not QUARANTINED: both withhold the staged copy, but only a rename
    says the pipeline repudiated the file.  A size mismatch is a re-run, a
    re-chunk or a new stage, and the page must not claim otherwise."""
    src = _src(tmp_path, "cloudc-f187n_i2d.fits", 2048)
    assert rf.source_state(src, 2048) == rf.LIVE
    # rebuilt: same name, different size
    Path(src).write_bytes(b"\0" * 2000)
    assert rf.source_state(src, 2048) == rf.REBUILT
    assert rf.is_superseded(rf.source_state(src, 2048))


def test_an_untouched_source_is_still_LIVE(tmp_path):
    src = _src(tmp_path, "brick-f212n_i2d.fits", 4096)
    assert rf.source_state(src, 4096) == rf.LIVE


def test_a_manifest_entry_with_no_size_is_not_condemned(tmp_path):
    """Older manifests may predate `size_bytes`.  Absent evidence is not
    evidence of staleness -- fall back to the presence check rather than
    withholding every image on a field whose manifest is old."""
    src = _src(tmp_path, "w51-f770w_i2d.fits", 1024)
    assert rf.source_state(src, None) == rf.LIVE


def test_the_rename_case_still_works(tmp_path):
    """The check this module was built for must not regress."""
    stem = tmp_path / "cloudc-f182m_i2d"
    quarantined = str(stem) + "_im0_badastrom.fits"
    Path(quarantined).write_bytes(b"\0" * 10)
    assert rf.source_state(str(stem) + ".fits", 1234) == rf.QUARANTINED


def test_a_missing_source_with_no_twin_is_MISSING(tmp_path):
    assert rf.source_state(str(tmp_path / "nope_i2d.fits"), 10) == rf.MISSING


def test_audit_manifest_passes_the_recorded_size_through(tmp_path):
    """The wiring: `audit_manifest` has to hand `size_bytes` to
    `source_state`, or the whole check is inert on real manifests -- and every
    real manifest carries it (115 of 115)."""
    fresh = _src(tmp_path, "a_i2d.fits", 512)
    stale = _src(tmp_path, "b_i2d.fits", 512)
    manifest = {"files": [
        {"dest": "a.fits", "src": fresh, "category": "image",
         "size_bytes": 512},
        {"dest": "b.fits", "src": stale, "category": "image",
         "size_bytes": 999},          # recorded != on disk -> rebuilt
    ]}
    states = rf.audit_manifest(manifest)
    assert states["a.fits"] == rf.LIVE
    assert states["b.fits"] == rf.REBUILT
    assert rf.superseded_files(manifest) == ["b.fits"]
