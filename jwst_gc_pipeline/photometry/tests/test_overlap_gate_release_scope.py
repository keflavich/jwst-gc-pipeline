"""A filter directory belonging to other observations is not a missing product.

``build_groups`` scopes frames to the release's observations by intersecting the
per-directory derivation with the keys the caller passed.  When that intersection
came out EMPTY the scope became an empty set, every frame was filtered out, and
``check_filter`` reported

    NO crf frames matched -- cannot verify inter-frame registration
    (glob mismatch / missing products?)

which names the wrong cause and, being fail-closed, refuses the whole field.

Live case (2026-08-19): cloudc releases NIRCam 2221-o002.  Its ``F770W``
directory holds 8 well-formed ``2526-o021`` crf -- a different program entirely,
imaging a non-overlapping pointing of the same cloud.  The empty intersection
refused a NIRCam-only release over a MIRI band that release never touched, while
the frames sat on disk and parsed cleanly.

The fail-closed rule must survive for the case it exists for: a directory whose
derivation yields NOTHING (no mosaic on disk) may really be glob drift, and still
has to block.
"""
import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    REPO_ROOT / "scripts" / "release" / "check_interframe_overlap.py")
cio = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cio)


def _derives(monkeypatch, keys):
    monkeypatch.setattr(cio, "_field_observations", lambda field, filt: set(keys))


def test_other_observations_are_not_missing_products(monkeypatch, capsys):
    """cloudc's shape: the directory derives 02526-021, the release claims
    02221-002, and the two do not meet."""
    _derives(monkeypatch, {"02526-021"})
    result = cio.check_filter("cloudc", "F770W", observations={"02221-002"})
    assert result["not_in_release"] is True
    assert result["PASS"] is None, "no verdict, rather than a claimed pass"
    assert result["derived"] == ["02526-021"]
    assert not result.get("could_not_verify")
    out = capsys.readouterr().out
    assert "NOT IN THIS RELEASE" in out
    assert "02526-021" in out and "02221-002" in out


def test_an_out_of_scope_band_neither_passes_nor_fails_the_scan(monkeypatch):
    """It must not block, and it must not be counted as a verified pass either:
    the gate has nothing to say about a band this release does not ship."""
    _derives(monkeypatch, {"02526-021"})
    monkeypatch.setattr(cio, "field_filters", lambda field: ["F770W"])
    rc = cio.main(["--field", "cloudc", "--scan",
                   "--observations", "02221-002"])
    assert rc == 0


def test_a_derivation_that_yields_nothing_still_fails_closed(monkeypatch,
                                                            capsys):
    """The case the fail-closed rule was written for: no mosaic on disk, so the
    directory cannot vouch for its own observations. Scoping is disabled and a
    genuinely empty crf glob still blocks."""
    _derives(monkeypatch, set())
    monkeypatch.setattr(cio.glob, "glob", lambda pat: [])
    result = cio.check_filter("cloudc", "F770W", observations={"02221-002"})
    assert result.get("could_not_verify") is True
    assert not result.get("not_in_release")
    assert "NO crf frames matched" in capsys.readouterr().out


def test_an_intersecting_scope_is_unaffected(monkeypatch):
    """The ordinary case: the release claims what the directory derives, so
    nothing is raised and the frames go through."""
    _derives(monkeypatch, {"02221-002", "02526-021"})
    monkeypatch.setattr(cio.glob, "glob", lambda pat: [])
    # no OutOfReleaseScope: an empty frame list here is the fail-closed path
    result = cio.check_filter("cloudc", "F182M", observations={"02221-002"})
    assert result.get("could_not_verify") is True
    assert not result.get("not_in_release")


def test_an_empty_requested_scope_is_not_an_out_of_release_band(monkeypatch,
                                                               capsys):
    """`derived and not scope`, not `not scope`: with NOTHING derived (no mosaic
    on disk) the passed set is used as-is, and a caller passing an empty set
    leaves an empty scope that must still fail closed rather than read as
    "belongs to another release"."""
    _derives(monkeypatch, set())
    monkeypatch.setattr(cio.glob, "glob", lambda pat: [])
    result = cio.check_filter("cloudc", "F770W", observations=set())
    assert result.get("could_not_verify") is True
    assert not result.get("not_in_release")
