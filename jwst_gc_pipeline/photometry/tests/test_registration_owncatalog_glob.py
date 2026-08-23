"""The own-catalog leg of the registration failsafe must FIND the catalog.

``registration_failsafes.py`` is wired BLOCKING into ``stage_release.py``, and its
own-catalog check resolved the vetted catalog with one literal pattern:

    {filt}_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits

which pins TWO things the name does not always spell -- the module token
(``merged``) and the absence of an observation token.  No match returned ``None``,
the truth simply never entered the check, and nothing said so.  Counted on
``/orange/adamginsburg/jwst`` (2026-08-23):

    sickle          5 vetted m7 catalogs, all `f*_nrcb_...`     0 matched
    gc2211_o023     2, both `..._dao_basic_o023_vetted.fits`    0 matched
    gc2211_o028     2                                           0 matched
    gc2211_o049     2                                           0 matched
    gc2211_o050     6, in BOTH token placements                 0 matched
    sgrb2          14, 8 of them `f*_nrcb_...`                   6 matched

So five fields had never once had this check run, and a green
``registration_failsafes`` meant nothing for them.
"""
import importlib.util
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "registration_failsafes",
    REPO_ROOT / "scripts" / "release" / "registration_failsafes.py")
rf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rf)

TAIL = "indivexp_merged_resbgsub_m7_dao_basic"


def _catalogs(tmp_path, field, *names):
    d = tmp_path / field / "catalogs"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")
    return d


def test_module_tokens_follow_the_view():
    """A per-module view wants that module's catalog, not ``merged``."""
    assert rf.catalog_module_tokens("merged") == ("merged",)
    assert rf.catalog_module_tokens("module-a") == ("nrca", "nrcalong")
    assert rf.catalog_module_tokens("module-b") == ("nrcb", "nrcblong")


def test_sickle_shape_is_found_in_its_module_view(tmp_path, monkeypatch):
    """sickle is nrcb-only: no ``merged`` catalog exists in any band, so the
    merged pattern could never match and its scan runs the module view."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _catalogs(tmp_path, "sickle", f"f187n_nrcb_{TAIL}_vetted.fits")
    assert rf.catalog_candidates("sickle", "F187N", "module-b") == [
        str(tmp_path / "sickle" / "catalogs" / f"f187n_nrcb_{TAIL}_vetted.fits")]
    # and the merged view legitimately finds nothing -- there is no merged catalog
    assert rf.catalog_candidates("sickle", "F187N", "merged") == []


def test_lw_module_spelling_is_covered(tmp_path, monkeypatch):
    """A field names its LW products ``nrcblong`` or ``nrcb``; both belong to
    module family b, exactly as ``module_family`` treats the mosaic tokens."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _catalogs(tmp_path, "sgrb2", f"f405n_nrcblong_{TAIL}_vetted.fits")
    assert len(rf.catalog_candidates("sgrb2", "F405N", "module-b")) == 1


@pytest.mark.parametrize("name", [
    # pre-#469 spelling: token at the END
    f"f200w_merged_{TAIL}_o023_vetted.fits",
    # post-#469 spelling: token after the MODULE
    f"f200w_merged_o023_{TAIL}_vetted.fits",
])
def test_both_observation_token_placements_are_found(tmp_path, monkeypatch, name):
    """gc2211 misses in BOTH placements before and after #469, so a pattern that
    pins one of them fixes nothing."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _catalogs(tmp_path, "gc2211_o023", name)
    assert rf.catalog_candidates("gc2211_o023", "F200W", "merged") == [
        str(tmp_path / "gc2211_o023" / "catalogs" / name)]


def test_a_field_carrying_both_placements_finds_both(tmp_path, monkeypatch):
    """gc2211_o050 has both on disk simultaneously.  Finding one and silently
    dropping the other is how a literal pattern matches half of its own field;
    the pick stays deterministic (sorted)."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _catalogs(tmp_path, "gc2211_o050",
              f"f200w_nrcb_{TAIL}_o050_vetted.fits",
              f"f200w_nrcb_o050_{TAIL}_vetted.fits")
    got = rf.catalog_candidates("gc2211_o050", "F200W", "module-b")
    assert len(got) == 2
    assert got == sorted(got)


def test_a_foreign_observation_token_is_not_swallowed(tmp_path, monkeypatch):
    """The token slot is a 3-digit class, not ``*``: a ``*`` there would also
    span a neighbouring name segment and pull in a different product."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _catalogs(tmp_path, "brick", f"f405n_merged_{TAIL}_o001_downsel_vetted.fits")
    assert rf.catalog_candidates("brick", "F405N", "merged") == []


def test_the_plain_merged_name_still_matches(tmp_path, monkeypatch):
    """The fields this check already covered must keep being covered."""
    monkeypatch.setattr(rf, "BASE", str(tmp_path))
    _catalogs(tmp_path, "brick", f"f405n_merged_{TAIL}_vetted.fits")
    assert len(rf.catalog_candidates("brick", "F405N", "merged")) == 1
    assert len(rf.catalog_candidates("brick", "F405N")) == 1


def test_a_missing_catalog_is_reported_rather_than_dropped():
    """The fail-open half: no catalog used to leave NOTHING in the record, so a
    check that never ran was indistinguishable from one that passed.  It is now
    stated in `unavailable` (reported, non-blocking -- a catalog that has not
    been produced yet is a state of the campaign, not a defect in the mosaics).
    """
    import inspect
    note = rf.no_catalog_note("sickle", "F187N", "module-b")
    for piece in ("sickle", "F187N", "module-b", "nrcb", "own-catalog",
                  "not run"):
        assert piece in note, piece
    # and the scan puts it in the report rather than dropping the band silently
    src = inspect.getsource(rf._scan_view)
    assert "unavailable.append(no_catalog_note(" in src
