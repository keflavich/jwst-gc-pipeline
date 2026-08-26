"""A release that ships no quality-filtered catalog has to SAY so.

The quality cut is written in exactly one place -- inside ``merge_catalogs``,
which runs at the m7 cross-band merge.  m8 is a forced cross-band FILL, not a
merge: it adds ``forced_filled_*``/``forced_snr_*`` columns to m7's rows and
writes a sibling table, so it never reaches the cut.  No field on either data
root has an m8 ``_qualcuts_oksep*`` table; every field that has one has it at m7.

``discover_catalogs`` takes the qualcuts OF THE SELECTED ITERATION, which is
right -- an m7-filtered subset beside an m8 combined table is two products of
different provenance under one release.  What was wrong is that advancing a
field from m7 to m8 then dropped its filtered catalog with nothing logged: six
fields have already advanced, and their READMEs still described a
``_qualcuts_oksep<proposal>`` file that is not in the tarball.

These pins are about the DISCLOSURE, not the selection: the selection is
unchanged, and a field that ships a filtered subset must still read "shipped".
Issue #450.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from jwst_gc_pipeline.photometry.merge_catalogs import _qualcuts_oksep_suffix

_SPEC = importlib.util.spec_from_file_location(
    "stage_release",
    Path(__file__).resolve().parents[3] / "scripts" / "release" / "stage_release.py")
stage_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_release)

CAT_BASE = stage_release.CAT_BASE


def _catdir(tmp_path, names):
    cat_dir = tmp_path / "catalogs"
    cat_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (cat_dir / name).write_text("x")
    return {"data_dir": tmp_path}


def _full_items(field_cfg, field="w51"):
    return [it for it in stage_release.discover_catalogs(field_cfg, field)
            if it.get("kind") in ("catalog_full", "catalog_qualcut")]


def _qualcut_name(field, iteration=None):
    """The quality-cut filename a given field really writes.

    The oksep suffix carries each field's own proposal, so the fixture asks
    the registry for it rather than spelling one program's token -- the same
    rule the readers follow (`jwst_gc_pipeline/tests/test_no_hardcoded_qualcuts_token.py`).
    """
    iter_token = f"_{iteration}" if iteration else ""
    return f"{CAT_BASE}{iter_token}{_qualcuts_oksep_suffix(field)}.fits"


def test_m8_combined_table_records_the_missing_subset(tmp_path):
    """w51's shape: m7 with its cut, m8 without.  m8 ships, and the record
    names the iteration whose cut exists but is not shipped."""
    cfg = _catdir(tmp_path, [
        f"{CAT_BASE}_resbgsub_m7.fits",
        _qualcut_name("w51", "resbgsub_m7"),
        f"{CAT_BASE}_resbgsub_m8.fits",
    ])
    items = _full_items(cfg)
    assert [it["kind"] for it in items] == ["catalog_full"]
    assert items[0]["iteration"] == "resbgsub_m8"
    assert items[0]["filtered_subset"] == "absent"
    assert items[0]["filtered_subset_at_iteration"] == "resbgsub_m7"

    state = stage_release.filtered_subset_state(items)
    assert state.startswith("absent("), state
    assert "resbgsub_m8" in state and "resbgsub_m7" in state


def test_a_field_that_ships_its_cut_still_reads_shipped(tmp_path):
    """wd1's shape: m7 throughout.  The disclosure must not fire on a release
    that does ship the filtered subset."""
    cfg = _catdir(tmp_path, [
        f"{CAT_BASE}_resbgsub_m7.fits",
        _qualcut_name("wd1", "resbgsub_m7"),
    ])
    items = _full_items(cfg, field="wd1")
    assert {it["kind"] for it in items} == {"catalog_full", "catalog_qualcut"}
    assert all("filtered_subset" not in it for it in items)
    assert stage_release.filtered_subset_state(items) == "shipped"


def test_a_field_with_no_cut_anywhere_says_so(tmp_path):
    """arches's shape: its only quality-cut file carries no iteration token, so
    it matches no iteration and there is nothing to point at.

    Which oksep token that file carries is beside the point -- `QUALCUTS_RE`
    matches any field's -- so the fixture takes the registry's.
    """
    cfg = _catdir(tmp_path, [f"{CAT_BASE}_resbgsub_m7.fits",
                             _qualcut_name("arches")])
    items = _full_items(cfg, field="arches")
    assert [it["kind"] for it in items] == ["catalog_full"]
    assert items[0]["filtered_subset_at_iteration"] is None
    assert "no quality cut written" in stage_release.filtered_subset_state(items)


@pytest.mark.parametrize("items,expected", [
    ([], "not_applicable(no-catalogs)"),
    ([{"category": "catalog", "kind": "seed"}], "not_applicable(no-combined-table)"),
])
def test_states_for_a_release_with_no_combined_table(items, expected):
    assert stage_release.filtered_subset_state(items) == expected


def _readme(tmp_path, items, name="rel"):
    out = tmp_path / name
    out.mkdir(exist_ok=True)
    stage_release.write_readme(out, "w51", "v1.0-test", items, "copy",
                               built_at="2026-08-25T00:00:00-04:00")
    return (out / "README.md").read_text()


def test_the_readme_stops_promising_a_file_that_is_not_there(tmp_path):
    """The downloader-facing half.  A machine key in MANIFEST.json is not the
    surface a downloader reads."""
    absent = [{"category": "catalog", "kind": "catalog_full", "filter": None,
               "iteration": "resbgsub_m8", "observation": None,
               "src": "/x/a.fits", "filtered_subset": "absent",
               "filtered_subset_at_iteration": "resbgsub_m7"}]
    text = _readme(tmp_path, absent)
    assert "ships NO quality-filtered subset" in text
    assert "UNFILTERED" in text
    assert "variant is the" not in text

    shipped = [{"category": "catalog", "kind": "catalog_full", "filter": None,
                "iteration": "resbgsub_m7", "observation": None, "src": "/x/a.fits"},
               {"category": "catalog", "kind": "catalog_qualcut", "filter": None,
                "iteration": "resbgsub_m7", "observation": None, "src": "/x/b.fits"}]
    text = _readme(tmp_path, shipped, name="rel2")
    assert "quality-filtered subset." in text
    assert "ships NO quality-filtered subset" not in text


def test_the_state_reaches_manifest_json(tmp_path, monkeypatch):
    """`stage()` writes it beside `continuity_gate`, so a consumer can tell an
    unfiltered release from one whose filtered table it simply did not fetch."""
    src = tmp_path / "a.fits"
    src.write_text("x")
    items = [{"category": "catalog", "kind": "catalog_full", "filter": None,
              "iteration": "resbgsub_m8", "observation": None, "src": str(src),
              "dest": "catalogs/a.fits",
              "filtered_subset": "absent",
              "filtered_subset_at_iteration": "resbgsub_m7"}]
    root = tmp_path / "releases"
    monkeypatch.setattr(stage_release, "GLOBUS_COLLECTION_ROOT", tmp_path)
    field_dir = stage_release.stage(items, "w51", "v1.0-test", root, "copy",
                                    False)
    manifest = json.loads((field_dir / "MANIFEST.json").read_text())
    assert manifest["filtered_subset"].startswith("absent(")
    assert "continuity_gate" in manifest, "placed beside the sibling gate record"
