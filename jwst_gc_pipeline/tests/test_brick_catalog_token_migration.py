"""The brick token migration renames the right files and nothing else.

PR #597 tokened brick's per-filter merged catalogs; every product on disk
predates it, so the m7 crossband seed's glob matches nothing and the chain is
dead (#625).  This tool renames the existing files to the tokened form, which is
cheaper and reversible where the code split was not (#628).
"""
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = (Path(__file__).resolve().parents[2] / "scripts" / "reduction"
        / "migrate_brick_catalog_tokens.py")
_spec = importlib.util.spec_from_file_location("brickmig", TOOL)
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)


def _touch(d, *names):
    for n in names:
        (d / n).write_text("")


def test_each_filter_maps_to_its_own_proposal():
    """The mapping is what makes the rename unambiguous -- disjoint filters."""
    assert {f for f, t in mig.OWNER.items() if t == "_o004"} == {
        "f115w", "f200w", "f356w", "f444w"}
    assert {f for f, t in mig.OWNER.items() if t == "_o001"} == {
        "f182m", "f187n", "f212n", "f405n", "f410m", "f466n"}
    assert len(mig.OWNER) == 10, "brick has ten bands across its two proposals"


def test_current_generation_files_are_renamed(tmp_path):
    _touch(tmp_path,
           "f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits",
           "f115w_nrca_indivexp_merged_m2_dao_basic.fits")
    pairs, _skipped = mig.plan(str(tmp_path))
    got = {Path(o).name: Path(n).name for o, n in pairs}
    assert got["f182m_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits"] == \
        "f182m_merged_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits"
    assert got["f115w_nrca_indivexp_merged_m2_dao_basic.fits"] == \
        "f115w_nrca_o004_indivexp_merged_m2_dao_basic.fits"


def test_legacy_products_are_left_alone(tmp_path):
    """LOCKED / XFILT / crowdsource predate this naming; nothing reads them."""
    _touch(tmp_path,
           "f115w_merged_indivexp_LOCKED_dao_basic.fits",
           "f115w_merged_indivexp_XFILT_dao_basic.fits",
           "f115w_merged_indivexp_merged_crowdsource_nsky0.fits")
    pairs, skipped = mig.plan(str(tmp_path))
    assert pairs == []
    assert len(skipped) == 3


def test_already_tokened_files_are_not_touched_twice(tmp_path):
    _touch(tmp_path,
           "f182m_merged_o001_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits")
    pairs, _ = mig.plan(str(tmp_path))
    assert pairs == []


def test_unrelated_catalogs_are_ignored(tmp_path):
    """The cross-band combined tables and consensus catalogs keep their names."""
    _touch(tmp_path,
           "basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits",
           "f182m_o001_consensus.fits")
    pairs, _ = mig.plan(str(tmp_path))
    assert pairs == []


def test_apply_renames_and_the_manifest_undoes_it(tmp_path):
    name = "f405n_merged_indivexp_merged_resbgsub_m7_dao_basic.fits"
    _touch(tmp_path, name)
    assert mig.main(["--catalogs", str(tmp_path), "--apply"]) == 0
    assert not (tmp_path / name).exists()
    assert (tmp_path / name.replace("_merged_indivexp",
                                    "_merged_o001_indivexp")).exists()

    manifest = next(tmp_path.glob("_rename_manifest_per_obs_token_*.json"))
    assert len(json.load(open(manifest))) == 1
    assert mig.main(["--catalogs", str(tmp_path), "--undo", str(manifest)]) == 0
    assert (tmp_path / name).exists(), "undo must restore the original name"


def test_it_refuses_rather_than_clobbering(tmp_path):
    """If a target name already exists, stop -- do not overwrite a catalog."""
    old = "f212n_merged_indivexp_merged_resbgsub_m7_dao_basic.fits"
    _touch(tmp_path, old, old.replace("_merged_indivexp", "_merged_o001_indivexp"))
    assert mig.main(["--catalogs", str(tmp_path), "--apply"]) == 2
    assert (tmp_path / old).exists(), "nothing may be renamed on refusal"


def test_dry_run_is_the_default(tmp_path):
    name = "f187n_merged_indivexp_merged_resbgsub_m5_dao_basic.fits"
    _touch(tmp_path, name)
    assert mig.main(["--catalogs", str(tmp_path)]) == 0
    assert (tmp_path / name).exists(), "a bare invocation must change nothing"
