"""The observation token is emitted for every observation (issues #298/#281/#285).

It used to be emitted only for proposals 2211/7213/6778.  Everywhere else the
token was `''`, and every guard keyed on it degraded to "keep everything"
because its comparison became `'' != ''`.  That is not a missing feature, it is
a family of silent no-ops.
"""
import os

import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.cataloging import (
    _catalog_source_frame, _drop_foreign_obs_duplicates)
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import obs_token


def test_every_observation_gets_a_token():
    assert obs_token("2092", "002") == "_o002"
    assert obs_token("2092", "005") == "_o005"
    assert obs_token("2221", "001") == "_o001"
    assert obs_token("5365", "001") == "_o001"
    assert obs_token("2211", "023") == "_o023"


def test_two_observations_of_one_proposal_differ():
    """cloudef 2092/002 and 2092/005 share a basepath; this is the collision."""
    assert obs_token("2092", "002") != obs_token("2092", "005")


def test_ngc6334_proposals_are_still_distinguished_by_proposal():
    """7213 and 6778 share a target dir, filters, obs number AND (visit,
    vgroup, exp) tuples -- `_o{field}` alone would still collide."""
    assert obs_token("7213", "001") == "_j7213"
    assert obs_token("6778", "001") == "_j6778"
    assert obs_token("7213", "001") != obs_token("6778", "001")


def test_no_field_still_gives_no_token():
    assert obs_token("2092", "") == ""
    assert obs_token("2092", None) == ""


def _catalog(tmp_path, name, source_frame):
    p = tmp_path / name
    t = Table({"x": [1.0]})
    t.meta["FILENAME"] = source_frame
    t.write(p)
    return str(p)


def test_untokened_legacy_catalogs_are_split_by_provenance(tmp_path, monkeypatch):
    """The cloudef case: 24 catalogs in one directory, 8 of them observation
    005's frames, none of them carrying the token in its NAME."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = ([_catalog(tmp_path, f"f360m_nrcblong_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                     f"/x/jw02092002001_02101_{i:05d}_nrcblong_destreak_o002_crf.fits")
            for i in range(1, 9)]
           + [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                       f"/x/jw02092005001_02101_{i:05d}_nrcblong_destreak_o005_crf.fits")
              for i in range(1, 9)])
    kept = _drop_foreign_obs_duplicates(fns, "_o002", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002")
    assert len(kept) == 8
    assert all("_o002_" in _catalog_source_frame(f) for f in kept)


def test_a_catalog_with_unreadable_provenance_is_KEPT(tmp_path, monkeypatch, capsys):
    """Fail-safe.  Dropping real exposures builds a consensus from half the
    detectors and PASSES, which is worse than the duplicate it avoids."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    good = _catalog(tmp_path, "f360m_nrcblong_visit001_vgroup02101_exp00001_m2_daophot_basic.fits",
                    "/x/jw02092002001_02101_00001_nrcblong_destreak_o002_crf.fits")
    blind = tmp_path / "f360m_nrcblong_visit001_vgroup02101_exp00002_m2_daophot_basic.fits"
    Table({"x": [1.0]}).write(blind)          # no FILENAME meta
    kept = _drop_foreign_obs_duplicates([good, str(blind)], "_o002", "f360m",
                                        "m2", "merged", "cloudef",
                                        target_obs="002")
    assert str(blind) in kept
    assert "KEEPING" in capsys.readouterr().out


def test_single_observation_fields_are_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 1)
    fns = [_catalog(tmp_path, f"f212n_nrcb1_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02221001001_02101_{i:05d}_nrcb1_destreak_o001_crf.fits")
           for i in range(1, 5)]
    kept = _drop_foreign_obs_duplicates(fns, "_o001", "f212n", "m2", "merged",
                                        "brick", target_obs="001")
    assert len(kept) == 4


def test_skip_if_done_still_finds_untokened_products(tmp_path):
    """Every product on disk predates the universal token; re-cataloging a
    campaign to rename files would be worse than the collisions it prevents."""
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as ccl

    class _Opt:
        daophot = True
        basic_only = True
        proposal_id = "2092"
        field = "002"
        target = "cloudef"
        iteration_label = None
        modules = "merged"
        desaturated = False
        bgsub = False
        epsf = False
        blur = False
        group = False
        each_exposure = True
        each_suffix = ""

    opt = _Opt()
    path = ccl._predict_tblfilename(str(tmp_path), "f360m", "nrcblong", opt,
                                    "1", "02101", "00001",
                                    method="daophot",
                                    basic_or_iterative="basic")
    assert "_o002" in path, path
    legacy = path.replace("_o002", "", 1)
    os.makedirs(os.path.dirname(legacy), exist_ok=True)
    Table({"x": [1.0]}).write(legacy)
    assert ccl._expected_output_exists(str(tmp_path), "f360m", "nrcblong", opt,
                                       "1", "02101", "00001")
