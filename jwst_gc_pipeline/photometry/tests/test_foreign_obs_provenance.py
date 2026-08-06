"""A per-frame catalog's OBSERVATION is read from its provenance, not its name.

The frozen-stage checkpoint globs ``{filt}_*visit*_vgroup*_exp*``.  The ``*``
after the filter swallows the per-observation token, so on a directory holding
two observations the glob is obs-blind.  The name-based filter meant to narrow
it compares this run's token against each file's, and the per-frame writer
emits a token for proposals 2211/7213/6778 only -- so on cloudef (2092,
observations 002 and 005 under one basepath) the comparison was ``'' != ''``
and all 24 F360M catalogs were kept, 8 of them the OTHER observation's frames.
That is the aliasing input behind issue #298.

Every per-frame catalog records the crf it was measured on in
``meta['FILENAME']``, and that path carries the observation.  Reading it splits
the directory correctly without renaming any of the ~110k catalogs on disk.
"""
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.cataloging import (
    _catalog_source_frame, _drop_foreign_obs_duplicates, _resolved_obsid)


def _catalog(dirpath, name, source_frame):
    p = str(dirpath) + "/" + name
    t = Table({"x": [1.0]})
    if source_frame is not None:
        t.meta["FILENAME"] = source_frame
    t.write(p)
    return p


# --------------------------------------------------------------- provenance

def test_untokened_legacy_catalogs_are_split_by_provenance(tmp_path, monkeypatch):
    """The cloudef case: 16 catalogs in one directory, 8 of them observation
    005's frames, NONE of them carrying a token in its name."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = ([_catalog(tmp_path, f"f360m_nrcblong_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                     f"/x/jw02092002001_02101_{i:05d}_nrcblong_destreak_o002_crf.fits")
            for i in range(1, 9)]
           + [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                       f"/x/jw02092005001_02101_{i:05d}_nrcblong_destreak_o005_crf.fits")
              for i in range(1, 9)])
    kept = _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002")
    assert len(kept) == 8
    assert all("_o002_" in _catalog_source_frame(f) for f in kept)


def test_the_untokened_writer_is_what_makes_the_name_check_a_no_op(tmp_path, monkeypatch):
    """Pins the premise.  This run's per-frame token on cloudef is '', so the
    name comparison keeps everything and the provenance read is the only thing
    doing the work.  If the token is ever made universal this test should be
    rewritten, not deleted: the split has to keep working either way."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import obs_token
    assert obs_token("2092", "002") == ""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02092005001_02101_{i:05d}_nrcblong_destreak_o005_crf.fits")
           for i in range(1, 4)]
    assert _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs=None) == fns
    assert _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002") == []


def test_a_catalog_with_unreadable_provenance_is_KEPT(tmp_path, monkeypatch, capsys):
    """Fail-safe.  Dropping real exposures builds a consensus from half the
    detectors and PASSES, which is worse than the duplicate it avoids."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    good = _catalog(tmp_path, "f360m_nrcblong_visit001_vgroup02101_exp00001_m2_daophot_basic.fits",
                    "/x/jw02092002001_02101_00001_nrcblong_destreak_o002_crf.fits")
    blind = _catalog(tmp_path, "f360m_nrcblong_visit001_vgroup02101_exp00002_m2_daophot_basic.fits",
                     None)
    kept = _drop_foreign_obs_duplicates([good, blind], "", "f360m",
                                        "m2", "merged", "cloudef",
                                        target_obs="002")
    assert blind in kept
    assert "KEEPING" in capsys.readouterr().out


def test_a_zero_length_catalog_reads_as_unreadable_not_as_a_crash(tmp_path):
    """What a killed job leaves, and what the repo's end-to-end test simulates
    with ``touch()``.  The two readers fail differently on it -- ``getheader``
    with OSError, ``Table.read`` with IORegistryError, because it cannot even
    guess the format -- so both have to be named and neither may propagate."""
    empty = tmp_path / "f360m_nrcb_visit001_vgroup02101_exp00001_m2_daophot_basic.fits"
    empty.touch()
    assert _catalog_source_frame(str(empty)) is None
    empty_ecsv = tmp_path / "f360m_nrcb_visit001_vgroup02101_exp00002_m2_daophot_basic.ecsv"
    empty_ecsv.touch()
    assert _catalog_source_frame(str(empty_ecsv)) is None


def test_provenance_is_read_from_the_header_not_the_rows(tmp_path, monkeypatch):
    """This runs once per candidate catalog before every frozen-stage
    checkpoint.  ``Table.read`` pulls the whole binary table to reach one
    keyword: 0.60 s/file against 0.071 s for ``fits.getheader(ext=1)``,
    measured on brick F212N -- ~4 core-hours per archive pass per stage."""
    fn = _catalog(tmp_path, "f360m_nrcb_visit001_vgroup02101_exp00001_m2_daophot_basic.fits",
                  "/x/jw02092002001_02101_00001_nrcblong_destreak_o002_crf.fits")

    def _no(*a, **k):
        raise AssertionError("read the whole table to get one header keyword")

    monkeypatch.setattr(Table, "read", staticmethod(_no))
    assert _catalog_source_frame(fn).endswith("_o002_crf.fits")


def test_single_observation_fields_are_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 1)
    fns = [_catalog(tmp_path, f"f212n_nrcb1_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02221001001_02101_{i:05d}_nrcb1_destreak_o001_crf.fits")
           for i in range(1, 5)]
    kept = _drop_foreign_obs_duplicates(fns, "", "f212n", "m2", "merged",
                                        "brick", target_obs="001")
    assert len(kept) == 4


# ------------------------------------------------------------ obsid resolution

class _Opt:
    def __init__(self, **kw):
        self.target = None
        self.proposal_id = None
        self.field = None
        self.modules = "merged"
        for k, v in kw.items():
            setattr(self, k, v)


def test_an_explicit_field_wins():
    assert _resolved_obsid(_Opt(target="cloudef", proposal_id="2092",
                                field="005")) == "005"


def test_a_multi_observation_target_refuses_to_guess():
    """``fields.default_field_token`` answers ``obsids[0]`` -- on cloudef that
    is the same value whichever observation is running.  Using it here would
    make one of the two runs read its own frames as foreign, drop them, and
    PASS on the other observation's exposures.  None means "keep everything",
    which is what happened before this filter existed."""
    got = _resolved_obsid(_Opt(target="cloudef", proposal_id="2092", field=None))
    assert got is None, got


def test_a_single_observation_target_resolves():
    """Nothing to guess: the registry lists one obsid."""
    assert _resolved_obsid(_Opt(target="brick", proposal_id="2221",
                                field=None)) == "001"


def test_an_unregistered_target_resolves_to_none():
    assert _resolved_obsid(_Opt(target="not-a-field", proposal_id="9999")) is None
    assert _resolved_obsid(_Opt()) is None


# ------------------------------------------------------------ production wiring

def test_the_checkpoint_passes_its_own_observation_down(tmp_path, monkeypatch):
    """BEHAVIOURAL, not a source grep.  There is exactly one production call of
    ``_drop_foreign_obs_duplicates``, and every unit test above supplies
    ``target_obs`` itself -- so deleting the line that wires it leaves them all
    passing.  This one goes through ``_run_astrometry_stage_checkpoint``."""
    cut_bp = tmp_path / "cutouts" / "merged"
    (cut_bp / "F360M").mkdir(parents=True)
    for i in (1, 2):
        _catalog(cut_bp / "F360M",
                 f"f360m_nrcblong_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                 f"/x/jw02092002001_02101_{i:05d}_nrcblong_destreak_o002_crf.fits")

    seen = {}

    def _spy(fns, obs_token, filt, merge_label, module, target, target_obs=None):
        seen["target_obs"] = target_obs
        seen["n_in"] = len(fns)
        return []          # empty -> m2 prints "cannot run" and returns

    monkeypatch.setattr(cataloging, "_drop_foreign_obs_duplicates", _spy)
    monkeypatch.delenv("ASTROM_CHECKPOINT", raising=False)
    opts = _Opt(target="cloudef", proposal_id="2092", field="005",
                each_exposure=True, cutout_region="")
    cataloging._run_astrometry_stage_checkpoint(
        "m2", "merged", "F360M", str(cut_bp), str(tmp_path), "2092",
        opts, {}, context="test")
    assert seen.get("target_obs") == "005", seen
    assert seen.get("n_in") == 2, seen


# ------------------------------------------- the fail-safe must not fail CLOSED

def test_a_source_that_names_no_observation_is_KEPT_not_dropped(tmp_path, monkeypatch, capsys):
    """`want not in basename(src)` treated "this path does not spell an
    observation at all" identically to "it spells a DIFFERENT one".

    That is fail-CLOSED inside the one branch whose stated design is that an
    unidentifiable catalog is KEPT, and it empties the input: a source that is
    not a crf (`..._cal.fits`) drops every catalog, and at m1/m12/m2
    `if not fns:` only prints and returns, so the checkpoint silently ceases to
    exist rather than raising.
    """
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02092002001_02101_{i:05d}_nrcblong_cal.fits")
           for i in range(1, 5)]
    kept = _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002")
    assert kept == fns, kept
    assert "KEEPING" in capsys.readouterr().out


def test_a_source_naming_an_unrelated_observation_is_still_dropped(tmp_path, monkeypatch):
    """The relaxation must not become a pardon: a source that DOES name an
    observation, and names one that is not ours, is foreign whether or not that
    observation is a registered sibling."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02092999001_02101_{i:05d}_nrcblong_destreak_o999_crf.fits")
           for i in range(1, 5)]
    assert _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002") == []


def test_a_joint_obsid_source_is_recognised(tmp_path, monkeypatch):
    """sgrb2 registers MIRI 002-998 and sickle 001-002 as JOINT obsids, so a
    provenance path can spell `_o002-998_`."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    ours = [_catalog(tmp_path, f"f770w_mirimage_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                     f"/x/jw05365002001_02101_{i:05d}_mirimage_destreak_o002-998_crf.fits")
            for i in range(1, 3)]
    theirs = [_catalog(tmp_path, f"f770w_mirimage_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits".replace("exp0000", "exp0001"),
                       f"/x/jw05365007001_02101_{i:05d}_mirimage_destreak_o007_crf.fits")
              for i in range(1, 3)]
    kept = _drop_foreign_obs_duplicates(ours + theirs, "", "f770w", "m2",
                                        "merged", "sgrb2",
                                        target_obs="002-998")
    assert kept == ours, kept
