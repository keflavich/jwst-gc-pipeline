"""The crf filename parser must accept a base-36 activity id.

A JWST product name is ``jw<PPPPP><OOO><VVV>_<GGSAA>_<EEEEE>_<detector>...``
where ``<GGSAA>`` is visit-group + parallel-sequence-id + ACTIVITY id, and the
activity id counts in BASE 36 -- ``0``-``9`` then ``a``-``z``.  The interframe
overlap gate captured that field as ``\\d+``, so every exposure whose activity
id had reached a letter failed to parse and was skipped.

Measured 2026-08-22 over ``/orange/adamginsburg/jwst/*/*/pipeline/*_crf.fits``:
368 real frames in 7 directories were rejected on this alone, and wd1/F200W lost
ALL 96 of its frames -- every one of them is ``_0210b_``.  ``check_filter``
fails closed on zero frames, so the band reported "NO crf frames matched --
cannot verify" and blocked; the cost is that its registration has never been
checked, under a message that names a glob mismatch rather than the parser.

These names are taken verbatim from disk.
"""
import importlib.util
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "check_interframe_overlap",
    REPO_ROOT / "scripts" / "release" / "check_interframe_overlap.py")
cio = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cio)


#: verbatim from /orange/adamginsburg/jwst/wd1/F200W/pipeline/
WD1_F200W = "jw01905001001_0210b_00001_nrca1_destreak_o001_crf.fits"
#: verbatim from /orange/adamginsburg/jwst/wd1/F115W/pipeline/ -- the all-digit
#: sibling that always parsed, so the test can tell "widened" from "broken".
WD1_F115W = "jw01905001001_02107_00001_nrca1_destreak_o001_crf.fits"
#: verbatim from /orange/adamginsburg/jwst/sickle/F210M/pipeline/
SICKLE_F210M = "jw03958007001_0310c_00001_nrcb1_align_o007_crf.fits"
#: verbatim from /orange/adamginsburg/jwst/w51/F560W/pipeline/ -- MIRI, bare
#: lineage, so the widened capture is exercised without a lineage token too.
W51_F560W = "jw06151002001_0210b_00001_mirimage_o002_crf.fits"


def test_a_base36_activity_id_parses():
    """The whole point: a letter in <GGSAA> is an ordinary filename."""
    parsed = cio._parse_crf(WD1_F200W)
    assert parsed is not None, (
        f"{WD1_F200W} is a real frame on disk and must parse; every one of "
        f"wd1/F200W's 96 frames has this shape")
    assert parsed["prop"] == "01905"
    assert parsed["obs"] == "001"
    assert parsed["visit"] == "001"
    assert parsed["det"] == "nrca1"
    assert parsed["module"] == "nrca"
    assert parsed["obs_key"] == "01905-001"


def test_the_activity_id_reaches_exposure_identity_unchanged():
    """``exposure_identity`` keys the lineage selection on the vgroup, so the
    letter has to survive the capture rather than being normalised away."""
    ident = cio.exposure_identity(WD1_F200W)
    assert ident is not None
    assert ident == ("01905", "001", "001", "0210b", "00001", "nrca1")


def test_all_digit_activity_ids_still_parse():
    """Widening must not cost the case that already worked."""
    parsed = cio._parse_crf(WD1_F115W)
    assert parsed is not None
    assert cio.exposure_identity(WD1_F115W)[3] == "02107"


def test_base36_across_instruments_and_lineages():
    for name, det, module in ((SICKLE_F210M, "nrcb1", "nrcb"),
                              (W51_F560W, "mirimage", "mirimage")):
        parsed = cio._parse_crf(name)
        assert parsed is not None, name
        assert parsed["det"] == det
        assert parsed["module"] == module


def test_the_two_all_digit_fields_stay_strict():
    """Only <GGSAA> is base 36.  Proposal, observation, visit and the exposure
    counter are decimal, and a letter in any of them is still a parse failure --
    otherwise this widening would start admitting names it has no business
    reading."""
    assert cio._parse_crf(
        "jw0190a001001_0210b_00001_nrca1_destreak_o001_crf.fits") is None
    assert cio._parse_crf(
        "jw01905001001_0210b_0000a_nrca1_destreak_o001_crf.fits") is None
    assert cio._parse_crf(
        "jw01905001001_0210b_00001_nrca1_destreak_o00a_crf.fits") is None


def test_the_observation_agreement_rule_still_applies():
    """A name whose leading observation and trailing ``_oOOO_`` disagree is a
    parse failure, base-36 activity id or not."""
    assert cio._parse_crf(
        "jw01905001001_0210b_00001_nrca1_destreak_o002_crf.fits") is None
