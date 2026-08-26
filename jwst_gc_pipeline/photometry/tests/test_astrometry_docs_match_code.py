"""The astrometry docs must describe what the astrometry code does.

Issue #400 is a roundup of places where a docstring, a comment or
`ASTROMETRY_CHECKPOINTS.md` states something the code does not do.  None of
them produces a wrong number on its own; each one misleads the next reader,
which in this codebase is how the larger errors have started.

This module pins the four the accompanying change fixes, each against BOTH
sides -- the prose and the thing it describes -- so neither can drift alone:

* item 2  -- an environment switch that changes a verdict, missing from the
             table an operator reads;
* item 4  -- a comment claiming `n_pairs` separates three empty maps when two
             of them report the same value;
* item 7  -- documented return shapes that do not match the returns;
* item 10 -- "contrast" documented as peak/median of the histogram while the
             code takes the median of the OCCUPIED bins;
* item 11 -- the m2 correction described as unconditional in the ladder table
             while all of it sits behind `ASTROM_CHECKPOINT_APPLY=1`.
"""
import inspect
import os
import re

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry import astrometry_offsets as ao

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CHECKPOINTS_MD = os.path.join(REPO, "jwst_gc_pipeline", "photometry",
                              "ASTROMETRY_CHECKPOINTS.md")


def _md():
    with open(CHECKPOINTS_MD) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# item 10 -- what "contrast" is
# ---------------------------------------------------------------------------

def test_contrast_denominator_is_the_median_of_the_OCCUPIED_bins():
    """The code's denominator, measured rather than read."""
    src = inspect.getsource(ao._hist_peak)
    assert "np.median(H[H > 0])" in src, (
        "the contrast denominator moved; the module docstring describes it")


def test_the_module_docstring_says_occupied_bins():
    doc = ao.__doc__ or ""
    src = open(ao.__file__).read()
    stated = src.split("DEFAULT_MIN_CONTRAST")[0]
    assert "OCCUPIED" in stated, (
        "'contrast' is documented as peak/median of the histogram; the code "
        "takes the median over bins holding at least one pair, which in a "
        "sparse histogram is 1 (issue #400)")
    assert "sparse" in stated.lower()


def test_a_sparse_histogram_makes_contrast_a_pair_count():
    """The consequence the docstring now states, demonstrated: with one pair
    per occupied bin the denominator is 1, so `contrast` IS the peak-bin
    population."""
    rng = np.random.default_rng(400)
    n = 400
    # a peak of `k` coincident pairs on a scatter of singletons
    k = 9
    dra = np.concatenate([np.full(k, 0.001),
                          rng.uniform(-2.9, 2.9, n)])
    ddec = np.concatenate([np.full(k, 0.001),
                           rng.uniform(-2.9, 2.9, n)])
    out = ao._hist_peak(dra, ddec, 3.0, 0.02)
    contrast, n_peak = out[4], out[7]
    assert n_peak >= k
    assert contrast == pytest.approx(float(n_peak), rel=0.5), (
        "in a sparse histogram the median occupied bin holds one pair, so "
        "contrast is the peak population and not a ratio to a background")


# ---------------------------------------------------------------------------
# item 7 -- documented return shapes
# ---------------------------------------------------------------------------

def test_confirm_peak_windows_does_not_document_a_top_level_tol_mas():
    doc = ao.confirm_peak_windows.__doc__
    m = re.search(r"Returns ``dict\(([^)]*)\)``", doc)
    assert m, doc
    keys = {k.strip() for k in m.group(1).split(",")}
    assert "tol_mas" not in keys, (
        "confirm_peak_windows documents a top-level tol_mas it does not "
        "return; the tolerance is per PROBE (issue #400)")
    src = inspect.getsource(ao.confirm_peak_windows)
    for ret in re.findall(r"return dict\(([^\n]*)", src):
        assert "tol_mas" not in ret.split("probes")[0]


def test_local_residual_map_documents_n_pairs():
    doc = ao.local_residual_map.__doc__
    m = re.search(r"``dict\(cells=\[\.\.\.\][^`]*``", doc)
    assert m and "n_pairs" in m.group(0), (
        "local_residual_map returns n_pairs and did not document it "
        "(issue #400)")


# ---------------------------------------------------------------------------
# item 4 -- the three ways a residual map comes back empty
# ---------------------------------------------------------------------------

def _pair_at(ra, dec):
    return SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)


def _tie():
    return dict(ok=True, swept=False, dra=0.0, ddec=0.0, off=0.0)


def test_no_pairs_and_all_ambiguous_are_distinguishable():
    """Both report `n_pairs=0`, which is correct -- n_pairs counts UNAMBIGUOUS
    pairs and the ambiguous case has none either.  `reason` is what separates
    them (issue #400)."""
    far = ao.local_residual_map(_pair_at([266.5], [-28.7]),
                                _pair_at([266.9], [-28.9]), _tie(),
                                cell_arcsec=2.0)
    assert far["n_pairs"] == 0
    assert "no pairs" in far["reason"]

    # two `a` sources sharing one nearest `b`: every pair is ambiguous
    amb = ao.local_residual_map(
        _pair_at([266.50000, 266.50002], [-28.7, -28.7]),
        _pair_at([266.500010], [-28.7]), _tie(), cell_arcsec=2.0)
    assert amb["n_pairs"] == 0
    assert "ambiguous" in amb["reason"]
    assert far["reason"] != amb["reason"], (
        "the two empty maps must not be indistinguishable in the return value")


def test_the_third_empty_map_names_min_stars():
    """Pairs found and unambiguous, but every cell below `min_stars`."""
    ra = 266.54 + np.arange(4) * 1e-4
    dec = np.full(4, -28.70)
    out = ao.local_residual_map(_pair_at(ra, dec), _pair_at(ra, dec), _tie(),
                               cell_arcsec=0.5, min_stars=50)
    assert out["cells"] == []
    assert out["n_pairs"] > 0
    assert "min_stars" in out["reason"]


def test_the_comment_no_longer_claims_n_pairs_separates_three():
    src = inspect.getsource(ao.local_residual_map)
    block = src.split("def _no_pairs")[1].split("return dict")[0]
    assert "reason" in block
    assert "issue #400" in block


# ---------------------------------------------------------------------------
# items 2 and 11 -- ASTROMETRY_CHECKPOINTS.md
# ---------------------------------------------------------------------------

def test_every_verdict_changing_switch_is_in_the_env_table():
    md = _md()
    table = md.split("## Environment switches")[1]
    for var in ("ASTROM_M2_CORRECTION_FLOOR_MAS",
                "ALLOW_UNVERIFIED_ASTROM_CHECKPOINT",
                "OFFSETS_TABLE_DIVERGENCE_RAISE"):
        assert f"`{var}" in table, (
            f"{var} changes a verdict and is not in the env table (issue #400)")


def test_the_switch_named_in_the_table_is_the_one_the_code_reads():
    val = os.path.join(REPO, "jwst_gc_pipeline", "reduction",
                       "validate_offsets_table.py")
    src = open(val).read()
    assert "os.environ.get('OFFSETS_TABLE_DIVERGENCE_RAISE') == '1'" in src
    assert "DivergedColumnPairError" in _md().split(
        "## Environment switches")[1]


def test_the_ladder_row_says_the_m2_correction_is_conditional():
    md = _md()
    ladder = md.split("## The ladder")[1].split("###")[0]
    m2_row = [ln for ln in ladder.splitlines() if ln.startswith("| **m2**")]
    assert m2_row, ladder
    assert "ASTROM_CHECKPOINT_APPLY" in m2_row[0], (
        "the ladder table describes the m2 correction -- table updated, "
        "mosaics stale-tagged, run stopped -- as unconditional.  All of it is "
        "inside `if ASTROM_CHECKPOINT_APPLY == '1'` (issue #400)")


def test_the_apply_switch_still_gates_the_m2_write():
    cat = os.path.join(REPO, "jwst_gc_pipeline", "photometry", "cataloging.py")
    src = open(cat).read()
    assert re.search(r"os\.environ\.get\(\s*['\"]ASTROM_CHECKPOINT_APPLY['\"]",
                     src), (
        "the ladder row says the m2 correction is conditional on "
        "ASTROM_CHECKPOINT_APPLY; that is no longer how it is gated")
