"""A refused m2 per-exposure measurement is not a frozen-stage baseline.

cloudc F182M o002, 2026-08-24 (issue #626).  Four exposures --
``('2', n, 'nrcb2', 'F182M', '06201')`` -- landed alone in one parity half of
``build_visit_consensus``, so they had no true pairs with the opposite half.
``measure_offset`` swept out to the footprint pair-density ridge and returned
the search-window edge: 9.86" at a 10" window, 29.50" at 30", 58.23" at 60",
0.97-0.99 of the window every time.  m2 recorded ``ok=False``,
``alias_rejected=True``, ``window_consistent=False``, ``component=-1``, wrote
"recorded, NOT applied" into ``unverified``, and emitted no correction.

``_m2_exposure_baseline`` admitted the number anyway, on ``np.isfinite`` alone.
Every frozen stage then computed ``hypot(now - 9.8")`` against frames that
actually read 1-2.5 mas and reported "MOVED 9858 mas since the m2 freeze" --
five FAILED records (m3, m4, m5, m6, m7) that no later stage could clear,
because nothing downstream can rewrite an m2 record.

The refusal has two kinds and they do NOT get the same verdict (#312):
"m2 could not measure a tie" is advisory, "m2 measured a tie and refused to
APPLY it" is a number saying the exposure may be misaligned and blocks.  The
end-to-end tests at the bottom drive ``run_visit_checkpoint`` itself, because a
reader-only test passes with the frozen-stage branch deleted.
"""
import json

import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    _m2_exposure_baseline, _m2_exposure_untrustworthy, run_visit_checkpoint)
from .test_astrometry_checkpoint import (
    _exp, _patch_consensus_exposures, _tiny_visit_table)

GOOD_KEY = ("2", 1, "nrca1", "F182M", "06201")
ALIAS_KEY = ("2", 2, "nrcb2", "F182M", "06201")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The magnitude bound reads the environment, and so does the enforcement
    mode.  Pin both: a developer with ASTROM_MAX_CORRECTION_ARCSEC exported
    would otherwise see these pass or fail for a reason that is not in the file.
    """
    monkeypatch.delenv("ASTROM_MAX_CORRECTION_ARCSEC", raising=False)
    monkeypatch.delenv("ALLOW_UNVERIFIED_ASTROM_CHECKPOINT", raising=False)
    monkeypatch.setenv("ASTROM_CHECKPOINT_ENFORCE", "release")


def _write_record(tmp_path, exposures):
    rec = dict(stage="m2", filtername="F182M",
               visits=[dict(visit="2", filtername="F182M",
                            exposures=exposures)])
    (tmp_path / "checkpoint_m2_F182M_o002_latest.json").write_text(
        json.dumps(rec))


def _good_entry():
    """The ordinary case: a certified mas-scale tie (cloudc's own nrca1)."""
    return dict(key=list(GOOD_KEY), dra=1.03, ddec=-1.80, off=2.08,
                ok=True, unverified=False, alias_suspect=False,
                alias_rejected=False, window_consistent=None,
                window_arcsec=3.0, contrast=823.6, component=0,
                internal_tie=True, misaligned=True)


def _alias_entry():
    """cloudc's real numbers for ('2', 2, 'nrcb2', 'F182M', '06201')."""
    return dict(key=list(ALIAS_KEY), dra=3957.630928198127,
                ddec=9028.917054278907, off=9858.203981297956,
                ok=False, unverified=True, alias_suspect=False,
                alias_rejected=True, window_consistent=False,
                window_arcsec=10.0, contrast=9.0, component=-1,
                internal_tie=False, misaligned=False,
                window_edge_fraction=0.9858203981297956)


def test_refused_alias_is_not_a_baseline_but_a_certified_tie_is(tmp_path):
    _write_record(tmp_path, [_good_entry(), _alias_entry()])
    base, refused = _m2_exposure_baseline(str(tmp_path), "F182M", "2", "_o002")
    assert GOOD_KEY in base
    assert base[GOOD_KEY] == (1.03, -1.80)
    assert ALIAS_KEY not in base, (
        "the 9.8 arcsec wide-sweep diagnostic m2 refused became the frozen "
        "baseline, so every later stage reads a ~9858 mas MOVED")
    assert ALIAS_KEY in refused
    reason = refused[ALIAS_KEY].reason
    assert "alias" in reason or "no measurable tie" in reason


def test_each_refusal_flag_alone_disqualifies_the_entry():
    for field, value in (("ok", False), ("unverified", True),
                         ("alias_rejected", True), ("alias_suspect", True)):
        entry = _good_entry()
        entry[field] = value
        assert _m2_exposure_untrustworthy(entry) is not None, field


def test_a_legacy_entry_without_the_flags_is_still_admitted():
    """Records written before ``alias_rejected``/``unverified`` existed carry
    neither, and a missing flag means "not stated", never "refused"."""
    entry = dict(key=list(GOOD_KEY), dra=1.03, ddec=-1.80, off=2.08)
    assert _m2_exposure_untrustworthy(entry) is None


def test_a_flagless_gross_baseline_is_bounded_by_the_write_limit():
    """w51's July m2 F444W record carries 29" per-exposure entries with
    ``ok=True`` and none of the later flags.  ``_assert_correction_magnitudes``
    refuses to WRITE a per-exposure correction past MAX_CORRECTION_ARCSEC, so a
    baseline past it is a value m2 could not have acted on either."""
    entry = dict(key=list(GOOD_KEY), dra=29033.0, ddec=-120.0, ok=True)
    refusal = _m2_exposure_untrustworthy(entry)
    assert refusal is not None and "500 mas" in refusal.reason


def test_the_bound_honours_its_environment_override(monkeypatch):
    entry = dict(key=list(GOOD_KEY), dra=29033.0, ddec=-120.0, ok=True)
    monkeypatch.setenv("ASTROM_MAX_CORRECTION_ARCSEC", "40")
    assert _m2_exposure_untrustworthy(entry) is None


# ---------------------------------------------------------------------------
# Which refusal BLOCKS.  "m2 could not measure" and "m2 measured and refused to
# apply" are different findings and _checkpoint_passed treats them differently
# (#312) -- routing both to the advisory bucket would rebuild that bug here.
# ---------------------------------------------------------------------------

def test_could_not_measure_is_advisory_but_measured_and_refused_blocks():
    assert _m2_exposure_untrustworthy(_alias_entry()).blocking is False
    gross = dict(key=list(GOOD_KEY), dra=29033.0, ddec=-120.0, ok=True)
    assert _m2_exposure_untrustworthy(gross).blocking is True
    anti = dict(_good_entry(), alias_suspect=True, dra=734.0, ddec=-30.0)
    assert _m2_exposure_untrustworthy(anti).blocking is True


def test_an_uncertified_gross_baseline_does_not_block():
    """cloudc's 9.9" diagnostic is past the write bound AND uncertified.  The
    magnitude alone is not the finding -- a number ``measure_offset`` never
    certified measures nothing, so it stays advisory."""
    refusal = _m2_exposure_untrustworthy(_alias_entry())
    assert "500 mas" in refusal.reason and refusal.blocking is False


# ---------------------------------------------------------------------------
# END TO END through run_visit_checkpoint.  The reader tests above all pass with
# the frozen-stage branch deleted; these are the ones that do not.
# ---------------------------------------------------------------------------

_E2E_KEY = ("001", 8, "nrca2", "F212N")


def _write_m2_entries(record_dir, entries, visit="001", filt="F212N"):
    rec = dict(visits=[dict(visit=visit, exposures=entries)])
    (record_dir / f"checkpoint_m2_{filt}_latest.json").write_text(
        json.dumps(rec))


def test_frozen_stage_refused_m2_baseline_is_not_a_movement(
        tmp_path, monkeypatch):
    """cloudc F182M o002, end to end.  m2's entry for the exposure is the 9858
    mas diagnostic it refused; m3 measures the frame 2.4 mas off consensus.

    Before this fix the stage reported "MOVED 9858 mas since the m2 freeze" and
    FAILED -- an m3-m7 stop no later stage could clear.  It is UNVERIFIED, and
    ADVISORY: nothing was measured, so nothing is being ignored."""
    _write_m2_entries(tmp_path, [dict(
        key=list(_E2E_KEY), dra=3957.63, ddec=9028.92, off=9858.20,
        ok=False, unverified=True, alias_rejected=True, alias_suspect=False,
        window_consistent=False, contrast=9.0, component=-1)])
    _patch_consensus_exposures(monkeypatch,
                               [_exp(_E2E_KEY, 2.4, 0.3, misaligned=True)])
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["failures"] == [], rec["failures"]
    assert rec["unverified_blocking"] == [], rec["unverified_blocking"]
    assert rec["passed"] is True
    assert not rec["all_verified"]
    msg = "\n".join(rec["unverified"])
    assert "no measurable tie" in msg, msg
    assert "not a movement" in msg, msg


def test_frozen_stage_gross_certified_m2_baseline_blocks(tmp_path, monkeypatch):
    """w51 F444W's July m2 record: ``ok=True``, no flags, ~29" per exposure --
    a number m2 MEASURED and could not write.

    That is #312's other half.  It is still not a "movement" (there was never a
    frozen value to move from), but it must not read as a pass either: routing
    it to the advisory bucket makes an m3 exposure 29" off its own visit
    consensus report ``passed=True`` with empty ``failures``."""
    _write_m2_entries(tmp_path, [dict(key=list(_E2E_KEY), dra=29033.5,
                                      ddec=-120.0, ok=True)])
    _patch_consensus_exposures(
        monkeypatch, [_exp(_E2E_KEY, 29033.5, -120.0, misaligned=True)])
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["failures"] == [], rec["failures"]
    assert len(rec["unverified_blocking"]) == 1, rec["unverified_blocking"]
    assert "500 mas" in rec["unverified_blocking"][0]
    assert rec["passed"] is False


def test_frozen_stage_antisymmetric_m2_baseline_blocks(tmp_path, monkeypatch):
    """cloudc F410M: 16 entries m2 certified (contrast 4010-4152) at ~734 mas
    and refused to apply as a #158 footprint alias -- the visit whose exposures
    drizzle 4.06" out of place.  m2 filed it ``unverified_blocking`` at the
    visit level, so its frozen-stage mirror blocks too."""
    _write_m2_entries(tmp_path, [dict(
        key=list(_E2E_KEY), dra=-734.4, ddec=31.2, ok=True,
        alias_suspect=True, unverified=False, alias_rejected=False)])
    _patch_consensus_exposures(monkeypatch,
                               [_exp(_E2E_KEY, 5.0, 0.0, misaligned=True)])
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["failures"] == [], rec["failures"]
    assert len(rec["unverified_blocking"]) == 1, rec["unverified_blocking"]
    assert "ANTISYMMETRIC" in rec["unverified_blocking"][0]
    assert rec["passed"] is False


def test_frozen_stage_certified_m2_baseline_still_gates(tmp_path, monkeypatch):
    """The exemption is only for a refused entry.  A certified mas-scale
    baseline is a real freeze point and moving away from it still FAILS."""
    _write_m2_entries(tmp_path, [dict(key=list(_E2E_KEY), dra=0.0, ddec=0.0,
                                      ok=True, unverified=False,
                                      alias_rejected=False,
                                      alias_suspect=False)])
    _patch_consensus_exposures(monkeypatch,
                               [_exp(_E2E_KEY, 5.0, 0.0, misaligned=True)])
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert len(rec["failures"]) == 1, rec["failures"]
    assert "MOVED" in rec["failures"][0]
    assert rec["passed"] is False


def test_the_frozen_perexposure_table_documents_both_refusal_kinds():
    """The doc's frozen per-exposure table is what an operator reads to find out
    why a stage did what it did; #400 is the roundup of the times it said
    something the code does not do.  Pin the two rows this change adds."""
    import os
    md = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "ASTROMETRY_CHECKPOINTS.md")
    with open(md) as fh:
        text = fh.read()
    table = text.split("### What the frozen (m3–m6) per-exposure gate compares"
                       " against")[1].split("\n## ")[0]
    assert "refused its own measurement" in table
    assert "refused to APPLY it" in table
    assert "unverified_blocking" in table
    assert "_m2_exposure_untrustworthy" in table
