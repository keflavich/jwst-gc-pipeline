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
"""
import json

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    _m2_exposure_baseline, _m2_exposure_untrustworthy)

GOOD_KEY = ("2", 1, "nrca1", "F182M", "06201")
ALIAS_KEY = ("2", 2, "nrcb2", "F182M", "06201")


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
    assert "alias" in refused[ALIAS_KEY] or "no measurable tie" in refused[ALIAS_KEY]


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
    reason = _m2_exposure_untrustworthy(entry)
    assert reason is not None and "500 mas" in reason


def test_the_bound_honours_its_environment_override(monkeypatch):
    entry = dict(key=list(GOOD_KEY), dra=29033.0, ddec=-120.0, ok=True)
    monkeypatch.setenv("ASTROM_MAX_CORRECTION_ARCSEC", "40")
    assert _m2_exposure_untrustworthy(entry) is None
