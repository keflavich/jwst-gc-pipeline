"""The ASTROM_M2_CORRECTION_FLOOR_MAS filter must read the correction magnitude
LOUDLY (#111 item 1).

A ``c.get('dra_onsky_mas', 0.0)`` default reads magnitude 0 for any correction
missing the on-sky keys, so every such correction is silently "sub-floor",
nothing is actionable, and the checkpoint PASSes applying nothing -- a real
misalignment shipped as clean.  ``_floor_actionable_corrections`` raises instead.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry.cataloging import _floor_actionable_corrections


def _corr(dra, ddec, **extra):
    d = {'dra_onsky_mas': dra, 'ddec_onsky_mas': ddec}
    d.update(extra)
    return d


def test_filters_below_floor():
    corrs = [_corr(1.0, 1.0), _corr(5.0, 5.0)]         # ~1.41, ~7.07 mas
    out = _floor_actionable_corrections(corrs, 4.0, "m2] F212N/merged")
    assert out == [corrs[1]]                            # only the >=4 mas one


def test_raises_on_missing_key():
    corrs = [_corr(5.0, 5.0), {'vgroup': 3}]           # second lacks both keys
    with pytest.raises(ValueError) as exc:
        _floor_actionable_corrections(corrs, 4.0, "m2] F212N/merged")
    assert "1/2" in str(exc.value)                      # names the count
    assert "F212N" in str(exc.value)                    # names the label


def test_raises_on_none_valued_key():
    corrs = [_corr(None, 2.0)]                          # present but unreadable
    with pytest.raises(ValueError):
        _floor_actionable_corrections(corrs, 4.0, "m2] F405N/merged")


def test_raises_on_nonfinite_magnitude():
    # nan reaches the same silent-sub-floor hole (nan >= floor is False)
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError):
            _floor_actionable_corrections([_corr(bad, 2.0)], 4.0, "lbl")
        with pytest.raises(ValueError):
            _floor_actionable_corrections([_corr(2.0, bad)], 4.0, "lbl")


def test_all_below_floor_returns_empty_not_raise():
    corrs = [_corr(0.5, 0.5), _corr(1.0, 1.0)]
    assert _floor_actionable_corrections(corrs, 4.0, "lbl") == []


def test_a_missing_key_does_not_read_as_zero_magnitude():
    # the whole point: a missing-key correction must NOT be silently sub-floor
    corrs = [{'vgroup': 1}]                             # no on-sky magnitude
    with pytest.raises(ValueError):
        _floor_actionable_corrections(corrs, 4.0, "lbl")


# ---------------------------------------------------------------------------
# What the floor may NOT suppress
# ---------------------------------------------------------------------------

def _ref_tie(dra, ddec):
    """The consensus-to-reference correction: no exposure, no module.

    Every other correction names both.  This one shifts a whole visit rigidly
    onto the reference catalog.
    """
    return _corr(dra, ddec, visit='jw02092002001', exposure=None, module=None,
                 source='m2 consensus->reference')


def test_the_reference_tie_is_never_floored():
    """The floor's rationale is that a per-detector distortion term cannot be
    expressed by a table holding one rigid shift per exposure.  A whole-visit
    shift onto the reference catalog is exactly what such a table DOES express,
    so the argument does not reach it.

    Suppressing it would let an absolute frame error up to the floor through
    with nothing downstream to catch it: a common-mode shift moves every band
    equally, so the cross-band gate sees agreement, and the ~100 mas gross gate
    is two orders of magnitude away.
    """
    tie = _ref_tie(3.0, 0.0)                              # 3 mas, under a 7.6 floor
    perexp = _corr(1.0, 1.0, exposure=1, module='nrca1')  # ~1.41 mas, sub-floor
    out = _floor_actionable_corrections([tie, perexp], 7.6, "m2] F162M/nrca")
    assert out == [tie], (
        'the whole-visit reference tie must survive the floor; only '
        'per-exposure residuals are floored')


def test_per_exposure_corrections_are_still_floored_alongside_it():
    tie = _ref_tie(0.2, 0.0)
    big = _corr(9.0, 0.0, exposure=2, module='nrcb3')
    small = _corr(1.0, 0.0, exposure=3, module='nrcb3')
    out = _floor_actionable_corrections([tie, big, small], 7.6, "lbl")
    assert out == [tie, big]


def test_an_annotated_reference_tie_is_still_exempt():
    """w51's live table carries three rows reading

        'm2 consensus->reference (cross-band tied-F210M, contrast>2900)'

    No code at this head writes that parenthetical, so an `endswith` test on the
    source string fails OPEN on it -- silently putting the absolute frame tie
    back under the floor, which is the defect the exemption exists to fix.
    """
    annotated = _corr(3.0, 0.0, visit='jw06151002001', exposure=None,
                      module=None,
                      source='m2 consensus->reference '
                             '(cross-band tied-F210M, contrast>2900)')
    out = _floor_actionable_corrections([annotated], 7.6, "m2] F210M/nrcb")
    assert out == [annotated], (
        'a reference tie carrying its cross-band annotation was floored')
