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
    with pytest.raises(KeyError) as exc:
        _floor_actionable_corrections(corrs, 4.0, "m2] F212N/merged")
    assert "1/2" in str(exc.value)                      # names the count
    assert "F212N" in str(exc.value)                    # names the label


def test_raises_on_none_valued_key():
    corrs = [_corr(None, 2.0)]                          # present but unreadable
    with pytest.raises(KeyError):
        _floor_actionable_corrections(corrs, 4.0, "m2] F405N/merged")


def test_all_below_floor_returns_empty_not_raise():
    corrs = [_corr(0.5, 0.5), _corr(1.0, 1.0)]
    assert _floor_actionable_corrections(corrs, 4.0, "lbl") == []


def test_a_missing_key_does_not_read_as_zero_magnitude():
    # the whole point: a missing-key correction must NOT be silently sub-floor
    corrs = [{'vgroup': 1}]                             # no on-sky magnitude
    with pytest.raises(KeyError):
        _floor_actionable_corrections(corrs, 4.0, "lbl")
