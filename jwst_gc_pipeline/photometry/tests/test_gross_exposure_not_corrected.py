"""One arcsecond-scale exposure must not discard a visit's real corrections.

gc2211 o023's m12 finalize (38963501) measured 25 exposures.  Twenty-four were
mas-scale.  One -- nrcb1 exposure 4, F200W -- read -9.28", and the batch check
at WRITE time rejected all 25::

    OffsetsTableUpdateError: 1 correction(s) exceed their magnitude limit and
    will NOT be applied ... [('1', 'F200W', 4, 'nrcb1', -9.2773, -1.0838,
    'limit=0.5"')]

That exposure is not a per-exposure misalignment.  Its three siblings on the
same detector measured 9.85-9.98" at the same 10" search window and were
rejected as window-edge aliases (#158); exposure 4 differs only in that one
confirmation probe reproduced its peak, at the LOWEST contrast of the four
(5.0, exactly the floor).  All four sit at off/window 0.93-0.998:

    exp 1   9983 mas  contrast 8.0  off/window 0.998  consistent False -> alias
    exp 2   9956 mas  contrast 7.0  off/window 0.996  consistent False -> alias
    exp 3   9852 mas  contrast 8.0  off/window 0.985  consistent False -> alias
    exp 4   9340 mas  contrast 5.0  off/window 0.934  consistent True  -> CORRECTED

The checkpoint already states the policy -- "a per-exposure tie is mas-scale,
so a gross frame belongs to the per-visit bulk path" -- but only on the
UNVERIFIED path, which an exposure reaches only when it has no measurable tie
at all.  One that produces a peak at the same arcsecond scale fell through to
the correcting path.

Asking the same limit at measurement time refuses that exposure on its own:
blocking-unverified, so the run still stops for it, while the visit's 24 real
corrections are still written.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    MAX_CORRECTION_ARCSEC, _assert_correction_magnitudes,
    _gross_per_exposure_offset)

#: The o023 nrcb1 measurements, as recorded in checkpoint_m2_F200W_o023_latest.
O023_NRCB1 = {
    1: dict(off=9983.442903629702, contrast=8.0, window_edge_fraction=0.998,
            window_consistent=False),
    2: dict(off=9956.292418305235, contrast=7.0, window_edge_fraction=0.996,
            window_consistent=False),
    3: dict(off=9851.760102112683, contrast=8.0, window_edge_fraction=0.985,
            window_consistent=False),
    4: dict(off=9340.436406413199, contrast=5.0, window_edge_fraction=0.934,
            window_consistent=True),
}


@pytest.mark.parametrize('exp', sorted(O023_NRCB1))
def test_every_o023_nrcb1_exposure_is_gross(exp):
    """Including exposure 4, whose confirmation probe reproduced the peak.
    Reproducing at one probe does not make a 9.3" per-exposure tie mas-scale."""
    assert _gross_per_exposure_offset(O023_NRCB1[exp]) is not None


def test_the_message_names_the_limit_it_failed():
    msg = _gross_per_exposure_offset(O023_NRCB1[4])
    assert '9.34"' in msg
    assert '0.5"' in msg


@pytest.mark.parametrize('off_mas', [0.25, 2.19, 154.7, 198.5, 499.0])
def test_a_mas_SCALE_offset_is_not_gross(off_mas):
    """o050's exposures read 0.25-2.8 mas and o023's other detectors 6-199 mas;
    every one of those must still be correctable."""
    assert _gross_per_exposure_offset(dict(off=off_mas)) is None


def test_the_boundary_is_the_writers_limit():
    lim = MAX_CORRECTION_ARCSEC * 1000.0
    assert _gross_per_exposure_offset(dict(off=lim)) is None
    assert _gross_per_exposure_offset(dict(off=lim + 1.0)) is not None


def test_a_missing_or_nonfinite_offset_is_not_called_gross():
    """An unmeasured exposure is the UNVERIFIED path's business, not this one;
    calling it gross here would relabel 'no measurement' as 'bad measurement'."""
    assert _gross_per_exposure_offset({}) is None
    assert _gross_per_exposure_offset(dict(off=None)) is None
    assert _gross_per_exposure_offset(dict(off=float('nan'))) is None


def test_the_env_override_moves_this_limit_TOO(monkeypatch):
    """A deliberate gross re-authoring raises ASTROM_MAX_CORRECTION_ARCSEC.  If
    that raised the writer's limit but not this one, the correction would be
    refused here and the override would do nothing."""
    monkeypatch.setenv('ASTROM_MAX_CORRECTION_ARCSEC', '20')
    assert _gross_per_exposure_offset(O023_NRCB1[4]) is None
    monkeypatch.setenv('ASTROM_MAX_CORRECTION_ARCSEC', '0.5')
    assert _gross_per_exposure_offset(O023_NRCB1[4]) is not None


def test_what_this_refuses_is_what_the_WRITER_would_have_refused():
    """The two limits must not drift: anything refused here has to be something
    `_assert_correction_magnitudes` would have rejected, or this would be
    dropping corrections the table would have accepted."""
    corr = dict(visit='jw02211023001', filtername='F200W', exposure=4,
                module='nrcb1', dra_onsky_mas=-9277.34,
                ddec_onsky_mas=-1083.81, dec_deg=-28.9)
    assert _gross_per_exposure_offset(
        dict(off=float(np.hypot(corr['dra_onsky_mas'],
                                corr['ddec_onsky_mas'])))) is not None
    with pytest.raises(Exception) as ex:
        _assert_correction_magnitudes([corr], 'Offsets_test.csv')
    assert 'limit' in str(ex.value)


def test_a_BULK_correction_is_not_judged_by_the_per_exposure_limit():
    """A per-visit bulk tie may legitimately be arcseconds (cloudef +102",
    sgra ~14.8").  This helper is only ever asked about a per-exposure
    measurement, so nothing here may narrow the bulk path -- guarded by the
    writer keeping its separate, larger bulk limit."""
    bulk = dict(visit='jw01939001001', filtername='F212N', exposure=None,
                module=None, dra_onsky_mas=-14830.0, ddec_onsky_mas=120.0,
                dec_deg=-29.0)
    _assert_correction_magnitudes([bulk], 'Offsets_test.csv')


# ---------------------------------------------------------------------------
# The WIRING.  Every test above drives _gross_per_exposure_offset directly, so
# the whole fix reverts at the call site with the suite green:
#
#     gross = _gross_per_exposure_offset(res)   ->   gross = None      29 passed
#
# The inherited unverified_blocking site-count pin does not help: the mutant
# changes the CONDITION, not the append, so the count stays at 3.
# ---------------------------------------------------------------------------

def test_the_correcting_branch_is_GUARDED_by_the_gross_check():
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as A
    src = inspect.getsource(A.run_visit_checkpoint)

    assert 'gross = _gross_per_exposure_offset(res)' in src, (
        'the gross check must be computed from the measurement, not stubbed')
    guard = src.index('if correcting and gross is not None:')
    plain = src.index('elif correcting:')
    assert guard < plain, (
        'the correcting branch must be reachable only past the gross test, or '
        'an arcsecond-scale peak is emitted as a per-exposure correction again')


def test_the_gross_branch_blocks_rather_than_only_advising():
    """A gross exposure that merely warned would let the run continue on a
    frame nobody measured -- it goes on the blocking list, which is what makes
    the checkpoint not-a-pass."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as A
    src = inspect.getsource(A.run_visit_checkpoint)
    start = src.index('if correcting and gross is not None:')
    block = src[start:src.index('elif correcting:')]
    assert 'unverified_blocking.append(' in block
    assert 'corrections.append(' not in block
