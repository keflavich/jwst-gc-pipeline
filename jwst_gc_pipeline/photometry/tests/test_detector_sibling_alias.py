"""A contrast-5 tie must not outrank its own detector's contrast-8 rejections.

gc2211 o023, F200W, nrcb1 -- four exposures, one detector, one search window
(`checkpoint_m2_F200W_o023_latest.json`)::

    contrast 8, edge 1.00, reproduced=False  ->  alias_rejected
    contrast 7, edge 1.00, reproduced=False  ->  alias_rejected
    contrast 8, edge 0.99, reproduced=False  ->  alias_rejected
    contrast 5, edge 0.93, reproduced=True   ->  ACCEPTED -> -9.28" correction

The accepted one has the LOWEST contrast of the four -- exactly
DEFAULT_MIN_CONTRAST, clearing the floor by nothing -- and differs only in that
one confirmation probe reproduced its peak.  nrcb1 and nrcb3 carry normal source
counts (n_reliable 2150-6103) but ~10x fewer PAIRS (5.4k-17k against 60k-272k on
the six detectors that tie cleanly in the same visit).  That is the #158
footprint ridge: a pair-density property of the detector's geometry, shared by
every exposure of it, and still there at the second probe window -- so
reproducing does not distinguish it (#347 item 1).

This rejects such a peak WITH its siblings.  It needs no new threshold: the
question it asks is whether the detector produced a tie anywhere in the visit.
"""
import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    detector_sibling_alias_keys)
from jwst_gc_pipeline.photometry.astrometry_offsets import (
    DEFAULT_MIN_CONTRAST, WINDOW_EDGE_FRACTION)


def _exp(det, exposure, off, contrast, edge, rejected, swept=True, window=10.0):
    return {
        'key': ('1', exposure, det, 'F200W', '02201'),
        'vs_consensus': {
            'off': off, 'dra': -off * 0.99, 'ddec': -off * 0.12,
            'contrast': contrast, 'window_edge_fraction': edge,
            'window_arcsec': window, 'swept': swept,
            'alias_rejected': rejected, 'window_consistent': not rejected,
        },
    }


#: nrcb1's four exposures, as recorded.
NRCB1 = [
    _exp('nrcb1', 1, 9983.4, 8.0, 0.998, True),
    _exp('nrcb1', 2, 9956.3, 7.0, 0.996, True),
    _exp('nrcb1', 3, 9851.8, 8.0, 0.985, True),
    _exp('nrcb1', 4, 9340.4, 5.0, 0.934, False),      # the one that got through
]

#: a detector that ties cleanly -- mas-scale, narrow window, not swept.
def _clean(det):
    return [_exp(det, e, 6.0 + e, 200.0, 0.0006, False, swept=False, window=3.0)
            for e in (1, 2, 3, 4)]


def test_the_o023_exposure_is_rejected_with_its_siblings():
    flagged = detector_sibling_alias_keys(NRCB1)
    assert ('1', 4, 'nrcb1', 'F200W', '02201') in flagged
    assert len(flagged) == 1, 'only the accepted one is added; the rest already were'


def test_it_really_is_the_weakest_of_the_four():
    """Named so the rule's premise stays checkable: the accepted measurement
    clears the contrast floor by nothing while its rejected siblings beat it."""
    accepted = NRCB1[3]['vs_consensus']
    assert accepted['contrast'] == DEFAULT_MIN_CONTRAST
    assert all(e['vs_consensus']['contrast'] > accepted['contrast']
               for e in NRCB1[:3])


def test_a_CLEAN_detector_is_untouched():
    assert detector_sibling_alias_keys(_clean('nrca1')) == set()


def test_a_clean_detector_in_the_SAME_visit_is_untouched():
    """The visit has both; rejecting nrcb1's peak must not touch nrca1's ties."""
    flagged = detector_sibling_alias_keys(NRCB1 + _clean('nrca1'))
    assert all(k[2] == 'nrcb1' for k in flagged)


def test_a_detector_that_mostly_TIES_keeps_its_odd_one_out():
    """One bad exposure among three good ones is a real per-exposure problem
    and must stay visible -- this rule is about a detector that produced no tie
    anywhere, not about outvoting a minority."""
    mixed = _clean('nrca2')[:3] + [
        _exp('nrca2', 4, 9000.0, 6.0, 0.9, True, window=3.0)]
    assert detector_sibling_alias_keys(mixed) == set()


def test_ONE_rejected_sibling_is_not_enough():
    """A single unlucky neighbour cannot condemn an exposure."""
    two = [_exp('nrcb1', 1, 9983.4, 8.0, 0.998, True),
           _exp('nrcb1', 4, 9340.4, 5.0, 0.934, False)]
    assert detector_sibling_alias_keys(two) == set()


def test_a_measurement_that_is_not_SWEPT_is_untouched():
    """A narrow-window tie was never in the ridge regime."""
    grp = NRCB1[:3] + [_exp('nrcb1', 4, 40.0, 5.0, 0.95, False, swept=False)]
    assert detector_sibling_alias_keys(grp) == set()


def test_a_measurement_away_from_the_window_EDGE_is_untouched():
    """The edge fraction is the ridge's signature; a peak well inside the
    window is a different claim and this rule says nothing about it."""
    grp = NRCB1[:3] + [_exp('nrcb1', 4, 2000.0, 5.0, 0.2, False)]
    assert detector_sibling_alias_keys(grp) == set()
    assert 0.2 < WINDOW_EDGE_FRACTION


def test_a_DIFFERENT_window_is_a_different_group():
    """Siblings only count as evidence at the same search window -- a rejection
    at 10" says nothing about a measurement made at 3"."""
    grp = NRCB1[:3] + [_exp('nrcb1', 4, 9340.4, 5.0, 0.934, False, window=3.0)]
    assert detector_sibling_alias_keys(grp) == set()


def test_an_exposure_with_no_measurement_is_skipped():
    grp = list(NRCB1) + [{'key': ('1', 5, 'nrcb1', 'F200W', '02201'),
                          'vs_consensus': None}]
    flagged = detector_sibling_alias_keys(grp)
    assert ('1', 5, 'nrcb1', 'F200W', '02201') not in flagged
    assert len(flagged) == 1


def test_a_malformed_key_is_skipped():
    grp = list(NRCB1) + [{'key': ('1',), 'vs_consensus': NRCB1[3]['vs_consensus']}]
    assert len(detector_sibling_alias_keys(grp)) == 1


@pytest.mark.parametrize('n_rejected,n_total,expect', [
    (2, 3, True),      # majority
    (2, 4, False),     # exactly half is not a majority
    (3, 4, True),
    (1, 4, False),
])
def test_the_majority_rule(n_rejected, n_total, expect):
    grp = [_exp('nrcb3', i + 1, 9500.0, 8.0, 0.99, i < n_rejected)
           for i in range(n_total)]
    flagged = detector_sibling_alias_keys(grp)
    assert bool(flagged) is expect, (n_rejected, n_total, flagged)


def test_the_checkpoint_consults_it():
    """Pinned by source: the helper is worthless if run_visit_checkpoint does
    not narrow with it before emitting a correction."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as A
    src = inspect.getsource(A.run_visit_checkpoint)
    assert 'detector_sibling_alias_keys' in src
    assert src.index('sibling_alias_keys = ') < src.index('corrections.append(')


def test_a_sibling_alias_BLOCKS_rather_than_only_advising():
    """Review of #359: routed to the advisory list, this branch caught o023's
    exposure ('1', 4, 'nrcb1') one step before #355's gross branch and took the
    F200W checkpoint from not-passing to PASSING -- two detectors that produced
    no tie anywhere in the visit, reported as advisory lines.

    A peak was measured and deliberately refused, which is the same class as
    #341's and #355's items, so it belongs on the blocking list with them."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as A
    src = inspect.getsource(A.run_visit_checkpoint)
    start = src.index('in sibling_alias_keys:')
    block = src[start:src.index('elif exp["misaligned"] and tuple(exp["key"]) '
                                'in antisym_keys:', start)]
    assert 'unverified_blocking.append(' in block, (
        'a sibling alias that only advises lets a field pass its astrometry '
        'gate with a detector that never tied')
    assert 'corrections.append(' not in block


def test_the_sibling_branch_still_comes_FIRST():
    """Order is deliberate: the alias diagnosis is more specific than 'gross',
    so it should be the message the operator gets -- now that both block, the
    ordering costs nothing."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as A
    src = inspect.getsource(A.run_visit_checkpoint)
    assert (src.index('in sibling_alias_keys:')
            < src.index('if correcting and gross is not None:'))
