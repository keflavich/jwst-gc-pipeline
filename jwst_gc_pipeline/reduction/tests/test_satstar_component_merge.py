"""One bright saturated star embedded in nebulosity fragments into COMPARABLE-
size DQ-SATURATED components split by a thin gap (W51 darkfil F480M blob: 659 +
370 px, centroids ~3px apart).  ``_merge_spike_satellites`` only folds a small
satellite into a >=ratio-larger core, so it leaves the pair split -> each is fit
as a satstar -> overlapping PSFs -> oversubtraction crater.  ``_merge_overlapping
_components`` merges b into the larger a when b's centroid lies within
overlap_frac x a's footprint radius, size-ratio-agnostic, but ONLY for large
cores (>= min_big_px) so a real crowded cluster of compact separate cores is
preserved.
"""
import numpy as np

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    _merge_overlapping_components)


def test_fragmented_bright_core_merges_to_one():
    # blob1 geometry: a big core + a comparable core whose centroid sits INSIDE
    # the big core's footprint radius, as two disjoint labels (thin DQ gap).
    yy, xx = np.mgrid[0:80, 0:80]
    big = np.hypot(xx - 40, yy - 40) < 15         # r15 ~ 700 px, centred (40,40)
    small = np.hypot(xx - 43, yy - 40) < 7        # r7 ~ 150 px, centre 3px away
    big = big & ~small                            # disjoint pixel sets (thin gap)
    src = np.zeros((80, 80), int)
    src[big] = 1
    src[small] = 2
    sat = src > 0
    _, n = _merge_overlapping_components(sat, src, 2, overlap_frac=0.5,
                                         min_big_px=250)
    assert n == 1


def test_distinct_compact_cluster_preserved():
    # a real crowded cluster: separate COMPACT saturated cores (< min_big_px).
    # Their radii are small so the containment distance is tiny -> never fused,
    # even at ~0.3" (5px) separation.
    yy, xx = np.mgrid[0:80, 0:80]
    src = np.zeros((80, 80), int)
    for i, cx in enumerate((30, 35, 40, 45)):     # 4 cores, 5px apart, r=3 (~28px)
        src[np.hypot(xx - cx, yy - 40) < 3] = i + 1
    sat = src > 0
    _, n = _merge_overlapping_components(sat, src, 4, overlap_frac=0.5,
                                         min_big_px=250)
    assert n == 4


def test_off_by_default():
    yy, xx = np.mgrid[0:80, 0:80]
    big = np.hypot(xx - 34, yy - 40) < 15
    small = np.hypot(xx - 40, yy - 40) < 8
    big = big & ~small
    src = np.zeros((80, 80), int)
    src[big] = 1
    src[small] = 2
    sat = src > 0
    # overlap_frac=0 -> no-op (other fields unchanged)
    _, n = _merge_overlapping_components(sat, src, 2, overlap_frac=0.0)
    assert n == 2
