"""The >16-exposure visit consensus splits exposures into two parity halves.

Each exposure is measured against the pooled OPPOSITE half, so the half an
exposure lands in decides which frames its m2 reading is referenced to.  The
assignment used to alternate by rank of ``n_reliable``; a re-catalog that
changed a few stars reshuffled it, and readings moved 1-2 mas with no frame
changed.  ``parity_halves`` assigns from exposure identity alone.  These tests
pin that it depends on nothing else, that each detector splits across both
halves, and that the builder records the half it used.
"""
import numpy as np

from jwst_gc_pipeline.photometry.visit_consensus import (
    build_visit_consensus, parity_halves,
)
from jwst_gc_pipeline.photometry.tests.test_visit_consensus import (
    _exposure_table, _field,
)

DETECTORS = ["nrca1", "nrca2", "nrca3", "nrca4", "nrcb1", "nrcb2", "nrcb3", "nrcb4"]


def _keys(detectors=DETECTORS, exposures=range(1, 7), visit="001",
          filtername="F212N", vgroup="02101"):
    return [(visit, e, d, filtername, vgroup) for d in detectors for e in exposures]


def _by_detector(keys, parity):
    out = {}
    for k, p in zip(keys, parity):
        out.setdefault(k[2], {})[k[1]] = int(p)
    return out


def test_each_detector_alternates_by_exposure_number():
    """10678 F212N: 8 detectors x 6 dithers -> every detector 1,3,5 | 2,4,6."""
    keys = _keys()
    halves = _by_detector(keys, parity_halves(keys))
    for det in DETECTORS:
        assert [halves[det][e] for e in range(1, 7)] == [0, 1, 0, 1, 0, 1], det


def test_assignment_ignores_input_order():
    keys = _keys()
    shuffled = list(keys)
    np.random.default_rng(3).shuffle(shuffled)
    a = dict(zip(keys, parity_halves(keys)))
    b = dict(zip(shuffled, parity_halves(shuffled)))
    assert a == b


def test_no_detector_lands_in_one_half():
    """The old rank-by-count rule put o043 nrca3 at 6+0 (and cloudc F182M
    o002's four nrcb2 exposures in one half, #626)."""
    keys = _keys(exposures=range(1, 5))
    halves = _by_detector(keys, parity_halves(keys))
    for det, h in halves.items():
        assert sorted(h.values()) == [0, 0, 1, 1], det


def test_odd_groups_hand_over_the_starting_half():
    """Singletons (one exposure per pointing) still alternate, and three
    detectors of three dithers stay balanced 5 + 4."""
    singles = [("001", 1, "nrca1", "F770W", f"{v:05d}") for v in range(1, 9)]
    assert list(parity_halves(singles)) == [0, 1] * 4
    keys = _keys(detectors=DETECTORS[:3], exposures=range(1, 4))
    parity = parity_halves(keys)
    assert sorted(np.bincount(parity)) == [4, 5]
    halves = _by_detector(keys, parity)
    assert [halves["nrca2"][e] for e in (1, 2, 3)] == [1, 0, 1]


def test_none_exposure_number_does_not_raise():
    keys = [("001", None, "nrca1", "F212N", "None"),
            ("001", 2, "nrca1", "F212N", "None"),
            ("001", 1, "nrca1", "F212N", "None")]
    # the unnumbered exposure sorts first, then exposures 1 and 2
    assert list(parity_halves(keys)) == [0, 0, 1]


def _visit_tables(n_reliable_by_key):
    """One catalog per (detector, exposure), each with its own star count."""
    ra, dec = _field(n=600)
    tables = []
    for i, ((det, e), n) in enumerate(n_reliable_by_key.items()):
        rng = np.random.default_rng(1000 + i)
        idx = np.sort(rng.choice(len(ra), n, replace=False))
        tables.append(_exposure_table(ra[idx], dec[idx], exposure=e, module=det,
                                      rng=rng))
    return tables


def _halves_from_build(tables):
    cons = build_visit_consensus(tables, context="test-parity-halves")
    return {tuple(e["key"]): e["parity_half"] for e in cons["exposures"]}, cons


def test_builder_records_parity_and_it_does_not_follow_star_counts():
    """Two catalog sets with the same exposures and reversed star counts:
    the rank rule would swap every half, the identity rule keeps them."""
    dets = ["nrca1", "nrca2", "nrcb1", "nrcb2"]
    pairs = [(d, e) for d in dets for e in range(1, 7)]
    counts = {p: 200 + 10 * i for i, p in enumerate(pairs)}
    reverse = {p: 200 + 10 * (len(pairs) - 1 - i) for i, p in enumerate(pairs)}
    a, cons = _halves_from_build(_visit_tables(counts))
    b, _ = _halves_from_build(_visit_tables(reverse))
    assert len(a) == 24 and set(a.values()) == {0, 1}
    assert a == b
    for k, half in a.items():
        assert half == (k[1] - 1) % 2, k
    assert all(e["vs_consensus"] is not None for e in cons["exposures"])


def test_an_unusable_exposure_does_not_shift_its_siblings():
    """Halves are computed over every ingested catalog, so one falling below
    min_stars leaves the others where they were."""
    dets = ["nrca1", "nrca2", "nrcb1", "nrcb2"]
    pairs = [(d, e) for d in dets for e in range(1, 7)]
    counts = {p: 300 for p in pairs}
    full, _ = _halves_from_build(_visit_tables(counts))
    counts[("nrca1", 3)] = 20             # below the default min_stars=50
    starved, _ = _halves_from_build(_visit_tables(counts))
    assert len(starved) == 23
    assert all(starved[k] == full[k] for k in starved)


def test_small_visits_record_no_parity_half():
    """<=16 exposures use union growth, which has no halves."""
    ra, dec = _field()
    tables = [_exposure_table(ra, dec, exposure=e) for e in range(1, 5)]
    cons = build_visit_consensus(tables, context="test-small")
    assert all(e["parity_half"] is None for e in cons["exposures"])
