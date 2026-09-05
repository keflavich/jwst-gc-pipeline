"""An UNSWEPT peak that rides its own window's edge must be confirmed too (#600).

``confirm_peak_windows`` was reached only from ``best["swept"]`` -- "the peak is
bigger than the FIRST search window, so widening is what found it".  A peak does
not need a sweep to be an artifact of its window: at the 3" first window a
sparse offset histogram (~30k pairs over 90k bins, median occupied bin 1) has a
chance bin of 5-8 pairs somewhere, ``contrast`` IS that bin count, so it clears
the floor, and the arg-max lands anywhere in the search disc -- most often far
out, because the area goes as r.  ``swept`` is False for all of it.

That is what every gc2211 checkpoint record is.  Across the five per-obs fields
they all carry ``swept: false`` and ``window_consistent: null``, and the two
populations separate on the edge fraction where contrast cannot:

    good ties    47-116 mas   edge 0.016-0.037   n_peak  94-914
    artifacts   895-2973 mas  edge 0.298-0.991   n_peak   6-114   contrast 2.0-8.0

and the recorded ``windows`` lists say what the probe would have found: of the
reference-tie legs at edge >= 0.25, every one has its peak MOVE between the 3"
and 10" windows, by 1.1" to 11.0" -- gc2211_o050 m7 F277W reads (+1728, -1669)
mas at 3" and (+5113, +2833) mas at 10".  No leg below 0.25 is affected.

The fields here are UNCORRELATED -- there is no tie to find -- and reproduce the
recorded signature under the same estimator.  Issue #158's other route to the
same place (two adjacent footprints, ridge truncated by a SWEPT window) is
covered by ``test_sweep_window_alias.py``.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    CONFIRM_EDGE_FRACTION, confirm_peak_windows, measure_offset)

# (3, 10) rather than the full (3, 10, 30, 60): the same bounded sweep the
# per-exposure consensus tie uses, and it keeps this file to a few seconds.
_WINDOWS = (3.0, 10.0)

#: the probe-window COLLAPSE BAND: below this edge fraction both
#: ``CONFIRM_WINDOW_FACTORS`` fall under a single ``base_w * 1.25`` floor, which
#: is where the paired ``CONFIRM_MIN_PROBE_FACTORS`` changes the probe windows.
_COLLAPSE_BAND_MAX = 1.25 / 2.2


def _uniform_field(seed, n=2000, width_arcsec=60.0, ra0=266.4, dec0=-28.9):
    rng = np.random.RandomState(seed)
    cosd = np.cos(np.radians(dec0))
    ra = ra0 + (rng.rand(n) - 0.5) * width_arcsec / 3600.0 / cosd
    dec = dec0 + (rng.rand(n) - 0.5) * width_arcsec / 3600.0
    return SkyCoord(ra * u.deg, dec * u.deg)


def _unrelated_pair(seed=1):
    """Two source lists with NO common stars and no offset between them: every
    pair in the histogram is a wrong pair."""
    return _uniform_field(seed), _uniform_field(seed + 400)


def test_unswept_edge_peak_looks_like_a_tie_without_the_probe():
    """The raw material: ``ok`` True at contrast 5 on a 6-pair bin, from two
    catalogues that share nothing -- and ``swept`` False, so the swept-only
    trigger skipped the one guard built for this."""
    a, b = _unrelated_pair()
    r = measure_offset(a, b, sweep=True, sweep_windows=_WINDOWS)
    assert r is not None and r["ok"], r
    assert not r["swept"], r
    assert r["window_arcsec"] == 3.0, r
    assert r["n_peak"] < 20, r                       # gc2211 records: 6-16
    assert r["window_edge_fraction"] >= CONFIRM_EDGE_FRACTION, r
    assert r["off"] > 750.0, r


def test_unswept_edge_peak_is_probed_and_rejected():
    """The fix: the edge fraction triggers the confirmation on an UNSWEPT peak,
    the chance bin does not reproduce at a wider window, and the result is
    rejected instead of being recorded as a 2.2" tie."""
    a, b = _unrelated_pair()
    r = measure_offset(a, b, sweep=True, sweep_windows=_WINDOWS,
                       confirm_windows=True)
    assert r is not None and not r["swept"], r
    assert r["window_consistent"] is False, r
    assert r["alias_rejected"], r
    assert not r["ok"], f"unswept window-edge peak survived the probe: {r}"
    probes = [p for p in r["window_confirmation"]["probes"] if p["dra"] is not None]
    assert probes and not any(p["agrees"] for p in probes), probes
    # the probes must be at DIFFERENT windows: a measurement repeated at one
    # window is one measurement, not an independent confirmation
    assert len({round(p["window_arcsec"], 3) for p in probes}) == len(probes), probes


def test_rejection_is_stable_across_realizations():
    """Not one lucky seed: the chance-bin peak appears and is rejected on every
    realization of the same uncorrelated geometry."""
    for seed in (1, 2, 3, 4):
        a, b = _unrelated_pair(seed)
        plain = measure_offset(a, b, sweep=True, sweep_windows=_WINDOWS)
        assert plain["ok"] and not plain["swept"], (seed, plain)
        assert plain["window_edge_fraction"] >= CONFIRM_EDGE_FRACTION, (seed, plain)
        guarded = measure_offset(a, b, sweep=True, sweep_windows=_WINDOWS,
                                 confirm_windows=True)
        assert not guarded["ok"] and guarded["alias_rejected"], (seed, guarded)


def test_good_tie_below_the_trigger_is_untouched():
    """A real sub-arcsec tie sits at edge fraction ~0.01, well under the
    trigger, and must not pay for a probe nor change by one bit.  This matters:
    the gc2211 records include good ties measured at the 30"/60" window whose
    NEXT window peak is 23-39 arcsec away -- probing those would reject them."""
    rng = np.random.RandomState(11)
    ra = 266.4 + rng.rand(900) * 0.05
    dec = -28.9 + rng.rand(900) * 0.05
    a = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(dec))
    b = SkyCoord((ra + 0.05 / 3600.0 / cosd) * u.deg, (dec + 0.02 / 3600.0) * u.deg)
    plain = measure_offset(a, b, sweep=True)
    guarded = measure_offset(a, b, sweep=True, confirm_windows=True)
    assert plain["window_edge_fraction"] < CONFIRM_EDGE_FRACTION, plain
    for k in ("dra", "ddec", "off", "contrast", "npairs", "window_arcsec", "ok"):
        assert plain[k] == guarded[k], (k, plain[k], guarded[k])
    assert guarded["window_consistent"] is None and not guarded["alias_rejected"]


def test_real_offset_at_a_high_edge_fraction_survives_the_probe():
    """The widened trigger must not become a rejection.  A genuine rigid 2.7"
    shift also reads edge fraction ~0.9 at the 3" window, so it is now probed --
    and it reproduces, so it is kept.  Only a MEASURED disagreement rejects."""
    rng = np.random.RandomState(5)
    ra = 266.4 + rng.rand(1200) * 0.02
    dec = -28.9 + rng.rand(1200) * 0.02
    a = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(dec))
    b = SkyCoord((ra + 2.0 / 3600.0 / cosd) * u.deg, (dec - 1.8 / 3600.0) * u.deg)
    r = measure_offset(a, b, sweep=True, confirm_windows=True)
    assert r["window_edge_fraction"] >= CONFIRM_EDGE_FRACTION, r
    assert r["window_consistent"] is True, r
    assert r["ok"] and not r["alias_rejected"], r
    assert abs(r["dra"] - 2000) < 100 and abs(r["ddec"] + 1800) < 100, r


# --- the probe WINDOWS, which the widened trigger is what first reaches ------
#
# Six live ``_latest`` reference-tie records whose sparse leg is an UNSWEPT
# edge-riding peak, read on 2026-09-04 from
# ``/orange/adamginsburg/jwst/<field>/astrometry_checkpoints/``.  Under the old
# single ``base_w * 1.25`` floor five of the six place BOTH probes at 3.75" --
# one window measured twice.  That is a BOOKKEEPING defect, not a hole in the
# verdict: ``consistent = any(p['agrees'])`` and ``any({p, p}) == any({p})``, so
# the duplicate could never confirm anything the single probe did not already
# confirm; what it did was report ``n_probes`` 2 for one measured window.
# ``CONFIRM_MIN_PROBE_FACTORS`` pairs a floor to each factor so they land at
# 3.75" and 6.0" instead -- a genuinely second window.  The direction of that,
# and the one path on which it is stricter rather than more permissive, are
# pinned by the two tests below it.
#
# (field/stage/filter, off_mas, window_arcsec, dra_mas, ddec_mas)
_LIVE_UNSWEPT_EDGE_PEAKS = [
    ("sgrc m2 F182M", 939.84, 3.0, -473.91, 811.61),
    ("sgrc m3 F182M", 938.41, 3.0, -474.03, 809.88),
    ("sgrc m4 F182M", 938.30, 3.0, -473.85, 809.86),
    ("sgrc m5 F182M", 937.42, 3.0, -474.11, 808.69),
    ("sgrc m6 F182M", 2574.26, 3.0, -1129.58, -2313.20),
    ("gc2211_o028 m2 F150W", 1147.61, 3.0, -375.02, 1084.61),
]


@pytest.mark.parametrize("label,off,window,dra,ddec", _LIVE_UNSWEPT_EDGE_PEAKS)
def test_live_edge_peak_is_probed_at_two_distinct_windows(label, off, window,
                                                          dra, ddec):
    """Every live record the widened trigger newly reaches must get TWO probes
    at DIFFERENT windows.

    An unswept peak is by definition SMALLER than its own window, so
    ``f * off`` is under the floor for both factors and a single floor collapses
    them.  These are the exact numbers on disk; ``CONFIRM_WINDOW_FACTORS``
    (1.4, 2.2) never separates them, only the paired
    ``CONFIRM_MIN_PROBE_FACTORS`` (1.25, 2.0) does.
    """
    a, b = _unrelated_pair()
    conf = confirm_peak_windows(
        a, b, dict(off=off, window_arcsec=window, dra=dra, ddec=ddec))
    windows = sorted(round(p["window_arcsec"], 6) for p in conf["probes"])
    assert windows == [window * 1.25, window * 2.0], (label, conf)
    assert conf["n_probes"] == 2, (label, conf)


def _sparse_pair(n=50, seed=11):
    """Two uncorrelated fields thin enough that a 3.75" window cannot reach
    ``min_pairs`` (30) but a 6.0" one can -- the area goes as r^2, so 50 sources
    over 60" give ~20 pairs at 3.75" and ~60 at 6.0"."""
    return _uniform_field(seed, n=n), _uniform_field(seed + 500, n=n)


def test_in_the_collapse_band_the_second_probe_is_added_not_moved():
    """The DIRECTION of the probe-window change, which the campaign's guard
    rules require be stated: inside the collapse band it is PERMISSIVE.

    Probe 1 is ``max(1.4 * off, 1.25 * base_w)`` under both the old single floor
    and the new paired one, so it does not move; the second floor only lifts
    probe 2 off it.  Below ``1.25 / 2.2`` the new probe set is therefore a
    strict SUPERSET of the old one, and since ``consistent = any(p['agrees'])``
    over the measured probes, adding a probe can only turn ``consistent``
    False -> True -- the alias guard can only reject LESS often here, never
    more.  ``base_w * 1.25`` for both factors is exactly the pre-#751
    arithmetic.

    Five of the six live records sit in that band.  The sixth (sgrc m6, edge
    0.858) is ABOVE it, where the old second probe was already distinct
    (2.2 * off) and the new floor MOVES it outward instead of adding to it; the
    verdict direction there is not fixed by the arithmetic, so it is asserted as
    a move, not as a superset.

    What stops the permissive half from being a gate weakened into a pass is
    that rejection is still a MEASURED disagreement at both windows:
    ``test_collapse_band_peak_is_rejected_on_two_independent_probes`` and
    ``test_swept_footprint_alias_in_the_collapse_band_is_still_rejected`` hold
    the unswept and swept halves of this same band.
    """
    a, b = _unrelated_pair()
    in_band = []
    for label, off, window, dra, ddec in _LIVE_UNSWEPT_EDGE_PEAKS:
        best = dict(off=off, window_arcsec=window, dra=dra, ddec=ddec)
        new = confirm_peak_windows(a, b, best)
        old = confirm_peak_windows(a, b, best, min_probe_factors=(1.25, 1.25))
        new_w = {round(p["window_arcsec"], 6) for p in new["probes"]}
        old_w = {round(p["window_arcsec"], 6) for p in old["probes"]}
        if off / 1000.0 / window < _COLLAPSE_BAND_MAX:
            in_band.append(label)
            assert old_w < new_w, (label, old_w, new_w)
            assert old_w == {round(window * 1.25, 6)}, (label, old_w)
            # ...and the verdict is monotone in that: every probe the old set
            # measured is still measured, with the same agreement.
            old_agree = {round(p["window_arcsec"], 6): p["agrees"]
                         for p in old["probes"] if p["dra"] is not None}
            new_agree = {round(p["window_arcsec"], 6): p["agrees"]
                         for p in new["probes"] if p["dra"] is not None}
            for w, ag in old_agree.items():
                assert new_agree[w] == ag, (label, w, ag, new_agree)
        else:
            # above the band: probe 2 was already its own window and is widened
            assert len(old_w) == len(new_w) == 2, (label, old_w, new_w)
            assert max(new_w) > max(old_w), (label, old_w, new_w)
    assert len(in_band) == 5, in_band


def test_a_first_probe_with_no_pairs_no_longer_leaves_the_verdict_undetermined():
    """The one path on which the paired floors are STRICTER, and the only place
    the probe-window change moves a verdict rather than a count.

    When the first probe window holds fewer than ``min_pairs`` pairs it returns
    no result.  With both probes collapsed onto it the whole confirmation was
    (None, None) -> ``measured`` empty -> ``consistent`` None, i.e. UNDETERMINED,
    and ``measure_offset`` leaves an undetermined confirmation alone (``ok``
    unchanged, ``alias_rejected`` False -- only ``conf["consistent"] is False``
    rejects in ``measure_offset``; see
    ``test_sweep_window_alias.test_small_tie_is_numerically_untouched_by_confirmation``
    for the same mapping on a peak whose probe never ran).
    The second, wider probe can measure where the first could not, and here it
    disagrees, so the peak is rejected instead of waved through.

    Reverting the hunk to a single ``base_w * 1.25`` floor makes this test read
    ``consistent`` None.
    """
    a, b = _sparse_pair()
    best = dict(off=939.84, window_arcsec=3.0, dra=-473.91, ddec=811.61)
    conf = confirm_peak_windows(a, b, best)
    # THE VERDICT, asserted first so a revert fails on it and not on a window
    # list: rejected, where the collapsed arithmetic returned UNDETERMINED.
    assert conf["consistent"] is False, conf
    collapsed = confirm_peak_windows(a, b, best, min_probe_factors=(1.25, 1.25))
    assert collapsed["consistent"] is None, collapsed
    # ...and why: the narrow probe cannot be measured at this source density
    windows = sorted(round(p["window_arcsec"], 6) for p in conf["probes"])
    assert windows == [3.75, 6.0], conf
    narrow = [p for p in conf["probes"] if round(p["window_arcsec"], 6) == 3.75]
    assert narrow and narrow[0]["dra"] is None, conf
    # ...while the wide one can, and it disagrees
    wide = [p for p in conf["probes"] if round(p["window_arcsec"], 6) == 6.0]
    assert wide and wide[0]["dra"] is not None and wide[0]["agrees"] is False, conf
    assert conf["n_probes"] == 1, conf


def test_a_probe_window_is_never_measured_twice():
    """The dedup.  With the floors collapsed onto one value the second probe
    lands on the first's window; it is dropped rather than recorded as a second
    independent measurement.  The shipped constants never produce this (the
    factors and floors are separated by 1.57x and 1.6x), so it guards a caller
    that passes its own ``factors``/``min_probe_factors``, and it keeps
    ``n_probes`` an honest count of independent windows.
    """
    a, b = _unrelated_pair()
    best = dict(off=939.84, window_arcsec=3.0, dra=-473.91, ddec=811.61)
    conf = confirm_peak_windows(a, b, best, min_probe_factors=(1.25, 1.25))
    assert [round(p["window_arcsec"], 6) for p in conf["probes"]] == [3.75], conf
    assert conf["n_probes"] == 1, conf


def test_collapse_band_peak_is_rejected_on_two_independent_probes():
    """The band the live records sit in -- edge fraction between the trigger
    (0.25) and ``1.25 / 2.2 = 0.568``, below which BOTH factors fall under the
    old single floor.  Seed 2 of the uncorrelated fixture lands at 0.45 with the
    recorded signature (unswept, 3" window, n_peak 8, contrast 6) -- alongside
    the live sgrc 0.313 / gc2211_o028 0.383 -- and it is rejected on probes at
    two different windows, not one window twice.
    """
    a, b = _unrelated_pair(2)
    r = measure_offset(a, b, sweep=True, sweep_windows=_WINDOWS,
                       confirm_windows=True)
    assert r is not None and not r["swept"], r
    assert CONFIRM_EDGE_FRACTION <= r["window_edge_fraction"] <= _COLLAPSE_BAND_MAX, r
    assert r["alias_rejected"] and not r["ok"], r
    probes = r["window_confirmation"]["probes"]
    assert sorted(round(p["window_arcsec"], 6) for p in probes) == [3.75, 6.0], probes
    assert not any(p["agrees"] for p in probes if p["dra"] is not None), probes


def _clumpy_field(ra0, dec0, width_arcsec, seed, n_clump=300, per_clump=6,
                  sigma_arcsec=1.5):
    """A clustered star field -- the regime that makes the footprint-geometry
    ridge sharp enough to clear the contrast floor (see test_sweep_window_alias).
    """
    rng = np.random.RandomState(seed)
    cra = ra0 + (rng.rand(n_clump) - 0.5) * width_arcsec / 3600.0
    cdec = dec0 + (rng.rand(n_clump) - 0.5) * width_arcsec / 3600.0
    ra = (np.repeat(cra, per_clump)
          + rng.randn(n_clump * per_clump) * sigma_arcsec / 3600.0)
    dec = (np.repeat(cdec, per_clump)
           + rng.randn(n_clump * per_clump) * sigma_arcsec / 3600.0)
    return SkyCoord(ra * u.deg, dec * u.deg)


def test_swept_footprint_alias_in_the_collapse_band_is_still_rejected():
    """The SWEPT path, which the probe-window change also touches.

    ``CONFIRM_MIN_PROBE_FACTORS`` moves the second probe outward wherever both
    factors used to fall under the single floor -- for a swept peak that is
    ``maxsep < off < 0.568 * best_window``, e.g. a 3.75" peak found at the 10"
    window (the cloudc F410M nrcb shape).  There the probes were 12.5" and 12.5"
    and are now 12.5" and 20".  Since ``consistent = any(agrees)`` a further
    probe could in principle rescue an alias, so the #158 case has to be shown
    still rejected in exactly that band: two offset 160" footprints with NO
    shared stars, 8" apart, give a swept ridge peak at edge fraction 0.375 and
    both probes disagree.
    """
    a = _clumpy_field(290.915, 14.529, 160.0, seed=1)
    b = _clumpy_field(290.915, 14.529 - 8.0 / 3600.0, 160.0, seed=2)
    r = measure_offset(a, b, sweep=True, confirm_windows=True)
    assert r["swept"], r
    assert CONFIRM_EDGE_FRACTION <= r["window_edge_fraction"] <= _COLLAPSE_BAND_MAX, r
    probes = r["window_confirmation"]["probes"]
    assert sorted(round(p["window_arcsec"], 6) for p in probes) == [12.5, 20.0], probes
    assert not any(p["agrees"] for p in probes if p["dra"] is not None), probes
    assert r["window_consistent"] is False and r["alias_rejected"], r
    assert not r["ok"], r
