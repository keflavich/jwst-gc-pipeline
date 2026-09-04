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
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    CONFIRM_EDGE_FRACTION, measure_offset)

# (3, 10) rather than the full (3, 10, 30, 60): the same bounded sweep the
# per-exposure consensus tie uses, and it keeps this file to a few seconds.
_WINDOWS = (3.0, 10.0)


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
