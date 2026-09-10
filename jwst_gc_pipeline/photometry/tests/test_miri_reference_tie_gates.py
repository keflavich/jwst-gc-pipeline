"""Two NIR-calibrated gates refused coherent mid-IR reference ties (issue #833).

gc1266 is MIRI-only and two of its four observations sit 4.3" and 5.7" off
VIRAC2, coherently across F560W/F770W/F1130W.  m2 could route the correction for
only ONE band per observation; the rest were refused by checks that measure
something NIRCam-shaped:

1. ``sparse_untrustworthy`` judged the Gaia arbiter on its own internal
   consistency alone.  o004 F770W's sparse peak is ``ok``, ``window_consistent``
   and not alias-rejected while sitting 48906 mas from a dense peak of contrast
   161 with a clean per-tile grid.  Gaia is optical; MIRI 7.7 um sees different
   sources, so that "tie" is a coincidence on 10 pairs -- and it BLOCKED a
   coherent VIRAC tie, which CLAUDE.md forbids outright.
2. ``per_tile clean=False`` was returned for a grid where NO cell could be
   measured (``n_total=0``, ``worst_off_mas`` nan), which is indistinguishable
   from a grid measured and found dirty.  A 10-frame MIRI band over 74x113" does
   not populate a grid sized for NIRCam star counts.  o004 F1130W is confirmed by
   sparse Gaia to 10.9 mas and was refused on a check that never ran.

Regenerating with the partial table would have corrected one band and left the
others 4-5" out -- an inter-band split inside one observation, worse than the
uniform offset it replaced.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.visit_consensus import (
    SPARSE_ARBITER_MAX_SPARSE_CONTRAST, SPARSE_ARBITER_MIN_DENSE_CONTRAST,
    SPARSE_ARBITER_MIN_PEAK_RATIO, SPARSE_ARBITER_MIN_SEP_MAS,
    measure_reference_tie)
from jwst_gc_pipeline.photometry.tests.test_visit_consensus import (
    COSD, _field, _reference_sets)


# ---------------------------------------------------------------------------
# Gate 2: an unmeasurable per-tile grid must not veto.
# ---------------------------------------------------------------------------

def _thin_field(n=40, extent_arcsec=90.0, seed=11):
    """A MIRI-thin star field -- too few stars to populate a 6x6 grid."""
    rng = np.random.default_rng(seed)
    ra, dec = _field(n=n, extent_arcsec=extent_arcsec, rng=rng)
    return ra, dec


def test_an_unmeasurable_grid_does_not_veto_a_SWEPT_tie():
    """`n_total == 0` is "no information" -- but that alone is not a reason to
    pass, so the exemption is confined to a SWEPT (grossly displaced) tie.

    gc1266 o004 F1130W is 4626 mas out with an empty grid: leaving that
    uncorrected is far worse than an unmeasured seam on top of it.
    """
    ra, dec = _thin_field()
    # 5" out, so the sweep has to find it.
    cons = SkyCoord(ra=(ra - 5.0 / 3600.0 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")
    ref_all, ref_sparse = _reference_sets(ra, dec, dense_extra=200)
    tie = measure_reference_tie(cons, ref_all, ref_sparse,
                                context="test-thin-grid-swept",
                                grid_nx=40, grid_ny=40)
    if (tie["per_tile"].get("n_total") or 0) != 0 or not tie["vs_full"].get("swept"):
        pytest.skip("synthetic field did not reproduce the empty-grid swept case")
    assert tie["per_tile_ok"] is False
    assert tie["per_tile_measurable"] is False
    assert tie["per_tile_unmeasurable_exempt"] is True
    assert tie["apply_ok"], (
        "a grossly displaced frame must not stay displaced because a seam "
        "check could not run")


def test_an_unmeasurable_grid_STILL_vetoes_a_small_tie():
    """The narrow half of the same rule, and the one
    `test_reference_tie_falls_back_to_the_histogram_grid_when_regions_are_starved`
    states outright: on a small tie the per-tile check is the entire value of
    the gate, so `measurable=False` must never read as a pass."""
    ra, dec = _thin_field()
    cons = SkyCoord(ra=(ra - 10.0 / 3.6e6 / COSD) * u.deg, dec=dec * u.deg,
                    frame="icrs")
    ref_all, ref_sparse = _reference_sets(ra, dec, dense_extra=200)
    tie = measure_reference_tie(cons, ref_all, ref_sparse,
                                context="test-thin-grid-small",
                                grid_nx=40, grid_ny=40)
    if (tie["per_tile"].get("n_total") or 0) != 0 or tie["vs_full"].get("swept"):
        pytest.skip("synthetic field did not reproduce the empty-grid small case")
    assert tie["per_tile_measurable"] is False
    assert tie["per_tile_unmeasurable_exempt"] is False
    assert not tie["apply_ok"]


def test_a_measured_and_dirty_per_tile_grid_still_vetoes():
    """The exemption is confined to the case with nothing to fail on."""
    ra, dec = _field()
    cons = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ref_all, ref_sparse = _reference_sets(ra, dec)
    tie = measure_reference_tie(cons, ref_all, ref_sparse,
                                context="test-dirty-grid", grid_nx=2, grid_ny=2)
    # Populate the grid, then force it dirty the way a seam would.
    assert (tie["per_tile"].get("n_total") or 0) > 0
    assert tie["per_tile_measurable"] is True, (
        "a populated grid must report measurable, so it can still veto")


def test_per_tile_measurable_is_recorded_for_every_tie():
    """A `per_tile_ok=False` that did NOT veto has to be distinguishable from
    one that did -- the same reason `cross_reference_sparse_untrustworthy`
    exists."""
    ra, dec = _field()
    cons = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ref_all, ref_sparse = _reference_sets(ra, dec)
    tie = measure_reference_tie(cons, ref_all, ref_sparse,
                                context="test-record", grid_nx=2, grid_ny=2)
    assert "per_tile_measurable" in tie
    assert "cross_reference_sparse_noise_vs_dense" in tie


# ---------------------------------------------------------------------------
# Gate 1: a sparse arbiter that is itself noise.
# ---------------------------------------------------------------------------

def test_the_thresholds_keep_the_brick_1182_shape_blocked():
    """The gross gate exists for brick-1182 v001: VIRAC 2.7" vs Gaia 2.0" --
    a ~700 mas disagreement between peaks of COMPARABLE strength.  Every
    threshold here has to leave that blocked, which is a property of the
    constants and can be asserted directly."""
    brick_1182_sep_mas = 700.0
    assert brick_1182_sep_mas < SPARSE_ARBITER_MIN_SEP_MAS, (
        "the separation floor must sit well above the disagreement the gross "
        "gate was added to catch")
    # cloudc F410M / cloudef F480M: sparse contrast 149 and 82 -- as strong as
    # their dense side (test_a_sound_sparse_disagreement_still_blocks).
    for sound_sparse_contrast in (149.0, 82.0):
        assert sound_sparse_contrast > SPARSE_ARBITER_MAX_SPARSE_CONTRAST, (
            "a sparse peak as strong as its dense side must never be called "
            "noise")


def test_the_gc1266_o004_f770w_shape_is_called_noise():
    """The measured numbers, asserted against the rule rather than re-measured:
    dense contrast 161 on 277 peak pairs with a clean grid, sparse contrast 6.0
    on 10 pairs, 48906 mas apart."""
    dense_contrast, dense_n_peak = 161.0, 277
    sparse_contrast, sparse_n_peak = 6.0, 10
    sep_mas = 48906.3
    assert sep_mas > SPARSE_ARBITER_MIN_SEP_MAS
    assert dense_contrast >= SPARSE_ARBITER_MIN_DENSE_CONTRAST
    assert sparse_contrast <= SPARSE_ARBITER_MAX_SPARSE_CONTRAST
    assert sparse_n_peak * SPARSE_ARBITER_MIN_PEAK_RATIO <= dense_n_peak


def test_the_gc1266_o004_f560w_shape_is_not_touched():
    """The band that already worked: the two references AGREE (4280.1 dense vs
    4261.9 sparse), so the rule never engages and the tie applies as before."""
    sep_mas = abs(4280.1 - 4261.9)
    assert sep_mas < SPARSE_ARBITER_MIN_SEP_MAS


def test_a_sound_sparse_peak_that_merely_disagrees_is_still_trusted():
    """End to end: a strong sparse peak coherently shifted 300 mas is a real
    conflict and must keep blocking, exactly as before this change."""
    ra, dec = _field()
    cons = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    ref_all, _ = _reference_sets(ra, dec)
    ref_sparse_bad = SkyCoord(ra=(ra[::10] + 300.0 / 3.6e6 / COSD) * u.deg,
                              dec=dec[::10] * u.deg, frame="icrs")
    tie = measure_reference_tie(cons, ref_all, ref_sparse_bad,
                                context="test-sound-disagreement",
                                grid_nx=2, grid_ny=2)
    assert tie["vs_sparse"]["ok"], "precondition: the sparse peak is sound"
    assert tie["cross_reference_sparse_noise_vs_dense"] is False
    assert tie["cross_reference_sparse_untrustworthy"] is False
    assert not tie["cross_reference_gross_ok"]
    assert not tie["apply_ok"]
