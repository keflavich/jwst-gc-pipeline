"""Unit tests for the VIRAC2/Gaia predicted-sigma math (issue #965 item 1).

Synthetic only: no VizieR query, no FITS I/O.  Covers the propagation formula
itself, the NaN-safe fallback a refcat without the sigma columns must still
give (old refcats keep loading, CLAUDE.md "no silent frame drops" spirit
applied to sigma rather than to a source), and the weighted-median cell
statistic the region-map cut will use.
"""
import numpy as np

from jwst_gc_pipeline.photometry.reference_uncertainty import (
    predicted_axis_sigma_mas, combined_sigma_pred_mas, sigma_pred_mas,
    weighted_median, inverse_variance_weights, weights_with_unknown_fallback)


def test_predicted_axis_sigma_matches_hand_computation():
    # sigma_pos=10 mas, sigma_pm=2 mas/yr, dt=12.7 yr ->
    # sqrt(10^2 + (2*12.7)^2) = sqrt(100 + 645.16) = sqrt(745.16)
    got = predicted_axis_sigma_mas(10.0, 2.0, 12.7)
    assert np.isclose(got, np.sqrt(745.16))


def test_predicted_axis_sigma_broadcasts_over_arrays():
    sigma_pos = np.array([5.0, 10.0, 0.0])
    sigma_pm = np.array([1.0, 2.0, 3.0])
    dt = 12.7
    got = predicted_axis_sigma_mas(sigma_pos, sigma_pm, dt)
    expect = np.sqrt(sigma_pos ** 2 + (dt * sigma_pm) ** 2)
    np.testing.assert_allclose(got, expect)


def test_predicted_axis_sigma_zero_dt_returns_position_sigma_only():
    got = predicted_axis_sigma_mas(7.5, 100.0, 0.0)
    assert np.isclose(got, 7.5)


def test_predicted_axis_sigma_propagates_nan_rather_than_masking_it():
    got = predicted_axis_sigma_mas(np.array([10.0, np.nan]),
                                   np.array([np.nan, 2.0]), 12.7)
    assert np.isnan(got[0]) and np.isnan(got[1])


def test_combined_sigma_pred_is_hypot_of_axes():
    got = combined_sigma_pred_mas(3.0, 4.0)
    assert np.isclose(got, 5.0)


def test_sigma_pred_mas_end_to_end_matches_manual_quadrature():
    # RA: sigma_pos=8, sigma_pm=1 mas/yr; Dec: sigma_pos=6, sigma_pm=1.5 mas/yr
    dt = 12.7192
    got = sigma_pred_mas(8.0, 6.0, 1.0, 1.5, dt)
    sra = np.sqrt(8.0 ** 2 + (dt * 1.0) ** 2)
    sdec = np.sqrt(6.0 ** 2 + (dt * 1.5) ** 2)
    expect = np.hypot(sra, sdec)
    assert np.isclose(got, expect)


def test_sigma_pred_mas_vectorized_over_a_table_of_stars():
    n = 5
    dt = 12.7
    rng = np.random.RandomState(0)
    sra = rng.uniform(3, 20, n)
    sdec = rng.uniform(3, 20, n)
    pmra = rng.uniform(0.5, 5, n)
    pmdec = rng.uniform(0.5, 5, n)
    got = sigma_pred_mas(sra, sdec, pmra, pmdec, dt)
    assert got.shape == (n,)
    assert np.all(got > np.maximum(sra, sdec))  # PM term only adds


def test_weighted_median_recovers_plain_median_with_equal_weights():
    v = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    w = np.ones_like(v)
    assert np.isclose(weighted_median(v, w), np.median(v))


def test_weighted_median_even_count_tie_break_matches_np_median():
    """Blocker (round 3): the odd-count test above never lands the cumulative
    weight exactly on the half-total boundary, so it cannot see the even-count
    tie-break at all -- that boundary case "currently survives disabling"
    (removing the ``np.isclose(cum[idx], half)`` branch and always returning
    the lower straddling value passed the rest of the suite).  Two equally
    weighted values must average, exactly like ``np.median`` averages the two
    middle elements of an even-length array: not the lower value 0.0 an
    unbroken "first index reaching half" rule would return.
    """
    assert weighted_median(np.array([0.0, 10.0]), np.array([1.0, 1.0])) == 5.0
    assert weighted_median(np.array([0.0, 10.0]), np.array([1.0, 1.0])) == \
        np.median(np.array([0.0, 10.0]))


def test_weighted_median_downweights_high_sigma_outlier():
    # four good stars near 0, one wild outlier with a huge sigma (tiny weight)
    v = np.array([0.5, -0.5, 1.0, -1.0, 500.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 1e-6])
    m = weighted_median(v, w)
    assert abs(m) < 2.0


def test_weighted_median_falls_back_to_plain_median_with_no_valid_weights():
    v = np.array([1.0, 2.0, 3.0])
    w = np.full(3, np.nan)
    assert np.isclose(weighted_median(v, w), np.median(v))


def test_weighted_median_empty_input_returns_nan():
    assert np.isnan(weighted_median(np.array([]), np.array([])))


def test_inverse_variance_weights_basic():
    sigma = np.array([1.0, 2.0, 10.0])
    w = inverse_variance_weights(sigma, floor_mas=0.5)
    np.testing.assert_allclose(w, 1.0 / sigma ** 2)


def test_inverse_variance_weights_floors_tiny_sigma():
    sigma = np.array([0.01, 5.0])
    w = inverse_variance_weights(sigma, floor_mas=1.0)
    # the 0.01 mas star is floored to 1.0 mas before inverting, so it cannot
    # get an unphysically huge weight from a suspiciously small reported sigma
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 1.0 / 25.0)


def test_inverse_variance_weights_zero_for_nonfinite_or_nonpositive():
    sigma = np.array([np.nan, 0.0, -3.0, 4.0])
    w = inverse_variance_weights(sigma)
    assert w[0] == 0.0 and w[1] == 0.0 and w[2] == 0.0
    assert w[3] > 0.0


def test_weights_with_unknown_fallback_all_known_matches_plain_ivw():
    sigma = np.array([2.0, 4.0, 6.0])
    w1 = weights_with_unknown_fallback(sigma)
    w2 = inverse_variance_weights(sigma)
    np.testing.assert_allclose(w1, w2)


def test_weights_with_unknown_fallback_gives_unknowns_the_median_known_weight():
    sigma = np.array([2.0, np.nan, 8.0])
    w = weights_with_unknown_fallback(sigma)
    known_w = inverse_variance_weights(np.array([2.0, 8.0]))
    assert np.isclose(w[1], np.median(known_w))
    assert np.isclose(w[0], known_w[0]) and np.isclose(w[2], known_w[1])


def test_weights_with_unknown_fallback_all_unknown_is_uniform_one():
    sigma = np.full(5, np.nan)
    w = weights_with_unknown_fallback(sigma)
    np.testing.assert_allclose(w, 1.0)
    # uniform weights -> weighted_median degenerates to the plain median
    # (odd length, so the two definitions cannot differ on which side of a
    # tie they pick)
    v = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    assert np.isclose(weighted_median(v, w), np.median(v))
