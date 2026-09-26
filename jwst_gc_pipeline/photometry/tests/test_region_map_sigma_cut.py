"""``local_residual_map``'s sigma cut / inverse-variance weighting (issue #965
item 1): pairs whose predicted reference sigma exceeds a cap are dropped
before a cell's statistic is formed, and a surviving pair's weight in the
cell's median is ``1/sigma^2`` (unknown sigma gets the cell's median known
weight, never zero and never a cut).

These are BEHAVIOUR tests against ``local_residual_map`` itself -- the pure
math (the propagation formula, the weighted median, the unknown-sigma
fallback) is already covered by ``test_reference_uncertainty.py``.  This file
answers: does the estimator actually cut/weight the pairs it is handed.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    local_residual_map, DEFAULT_SIGMA_CAP_MAS)

RA0, DEC0 = 266.60, -28.50
COSD = float(np.cos(np.radians(DEC0)))


def _sky(x_arcsec, y_arcsec):
    x_arcsec = np.asarray(x_arcsec, dtype=float)
    y_arcsec = np.asarray(y_arcsec, dtype=float)
    return SkyCoord((RA0 + x_arcsec / 3600.0 / COSD) * u.deg,
                    (DEC0 + y_arcsec / 3600.0) * u.deg, frame="icrs")


def _tie():
    return dict(ok=True, swept=False, dra=0.0, ddec=0.0, off=0.0)


def _field(dra_mas, ddec_mas=None):
    """``n = len(dra_mas)`` stars, ``a`` all at the origin, ``b`` offset by the
    given per-star dRA (mas); one 100" cell holds everything.  Returns
    ``(a, b)``.
    """
    dra_mas = np.asarray(dra_mas, dtype=float)
    if ddec_mas is None:
        ddec_mas = np.zeros_like(dra_mas)
    n = len(dra_mas)
    # spread the stars out a little so they are not all coincident, well
    # inside one 100" cell and well inside the 0.3" match radius
    x = np.linspace(-5.0, 5.0, n)
    y = np.linspace(-5.0, 5.0, n)
    a = _sky(x, y)
    b = _sky(x + dra_mas / 1000.0, y + ddec_mas / 1000.0)
    return a, b


def _one_cell(out):
    assert out["n_cells"] == 1, out
    return out["cells"][0]


def test_pairs_over_the_cap_are_dropped_and_counted():
    n = 20
    dra = np.zeros(n)
    sigma = np.where(np.arange(n) < 10, 200.0, 10.0)  # 10 over cap, 10 under
    out = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                             min_stars=5, tol_mas=float("inf"),
                             sigma_b_mas=sigma)
    c = _one_cell(out)
    assert c["n"] == 10, c
    assert c["n_cut"] == 10, c
    assert out["n_sigma_cut"] == 10, out
    assert out["n_sigma_unknown"] == 0, out
    assert out["sigma_cap_mas"] == DEFAULT_SIGMA_CAP_MAS, out


def test_unknown_sigma_pairs_are_kept_never_cut():
    n = 20
    dra = np.zeros(n)
    sigma = np.where(np.arange(n) < 10, np.nan, 10.0)  # 10 unknown, 10 known
    out = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                             min_stars=5, tol_mas=float("inf"),
                             sigma_b_mas=sigma)
    c = _one_cell(out)
    assert c["n"] == 20, c            # nothing cut for lacking a sigma
    assert c["n_cut"] == 0, c
    assert out["n_sigma_cut"] == 0, out
    assert out["n_sigma_unknown"] == 10, out


def test_a_custom_cap_overrides_the_default():
    n = 10
    dra = np.zeros(n)
    sigma = np.full(n, 50.0)
    out_default = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                                     min_stars=2, tol_mas=float("inf"),
                                     sigma_b_mas=sigma)
    assert out_default["n_sigma_cut"] == 0   # 50 < 100 default cap
    out_tight = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                                   min_stars=2, tol_mas=float("inf"),
                                   sigma_b_mas=sigma, sigma_cap_mas=25.0)
    assert out_tight["n_sigma_cut"] == n      # 50 > 25 custom cap
    assert out_tight["sigma_cap_mas"] == 25.0


def test_sigma_none_reproduces_the_plain_median_exactly():
    """``sigma_b_mas=None`` must be bit-identical to the pre-#965 code path."""
    n = 9
    rng = np.random.RandomState(3)
    dra = rng.uniform(-20, 20, n)
    ddec = rng.uniform(-20, 20, n)
    out = local_residual_map(*_field(dra, ddec), _tie(), cell_arcsec=100.0,
                             min_stars=3, tol_mas=float("inf"))
    c = _one_cell(out)
    # sub-micro-arcsec round-trip noise from the SkyCoord deg<->arcsec
    # conversion in `_field`, not a difference in the statistic itself.
    assert np.isclose(c["dra_mas"], np.median(dra), atol=1e-4)
    assert np.isclose(c["ddec_mas"], np.median(ddec), atol=1e-4)
    assert c["n_cut"] == 0
    assert out["sigma_cap_mas"] is None
    assert out["n_sigma_cut"] == 0 and out["n_sigma_unknown"] == 0


def test_weighted_median_recovers_the_truth_a_plain_median_misses():
    """4 well-measured pairs read the true ~0 offset; 5 poorly-measured pairs
    (large sigma, still under the cap) read a spurious 95 mas.  The plain
    median is outvoted by the majority-but-uncertain group; the
    inverse-variance-weighted median is not.
    """
    dra = np.concatenate([np.full(4, 5.0), np.full(5, 95.0)])
    sigma = np.concatenate([np.full(4, 5.0), np.full(5, 90.0)])  # 90 < 100 cap
    plain = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                               min_stars=5, tol_mas=float("inf"))
    weighted = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                                  min_stars=5, tol_mas=float("inf"),
                                  sigma_b_mas=sigma)
    c_plain = _one_cell(plain)
    c_weighted = _one_cell(weighted)
    assert np.isclose(c_plain["dra_mas"], 95.0, atol=1e-4), c_plain  # outvoted
    assert np.isclose(c_weighted["dra_mas"], 5.0, atol=1e-4), c_weighted  # trusts sigma
    assert c_weighted["n_cut"] == 0                     # weighted, not cut


def test_a_cell_can_drop_below_min_stars_after_the_cut():
    n = 8
    dra = np.zeros(n)
    sigma = np.where(np.arange(n) < 4, 200.0, 10.0)  # 4 over cap, 4 under
    uncut = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                               min_stars=5, tol_mas=float("inf"))
    assert uncut["n_cells"] == 1, uncut   # 8 >= min_stars=5 with no cut
    cut = local_residual_map(*_field(dra), _tie(), cell_arcsec=100.0,
                             min_stars=5, tol_mas=float("inf"),
                             sigma_b_mas=sigma)
    assert cut["n_cells"] == 0, cut       # 4 kept < min_stars=5: cell vanishes
    assert cut["n_sigma_cut"] == 4, cut
