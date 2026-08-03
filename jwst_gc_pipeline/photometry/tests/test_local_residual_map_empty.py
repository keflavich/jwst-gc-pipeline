"""``local_residual_map`` must answer "could not measure", never crash.

It guards the no-pairs case at ``search_around_sky``, but the uniqueness filter
that runs afterwards can also empty the pair set: in a crowded field one ``b``
can be the nearest partner of several ``a``, and when that is true of all of
them, nothing survives. Pairs WERE found -- they were just all ambiguous -- so
the early guard does not fire, and ``ra_deg.min()`` on the zero-size array
raises.

That crashed the whole ``--refcat`` arbiter run of
``scripts/release/check_interframe_overlap.py`` on brick F187N (2026-08-03),
*after* it had already produced a clean verdict for F182M's deferred sliver pair
(worst 16 mas, 260 same-star matches). The callers are written to handle
``cells: []``; an unhandled ValueError takes the whole release gate down and
loses the results of the filters it had already finished.
"""
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import local_residual_map


GLOBAL_OK = dict(ok=True, swept=False, dra=0.0, ddec=0.0, off=0.0)


def _empty_result_shape(res):
    assert res["cells"] == []
    assert res["n_cells"] == 0
    assert res["n_measured"] == 0
    assert res["n_flagged"] == 0
    assert np.isnan(res["worst_off_mas"])
    assert res["clean"] is False


def test_no_pairs_at_all_returns_empty():
    """The already-guarded case: nothing within the match radius."""
    a = SkyCoord([266.0, 266.001] * u.deg, [-28.0, -28.001] * u.deg)
    b = SkyCoord([267.0, 267.001] * u.deg, [-27.0, -27.001] * u.deg)
    _empty_result_shape(
        local_residual_map(a, b, GLOBAL_OK, match_radius=0.3 * u.arcsec))


def test_all_pairs_ambiguous_returns_empty_not_valueerror():
    """Every ``a`` has the SAME single nearest ``b``, so the uniqueness filter
    drops all of them and the surviving pair set is empty.

    Before the fix this raised
    ``ValueError: zero-size array to reduction operation minimum which has no
    identity`` from ``ra_deg.min()``.
    """
    # three sources packed well inside the match radius of one reference star
    a = SkyCoord([266.0, 266.00001, 266.00002] * u.deg,
                 [-28.0, -28.00001, -28.00002] * u.deg)
    b = SkyCoord([266.00001] * u.deg, [-28.00001] * u.deg)
    res = local_residual_map(a, b, GLOBAL_OK, match_radius=1.0 * u.arcsec,
                             context="all-ambiguous")
    _empty_result_shape(res)


def test_unambiguous_pairs_still_measure():
    """The guard must not swallow the working case: well-separated one-to-one
    pairs still produce a measured cell."""
    n = 60
    rng = np.random.default_rng(0)
    ra = 266.0 + rng.uniform(0, 20, n) / 3600.0
    dec = -28.0 + rng.uniform(0, 20, n) / 3600.0
    a = SkyCoord(ra * u.deg, dec * u.deg)
    # shift every partner by the same small amount: one-to-one, unambiguous
    b = SkyCoord((ra + 2e-6) * u.deg, dec * u.deg)
    res = local_residual_map(a, b, GLOBAL_OK, cell_arcsec=1e9,
                             match_radius=0.3 * u.arcsec, min_stars=10)
    assert res["n_measured"] >= 1
    assert res["cells"]
