"""`measure_offsets` must not report "no shift needed" when it measured nothing.

Issue #394. The too-few-matches branch used to return `0 * u.arcsec` for the
two ACCUMULATORS, which does two separate wrong things:

  1. zero is a positive claim -- for an offsets table it says the frame needs
     no correction -- and it is indistinguishable by value from a real
     measurement of zero;
  2. `total_dra` / `total_ddec` are summed across the passes of the loop, so
     returning zero DELETED the shift the earlier passes had already found.

Both are pinned here, together with the caller guard in
`make_reference_from_pipeline_catalogs` that the old zero walked straight
through (it tested `isinstance(total_dra, u.Quantity)`, and `0 * u.arcsec` is
one).
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.measure_offsets import measure_offsets

# Sparse enough to clear assert_sparse_reference_for_nn_median (median NN
# spacing must exceed 3"): 40 sources over 400" x 400".
N_REF = 40
EXTENT_ARCSEC = 400.0
RA0, DEC0 = 266.5, -28.85
COSD = np.cos(np.deg2rad(DEC0))


def _sparse_pair(offset_arcsec, seed=11):
    """A sparse reference and a copy of it displaced by `offset_arcsec` in RA."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, EXTENT_ARCSEC, N_REF)
    y = rng.uniform(0, EXTENT_ARCSEC, N_REF)
    ref = SkyCoord(ra=(RA0 + x / 3600.0 / COSD) * u.deg,
                   dec=(DEC0 + y / 3600.0) * u.deg, frame="icrs")
    cat = SkyCoord(ra=ref.ra + offset_arcsec * u.arcsec,
                   dec=ref.dec, frame="icrs")
    flux = rng.uniform(1e3, 1e5, N_REF)
    return ref, cat, flux


def test_a_measurable_pair_reports_measured_true():
    """Control: a 0.05" shift is inside the 0.2" match radius and converges."""
    ref, cat, flux = _sparse_pair(0.05)
    out = measure_offsets(reference_coordinates=ref, skycrds_cat=cat,
                          refflux=flux, skyflux=flux)
    assert len(out) == 11
    total_dra, total_ddec, measured = out[0], out[1], out[10]
    assert measured is True
    assert total_dra.to_value(u.arcsec) == pytest.approx(-0.05, abs=1e-3)
    assert total_ddec.to_value(u.arcsec) == pytest.approx(0.0, abs=1e-3)


def test_too_few_matches_does_not_report_a_zero_shift():
    """A 10" displacement leaves nothing inside the 0.2" match radius.

    Before the fix this returned `0 * u.arcsec` -- "already in the right
    place" -- for a pair 10 arcseconds apart.
    """
    ref, cat, flux = _sparse_pair(10.0)
    out = measure_offsets(reference_coordinates=ref, skycrds_cat=cat,
                          refflux=flux, skyflux=flux)
    total_dra, total_ddec, measured = out[0], out[1], out[10]
    assert measured is False
    # nothing measured, so nothing accumulated -- and the caller can now tell
    # this apart from a genuine zero, which it could not when the flag and the
    # value were the same thing
    assert total_dra.to_value(u.arcsec) == 0.0
    assert total_ddec.to_value(u.arcsec) == 0.0
    # the per-pass statistics say "nothing", not "converged to nothing"
    med_dra, med_ddec, std_dra, std_ddec = out[2], out[3], out[4], out[5]
    for q in (med_dra, med_ddec, std_dra, std_ddec):
        assert np.isnan(q.to_value(u.arcsec))


def test_an_unmeasurable_pass_does_not_delete_an_accumulated_shift():
    """The second half of the defect: the accumulators arrive as arguments.

    `total_dra` / `total_ddec` are what earlier passes established. Returning
    zero for them threw that away. Passing them in explicitly reproduces the
    same state the loop reaches partway through.
    """
    ref, cat, flux = _sparse_pair(10.0)
    already = -1.234 * u.arcsec
    out = measure_offsets(reference_coordinates=ref, skycrds_cat=cat,
                          refflux=flux, skyflux=flux,
                          total_dra=already, total_ddec=already)
    total_dra, total_ddec, measured = out[0], out[1], out[10]
    assert measured is False
    assert total_dra.to_value(u.arcsec) == pytest.approx(-1.234)
    assert total_ddec.to_value(u.arcsec) == pytest.approx(-1.234)


def test_the_caller_guard_now_sees_the_failure():
    """`_refine_against_vvv`'s own guard, exercised through the routine.

    The guard reads `measured`; the old one read `isinstance(total_dra,
    u.Quantity)`, which `0 * u.arcsec` satisfies.
    """
    ref, cat, flux = _sparse_pair(10.0)
    out = measure_offsets(reference_coordinates=ref, skycrds_cat=cat,
                          refflux=flux, skyflux=flux)
    total_dra, measured = out[0], out[10]
    assert isinstance(total_dra, u.Quantity)   # the old guard still passes
    assert not measured                        # the new one does not
