"""The refcat builder refuses a query whose usable coverage is not sky.

Issue #415 gap 4.  The builder's only refusals were "VIRAC2 query returned
nothing", "VizieR Gaia DR3 query returned nothing" and the 2000-row sync cap:
between the query and ``ref.write`` nothing looked at how much came back.  A
truncated VizieR response, a wrong ``--radius`` or a cone placed off the field
all produce a small-but-nonzero catalog, and the m2 tie is then measured against
it.

The floor is a DENSITY, not a row count, which is what makes it safe on non-GC
fields at other radii.  Its calibration is the 139-tile cone survey recorded on
the issue: every program-10678 tile returns 1.7e6-4.9e6 usable sources/deg^2 and
the brick's working refcat (same-star tie ~0.6 mas) 2.3e6, so 8e5 sits about a
factor of two below the thinnest real sky and two orders above a broken query.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.reduction.build_gaia_virac2_refcat_byquery import (
    MIN_REF_DENSITY_PER_SQDEG, ThinReferenceCoverageError,
    check_reference_coverage)


def _n_for(density, radius_deg):
    return int(round(density * np.pi * radius_deg ** 2))


@pytest.mark.parametrize("radius_deg", [0.02, 0.1, 0.35])
def test_the_thinnest_real_tile_builds_at_every_radius(radius_deg):
    """GC_130, the thinnest of the 139 tiles, at 1.7e6 deg^-2.  Because the floor
    is a density it clears at every build radius, which an absolute row count
    tuned for a 0.1 deg cone would not."""
    d = check_reference_coverage(_n_for(1.7e6, radius_deg), radius_deg)
    assert d == pytest.approx(1.7e6, rel=1e-3)


def test_the_brick_refcat_density_builds():
    """The reference point that makes the number mean something: the field whose
    same-star tie reads ~0.6 mas."""
    check_reference_coverage(_n_for(2.3e6, 0.1), 0.1)


def test_a_truncated_query_is_refused():
    """A VizieR response cut to a few hundred rows over a 0.1 deg cone is two
    orders below any real GC sky."""
    with pytest.raises(ThinReferenceCoverageError) as ex:
        check_reference_coverage(400, 0.1, context="gc-treasury o037")
    assert "BROKEN QUERY" in str(ex.value)
    assert "gc-treasury o037" in str(ex.value)


def test_a_cone_off_the_field_is_refused():
    """Zero usable rows is the same failure, not a special case."""
    with pytest.raises(ThinReferenceCoverageError):
        check_reference_coverage(0, 0.1)


def test_the_floor_sits_between_the_two_populations():
    """Pin the calibration itself: the default is below the thinnest measured
    tile and above a truncated query, so moving it needs a reason."""
    assert MIN_REF_DENSITY_PER_SQDEG < 1.7e6
    assert MIN_REF_DENSITY_PER_SQDEG > 400 / (np.pi * 0.1 ** 2)


def test_zero_records_a_deliberate_override():
    check_reference_coverage(1, 0.1, min_density=0.0)


def test_a_non_positive_radius_is_an_error_not_a_pass():
    with pytest.raises(ValueError):
        check_reference_coverage(10_000, 0.0)
