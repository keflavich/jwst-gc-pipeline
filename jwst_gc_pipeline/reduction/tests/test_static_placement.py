"""Unit tests for the static SIAF placement-field correction (opt-in)."""
import numpy as np
from astropy.io import fits

from jwst_gc_pipeline.reduction.static_placement_correction import (
    MARKER, PLACEMENT_MAS, placement_shift_deg, placement_shift_needed)


def test_solved_detectors_have_shifts():
    for det, (dra_star, ddec) in PLACEMENT_MAS.items():
        s = placement_shift_deg(det, -28.7)
        assert s is not None
        # correction is the NEGATIVE of the measured placement, with cos(dec)
        cosd = np.cos(np.radians(-28.7))
        assert np.isclose(s[0] * 3.6e6 * cosd, -dra_star, atol=1e-6)
        assert np.isclose(s[1] * 3.6e6, -ddec, atol=1e-9)


def test_lw_and_unknown_detectors_skipped():
    assert placement_shift_deg('NRCALONG', -28.7) is None
    assert placement_shift_deg('NRCBLONG', -28.7) is None
    assert placement_shift_deg('MIRIMAGE', -28.7) is None


def test_needed_logic():
    h = fits.Header()
    assert not placement_shift_needed(h)          # no detector
    h['DETECTOR'] = 'NRCA3'
    assert placement_shift_needed(h)
    h[MARKER] = True
    assert not placement_shift_needed(h)          # idempotent


def test_amplitudes_are_small():
    # sanity: the whole field is a 1-2.5 mas effect; a typo adding a zero
    # would be caught here
    for det in PLACEMENT_MAS:
        dra, ddec = placement_shift_deg(det, 0.0)
        assert abs(dra) * 3.6e6 < 5 and abs(ddec) * 3.6e6 < 5
