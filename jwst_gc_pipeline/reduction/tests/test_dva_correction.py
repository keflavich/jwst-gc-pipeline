"""Unit tests for the inter-detector DVA correction (dva_correction.py).

Hermetic: exercise the pure shift math and the idempotency/keyword gating;
no jwst datamodels or real files needed.
"""
import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction.dva_correction import (
    DVA_MARKER, dva_shift_needed, interdetector_dva_shift_deg)

# Representative Brick values: VA_SCALE from jw02221001001 headers, detector
# ref ~8' from the V1 axis.
VA = 0.9999081898292915
RA_V1, DEC_V1 = 266.40, -28.62
RA_REF, DEC_REF = 266.5268616, -28.7397249   # ~0.13 deg from V1


def test_shift_magnitude_and_sign():
    dra, dde = interdetector_dva_shift_deg(VA, RA_REF, DEC_REF, RA_V1, DEC_V1)
    s = 1 - VA
    # correction pulls the detector TOWARD the common point (contraction,
    # since assign_wcs leaves positions displaced outward by s * lever)
    assert dra == pytest.approx(-s * (RA_REF - RA_V1), rel=1e-12)
    assert dde == pytest.approx(-s * (DEC_REF - DEC_V1), rel=1e-12)
    # ~0.13 deg lever * 9.2e-5 -> tens of mas, not arcsec, not micro-mas
    assert 1 < abs(dra) * 3.6e6 < 60
    assert 1 < abs(dde) * 3.6e6 < 60


def test_two_symmetric_detectors_get_opposite_shifts():
    # detectors placed symmetrically about the common point -> equal and
    # opposite corrections (the anti-symmetric module A/B signature)
    d1 = interdetector_dva_shift_deg(VA, RA_V1 + 0.02, DEC_V1 + 0.03, RA_V1, DEC_V1)
    d2 = interdetector_dva_shift_deg(VA, RA_V1 - 0.02, DEC_V1 - 0.03, RA_V1, DEC_V1)
    assert d1[0] == pytest.approx(-d2[0], rel=1e-12)
    assert d1[1] == pytest.approx(-d2[1], rel=1e-12)


def test_restores_interdetector_consistency():
    # Forward model of the defect: assign_wcs-computed position of a star at
    # true position p, seen by detector d, is displaced by +s*(ref_d - C).
    # After adding the module's correction, both detectors must agree.
    s = 1 - VA
    p = np.array([266.45, -28.66])
    refs = [np.array([RA_V1 + 0.05, DEC_V1 - 0.04]),
            np.array([RA_V1 - 0.06, DEC_V1 + 0.07])]
    seen = []
    for ref in refs:
        delivered = p + s * (ref - np.array([RA_V1, DEC_V1]))
        corr = interdetector_dva_shift_deg(VA, ref[0], ref[1], RA_V1, DEC_V1)
        seen.append(delivered + np.array(corr))
    assert np.allclose(seen[0], seen[1], atol=1e-12)
    assert np.allclose(seen[0], p, atol=1e-12)


def test_trivial_va_scale_no_shift():
    assert interdetector_dva_shift_deg(None, RA_REF, DEC_REF, RA_V1, DEC_V1) == (0.0, 0.0)
    assert interdetector_dva_shift_deg(1, RA_REF, DEC_REF, RA_V1, DEC_V1) == (0.0, 0.0)


def test_invalid_va_scale_raises():
    with pytest.raises(ValueError):
        interdetector_dva_shift_deg(-0.5, RA_REF, DEC_REF, RA_V1, DEC_V1)


def _hdr(**kw):
    h = fits.Header()
    base = dict(VA_SCALE=VA, RA_REF=RA_REF, DEC_REF=DEC_REF,
                RA_V1=RA_V1, DEC_V1=DEC_V1)
    base.update(kw)
    for k, v in base.items():
        if v is not None:
            h[k] = v
    return h


def test_needed_on_fresh_header():
    assert dva_shift_needed(_hdr())


def test_idempotent_marker_blocks_reapplication():
    h = _hdr()
    h[DVA_MARKER] = True
    assert not dva_shift_needed(h)


def test_missing_keywords_mean_no_op():
    for missing in ('VA_SCALE', 'RA_REF', 'DEC_REF', 'RA_V1', 'DEC_V1'):
        h = _hdr()
        del h[missing]
        assert not dva_shift_needed(h)


def test_trivial_va_in_header_means_no_op():
    assert not dva_shift_needed(_hdr(VA_SCALE=1))
