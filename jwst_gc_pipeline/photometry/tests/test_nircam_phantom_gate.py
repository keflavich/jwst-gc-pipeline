"""Regression tests for the NIRCam in-FOV satstar diffraction-spike PHANTOM gate
(`nircam_phantom_reject` in reduction/saturated_star_finding.py).

The non-negotiable property (user requirement): the gate MUST NOT remove real
saturated stars.  On sickle F335M, every real saturated star (flux>1e6) has
qfit<=0.17 regardless of brightness, while diffraction-spike phantoms (no real
core) fit with qfit~1.2 and over-subtract the deep coadd.  The gate rejects only
when BOTH qfit>qfit_min AND model_peak>oversub_ratio*coadd_localpk, so real stars
(low qfit) and correctly-fit faint stars (model~=coadd) are always kept.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.reduction.saturated_star_finding import nircam_phantom_reject


# ---- real saturated stars measured on sickle F335M (must ALWAYS be kept) -----
# (qfit, model_peak, coadd_localpk): clipped coadd cores make model>>core, so
# ONLY the qfit exemption protects them -- this is the regression that matters.
REAL_SATSTARS = [
    (0.04, 1.0e6, 3534.0),   # flux 1e7 star, saturation-clipped coadd core
    (0.03, 5.0e6, 2255.0),   # flux 2.8e6 star
    (0.17, 8.0e5, 7493.0),   # flux 1.2e6 star
    (0.00, 2.0e7, 3000.0),   # flux 2.6e8 (off-FOV-bright) -- extreme model/core
    (0.16, 1.0e5, 1800.0),   # flux 1.2e6 star, faint coadd core
]


@pytest.mark.parametrize("qfit,model_peak,coadd", REAL_SATSTARS)
def test_real_satstars_never_rejected(qfit, model_peak, coadd):
    # default thresholds, and a deliberately AGGRESSIVE pair -- a real star must
    # survive both because its qfit is far below qfit_min.
    assert not nircam_phantom_reject(qfit=qfit, model_peak=model_peak,
                                     coadd_localpk=coadd)
    assert not nircam_phantom_reject(qfit=qfit, model_peak=model_peak,
                                     coadd_localpk=coadd,
                                     qfit_min=0.5, oversub_ratio=1.1)


def test_phantom_rejected():
    # F335M spike phantom: bad qfit AND model peak over-subtracts the coadd.
    assert nircam_phantom_reject(qfit=1.27, model_peak=5995.0,
                                 coadd_localpk=3249.0,
                                 qfit_min=0.5, oversub_ratio=1.2)
    assert nircam_phantom_reject(qfit=1.2, model_peak=6000.0,
                                 coadd_localpk=3000.0,
                                 qfit_min=0.5, oversub_ratio=1.5)


def test_faint_well_fit_star_kept_despite_high_qfit():
    # A correctly-fit faint saturated star: elevated qfit (crowded field) but the
    # model does NOT over-subtract (model ~= coadd peak) -> KEEP.
    assert not nircam_phantom_reject(qfit=0.9, model_peak=5000.0,
                                     coadd_localpk=4800.0,
                                     qfit_min=0.5, oversub_ratio=1.5)


def test_qfit_just_below_threshold_kept():
    assert not nircam_phantom_reject(qfit=0.49, model_peak=9e9,
                                     coadd_localpk=1.0, qfit_min=0.5,
                                     oversub_ratio=1.5)


def test_disabled_gate_keeps_everything():
    assert not nircam_phantom_reject(qfit=5.0, model_peak=1e9, coadd_localpk=1.0,
                                     qfit_min=0.0, oversub_ratio=1.5)
    assert not nircam_phantom_reject(qfit=5.0, model_peak=1e9, coadd_localpk=1.0,
                                     qfit_min=0.5, oversub_ratio=0.0)


@pytest.mark.parametrize("mp,cp", [(np.nan, 3000.0), (5995.0, np.nan),
                                   (5995.0, 0.0), (np.nan, np.nan)])
def test_nonfinite_inputs_failsafe_keep(mp, cp):
    # Cannot measure -> KEEP (never delete a star on an inability to measure).
    assert not nircam_phantom_reject(qfit=1.3, model_peak=mp, coadd_localpk=cp)
