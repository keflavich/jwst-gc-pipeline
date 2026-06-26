"""Regression: the extended-emission vetting must NOT drop a well-fit star whose
formal S/N is broken by group-fit covariance degeneracy.

Cause (sickle F480M, m3->m4): close-pair stars are fit in a group; the near-
degenerate joint normal matrix inflates flux_err 100-1000x (flux 6200/err 8076
-> S/N 0.8; flux 4402/err 144220 -> S/N 0.0) while flux and qfit (0.016) stay
excellent.  The old NIRCam keep-mask required S/N>=floor, so these confident
stars were dropped from the vetted catalog -> removed from the next phase's seed
AND the vetted residual mosaic -> they reappeared, strong and unsubtracted, and
were never re-fit.  A qfit-confident source must be kept regardless of S/N
(emission bumps have BAD qfit, so this cannot re-admit emission).
"""
import numpy as np
import pytest
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry.cataloging import (
    _emission_keep_nircam, _filter_extended_emission)


def test_keep_nircam_qfit_confident_survives_broken_snr():
    star_like = np.array([True, True, True])
    snr = np.array([0.8, 0.8, 200.0])           # broken, broken, good
    qfit_conf = np.array([True, False, False])   # confident, not, not
    keep = _emission_keep_nircam(star_like, snr, 5.0, qfit_confident=qfit_conf)
    # confident star with broken S/N -> KEPT (the regression)
    assert keep[0]
    # non-confident star_like (peakSB-only) with low S/N -> dropped
    assert not keep[1]
    # non-confident but high S/N -> kept
    assert keep[2]


def test_keep_nircam_backcompat_without_qfit():
    # old signature (no qfit_confident) keeps the prior behaviour
    star_like = np.array([True, True])
    snr = np.array([0.8, 200.0])
    keep = _emission_keep_nircam(star_like, snr, 5.0)
    assert not keep[0]
    assert keep[1]


def _mk(qfit, flux, flux_err, ra0=266.5):
    n = len(qfit)
    ras = ra0 + np.arange(n) * 0.001
    return Table({
        'skycoord': SkyCoord(ras * u.deg, np.full(n, -28.8) * u.deg),
        'qfit': np.asarray(qfit, float),
        'flux': np.asarray(flux, float),
        'flux_err': np.asarray(flux_err, float),
        'flags': np.zeros(n, int),
        'local_bkg': np.zeros(n),
    })


def test_filter_keeps_wellfit_star_with_inflated_fluxerr():
    # star1/star5 verbatim: excellent qfit, flux fine, flux_err blown up by group
    t = _mk(qfit=[0.016, 0.019], flux=[6200.0, 4402.0],
            flux_err=[8076.0, 144220.0])
    vet = _filter_extended_emission(t, qfit_max=0.2, local_snr_min=5.0,
                                    label='regr')
    assert len(vet) == 2, "well-fit stars with broken S/N must be kept"


def test_filter_still_drops_lowsnr_nonstar():
    # a NON-confident source (bad qfit, not bright) with low S/N is still dropped
    t = _mk(qfit=[0.5], flux=[40.0], flux_err=[40.0])  # snr=1, qfit>0.2
    vet = _filter_extended_emission(t, qfit_max=0.2, local_snr_min=5.0,
                                    label='regr')
    assert len(vet) == 0
