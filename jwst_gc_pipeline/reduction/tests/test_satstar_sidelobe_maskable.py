"""SATSTAR_SIDELOBE_MASK_FRAC_MIN: a fit that only the sidelobe cut rejects, and
that the fractional criterion of SATSTAR_SIDELOBE_FRAC_MIN would keep, stays out
of the model and the catalog and is marked ``sidelobe_maskable`` in the rejected
table, for the residual footprint mask.

``get_saturated_stars`` runs on the two-star frame of the wing-multiplier tests
with the deblend, the anchor and the PSF stubbed.  The sidelobe cut and the
annulus fraction are driven through ``accept_satstar_fit``, which the stub of
``_run`` wraps: the synthetic frame has no noise, so its own sidelobe sigma
(1e30) and fraction (NaN) say nothing about the gate.
"""
import builtins

import numpy as np
import pytest

import jwst_gc_pipeline.reduction.satstar_deblend as SD
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.tests.test_satstar_recovered_cap import (
    _two_star_frame)
from jwst_gc_pipeline.reduction.tests.test_satstar_wing_multiplier import (
    _EvalPRF)

MASK = 'SATSTAR_SIDELOBE_MASK_FRAC_MIN'
WAIVER = 'SATSTAR_SIDELOBE_FRAC_MIN'
# taken at import so that a second ``_run`` in one test wraps the predicate, not
# the first run's wrapper (which would override the second run's fraction)
_REAL_ACCEPT = SSF.accept_satstar_fit


def _run(monkeypatch, env=None, *, sidelobe_fails=True, frac=None,
         qfit_max_keep=None):
    """``(accepted table or None, rejected table or None)`` of one frame.

    ``sidelobe_fails`` puts the sidelobe cut out of reach, ``frac`` replaces the
    annulus fraction the gate sees, ``qfit_max_keep`` replaces the qfit cap.
    """
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    fh, sat = _two_star_frame()
    n = sat.shape[0]

    def _records(saturated, sources, coms, sizes, zeroframe, data, fwhm, **kw):
        return [{'com': c, 'label': 1, 'forced': False,
                 'sat_area': int(sat.sum())}
                for c in [(100.0, 93.0), (100.0, 107.0)]]

    def _anchor(data, dq, zeroframe, **kw):
        return data, sat.copy(), np.zeros_like(sat), None

    def _gate(**kw):
        if sidelobe_fails:
            kw['sidelobe_min_keep'] = 1e40
        if frac is not None:
            kw['sidelobe_resid_frac'] = frac
        if qfit_max_keep is not None:
            kw['qfit_max_keep'] = qfit_max_keep
        return _REAL_ACCEPT(**kw)

    monkeypatch.setattr(SD, 'build_deblended_source_records', _records)
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    monkeypatch.setattr(SSF, 'get_psf', lambda header, **kw: _EvalPRF(fwhm=1.6))
    monkeypatch.setattr(SSF.stpsf, 'NIRCam', lambda *a, **kw: object())
    monkeypatch.setattr(SSF, 'accept_satstar_fit', _gate)
    builtins.satstar_rejected = None
    tab = SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)),
                                  zeroframe_deblend=True, plot=False)
    rejected = getattr(builtins, 'satstar_rejected', None)
    builtins.satstar_rejected = None
    return tab, rejected


def test_the_switch_is_off_by_default_and_the_rejected_table_has_no_flag(monkeypatch):
    tab, rejected = _run(monkeypatch, frac=-0.1)
    assert tab is None
    assert len(rejected) == 2
    assert 'sidelobe_maskable' not in rejected.colnames
    assert 'sidelobe_resid_frac' not in rejected.colnames
    assert set(rejected['reject_reason']) == {'fit_quality_gate'}


@pytest.mark.parametrize('frac', [-0.1, -0.49])
def test_a_fit_inside_the_fraction_is_rejected_and_marked(monkeypatch, frac):
    tab, rejected = _run(monkeypatch, {MASK: '0.5'}, frac=frac)
    assert tab is None, "the marked fit must not reach the catalog"
    assert len(rejected) == 2
    assert list(rejected['sidelobe_maskable']) == [True, True]
    assert set(rejected['reject_reason']) == {'fit_quality_gate'}
    assert 'sidelobe_resid_frac' in rejected.colnames
    # the marked row keeps the columns the footprint mask reads
    assert np.all(np.asarray(rejected['sat_area'], dtype=float) > 0)
    assert 'skycoord_fit' in rejected.colnames


@pytest.mark.parametrize('frac', [-0.9, -0.5, float('nan')])
def test_a_fit_outside_the_fraction_or_without_one_is_rejected_and_not_marked(
        monkeypatch, frac):
    tab, rejected = _run(monkeypatch, {MASK: '0.5'}, frac=frac)
    assert tab is None
    assert list(rejected['sidelobe_maskable']) == [False, False]


def test_a_fit_the_cut_does_not_reject_is_accepted_and_has_no_rejected_row(monkeypatch):
    tab, rejected = _run(monkeypatch, {MASK: '0.5'}, sidelobe_fails=False)
    assert len(tab) == 2
    assert rejected is None


def test_a_fit_that_fails_another_gate_is_not_marked(monkeypatch):
    tab, rejected = _run(monkeypatch, {MASK: '0.5'}, frac=-0.1, qfit_max_keep=-1.0)
    assert tab is None
    assert list(rejected['sidelobe_maskable']) == [False, False]


def test_a_fit_that_fails_the_implied_peak_gate_is_not_marked(monkeypatch):
    """The waiver would run the severity gate on the fit; a fit it rejects there
    is an unsaturated star, whose footprint is not worth blanking."""
    tab, rejected = _run(monkeypatch, {MASK: '0.5', 'SATSTAR_SEVERITY_FLOOR': '1e12'},
                         frac=-0.1)
    assert tab is None
    assert set(rejected['reject_reason']) == {'implied_peak_gate'}
    assert list(rejected['sidelobe_maskable']) == [False, False]


def test_the_waiver_keeps_what_it_covers_and_the_mask_switch_marks_the_rest(monkeypatch):
    # fraction -0.1: inside both
    tab, rejected = _run(monkeypatch, {WAIVER: '0.5', MASK: '0.9'}, frac=-0.1)
    assert len(tab) == 2 and rejected is None
    # fraction -0.7: the waiver rejects it, the mask switch marks it
    tab, rejected = _run(monkeypatch, {WAIVER: '0.5', MASK: '0.9'}, frac=-0.7)
    assert tab is None
    assert list(rejected['sidelobe_maskable']) == [True, True]
    # fraction -0.95: outside both
    tab, rejected = _run(monkeypatch, {WAIVER: '0.5', MASK: '0.9'}, frac=-0.95)
    assert tab is None
    assert list(rejected['sidelobe_maskable']) == [False, False]


def test_a_mask_fraction_below_the_waiver_marks_nothing(monkeypatch):
    # inside the waiver: accepted by the live gate, so there is nothing to mark
    tab, rejected = _run(monkeypatch, {WAIVER: '0.5', MASK: '0.2'}, frac=-0.3)
    assert len(tab) == 2 and rejected is None
    # outside the waiver and the mask fraction: rejected and not marked
    tab, rejected = _run(monkeypatch, {WAIVER: '0.5', MASK: '0.2'}, frac=-0.7)
    assert tab is None
    assert list(rejected['sidelobe_maskable']) == [False, False]
