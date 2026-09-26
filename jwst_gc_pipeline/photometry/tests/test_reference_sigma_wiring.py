"""Wiring tests for the Gaia+VIRAC2 per-star predicted-sigma cut (issue #965
item 1 + the follow-up review on PR #970).

``test_reference_uncertainty.py`` covers the pure math (the propagation
formula, the weighted median, the unknown-sigma fallback).
``test_region_map_sigma_cut.py`` covers ``local_residual_map``'s cut/weight
behaviour, and ``test_same_star_region_map.py`` covers the region-map-level
hidden-seam regression.  This file answers a narrower question the review
found untested: does each of the 5 places the sigma column has to pass
through unmodified actually do so, and is the cut OFF by default everywhere
it is wired in.

Each test here is written to KILL a specific mutant (verified by hand: mutate
the real code, watch the corresponding test here fail, then verify it passes
again on the unmutated code):

  (a) ``_exclude_blended_references`` reindexing the sigma subset -- see
      ``test_astrometry_checkpoint.py::test_exclude_blended_references_keeps_sigma_row_aligned``.
  (b) ``load_reference_catalog`` always returning ``None`` -- this file.
  (c) the refcat builder with ``dt_virac``/``dt_gaia`` pinned to 0 -- this file.
  (d) ``measure_reference_tie``'s pass-through to ``same_star_region_map``
      replaced by ``None`` -- see
      ``test_same_star_region_map.py::test_measure_reference_tie_passes_sigma_through_to_the_region_map``.
  (e) ``local_residual_map``'s ``n_pairs`` reporting the pre-cut count -- see
      ``test_region_map_sigma_cut.py::test_n_pairs_reports_the_post_cut_count_not_the_pre_cut_total``.

Plus the opt-in gate (blocker 2): OFF by default, and byte-for-byte identical
to no sigma column at all until an operator sets ``REF_SIGMA_CUT_ENABLE=1``.
"""
import os
import tempfile

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.reference_uncertainty import (
    SIGMA_PRED_COLUMN, sigma_pred_mas, combined_sigma_pred_mas)
from jwst_gc_pipeline.photometry.visit_consensus import load_reference_catalog
from jwst_gc_pipeline.reduction import build_gaia_virac2_refcat_byquery as B
from jwst_gc_pipeline.photometry import astrometry_checkpoint as _ac

RA0, DEC0 = 266.60, -28.50


# ---------------------------------------------------------------------------
# (b) the loader
# ---------------------------------------------------------------------------

def _minimal_refcat_table(with_sigma):
    ref = Table()
    ref['RA'] = [RA0, RA0 + 0.001, RA0 + 0.002]
    ref['DEC'] = [DEC0, DEC0, DEC0]
    ref['source'] = ['GaiaDR3', 'VIRAC2', 'VIRAC2']
    ref['refmag'] = [15.0, 16.0, 17.0]
    if with_sigma:
        ref[SIGMA_PRED_COLUMN] = [3.0, 12.0, 25.0]
    return ref


def test_load_reference_catalog_returns_sigma_when_column_present(tmp_path):
    """Mutant (b): ``load_reference_catalog`` always returning ``None`` for
    ``sigma_pred_mas`` would pass every OTHER test that loads a refcat without
    the column; this one has the column and must actually read it back."""
    ref = _minimal_refcat_table(with_sigma=True)
    path = os.path.join(str(tmp_path), 'refcat_with_sigma.fits')
    ref.write(path)
    loaded = load_reference_catalog(path)
    assert loaded['sigma_pred_mas'] is not None
    np.testing.assert_allclose(loaded['sigma_pred_mas'], [3.0, 12.0, 25.0])
    # row-aligned with `all`, not with `sparse` (the Gaia-only subset)
    assert len(loaded['sigma_pred_mas']) == len(loaded['all']) == 3


def test_load_reference_catalog_returns_none_when_column_absent(tmp_path):
    """The pre-#965 refcat shape must still load, with sigma reported as
    unknown (``None``) rather than a crash or a fabricated zero."""
    ref = _minimal_refcat_table(with_sigma=False)
    path = os.path.join(str(tmp_path), 'refcat_no_sigma.fits')
    ref.write(path)
    loaded = load_reference_catalog(path)
    assert loaded['sigma_pred_mas'] is None
    assert len(loaded['all']) == 3


# ---------------------------------------------------------------------------
# (c) the builder's epoch-pinned propagation
# ---------------------------------------------------------------------------

def test_build_refcat_table_pins_the_proper_motion_term_to_the_epoch_baseline():
    """Mutant (c): with ``dt_virac``/``dt_gaia`` forced to 0, the PM term in
    ``sigma_pred_mas`` vanishes and every row's predicted sigma collapses to
    its bare position error.  Pin a known epoch baseline, give each star a
    large, distinct PM error, and assert the PM term actually enters the
    combined sigma -- both that it matches the independently-computed
    quadrature sum, and that it is measurably larger than the bare position
    term alone.
    """
    epoch = 2026.70
    dt_gaia = epoch - B.GAIA_EPOCH
    dt_virac = epoch - B.VIRAC2_EPOCH
    assert dt_gaia > 5 and dt_virac > 5   # a real, multi-year baseline

    g = Table()
    g['ra'] = [RA0]
    g['dec'] = [DEC0]
    g['pmra'] = [0.0]
    g['pmdec'] = [0.0]
    g['phot_g_mean_mag'] = [15.0]
    g['ra_error'] = [1.0]      # mas
    g['dec_error'] = [1.0]
    g['pmra_error'] = [2.0]    # mas/yr, deliberately large
    g['pmdec_error'] = [2.0]

    v = Table()
    v['RAJ2000'] = [RA0 + 0.01]     # ~36" away: no Gaia<->VIRAC2 dedup
    v['DEJ2000'] = [DEC0 + 0.01]
    v['pmRA'] = [0.0]
    v['pmDE'] = [0.0]
    v['Jmag'] = [16.0]
    v['e_RAJ2000'] = [3.0]
    v['e_DEJ2000'] = [3.0]
    v['e_pmRA'] = [4.0]
    v['e_pmDE'] = [4.0]

    ref = B.build_refcat_table(g, 'vizier', v, epoch, 0.02, min_ref_density=0.0)
    labels = np.asarray(ref['source']).astype(str)
    gaia_row = ref[labels == 'GaiaDR3'][0]
    virac_row = ref[labels == 'VIRAC2'][0]

    expected_gaia = sigma_pred_mas(1.0, 1.0, 2.0, 2.0, dt_gaia)
    expected_virac = sigma_pred_mas(3.0, 3.0, 4.0, 4.0, dt_virac)
    bare_gaia_pos = combined_sigma_pred_mas(1.0, 1.0)
    bare_virac_pos = combined_sigma_pred_mas(3.0, 3.0)

    assert np.isclose(float(gaia_row[SIGMA_PRED_COLUMN]), expected_gaia, rtol=1e-6), \
        (float(gaia_row[SIGMA_PRED_COLUMN]), expected_gaia)
    assert np.isclose(float(virac_row[SIGMA_PRED_COLUMN]), expected_virac, rtol=1e-6), \
        (float(virac_row[SIGMA_PRED_COLUMN]), expected_virac)
    # the PM term must actually be present at THIS epoch baseline, not just
    # correct in the formula for dt=0 (where it would vanish either way).
    assert gaia_row[SIGMA_PRED_COLUMN] > bare_gaia_pos + 1.0, gaia_row[SIGMA_PRED_COLUMN]
    assert virac_row[SIGMA_PRED_COLUMN] > bare_virac_pos + 1.0, virac_row[SIGMA_PRED_COLUMN]


# ---------------------------------------------------------------------------
# opt-in gate (blocker 2): off by default, explicit knob to turn on
# ---------------------------------------------------------------------------

def test_sigma_cut_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(_ac.REF_SIGMA_CUT_ENV, raising=False)
    assert _ac.ref_sigma_cut_enabled() is False
    refcat = dict(sigma_pred_mas=np.array([5.0, 500.0]))
    kwargs = _ac._sigma_cut_kwargs(refcat)
    assert kwargs == dict(ref_sigma_pred_mas=None, ref_sigma_cap_mas=None)


def test_sigma_cut_turns_on_only_with_the_explicit_env_knob(monkeypatch):
    monkeypatch.setenv(_ac.REF_SIGMA_CUT_ENV, "1")
    assert _ac.ref_sigma_cut_enabled() is True
    sigma = np.array([5.0, 500.0])
    refcat = dict(sigma_pred_mas=sigma)
    kwargs = _ac._sigma_cut_kwargs(refcat)
    assert kwargs["ref_sigma_pred_mas"] is sigma
    assert kwargs["ref_sigma_cap_mas"] is None    # default cap unless overridden


def test_sigma_cut_cap_override_env(monkeypatch):
    monkeypatch.setenv(_ac.REF_SIGMA_CUT_ENV, "1")
    monkeypatch.setenv(_ac.REF_SIGMA_CAP_ENV, "42.0")
    refcat = dict(sigma_pred_mas=np.array([5.0]))
    kwargs = _ac._sigma_cut_kwargs(refcat)
    assert kwargs["ref_sigma_cap_mas"] == 42.0


def test_sigma_cut_kwargs_is_none_safe(monkeypatch):
    """A refcat with no sigma column, or no refcat at all, must not raise --
    the gate is a no-op either way."""
    monkeypatch.setenv(_ac.REF_SIGMA_CUT_ENV, "1")
    assert _ac._sigma_cut_kwargs(None) == dict(ref_sigma_pred_mas=None,
                                               ref_sigma_cap_mas=None)
    assert _ac._sigma_cut_kwargs(dict(sigma_pred_mas=None)) == dict(
        ref_sigma_pred_mas=None, ref_sigma_cap_mas=None)
