"""A small, well-measured bulk with a handful of flagged region cells is
RECORDED, not BLOCKED (issue #965 item 2).

``REGION_BULK_DOMINANCE_RATIO`` (#921, ``test_reference_tie_bulk_dominance.py``)
already lets a dirty region map through when the bulk dominates the worst
cell, but only once the bulk reaches ``REGION_BULK_DOMINANCE_MIN_BULK_MAS``
(20 mas) -- the ratio's denominator is unstable below that.  Four gc-treasury
tiles (o075/o080/o084/o087) never reach it: their same-star bulks are
4.1-8.1 mas, measured to +/-1.6-1.9 mas, sitting on a region map with a
handful of cells over the adaptive tolerance (item 3) purely from VIRAC2's own
reference noise.  Nothing about that combination -- a bulk this SMALL, this
PRECISELY measured, with only a few flagged cells and no coverage failure --
looks like a seam, so it is recorded (``reference_tie.region_map.status =
"recorded_nonblocking"``) and applied rather than refused outright.

This mirrors ``test_reference_tie_bulk_dominance.py``'s fixture style: the
estimators (``measure_offset``, ``measure_offset_grid``, ``same_star_region_map``,
``local_residual_map``) are stubbed with canned shapes and the REAL decision
path in ``measure_reference_tie`` runs on top of them, so these tests exercise
shipping code rather than a synthetic approximation of it.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry import visit_consensus as vc
from jwst_gc_pipeline.photometry.visit_consensus import (
    REGION_SMALL_BULK_SIGMA_TOL_MAS, REGION_SMALL_BULK_MAX_FLAGGED_FRACTION,
    REGION_SMALL_BULK_WORST_SIG_CAP_MAS, measure_reference_tie)

_RA0, _DEC0 = 266.4, -28.9


def _coords(n=200):
    rng = np.random.RandomState(7)
    return SkyCoord((_RA0 + (rng.rand(n) - 0.5) / 60.0) * u.deg,
                    (_DEC0 + (rng.rand(n) - 0.5) / 60.0) * u.deg)


def _offset(dra_mas, ddec_mas, ok=True, swept=False):
    return dict(dra=dra_mas / 1000.0, ddec=ddec_mas / 1000.0,
                off=float(np.hypot(dra_mas, ddec_mas)) / 1000.0,
                ok=ok, swept=swept, alias_rejected=False,
                window_consistent=True, window_edge_fraction=0.01,
                contrast=60.0, n_peak=500, npairs=100000,
                window_arcsec=3.0, dra_err=0.0008, ddec_err=0.0008)


def _region_map(worst_mas, clean, n_flagged, n_measured=25, n_uncovered=0):
    return dict(clean=clean, measurable=True, n_cells=n_measured,
                n_measured=n_measured, n_flagged=n_flagged,
                n_uncovered=n_uncovered,
                uncovered_cells=[{'by': 'tight-pairs'}] * n_uncovered,
                worst_off_mas=worst_mas, worst_sig_off_mas=worst_mas,
                n_pairs=4000, cells=[], tol_mas=30.0, tol_k=3.0,
                tol_floor_mas=30.0,
                reason=f'{n_flagged} region cell(s) above 30.0 mas '
                       f'(adaptive tolerance) at 3.0-sigma')


def _tie(monkeypatch, bulk_mas, worst_mas, clean=False, n_flagged=2,
         n_measured=25, n_uncovered=0, dra_sem=0.8, ddec_sem=0.8,
         region_measurable=True, grid_clean=True):
    """Run the REAL decision over canned estimator output."""
    dra, ddec = bulk_mas
    monkeypatch.setattr(vc, 'measure_offset',
                        lambda *a, **k: _offset(dra, ddec))
    monkeypatch.setattr(vc, 'measure_offset_grid',
                        lambda *a, **k: dict(clean=grid_clean, n_total=4,
                                             worst_off_mas=worst_mas, cells=[]))
    if region_measurable:
        rm = _region_map(worst_mas, clean, n_flagged, n_measured, n_uncovered)
    else:
        rm = dict(clean=False, measurable=False, n_cells=0, n_measured=0,
                  n_flagged=0, worst_off_mas=float('nan'), cells=[])
    monkeypatch.setattr(vc, 'same_star_region_map', lambda *a, **k: rm)
    monkeypatch.setattr(vc, 'local_residual_map',
                        lambda *a, **k: dict(cells=[dict(
                            dra_mas=dra, ddec_mas=ddec, n=4000,
                            dra_sem=dra_sem, ddec_sem=ddec_sem)], n_pairs=4000))
    c = _coords()
    return measure_reference_tie(c, c, c, dense=True, context='test')


# ---------------------------------------------------------------------------
# the demotion itself
# ---------------------------------------------------------------------------

def test_a_small_precisely_measured_bulk_with_a_few_flagged_cells_is_demoted(
        monkeypatch):
    """o075/o080/o084/o087 class: a 4-8 mas bulk, measured to ~1.1 mas
    (dra_sem=ddec_sem=0.8 -> sigma=hypot(0.8,0.8)=1.13 mas), 2 of 25 cells
    (8%) over the adaptive tolerance, no uncovered cell."""
    r = _tie(monkeypatch, (-6.0, -3.0), 45.0, n_flagged=2, n_measured=25)
    assert r['per_tile_source'] == 'same-star-region'
    assert r['per_tile_ok'] is False, 'fixture must be a DIRTY map to be a test'
    assert r['region_bulk_dominance_ratio'] is None, (
        'bulk is well under REGION_BULK_DOMINANCE_MIN_BULK_MAS -- the #921 '
        'ratio exemption must not be the one doing the work here')
    assert r['per_tile_small_bulk_exempt'] is True
    assert r['region_map']['status'] == 'recorded_nonblocking'
    assert r['region_map']['demoted'] is True
    assert r['region_map']['n_flagged'] == 2
    assert r['region_map']['n_measured'] == 25
    assert r['region_map']['flagged_fraction'] == pytest.approx(0.08)
    assert r['region_map']['same_star_sigma_mas'] < REGION_SMALL_BULK_SIGMA_TOL_MAS
    assert r['apply_ok'] is True, (
        'a small bulk measured to ~1 mas was refused over reference noise in '
        '2 of 25 cells; the correction cannot be improved by refusing it')


def test_a_clean_map_reads_not_a_demotion(monkeypatch):
    """Unchanged behaviour: a clean map needs no exemption, and the record
    says so plainly rather than reporting a demotion that did not happen."""
    r = _tie(monkeypatch, (-6.0, -3.0), 5.0, clean=True, n_flagged=0)
    assert r['per_tile_ok'] is True
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'clean'
    assert r['region_map']['demoted'] is False
    assert r['apply_ok'] is True


def test_the_histogram_grid_path_is_not_applicable(monkeypatch):
    """When the region map cannot measure, the record must not claim a
    region-map verdict of any kind -- there is nothing to demote or block."""
    r = _tie(monkeypatch, (-6.0, -3.0), 45.0, region_measurable=False,
             grid_clean=True)
    assert r['per_tile_source'] == 'histogram-grid'
    assert r['region_map']['status'] == 'not_applicable'
    assert r['per_tile_small_bulk_exempt'] is False


# ---------------------------------------------------------------------------
# scope of the exemption
# ---------------------------------------------------------------------------

def test_the_exemption_requires_a_low_flagged_fraction(monkeypatch):
    """4 of 25 cells (16%) is over REGION_SMALL_BULK_MAX_FLAGGED_FRACTION
    (10%) -- a precisely measured bulk does not buy a pass on its own."""
    r = _tie(monkeypatch, (-6.0, -3.0), 45.0, n_flagged=4, n_measured=25)
    frac = r['region_map']['flagged_fraction']
    assert frac > REGION_SMALL_BULK_MAX_FLAGGED_FRACTION, (
        f'fixture flagged fraction {frac} must exceed the bar to be a test')
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'blocking'
    assert r['apply_ok'] is False


def test_the_exemption_requires_a_precise_bulk(monkeypatch):
    """A bulk measured only to ~2.8 mas (dra_sem=ddec_sem=2.0) is over
    REGION_SMALL_BULK_SIGMA_TOL_MAS (2.0 mas) even with a low flagged
    fraction -- CONFIDENCE in the bulk, not just its size, gates this
    exemption."""
    r = _tie(monkeypatch, (-6.0, -3.0), 45.0, n_flagged=2, n_measured=25,
             dra_sem=2.0, ddec_sem=2.0)
    sigma = r['region_map']['same_star_sigma_mas']
    assert sigma > REGION_SMALL_BULK_SIGMA_TOL_MAS, (
        f'fixture sigma {sigma} must exceed the bar to be a test')
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'blocking'
    assert r['apply_ok'] is False


def test_the_exemption_never_overrides_an_uncovered_cell(monkeypatch):
    """A displaced-beyond-the-match-radius region (brick-1182 v001's class)
    must keep blocking no matter how precisely the bulk elsewhere is
    measured, or how few cells are flagged -- see the same guard in
    ``test_reference_tie_bulk_dominance.test_a_coverage_only_dirty_map_is_
    still_refused`` for the #921 exemption's identical rule."""
    r = _tie(monkeypatch, (-6.0, -3.0), 4.0, n_flagged=0, n_measured=25,
             n_uncovered=1)
    assert r['per_tile_ok'] is False
    assert r['region_map']['same_star_sigma_mas'] < REGION_SMALL_BULK_SIGMA_TOL_MAS, (
        'fixture must look precisely measured to be a test of the coverage '
        'override specifically')
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'blocking'
    assert r['region_map']['n_uncovered'] == 1
    assert r['apply_ok'] is False


def test_the_exemption_does_not_reach_the_bulk_dominance_ratio_regime(
        monkeypatch):
    """A bulk large enough for the #921 ratio exemption is graded by THAT
    exemption, not this one -- ``per_tile_bulk_dominant_exempt`` is the one
    that goes True, and item 2 stays out of the way rather than double
    counting the same pass."""
    r = _tie(monkeypatch, (-61.66, -9.91), 25.29, n_flagged=6, n_measured=25)
    assert r['per_tile_bulk_dominant_exempt'] is True
    assert r['apply_ok'] is True
    # item 2's own flagged-fraction bar (6/25 = 24%) would have refused this
    # tile on its own -- the record must show the ratio exemption, not item 2,
    # as the reason it passed.
    assert r['region_map']['flagged_fraction'] == pytest.approx(0.24)
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'blocking', (
        'item 2 did not demote this map on its own numbers -- the #921 ratio '
        'is a separate exemption that applies_ok reads independently')


# ---------------------------------------------------------------------------
# the worst-cell cap (issue #968 review): sigma and flagged-fraction alone
# never look at the SIZE of the worst flagged cell, so a seam confined to a
# small minority of cells passed both bars regardless of how large that one
# cell actually was.
# ---------------------------------------------------------------------------

def test_a_90_mas_seam_in_a_small_minority_of_cells_stays_refused(monkeypatch):
    """The reviewer's exact counter-example: a 90 mas seam confined to 1 of 25
    cells (4%), with a bulk still measured to ~1.1 mas (dra_sem=ddec_sem=0.8),
    used to clear both the sigma and flagged-fraction bars and demote with
    apply_ok=True. It must now stay blocking: 90 mas is far over
    REGION_SMALL_BULK_WORST_SIG_CAP_MAS."""
    r = _tie(monkeypatch, (-6.0, -3.0), 90.0, n_flagged=1, n_measured=25)
    sigma = r['region_map']['same_star_sigma_mas']
    frac = r['region_map']['flagged_fraction']
    assert sigma <= REGION_SMALL_BULK_SIGMA_TOL_MAS, 'fixture must clear the sigma bar to be a test of the cap'
    assert frac <= REGION_SMALL_BULK_MAX_FLAGGED_FRACTION, 'fixture must clear the fraction bar to be a test of the cap'
    assert r['region_bulk_dominance_ratio'] is None, (
        'bulk is well under REGION_BULK_DOMINANCE_MIN_BULK_MAS -- the #921 '
        'ratio exemption must not be the one doing the work here')
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'blocking'
    assert r['region_map']['demoted'] is False
    assert r['region_map']['worst_sig_off_mas'] == pytest.approx(90.0)
    assert r['region_map']['worst_sig_off_cap_mas'] == REGION_SMALL_BULK_WORST_SIG_CAP_MAS
    assert r['apply_ok'] is False


def test_a_250_mas_seam_in_a_small_minority_of_cells_stays_refused(monkeypatch):
    """Same shape as the 90 mas case, at a size closer to brick-1182 v001's
    class of failure (~20" rigid half-mosaic shift) -- still one flagged cell
    of 25, still a precisely measured bulk. Must stay refused."""
    r = _tie(monkeypatch, (-6.0, -3.0), 250.0, n_flagged=1, n_measured=25)
    sigma = r['region_map']['same_star_sigma_mas']
    frac = r['region_map']['flagged_fraction']
    assert sigma <= REGION_SMALL_BULK_SIGMA_TOL_MAS, 'fixture must clear the sigma bar to be a test of the cap'
    assert frac <= REGION_SMALL_BULK_MAX_FLAGGED_FRACTION, 'fixture must clear the fraction bar to be a test of the cap'
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'blocking'
    assert r['region_map']['demoted'] is False
    assert r['apply_ok'] is False


def test_the_o084_treasury_tile_no_longer_demotes_under_the_cap(monkeypatch):
    """o084 (F212N, live gc-treasury checkpoint, 2026-09-25): sigma 1.61 mas
    and 1/28 (3.6%) flagged both clear their bars on their own, same as
    o075/o080/o087, but its worst SIGNIFICANT cell reads 57.35 mas (46 stars)
    -- over REGION_SMALL_BULK_WORST_SIG_CAP_MAS (50.0). Reported explicitly,
    not silently absorbed: unlike o075 (22.98 mas)/o080 (25.84 mas)/o087
    (23.54 mas), o084 now stays blocking rather than being demoted."""
    r = _tie(monkeypatch, (-6.0, -3.0), 57.35, n_flagged=1, n_measured=28,
             dra_sem=1.13, ddec_sem=0.0)
    assert r['region_map']['same_star_sigma_mas'] == pytest.approx(1.13, abs=0.01)
    assert r['region_map']['flagged_fraction'] == pytest.approx(1 / 28)
    assert r['per_tile_small_bulk_exempt'] is False
    assert r['region_map']['status'] == 'blocking'
    assert r['apply_ok'] is False


def test_the_worst_sig_cap_boundary(monkeypatch):
    """Just under the cap still demotes; just over it does not -- the cap is
    inclusive (<=), matching REGION_SMALL_BULK_SIGMA_TOL_MAS and
    REGION_SMALL_BULK_MAX_FLAGGED_FRACTION above."""
    under = _tie(monkeypatch, (-6.0, -3.0),
                 REGION_SMALL_BULK_WORST_SIG_CAP_MAS - 0.1,
                 n_flagged=1, n_measured=25)
    assert under['per_tile_small_bulk_exempt'] is True
    assert under['apply_ok'] is True

    over = _tie(monkeypatch, (-6.0, -3.0),
                REGION_SMALL_BULK_WORST_SIG_CAP_MAS + 0.1,
                n_flagged=1, n_measured=25)
    assert over['per_tile_small_bulk_exempt'] is False
    assert over['apply_ok'] is False


def test_a_noisy_insignificant_cell_does_not_defeat_the_exemption(
        monkeypatch):
    """The cap reads `worst_sig_off_mas` (only statistically SIGNIFICANT
    cells), not the unconditional `worst_off_mas`: a low-N cell too noisy to
    be told apart from zero must not itself block a demotion the significant
    cells otherwise clear. This mirrors o075/o080 in the live checkpoints,
    where `worst_off_mas` (31.6/25.3 mas) exceeds `worst_sig_off_mas`
    (25.8/23.0 mas) because of exactly such a cell."""
    rm = _region_map(worst_mas=REGION_SMALL_BULK_WORST_SIG_CAP_MAS + 20.0,
                     clean=False, n_flagged=1, n_measured=25)
    # Override worst_off_mas only, leaving worst_sig_off_mas (the value the
    # cap actually reads) well inside the cap -- an insignificant cell noisier
    # than any real flagged cell.
    rm['worst_off_mas'] = REGION_SMALL_BULK_WORST_SIG_CAP_MAS + 20.0
    rm['worst_sig_off_mas'] = REGION_SMALL_BULK_WORST_SIG_CAP_MAS - 5.0

    dra, ddec = -6.0, -3.0
    monkeypatch.setattr(vc, 'measure_offset', lambda *a, **k: _offset(dra, ddec))
    monkeypatch.setattr(vc, 'measure_offset_grid',
                        lambda *a, **k: dict(clean=True, n_total=4,
                                             worst_off_mas=rm['worst_off_mas'],
                                             cells=[]))
    monkeypatch.setattr(vc, 'same_star_region_map', lambda *a, **k: rm)
    monkeypatch.setattr(vc, 'local_residual_map',
                        lambda *a, **k: dict(cells=[dict(
                            dra_mas=dra, ddec_mas=ddec, n=4000,
                            dra_sem=0.8, ddec_sem=0.8)], n_pairs=4000))
    c = _coords()
    r = measure_reference_tie(c, c, c, dense=True, context='test')
    assert r['region_map']['worst_sig_off_mas'] == pytest.approx(
        REGION_SMALL_BULK_WORST_SIG_CAP_MAS - 5.0)
    assert r['per_tile_small_bulk_exempt'] is True
    assert r['apply_ok'] is True


# ---------------------------------------------------------------------------
# the constants
# ---------------------------------------------------------------------------

def test_the_thresholds_clear_the_named_treasury_tiles():
    """o075/o080/o084/o087, measured against the live gc-treasury
    checkpoints (2026-09-25): same-star bulk sigma 1.61-1.91 mas and 1-2 of
    26-28 cells (3.6-7.1%) flagged, both inside their bars. This only checks
    the sigma and flagged-fraction bars; see
    test_the_o084_treasury_tile_no_longer_demotes_under_the_cap above for
    o084's worst_sig_off_mas (57.35 mas), which is over the separate
    REGION_SMALL_BULK_WORST_SIG_CAP_MAS and keeps it blocking regardless."""
    measured = {'o075': (1.77, 1 / 26), 'o080': (1.61, 1 / 27),
                'o084': (1.61, 1 / 28), 'o087': (1.74, 1 / 27)}
    for key, (sigma, frac) in measured.items():
        assert sigma <= REGION_SMALL_BULK_SIGMA_TOL_MAS, key
        assert frac <= REGION_SMALL_BULK_MAX_FLAGGED_FRACTION, key


def test_the_worst_sig_cap_clears_the_non_outlier_treasury_tiles():
    """o075/o080/o087's worst_sig_off_mas (22.98/25.84/23.54 mas, live
    checkpoints 2026-09-25) sit well under REGION_SMALL_BULK_WORST_SIG_CAP_MAS
    (50.0); o084's (57.35 mas) does not -- see
    test_the_o084_treasury_tile_no_longer_demotes_under_the_cap."""
    clears = {'o075': 22.98, 'o080': 25.84, 'o087': 23.54}
    for key, worst_sig in clears.items():
        assert worst_sig <= REGION_SMALL_BULK_WORST_SIG_CAP_MAS, key
    assert 57.35 > REGION_SMALL_BULK_WORST_SIG_CAP_MAS, (
        'o084 must be the reported exception, not silently under the cap too')


def test_the_flagged_fraction_bar_is_a_minority_of_cells():
    assert 0.0 < REGION_SMALL_BULK_MAX_FLAGGED_FRACTION <= 0.25
