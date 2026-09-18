"""A dirty region map must not veto a bulk that dominates it (#921).

``per_tile_ok`` judges each region cell against a FIXED 15 mas at 3 sigma.  That
is the right question for "is this field flat" and the wrong one for "should
this rigid shift be applied", because it does not know how big the shift is.

A rigid shift by a correctly measured bulk moves every cell by the same vector:
the spatial structure is untouched and the bulk comes out from under it, so the
per-cell error goes from ``hypot(bulk, structure)`` to ``structure``.  Refusing
cannot make the field flatter -- it only keeps the bulk.  The veto earns its
keep only when the "bulk" might not be one, which is the SEAM case, and a seam
announces itself by a worst cell comparable to or larger than the bulk it
produces.

gc-treasury tiles, measured (#921): ratios 0.29-0.41 on bulks of 60-74 mas that
are real, rigid and reference-independent -- Gaia-only reads 65.8 mas where
dense VIRAC2 reads 70.9.  Every tile that DID receive its bulk went from 56-86
mas to 8-12.  o138 is the natural experiment: one tile, one visit, one night,
split by filter -- F212N got a bulk row and reads 10.7 mas, F480M got none and
reads 73.8.

## How these drive the decision

``measure_reference_tie`` calls four estimators (``measure_offset`` twice,
``measure_offset_grid``, ``same_star_region_map``) before it reaches the verdict
this PR changes, and those calls dominate its runtime.  The estimators are
stubbed with the RECORDED shapes from the fields in question; everything from
``per_tile_ok`` through ``apply_ok`` is the shipping code path, as in
``test_checkpoint_fails_when_gate_did_not_run``.  Stubbing the estimators also
lets the fixtures carry the real o135 numbers rather than a synthetic
approximation of them.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry import visit_consensus as vc
from jwst_gc_pipeline.photometry.visit_consensus import (
    REGION_BULK_DOMINANCE_RATIO, REGION_BULK_DOMINANCE_MIN_BULK_MAS,
    measure_reference_tie)

_RA0, _DEC0 = 266.4, -28.9


def _coords(n=200):
    rng = np.random.RandomState(7)
    return SkyCoord((_RA0 + (rng.rand(n) - 0.5) / 60.0) * u.deg,
                    (_DEC0 + (rng.rand(n) - 0.5) / 60.0) * u.deg)


def _offset(dra_mas, ddec_mas, ok=True, swept=False):
    """A `measure_offset` result in arcsec, as the real one returns."""
    return dict(dra=dra_mas / 1000.0, ddec=ddec_mas / 1000.0,
                off=float(np.hypot(dra_mas, ddec_mas)) / 1000.0,
                ok=ok, swept=swept, alias_rejected=False,
                window_consistent=True, window_edge_fraction=0.01,
                contrast=60.0, n_peak=500, npairs=100000,
                window_arcsec=3.0, dra_err=0.0008, ddec_err=0.0008)


def _region_map(worst_mas, clean, n_flagged, n_uncovered=0):
    return dict(clean=clean, measurable=True, n_cells=25, n_measured=25,
                n_flagged=n_flagged, n_uncovered=n_uncovered,
                uncovered_cells=[{'by': 'tight-pairs'}] * n_uncovered,
                worst_off_mas=worst_mas, worst_sig_off_mas=worst_mas,
                n_pairs=4000, cells=[],
                reason=f'{n_flagged} region cell(s) above 15.0 mas at 3.0-sigma')


def _tie(monkeypatch, bulk_mas, worst_mas, clean=False, n_flagged=6,
         region_measurable=True, grid_clean=True, n_uncovered=0):
    """Run the REAL decision over canned estimator output."""
    dra, ddec = bulk_mas
    monkeypatch.setattr(vc, 'measure_offset',
                        lambda *a, **k: _offset(dra, ddec))
    monkeypatch.setattr(vc, 'measure_offset_grid',
                        lambda *a, **k: dict(clean=grid_clean, n_total=4,
                                             worst_off_mas=worst_mas, cells=[]))
    if region_measurable:
        rm = _region_map(worst_mas, clean, n_flagged, n_uncovered)
    else:
        rm = dict(clean=False, measurable=False, n_cells=0, n_measured=0,
                  n_flagged=0, worst_off_mas=float('nan'), cells=[])
    monkeypatch.setattr(vc, 'same_star_region_map', lambda *a, **k: rm)
    # the same-star bulk refinement; returning it is what makes `same_star` and
    # therefore the region-map branch live
    monkeypatch.setattr(vc, 'local_residual_map',
                        lambda *a, **k: dict(cells=[dict(
                            dra_mas=dra, ddec_mas=ddec, n=4000,
                            dra_sem=0.8, ddec_sem=0.8)], n_pairs=4000))
    c = _coords()
    return measure_reference_tie(c, c, c, dense=True, context='test')


# ---------------------------------------------------------------------------
# the two regimes
# ---------------------------------------------------------------------------

def test_a_bulk_that_dominates_its_structure_is_applied(monkeypatch):
    """The #921 verdict change, at o135 F212N's recorded numbers: 62.45 mas
    bulk, worst cell 25.29 mas, region map dirty at 6 of 25 cells."""
    r = _tie(monkeypatch, (-61.66, -9.91), 25.29)
    assert r['per_tile_source'] == 'same-star-region'
    assert r['per_tile_ok'] is False, 'fixture must be a DIRTY map to be a test'
    # the fixture reproduces o135 F212N's magnitude; the adopted bulk goes
    # through the same-star refinement, so pin the RATIO against the bulk the
    # code actually adopted rather than against arithmetic on the input
    adopted = float(np.hypot(r['dra_mas'], r['ddec_mas']))
    assert adopted == pytest.approx(62.45, rel=2e-3)
    assert r['region_bulk_dominance_ratio'] == pytest.approx(25.29 / adopted,
                                                             rel=1e-9)
    assert r['per_tile_bulk_dominant_exempt'] is True
    assert r['apply_ok'] is True, (
        'a bulk 2.5x its own worst cell was refused; applying it cannot make '
        'the field less flat, it only removes the bulk')


def test_a_seam_is_still_refused(monkeypatch):
    """The case the veto exists for.  brick-1182 F200W's ~90 mas strip sits on a
    sub-mas bulk; applying that bulk would move the undisplaced part wrong."""
    r = _tie(monkeypatch, (-0.2, 0.1), 90.0)
    assert r['per_tile_bulk_dominant_exempt'] is False
    assert r['apply_ok'] is False


def test_a_half_mosaic_seam_is_still_refused(monkeypatch):
    """Half a field displaced by D reads bulk ~D/2 against a worst cell ~D/2 --
    ratio ~1, twice the bar.  This is the shape
    test_reference_tie_unswept_sparse_alias's synthetic half-mosaic has."""
    r = _tie(monkeypatch, (-300.0, 0.0), 300.0)
    assert r['region_bulk_dominance_ratio'] == pytest.approx(1.0, rel=1e-3)
    assert r['per_tile_bulk_dominant_exempt'] is False
    assert r['apply_ok'] is False


# ---------------------------------------------------------------------------
# scope of the exemption
# ---------------------------------------------------------------------------

def test_a_clean_map_does_not_need_the_exemption(monkeypatch):
    """Unchanged behaviour: a clean map passes on `per_tile_ok`, and the
    exemption stays False so the record does not claim it was used."""
    r = _tie(monkeypatch, (-61.66, -9.91), 5.0, clean=True, n_flagged=0)
    assert r['per_tile_ok'] is True
    assert r['per_tile_bulk_dominant_exempt'] is False
    assert r['apply_ok'] is True


def test_a_coverage_only_dirty_map_is_still_refused(monkeypatch):
    """The exemption may overrule the RESIDUAL arm and only that one.

    ``clean`` is ``not flagged and not uncovered``, and the two arms catch
    opposite failures.  A region displaced BEYOND the match radius keeps its
    sources, loses its pairs, and contributes no cell at all -- it lands in
    ``n_uncovered`` and never reaches ``worst_off_mas``, which is a max over
    MEASURED cells.  So its signature is a SMALL numerator, which is exactly
    what this exemption looks for, and brick-1182 v001's ~20" half-mosaic would
    have sailed through on a ratio near zero (PR #924 review).

    The ratio is still RECORDED here -- a refusal should say how the residuals
    compared to the bulk even when coverage was what blocked.
    """
    r = _tie(monkeypatch, (-61.66, -9.91), 4.0, n_flagged=0, n_uncovered=3)
    assert r['per_tile_ok'] is False
    assert r['region_bulk_dominance_ratio'] < REGION_BULK_DOMINANCE_RATIO, (
        'fixture must look bulk-dominant on the residual arm to be a test')
    assert r['per_tile_bulk_dominant_exempt'] is False, (
        'a coverage-dirty map was exempted on a residual-arm ratio'
    )
    assert r['apply_ok'] is False


def test_a_map_dirty_on_BOTH_arms_is_still_refused(monkeypatch):
    """Flagged cells within the bound do not buy a pass while coverage is also
    dirty -- the arms are independent, so clearing one is not clearing both."""
    r = _tie(monkeypatch, (-61.66, -9.91), 25.29, n_flagged=6, n_uncovered=2)
    assert r['per_tile_bulk_dominant_exempt'] is False
    assert r['apply_ok'] is False


def test_the_exemption_does_not_reach_the_histogram_grid(monkeypatch):
    """The grid's worst cell can be a swept per-tile NOISE peak -- `clean`
    cannot tell a noise cell from a seam cell (issues #392/#775) -- so a ratio
    built on it would exempt on noise.  When the region map is unmeasurable and
    the grid decides, a dirty grid keeps vetoing exactly as before."""
    r = _tie(monkeypatch, (-61.66, -9.91), 25.29,
             region_measurable=False, grid_clean=False)
    assert r['per_tile_source'] == 'histogram-grid'
    assert r['region_bulk_dominance_ratio'] is None
    assert r['per_tile_bulk_dominant_exempt'] is False
    assert r['apply_ok'] is False


def test_a_small_bulk_does_not_engage_the_exemption(monkeypatch):
    """The ratio's denominator is the bulk, so it is unstable when the bulk is
    small -- and a sub-floor bulk is one nothing would have applied anyway."""
    r = _tie(monkeypatch, (-3.0, 1.0), 1.0)
    assert r['region_bulk_dominance_ratio'] is None
    assert r['per_tile_bulk_dominant_exempt'] is False


def test_the_ratio_is_recorded_even_when_it_does_not_exempt(monkeypatch):
    """A refusal has to be readable as "structure comparable to the bulk"
    rather than only as "a cell exceeded 15 mas" -- the distinction #921 could
    not make from the records on disk."""
    r = _tie(monkeypatch, (-300.0, 0.0), 300.0)
    assert r['region_bulk_dominance_ratio'] is not None
    assert r['region_bulk_dominance_tol'] == REGION_BULK_DOMINANCE_RATIO


# ---------------------------------------------------------------------------
# the constants, against the fields they were chosen for
# ---------------------------------------------------------------------------

def test_the_floor_sits_well_above_the_apply_floor():
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        REFERENCE_APPLY_MIN_MAS)
    assert REGION_BULK_DOMINANCE_MIN_BULK_MAS >= 5 * REFERENCE_APPLY_MIN_MAS


def test_the_measured_treasury_ratios_fall_inside_the_bar():
    """The six tile-filter pairs #921 measured.  Lowering the constant below
    these stops the fix working on the fields it was written for; the margin is
    the point."""
    measured = {('o135', 'F212N'): (25.29, 62.45),
                ('o135', 'F480M'): (23.62, 67.91),
                ('o130', 'F212N'): (23.40, 74.40),
                ('o127', 'F212N'): (20.42, 61.15),
                ('o129', 'F212N'): (20.14, 59.70),
                ('o132', 'F212N'): (20.20, 69.90)}
    for key, (worst, bulk) in measured.items():
        assert worst / bulk <= REGION_BULK_DOMINANCE_RATIO, f'{key}'
        assert bulk >= REGION_BULK_DOMINANCE_MIN_BULK_MAS, f'{key}'


def test_seam_signatures_stay_far_outside_the_bar():
    """brick-1182 F200W's ~90 mas strip on a sub-mas bulk, and the 2500 mas
    synthetic half-mosaic -- both hundreds, so no threshold near 0.5 admits
    them."""
    for worst, bulk in ((90.0, 0.2), (2500.0, 0.5)):
        assert worst / bulk > 10 * REGION_BULK_DOMINANCE_RATIO
