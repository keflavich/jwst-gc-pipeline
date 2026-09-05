"""Rejecting an UNSWEPT edge-riding SPARSE leg changes ``apply_ok`` (#600).

``measure_reference_tie`` measures the consensus against the dense VIRAC2
reference (leg A) and against the sparse Gaia-only subset (leg B), and lets a
GROSS split between them block::

    sparse_untrustworthy = bool(res_b is None or res_b.get("ok") is False
                                or res_b.get("alias_rejected") is True
                                or res_b.get("window_consistent") is False)
    cross_gross_ok = bool((not np.isfinite(sep_mas)) or sparse_untrustworthy
                          or sep_mas <= gross_tol_mas)
    apply_ok = bool(res_a is not None and res_a.get("ok")
                    and per_tile_ok and cross_gross_ok)

so widening the confirmation trigger to an unswept edge-riding peak reaches
leg B as well, and where leg B was a chance bin the gate it was powering is
released: ``apply_ok`` flips False -> True.  That is the documented valve --
CLAUDE.md's GC rule is that a Gaia-sparse tie "must never BLOCK a coherent VIRAC
tie" -- but it is a live behaviour change, not a diagnostic one, so it is pinned
here.

Six ``_latest`` records were in that state when this was written (2026-09-04),
five of them sub-threshold at the checkpoint's ``REFERENCE_APPLY_MIN_MAS`` and
one not:

    sgrc m2-m6 F182M o012   leg A 11.0-12.7 mas, adopted bulk 0.34-1.15 mas,
                            leg B 937-2574 mas on n_peak 6-7 at edge 0.31-0.86
    gc2211_o028 m2 F150W    leg A 51.2 mas (contrast 18.9, n_peak 914),
                            adopted bulk 43.88 mas, leg B 1147.6 mas on
                            n_peak 5 at edge 0.383

Every one of those sparse peaks MOVES with the window in its own recorded
``windows`` list (gc2211_o028: 1147.6 -> 5072 -> 27250 -> 49556 mas at
3/10/30/60"), so they are rejected on a measurement.  The natural experiment is
in the same field: gc2211_o028 F150W m3-m7 already read ``apply_ok: true`` on
the SAME ~51 mas tie because their sparse leg happened to land at a swept window
and was already disqualified; only m2's stayed inside the first 3" window.

What the released backstop leaves uncovered is at the bottom of this file
(PR #757 review, B4): ``per_tile['clean']`` is a CONTRAST statement, so a
half-mosaic seam still reaches ``apply_ok`` once an untrustworthy sparse leg has
switched the gross cross-reference check off.  The mechanism is asserted; the
verdict that should follow is an ``xfail(strict=True)``.

The fixtures are 6000 consensus sources over 60" against a 1000-source dense
reference and a 700-source sparse one, which reproduces the recorded regime:
~33k pairs at the 3" window over 90k bins, median occupied bin 1, so ``contrast``
IS the peak bin count and a chance bin of 5-8 clears the floor while the arg-max
lands far out (the area goes as r).  Each ``measure_reference_tie`` call here
costs ~2.5 min -- there is no cheaper way to exercise this decision on real
estimator output.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import CONFIRM_EDGE_FRACTION
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    REFERENCE_APPLY_MIN_MAS)
from jwst_gc_pipeline.photometry.visit_consensus import (
    REFERENCE_CROSSCHECK_GROSS_MAS, measure_reference_tie)

_W = 60.0
_RA0, _DEC0 = 266.4, -28.9
_COSD = float(np.cos(np.radians(_DEC0)))
#: the collapse band -- both CONFIRM_WINDOW_FACTORS fall under a single
#: ``base_w * 1.25`` floor below 1.25 / 2.2
_COLLAPSE_BAND_MAX = 1.25 / 2.2


def _field(seed, n, width=_W):
    rng = np.random.RandomState(seed)
    return SkyCoord((_RA0 + (rng.rand(n) - 0.5) * width / 3600.0 / _COSD) * u.deg,
                    (_DEC0 + (rng.rand(n) - 0.5) * width / 3600.0) * u.deg)


def _shift(coords, dra_arcsec, ddec_arcsec):
    return SkyCoord((coords.ra.deg + dra_arcsec / 3600.0 / _COSD) * u.deg,
                    (coords.dec.deg + ddec_arcsec / 3600.0) * u.deg)


def _consensus():
    return _field(1, 6000)


@pytest.fixture(scope="module")
def flip_case():
    """A clean 15 mas dense tie whose SPARSE leg is an unswept chance peak --
    the sgrc / gc2211_o028 shape.  grid 2x2 rather than 6x6 only to keep the
    per-tile check affordable at this source count; it is the DENSE per-tile
    gate either way."""
    cons = _consensus()
    ref_all = _shift(cons[::6], 0.012, -0.009)
    sparse = _field(402, 700)
    return measure_reference_tie(cons, ref_all, sparse, dense=True,
                                 grid_nx=2, grid_ny=2, context="flip")


@pytest.fixture(scope="module")
def coherent_sparse_case():
    """The sparse leg is a REAL 2.5" offset measured on 700 of the same stars
    -- the cloudc F410M shape.  It is unswept and edge-riding too, so the
    widened trigger probes it; it REPRODUCES, so nothing is released."""
    cons = _consensus()
    ref_all = _shift(cons[::6], 0.012, -0.009)
    sparse = _shift(cons[::9][:700], 2.0, -1.5)
    return measure_reference_tie(cons, ref_all, sparse, dense=True,
                                 grid_nx=2, grid_ny=2, context="coherent-sparse")


@pytest.fixture(scope="module")
def real_misregistration_case():
    """The consensus is rigidly 1.25" off the DENSE reference.  At the 3" window
    that reads edge fraction 0.42 -- inside the band the widened trigger newly
    probes -- so the probe now runs on a real misregistration."""
    cons = _consensus()
    ref_all = _shift(cons[::6], 1.20, -0.35)
    sparse = _field(402, 700)
    return measure_reference_tie(cons, ref_all, sparse, dense=True,
                                 grid_nx=2, grid_ny=2, context="real-misreg")


@pytest.fixture(scope="module")
def seam_case():
    """The corner the PR #757 review named (B4): the released gross-Gaia
    backstop, with a DENSE leg that is coherent and a field that is not rigidly
    offset.

    Half the reference is at the clean 15 mas tie and half is displaced 2.5" --
    the brick-1182 v001 half-mosaic class the gross cross-reference check is
    credited with catching.  The sparse leg is the same uncorrelated chance peak
    as ``flip_case``, so it is alias-rejected and the backstop is switched off.
    """
    cons = _consensus()
    sub = cons[::6]
    top = sub.dec.deg > _DEC0
    ra = sub.ra.deg.copy()
    dec = sub.dec.deg.copy()
    ra[~top] += 0.012 / 3600.0 / _COSD
    dec[~top] += -0.009 / 3600.0            # clean half: the 15 mas tie
    ra[top] += 2.000 / 3600.0 / _COSD
    dec[top] += -1.500 / 3600.0             # displaced half: 2.5"
    ref_all = SkyCoord(ra * u.deg, dec * u.deg)
    sparse = _field(402, 700)
    return measure_reference_tie(cons, ref_all, sparse, dense=True,
                                 grid_nx=2, grid_ny=2, context="seam")


def test_the_sparse_leg_is_the_newly_probed_shape(flip_case):
    """It has to be the case the trigger newly reaches: UNSWEPT (so the old
    ``best["swept"]`` condition skipped it) and riding its first window's edge.
    It also sits in the probe-window collapse band, where the old single floor
    put both probes at one window."""
    vs = flip_case["vs_sparse"]
    assert vs is not None and not vs["swept"], vs
    assert vs["window_arcsec"] == 3.0, vs
    assert CONFIRM_EDGE_FRACTION <= vs["window_edge_fraction"] <= _COLLAPSE_BAND_MAX, vs
    assert vs["n_peak"] < 20, vs            # live records: 5-7
    probes = vs["window_confirmation"]["probes"]
    assert sorted(round(p["window_arcsec"], 6) for p in probes) == [3.75, 6.0], probes


def test_the_sparse_leg_is_rejected_and_that_releases_the_dense_tie(flip_case):
    """The consequence the PR body has to state: rejecting leg B sets
    ``sparse_untrustworthy``, which disables the gross cross-reference gate, and
    ``apply_ok`` flips to True on a coherent dense tie."""
    vs = flip_case["vs_sparse"]
    assert vs["window_consistent"] is False and vs["alias_rejected"], vs
    assert not vs["ok"], vs
    assert flip_case["cross_reference_sparse_untrustworthy"] is True, flip_case
    assert flip_case["cross_reference_gross_ok"] is True, flip_case
    assert flip_case["per_tile_ok"] is True, flip_case
    assert flip_case["apply_ok"] is True, flip_case
    # ...on the dense tie's own value, not the sparse peak's
    assert abs(flip_case["dra_mas"] - 12.0) < 2.0, flip_case
    assert abs(flip_case["ddec_mas"] + 9.0) < 2.0, flip_case


def test_without_the_rejection_that_leg_would_have_blocked(flip_case):
    """What the flip is worth: the cross-reference split is GROSS, so before the
    rejection this row was ``gross_ok=False`` -> ``apply_ok=False`` -- a 15 mas
    dense tie with a clean per-tile map withheld on a 8-pair sparse chance bin.
    """
    sep = flip_case["cross_reference"]["sep_mas"]
    assert np.isfinite(sep) and sep > REFERENCE_CROSSCHECK_GROSS_MAS, flip_case
    assert flip_case["vs_full"]["ok"] and not flip_case["vs_full"]["swept"], flip_case
    assert (flip_case["vs_full"]["window_edge_fraction"]
            < CONFIRM_EDGE_FRACTION), flip_case


def test_a_coherent_sparse_disagreement_still_blocks(coherent_sparse_case):
    """The gate that must survive.  A sparse leg that is a REAL offset -- same
    stars, high contrast -- is probed by the widened trigger and REPRODUCES, so
    it is not rejected, ``sparse_untrustworthy`` stays False, and the gross
    split still sets ``apply_ok`` False.  Only a chance peak is released.
    """
    vs = coherent_sparse_case["vs_sparse"]
    assert vs is not None and vs["ok"], vs
    assert vs["window_edge_fraction"] >= CONFIRM_EDGE_FRACTION, vs
    assert vs["alias_rejected"] is False, vs
    assert vs["window_consistent"] is True, vs
    assert coherent_sparse_case["cross_reference_sparse_untrustworthy"] is False, \
        coherent_sparse_case
    sep = coherent_sparse_case["cross_reference"]["sep_mas"]
    assert np.isfinite(sep) and sep > REFERENCE_CROSSCHECK_GROSS_MAS, sep
    assert coherent_sparse_case["cross_reference_gross_ok"] is False, \
        coherent_sparse_case
    assert coherent_sparse_case["apply_ok"] is False, coherent_sparse_case


def test_a_real_rigid_misregistration_still_reaches_the_apply_threshold(
        real_misregistration_case):
    """The other gate that must survive.  A genuine rigid 1.25" consensus ->
    reference misregistration reads edge fraction ~0.42 at the 3" window, so it
    is now probed -- and it reproduces at both probes, keeps ``ok``, and is
    reported at its true magnitude, well over ``REFERENCE_APPLY_MIN_MAS``.  A
    rejection here would have made the checkpoint blind to it.
    """
    vf = real_misregistration_case["vs_full"]
    assert vf is not None and not vf["swept"], vf
    assert vf["window_edge_fraction"] >= CONFIRM_EDGE_FRACTION, vf
    assert vf["window_consistent"] is True, vf
    assert vf["alias_rejected"] is False and vf["ok"], vf
    probes = vf["window_confirmation"]["probes"]
    assert all(p["agrees"] for p in probes if p["dra"] is not None), probes
    assert real_misregistration_case["off_mas"] > REFERENCE_APPLY_MIN_MAS
    assert abs(real_misregistration_case["dra_mas"] - 1200.0) < 60.0, \
        real_misregistration_case
    assert abs(real_misregistration_case["ddec_mas"] + 350.0) < 60.0, \
        real_misregistration_case
    assert real_misregistration_case["apply_ok"] is True, real_misregistration_case


# ---------------------------------------------------------------------------
# B4 of the PR #757 review: what covers the gate that the sparse-untrustworthy
# valve switches OFF.
#
# The review asked for either a test asserting ``apply_ok`` False in that corner
# or a written statement that per-tile + same-star is the accepted cover.
# Measured, the statement would be wrong.  ``measure_offset_grid`` is called
# with ``max_off_mas=None`` by every production caller (issue #392), so
# ``per_tile['clean']`` asks only "does each tile have a coherent peak?", never
# "is that peak where the bulk says it should be" -- its own docstring says so.
# A half-mosaic seam gives every tile a razor-sharp peak, so the map reads
# clean while two of its four cells sit 2.5" away.
#
# The other half of the brick-1182 v001 signature does not need the backstop: a
# window-LIMITED dense peak (2.7" measured at the 3" window against a true ~20"
# offset) rides edge 0.9, so #600's widened trigger probes it and it does not
# reproduce at a window that can hold the true offset -- alongside the sweep,
# which escalates to that window in the first place.  The uncovered residue is a
# dense peak that is coherent and DOES reproduce while the field is not rigidly
# offset.
_DISPLACED_ARCSEC = 2.5


def test_a_seam_reads_per_tile_clean_while_two_cells_sit_2500_mas_out(seam_case):
    """The mechanism, asserted on real estimator output: ``per_tile_ok`` is a
    statement about CONTRAST alone, so it does not cover a seam.

    Each 2x2 cell finds a sharp peak -- 15 mas on the clean half, 2500 mas on
    the displaced half -- so every cell is ``ok`` and ``clean`` is True with a
    ``worst_off_mas`` of 2.5".  That is the whole of what ``per_tile_ok``
    contributes to ``apply_ok`` here.
    """
    pt = seam_case["per_tile"]
    assert pt["clean"] is True, pt
    assert seam_case["per_tile_ok"] is True, seam_case
    assert pt["n_ok"] == pt["n_total"] == 4, pt
    offs = sorted(c["off_mas"] for c in pt["cells"])
    assert all(abs(o - 15.0) < 3.0 for o in offs[:2]), offs
    assert all(abs(o - _DISPLACED_ARCSEC * 1000.0) < 50.0 for o in offs[2:]), offs
    assert pt["worst_off_mas"] > 2000.0, pt
    assert pt["min_contrast_seen"] > 100.0, pt      # every cell is razor-sharp
    # ...and the backstop that would have seen it is switched off, by the same
    # route #751 opened: an alias-rejected sparse chance peak.
    assert seam_case["vs_sparse"]["alias_rejected"] is True, seam_case
    assert seam_case["cross_reference_sparse_untrustworthy"] is True, seam_case
    sep = seam_case["cross_reference"]["sep_mas"]
    assert np.isfinite(sep) and sep > REFERENCE_CROSSCHECK_GROSS_MAS, sep
    assert seam_case["cross_reference_gross_ok"] is True, seam_case


@pytest.mark.xfail(strict=True, reason="known gap, issue #775 (see also #392): "
                   "per_tile['clean'] is contrast-only, so nothing left in "
                   "apply_ok sees a half-mosaic seam once the gross-Gaia "
                   "backstop is released by an untrustworthy sparse leg")
def test_a_half_displaced_field_should_not_reach_apply_ok(seam_case):
    """The desired end state, marked xfail(strict) so that closing the gap makes
    this file fail until the marker is removed.

    Today the run signs off: ``apply_ok`` True on a field where half the mosaic
    is 2.5" out, and the adopted bulk is the DISPLACED half's peak rather than
    the clean half's tie.  Nothing here weakens a gate -- the assertion is the
    gate that does not exist yet.
    """
    assert seam_case["apply_ok"] is False, seam_case


def test_the_seam_is_recorded_even_though_it_does_not_block(seam_case):
    """What a reader of the record CAN see today, so the gap is diagnosable
    from a written checkpoint without re-running the estimator: the per-tile map
    carries the seam in ``worst_off_mas`` and in its cells, and the adopted bulk
    disagrees with half of them by the full displacement.  This asserts the
    record's CONTENT, not the verdict -- the verdict is the xfail above.
    """
    bulk = seam_case["off_mas"]
    spread = (seam_case["per_tile"]["worst_off_mas"]
              - min(c["off_mas"] for c in seam_case["per_tile"]["cells"]))
    assert spread > 2000.0, seam_case["per_tile"]
    # the bulk landed on the displaced half, not on the 15 mas tie
    assert abs(bulk - _DISPLACED_ARCSEC * 1000.0) < 100.0, bulk
