"""The frozen-stage (m3+) reference-tie gate must compare the two stages on the
SAME stars.

Issue #285 restricted the later stage's consensus to m2's star LIST but left the
baseline it is differenced against as the number m2 measured over m2's FULL set.
One-sided restriction manufactures a shift out of two correct measurements: the
stars a later stage cannot re-detect drag the BASELINE and nothing else, and the
gate reports the difference as "the solution moved after it was frozen".

Measured on sickle F335M m2 vs m4 (2026-08-10):

    m2 ALL            ddec -0.80   n=2958
    m4 ALL            ddec +1.43   n=2813    <- gate called this 2.23 mas of movement
    m2 SHARED         ddec +1.50   n=2819
    m4 SHARED         ddec +1.42   n=2819    <- same stars: 0.08 mas
    m2 DROPOUTS only  ddec -18.58  n=139

The same field blocked twice in the 2026-08 campaign: m3 F187N at 2.34 mas and
m5 F335M at 2.23 mas, both on a solution nothing had touched since m2.

These tests drive ``run_visit_checkpoint`` with a reference tie whose value
DEPENDS on which stars are handed to it, which is the only way the asymmetry can
show up at all -- a fake that returns a constant is blind to it.
"""
import json
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import astrometry_checkpoint as _ac
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AstrometryRegressionError, run_visit_checkpoint)
from .test_visit_consensus import RA0, DEC0, _exposure_table, _field

_DUMMY_REFCAT = dict(all=None, sparse=None, mag=None, dense=True)

#: n_dropouts of these carry the drop-out bias; the rest are the shared stars.
N_M2_STARS = 400
N_DROPOUTS = 40


def _tiny_visit_table():
    ra, dec = _field(n=5)
    return _exposure_table(ra, dec, exposure=1)


def _m2_star_grid(n=N_M2_STARS):
    """n stars on one RA row, spaced 1" apart -- far enough that each pairs
    uniquely inside SURVIVOR_MATCH_TOL_MAS (200 mas)."""
    ra = RA0 + np.arange(n) / 3600.0
    dec = np.full(n, DEC0)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _write_m2_consensus_catalog(basepath, coords, filt="F212N", refmag=None):
    """Write the pooled m2 consensus catalog where ``consensus_path`` looks."""
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_path
    path = consensus_path(basepath, filt, obs_token="")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tbl = Table()
    tbl["skycoord"] = coords
    tbl["refmag"] = (np.full(len(coords), 16.0) if refmag is None else refmag)
    tbl.write(path, overwrite=True)
    return path


def _write_m2_record(record_dir, dra, ddec, consensus_catalog,
                     visit="001", filt="F212N"):
    """m2's record: the tie it measured over its FULL star set, and the path to
    the catalog those stars are in."""
    rec = dict(consensus_catalog=consensus_catalog,
               visits=[dict(visit=visit, reference_tie=dict(
                   apply_ok=True, dra_mas=dra, ddec_mas=ddec,
                   vs_full=dict(dra=dra, ddec=ddec)))])
    with open(os.path.join(record_dir, f"checkpoint_m2_{filt}_latest.json"),
              "w") as fh:
        json.dump(rec, fh)


def _patch_population_dependent_tie(monkeypatch, m2_coords, survivor_mask,
                                    shared_ddec, dropout_ddec, moved_ddec=0.0):
    """Patch the consensus + the tie so BOTH depend on the star set.

    ``build_visit_consensus`` returns the survivors (what a later stage
    re-detects).  ``measure_reference_tie`` returns the mean per-star Dec offset
    of whatever coordinates it is given, so handing it m2's full set, m2's
    survivors, or this stage's consensus gives three different answers -- which
    is the mechanism under test.  ``moved_ddec`` is added for the stage's own
    consensus only, modelling a solution that really did move.
    """
    survivors = m2_coords[survivor_mask]
    per_star = np.where(survivor_mask, shared_ddec, dropout_ddec)
    # RA is unique per star by construction, so it keys the lookup.
    lookup = {round(float(r), 9): float(v)
              for r, v in zip(m2_coords.ra.deg, per_star)}

    def _fake_consensus(tables, context="", **kw):
        return dict(coords=survivors, mag=np.full(len(survivors), 16.0),
                    exposures=[], anchor_key=("001", 1, "nrcb1", "F212N"),
                    scatter_mas=np.array([1.0]), consensus_ok=True, skipped=[])

    def _fake_tie(cons_coords, ref_all, ref_sparse, context="", **kw):
        # A coordinate neither catalog knows (a padded far star in the
        # match-coverage test) reads as a shared star, so padding changes the
        # star COUNT without moving the tie.
        vals = [lookup.get(round(float(r), 9), shared_ddec)
                for r in cons_coords.ra.deg]
        ddec = float(np.mean(vals))
        # The stage's own consensus (not the m2 re-measure) carries any real
        # movement.  The re-measure is labelled by its context.
        if "re-measured on survivors" not in context:
            ddec += moved_ddec
        return dict(off_mas=float(abs(ddec)), apply_ok=True,
                    dra_mas=0.0, ddec_mas=ddec,
                    cross_reference={"agree": True, "sep_mas": 0.0},
                    cross_reference_gross_ok=True, per_tile={"clean": True},
                    swept=False, reference_dense=True,
                    vs_full={"dra": 0.0, "ddec": ddec})

    monkeypatch.setattr(_ac, "build_visit_consensus", _fake_consensus)
    monkeypatch.setattr(_ac, "measure_reference_tie", _fake_tie)


def _sickle_case(tmp_path, monkeypatch, moved_ddec=0.0,
                 n_stars=N_M2_STARS, n_dropouts=N_DROPOUTS):
    """The sickle shape, with the drop-outs' measured -18.58 mas Dec bias: the
    shared stars agree between the stages, m2's FULL-set mean is dragged away
    from them by the drop-outs, and the raw comparison exceeds the 2 mas
    tolerance.  The shared value sits above ``REFERENCE_APPLY_MIN_MAS`` so the
    tie reaches the frozen-stage branch at all."""
    coords = _m2_star_grid(n_stars)
    mask = np.ones(n_stars, dtype=bool)
    mask[:n_dropouts] = False
    shared, dropout = 2.60, -18.58
    m2_full_mean = float(np.mean(np.where(mask, shared, dropout)))
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    _write_m2_record(basepath, 0.0, m2_full_mean, cat)
    _patch_population_dependent_tie(monkeypatch, coords, mask, shared, dropout,
                                    moved_ddec=moved_ddec)
    return basepath, m2_full_mean, shared


def _run(basepath, stage="m3"):
    return run_visit_checkpoint([_tiny_visit_table()], stage,
                                refcat=_DUMMY_REFCAT, filtername="F212N",
                                basepath=basepath, record_dir=basepath,
                                context="test")


def test_population_change_alone_is_not_a_regression(tmp_path, monkeypatch):
    """The sickle case: the raw comparison exceeds tolerance, the same stars
    agree to well under it, and the stage passes."""
    basepath, m2_full_mean, shared = _sickle_case(tmp_path, monkeypatch)
    # The premise: without the fix this comparison is the one that raises.
    assert abs(shared - m2_full_mean) > _ac.STAGE_STABILITY_TOL_MAS
    rec = _run(basepath)
    assert rec["passed"]
    assert rec["failures"] == []


def test_without_the_re_measure_the_same_fixture_raises(tmp_path, monkeypatch):
    """The premise of every test above, pinned: on this fixture nothing moved,
    and the gate raises anyway as soon as the survivor re-measure is taken out.
    Without this, a re-measure that silently stopped running would leave the
    suite green only because the fixture stopped exercising the gate."""
    basepath, _m2_full_mean, _shared = _sickle_case(tmp_path, monkeypatch)
    monkeypatch.setattr(_ac, "_survivor_baseline_tie",
                        lambda *a, **k: (None, dict(reason="disabled for the test",
                                                    n_m2=0, n_stage=0,
                                                    n_survivors=0)))
    with pytest.raises(AstrometryRegressionError):
        _run(basepath)


def test_the_record_carries_both_numbers(tmp_path, monkeypatch):
    """A pass that needed the re-measure must SAY so in the record -- the counts
    and both ties -- or the next reader cannot tell it from a raw pass."""
    basepath, m2_full_mean, shared = _sickle_case(tmp_path, monkeypatch)
    rec = _run(basepath)
    sym = rec["visits"][0]["symmetric_baseline"]
    assert sym is not None
    assert sym["n_survivors"] == N_M2_STARS - N_DROPOUTS
    assert sym["n_m2"] == N_M2_STARS
    # m2 RE-MEASURED on the survivors reads the shared value, not its full-set one.
    assert sym["ddec_mas"] == pytest.approx(shared, abs=1e-6)
    assert sym["delta_mas"] < _ac.STAGE_STABILITY_TOL_MAS


def test_a_raw_pass_does_not_re_measure(tmp_path, monkeypatch):
    """No drop-outs, so the raw comparison already agrees: the re-measure never
    runs and the record says nothing about it.  This is what keeps the fix off
    the path of every passing stage."""
    basepath, _m2_full_mean, _shared = _sickle_case(tmp_path, monkeypatch,
                                                    n_dropouts=0)
    rec = _run(basepath)
    assert rec["passed"]
    assert rec["visits"][0]["symmetric_baseline"] is None


def test_real_movement_still_raises(tmp_path, monkeypatch):
    """Same population change, plus a 5 mas shift the stage's consensus really
    carries.  The same stars now disagree by 5 mas, so it raises."""
    basepath, _m2_full_mean, _shared = _sickle_case(tmp_path, monkeypatch,
                                                    moved_ddec=5.0)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "stars both stages share" in msg, msg
    assert "5.00 mas" in msg, msg


def test_movement_message_reports_the_raw_delta_too(tmp_path, monkeypatch):
    """The raw number is what the operator saw in earlier logs and in the m2
    record; dropping it would make the two irreconcilable."""
    basepath, m2_full_mean, shared = _sickle_case(tmp_path, monkeypatch,
                                                  moved_ddec=5.0)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    raw = abs((shared + 5.0) - m2_full_mean)
    assert f"read {raw:.2f} mas" in str(ex.value), str(ex.value)


def test_no_consensus_catalog_raises_and_says_why(tmp_path, monkeypatch):
    """Without m2's catalog the re-measure cannot run.  That is not evidence the
    solution stayed put, so the stage fails closed -- and names what was
    missing, since 'MOVED 2.23 mas' on its own sent two earlier investigations
    to the wrong cause (issue #368)."""
    coords = _m2_star_grid()
    mask = np.ones(N_M2_STARS, dtype=bool)
    mask[:N_DROPOUTS] = False
    m2_full_mean = float(np.mean(np.where(mask, 2.60, -18.58)))
    basepath = str(tmp_path)
    _write_m2_record(basepath, 0.0, m2_full_mean, None)   # no catalog on disk
    _patch_population_dependent_tie(monkeypatch, coords, mask, 2.60, -18.58)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "was unavailable" in msg, msg
    assert "no m2 consensus catalog" in msg, msg


def test_too_small_a_consensus_raises_and_says_why(tmp_path, monkeypatch):
    """A re-measure on a handful of stars is not a measurement.  Below
    SURVIVOR_MIN_STARS the raw comparison stands and the stage fails closed."""
    n = _ac.SURVIVOR_MIN_STARS + 10
    n_drop = n - (_ac.SURVIVOR_MIN_STARS - 5)      # stage set just under the floor
    basepath, _m2, _sh = _sickle_case(tmp_path, monkeypatch, n_stars=n,
                                      n_dropouts=n_drop)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "was unavailable" in msg, msg
    assert "too few stars to re-measure" in msg, msg


def test_too_few_stars_actually_MATCH_raises_and_says_why(tmp_path, monkeypatch):
    """Both catalogs are big enough, and they still overlap in almost nothing --
    a stage consensus that is mostly a DIFFERENT patch of sky.  The count that
    decides is what MATCHES, not what either side holds; a guard on the two
    input sizes alone passes this and re-measures on 30 stars."""
    coords = _m2_star_grid()
    mask = np.ones(N_M2_STARS, dtype=bool)
    mask[:N_M2_STARS - 30] = False                 # only 30 real survivors
    shared, dropout = 2.60, -18.58
    m2_full_mean = float(np.mean(np.where(mask, shared, dropout)))
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    _write_m2_record(basepath, 0.0, m2_full_mean, cat)
    _patch_population_dependent_tie(monkeypatch, coords, mask, shared, dropout)

    # Pad the stage consensus with stars a degree away, so it clears the size
    # guard while adding nothing that matches.
    survivors = coords[mask]
    far = SkyCoord(ra=(coords.ra.deg[:100] + 1.0) * u.deg,
                   dec=coords.dec.deg[:100] * u.deg, frame="icrs")
    padded = SkyCoord(ra=np.concatenate([survivors.ra.deg, far.ra.deg]) * u.deg,
                      dec=np.concatenate([survivors.dec.deg, far.dec.deg]) * u.deg,
                      frame="icrs")

    def _padded_consensus(tables, context="", **kw):
        return dict(coords=padded, mag=np.full(len(padded), 16.0), exposures=[],
                    anchor_key=("001", 1, "nrcb1", "F212N"),
                    scatter_mas=np.array([1.0]), consensus_ok=True, skipped=[])

    monkeypatch.setattr(_ac, "build_visit_consensus", _padded_consensus)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "was unavailable" in msg, msg
    assert "survive into this stage" in msg, msg
    assert "only 30 of 400" in msg, msg


def test_a_correcting_stage_never_re_measures(tmp_path, monkeypatch):
    """m2 itself has nothing to be frozen against; the whole path is m3+."""
    basepath, _m2_full_mean, _shared = _sickle_case(tmp_path, monkeypatch)
    rec = _run(basepath, stage="m2")
    assert rec["visits"][0]["symmetric_baseline"] is None


def test_survivor_matching_is_per_star_not_positional(tmp_path, monkeypatch):
    """The survivors are found by SKY MATCH, so which m2 rows they are does not
    have to be known -- shuffling the stage's consensus order changes nothing.
    An index-based implementation passes the tests above and fails this one."""
    coords = _m2_star_grid()
    mask = np.ones(N_M2_STARS, dtype=bool)
    mask[:N_DROPOUTS] = False
    shared, dropout = 2.60, -18.58
    m2_full_mean = float(np.mean(np.where(mask, shared, dropout)))
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    _write_m2_record(basepath, 0.0, m2_full_mean, cat)
    _patch_population_dependent_tie(monkeypatch, coords, mask, shared, dropout)

    shuffled = _ac.build_visit_consensus(None)["coords"]
    order = np.random.default_rng(0).permutation(len(shuffled))

    def _shuffled_consensus(tables, context="", **kw):
        return dict(coords=shuffled[order],
                    mag=np.full(len(shuffled), 16.0), exposures=[],
                    anchor_key=("001", 1, "nrcb1", "F212N"),
                    scatter_mas=np.array([1.0]), consensus_ok=True, skipped=[])

    monkeypatch.setattr(_ac, "build_visit_consensus", _shuffled_consensus)
    rec = _run(basepath)
    assert rec["passed"]
    assert rec["visits"][0]["symmetric_baseline"]["n_survivors"] == \
        N_M2_STARS - N_DROPOUTS
