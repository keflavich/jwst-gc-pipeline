"""The frozen-stage (m3+) reference-tie gate must compare the two stages on the
stars they have in common.

Issue #285 restricted the later stage's consensus to m2's star LIST but left the
baseline it is differenced against as the number m2 measured over m2's FULL set.
One-sided restriction manufactures a shift out of two correct measurements: the
stars a later stage cannot re-detect drag the BASELINE and nothing else, and the
gate reports the difference as "the solution moved after it was frozen".

From the live sickle records (2026-08-16), F335M m2 vs m5 -- the case where this
is the whole story:

    m2 over its full 2964 stars    (-0.013, +0.014) mas
    m5 over its      2644 stars    (+0.457, +2.194) mas    raw delta 2.230 -> RAISE
    both over the shared 2642      (-0.013, +1.764) mas -> delta 0.637 -> pass

and F187N m3, where it is NOT -- there the drop-outs are ~2.2 mag BRIGHTER than
the survivors, carry no baseline artefact, and the comparison on shared stars
reads 2.463 mas, slightly worse than the raw 2.342.  The gate is meant to keep
failing that one.

These tests drive ``run_visit_checkpoint`` with a reference tie whose value
depends BOTH on which stars it is handed and on where those stars are -- a fake
returning a constant is blind to the first, and a fake reading only the star
list is blind to the second, which is how a match tolerance of 1e-9 mas once
left the suite green.
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

N_M2_STARS = 400
N_DROPOUTS = 40

#: Per-star offset from the reference, in mas of Dec.  The shared stars agree
#: between the stages; the drop-outs carry the measured sickle F335M drop-out
#: bias.  ``SHARED`` sits above ``REFERENCE_APPLY_MIN_MAS`` so the tie reaches
#: the frozen-stage branch at all.
SHARED_DDEC = 2.60
DROPOUT_DDEC = -18.58

#: How far the stage's stars sit from m2's, per star.  Real per-star fit
#: differences between stages are a few mas and carry no preferred direction;
#: 20 mas with ALTERNATING SIGN is well inside SURVIVOR_MATCH_TOL_MAS, sums to
#: zero so it is not itself a shift, and is far outside "identical
#: coordinates" -- which is what makes the match tolerance testable at all.
POS_JITTER_MAS = 20.0


@pytest.fixture(autouse=True)
def _enforce_at_the_stage(monkeypatch):
    """These tests ask what the frozen-stage comparison MEASURES, and each says
    so with `pytest.raises(AstrometryRegressionError)`.

    #442 moved WHERE the stop happens: `ASTROM_CHECKPOINT_ENFORCE` now defaults
    to `release`, so a frozen-stage failure is recorded with `passed=false` and
    the chain continues, leaving the refusal to the release gate.  The detection
    is unchanged and is the whole subject of this file, so these run under
    `stage` enforcement and keep asserting on the exception -- the same fixture
    `test_astrometry_checkpoint.py` uses for the same reason.

    WHERE the stop happens is pinned in `test_checkpoint_enforcement.py`, not
    here.
    """
    monkeypatch.setenv('ASTROM_CHECKPOINT_ENFORCE', 'stage')


def _zero_mean_jitter(n):
    """+/-POS_JITTER_MAS alternating, in degrees; exactly zero mean for even n."""
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    if n % 2:
        sign[-1] = 0.0
    return POS_JITTER_MAS * sign / 3.6e6


def _tiny_visit_table():
    ra, dec = _field(n=5)
    return _exposure_table(ra, dec, exposure=1)


def _m2_star_grid(n=N_M2_STARS):
    """n stars on one RA row, 1" apart -- far enough to pair uniquely."""
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


def _install_fakes(monkeypatch, m2_coords, per_star_ddec, stage_coords,
                   apply_ok=True, swept=False, bulk_source="same-star",
                   stage_bulk_source=None, remeasure_apply_ok=None,
                   remeasure_swept=None):
    """Patch the consensus and the tie so both depend on the stars AND on where
    they are.

    ``measure_reference_tie`` here returns, for whatever coordinates it is given:

        mean(per-star offset of the nearest m2 star)  +  mean(Dec displacement
        from that m2 star)

    i.e. a population term plus a rigid term, which is the decomposition the
    real estimator makes.  Handing it m2's full set, m2's shared subset, or the
    stage's stars therefore gives three different answers, and moving the stage's
    stars changes the answer independently of which stars they are.
    """
    base_ra = np.asarray(m2_coords.ra.deg)
    base_dec = np.asarray(m2_coords.dec.deg)
    vals = np.asarray(per_star_ddec, dtype=float)

    def _fake_consensus(tables, context="", **kw):
        return dict(coords=stage_coords,
                    mag=np.full(len(stage_coords), 16.0), exposures=[],
                    anchor_key=("001", 1, "nrcb1", "F212N"),
                    scatter_mas=np.array([1.0]), consensus_ok=True, skipped=[])

    def _fake_tie(cons_coords, ref_all, ref_sparse, context="", **kw):
        ra = np.asarray(cons_coords.ra.deg)
        dec = np.asarray(cons_coords.dec.deg)
        idx = np.abs(ra[:, None] - base_ra[None, :]).argmin(axis=1)
        population = float(np.mean(vals[idx]))
        rigid = float(np.mean(dec - base_dec[idx])) * 3.6e6
        ddec = population + rigid
        # RA is measured the same way and is NOT pinned to zero.  With
        # `dra_mas=0` everywhere, `hypot(ddra, dddec)` -> `abs(dddec)` is
        # unobservable -- and on the real sickle F187N case that mutation turns
        # a fail into a pass (dra -1.664, ddec -1.914: hypot 2.536 raises,
        # |ddec| 1.914 does not).
        dra = (float(np.mean(ra - base_ra[idx])) * 3.6e6
               * float(np.cos(np.radians(DEC0))))
        src = bulk_source
        if stage_bulk_source is not None and "stage re-measured" in context:
            src = stage_bulk_source
        # `remeasure_*` apply ONLY to the re-measures on shared stars, so the
        # stage's own tie still reaches the frozen-stage branch.  Setting
        # apply_ok False everywhere sends it to unverified_blocking instead and
        # the branch under test never runs.
        ok, swp = apply_ok, swept
        if "re-measured on shared stars" in context:
            if remeasure_apply_ok is not None:
                ok = remeasure_apply_ok
            if remeasure_swept is not None:
                swp = remeasure_swept
        return dict(off_mas=float(np.hypot(dra, ddec)), apply_ok=ok,
                    dra_mas=dra, ddec_mas=ddec, bulk_source=src,
                    cross_reference={"agree": True, "sep_mas": 0.0},
                    cross_reference_gross_ok=True, per_tile={"clean": True},
                    swept=swp, window_arcsec=3.0, reference_dense=True,
                    vs_full={"dra": dra, "ddec": ddec})

    monkeypatch.setattr(_ac, "build_visit_consensus", _fake_consensus)
    monkeypatch.setattr(_ac, "measure_reference_tie", _fake_tie)


def _sickle_case(tmp_path, monkeypatch, moved_mas=0.0, n_stars=N_M2_STARS,
                 n_dropouts=N_DROPOUTS, write_catalog=True, **fake_kw):
    """The sickle F335M shape: the shared stars agree between the stages, m2's
    FULL-set baseline is dragged away from them by the drop-outs, and the raw
    comparison exceeds the 2 mas tolerance while the shared stars do not."""
    coords = _m2_star_grid(n_stars)
    keep = np.ones(n_stars, dtype=bool)
    keep[:n_dropouts] = False
    per_star = np.where(keep, SHARED_DDEC, DROPOUT_DDEC)
    m2_full_mean = float(np.mean(per_star))
    # The stage re-detects the survivors, at slightly different positions, plus
    # any rigid movement under test.
    surv = coords[keep]
    stage = SkyCoord(
        ra=surv.ra.deg * u.deg,
        dec=(surv.dec.deg + _zero_mean_jitter(len(surv))
             + moved_mas / 3.6e6) * u.deg, frame="icrs")
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords) if write_catalog else None
    _write_m2_record(basepath, 0.0, m2_full_mean, cat)
    _install_fakes(monkeypatch, coords, per_star, stage, **fake_kw)
    return basepath, m2_full_mean, coords, per_star, keep


def _run(basepath, stage="m3"):
    return run_visit_checkpoint([_tiny_visit_table()], stage,
                                refcat=_DUMMY_REFCAT, filtername="F212N",
                                basepath=basepath, record_dir=basepath,
                                context="test")


def _sym(rec):
    return rec["visits"][0]["symmetric_baseline"]


# ---------------------------------------------------------------------------
# the population change alone
# ---------------------------------------------------------------------------

def test_population_change_alone_is_not_a_regression(tmp_path, monkeypatch):
    """The raw comparison exceeds tolerance, the shared stars do not, and the
    stage passes."""
    basepath, m2_full_mean, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    # The premise: without the fix this is the comparison that raises.
    assert abs(SHARED_DDEC - m2_full_mean) > _ac.STAGE_STABILITY_TOL_MAS
    rec = _run(basepath)
    assert rec["passed"]
    assert rec["failures"] == []


def test_without_the_re_measure_the_same_fixture_raises(tmp_path, monkeypatch):
    """The premise of every test here, pinned: nothing moved on this fixture,
    and the gate raises as soon as the re-measure is taken out.  Without this a
    re-measure that silently stopped running would leave the suite green."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    monkeypatch.setattr(_ac, "_survivor_baseline_tie",
                        lambda *a, **k: (None, dict(reason="disabled for the test",
                                                    n_m2=0, n_stage=0,
                                                    n_survivors=0)))
    with pytest.raises(AstrometryRegressionError):
        _run(basepath)


def test_the_record_carries_both_ties_and_the_raw_delta(tmp_path, monkeypatch):
    """A pass that needed the re-measure must SAY so -- both ties, the counts,
    and the raw number -- or the next reader cannot tell it from a raw pass, nor
    reconcile it with the m2 record and the earlier logs."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    rec = _run(basepath)
    sym = _sym(rec)
    assert sym is not None
    assert sym["n_m2"] == N_M2_STARS
    assert sym["n_stage"] == N_M2_STARS - N_DROPOUTS
    assert sym["n_survivors"] == N_M2_STARS - N_DROPOUTS
    # m2 RE-MEASURED on the shared stars reads the shared value, not its
    # full-set one; the stage reads the same plus its positional jitter.
    assert sym["m2_ddec_mas"] == pytest.approx(SHARED_DDEC, abs=1e-6)
    assert sym["stage_ddec_mas"] == pytest.approx(SHARED_DDEC, abs=1e-3)
    assert sym["delta_mas"] == pytest.approx(0.0, abs=1e-3)
    assert sym["raw_delta_mas"] > _ac.STAGE_STABILITY_TOL_MAS
    assert sym["bulk_source"] == "same-star"


def test_a_raw_pass_does_not_re_measure(tmp_path, monkeypatch):
    """No drop-outs, so the raw comparison already agrees: the re-measure never
    runs and the record says nothing about it.  This is what keeps the check off
    the path of every passing stage."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch, n_dropouts=0)
    rec = _run(basepath)
    assert rec["passed"]
    assert _sym(rec) is None


def test_a_correcting_stage_never_re_measures(tmp_path, monkeypatch):
    """m2 has nothing to be frozen against; the whole path is m3+."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    rec = _run(basepath, stage="m2")
    assert _sym(rec) is None


# ---------------------------------------------------------------------------
# real movement
# ---------------------------------------------------------------------------

def test_real_movement_still_raises(tmp_path, monkeypatch):
    """Same population change, plus a 30 mas rigid shift the stage's stars
    really carry.  The shared stars now disagree, so it raises."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch, moved_mas=30.0)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "the two consensi share" in msg, msg
    assert f"{30.0:.2f} mas" in msg, msg


def test_movement_message_reports_the_raw_delta_too(tmp_path, monkeypatch):
    """The raw number is what earlier logs and the m2 record show; dropping it
    would leave the two irreconcilable."""
    basepath, m2_full_mean, _c, _p, _k = _sickle_case(tmp_path, monkeypatch,
                                                      moved_mas=30.0)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    raw = abs((SHARED_DDEC + 30.0) - m2_full_mean)
    assert f"read {raw:.2f} mas" in str(ex.value), str(ex.value)


# ---------------------------------------------------------------------------
# the comparison is symmetric: BOTH sides restricted
# ---------------------------------------------------------------------------

def test_the_stage_side_is_restricted_too(tmp_path, monkeypatch):
    """The stage consensus equals m2's star list only while
    ``_restrict_to_same_stars`` succeeds on every exposure; it legitimately
    refuses, after which the stage set holds stars m2 never had.

    The extras must resolve to the DROP-OUT offset for this to test anything.
    An earlier version placed them at ``ra + 1 degree``, where the fake tie's
    nearest-base lookup mapped every one of them to the LAST base star -- a
    survivor carrying SHARED_DDEC -- so the restricted and unrestricted stage
    ties were identical to 0.0 mas and reverting the restriction passed.
    """
    coords = _m2_star_grid()
    keep = np.ones(N_M2_STARS, dtype=bool)
    keep[:N_DROPOUTS] = False
    per_star = np.where(keep, SHARED_DDEC, DROPOUT_DDEC)
    m2_full_mean = float(np.mean(per_star))

    surv = coords[keep]
    # Extras placed ON the drop-out stars' RA (offset by half their spacing, so
    # they are their own objects and cannot mutually match m2) -- the nearest
    # base star for each is a DROP-OUT, so an unrestricted stage tie is dragged
    # toward DROPOUT_DDEC exactly as a real refusal would drag it.
    extra_ra = coords.ra.deg[:N_DROPOUTS] + 0.5 / 3600.0
    stage = SkyCoord(
        ra=np.concatenate([surv.ra.deg, extra_ra]) * u.deg,
        dec=np.concatenate([surv.dec.deg + _zero_mean_jitter(len(surv)),
                            np.full(len(extra_ra), DEC0)]) * u.deg,
        frame="icrs")

    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    _write_m2_record(basepath, 0.0, m2_full_mean, cat)
    _install_fakes(monkeypatch, coords, per_star, stage)

    # The premise: an UNRESTRICTED stage tie is dragged past tolerance by the
    # extras, so restricting the stage side is what makes this pass.
    unrestricted = ((len(surv) * SHARED_DDEC + len(extra_ra) * DROPOUT_DDEC)
                    / (len(surv) + len(extra_ra)))
    assert abs(unrestricted - SHARED_DDEC) > _ac.STAGE_STABILITY_TOL_MAS, (
        'the fixture must place the extras where they actually move the tie')

    rec = _run(basepath)
    assert rec["passed"], rec["failures"]
    sym = _sym(rec)
    assert sym["n_stage"] == len(stage)
    assert sym["n_survivors"] == N_M2_STARS - N_DROPOUTS
    assert sym["stage_ddec_mas"] == pytest.approx(SHARED_DDEC, abs=1e-3)


def test_a_purely_RA_shift_is_movement_too(tmp_path, monkeypatch):
    """The comparison is a separation, not a declination difference.

    With the fixture's shift confined to Dec, `hypot(ddra, dddec)` and
    `abs(dddec)` are the same function.  On the real sickle F187N case they are
    not: dra -1.664, ddec -1.914, so the separation is 2.536 and raises while
    |ddec| is 1.914 and passes -- the gate this exists to keep failing is one
    uncaught edit from passing.
    """
    coords = _m2_star_grid()
    keep = np.ones(N_M2_STARS, dtype=bool)
    keep[:N_DROPOUTS] = False
    per_star = np.where(keep, SHARED_DDEC, DROPOUT_DDEC)
    m2_full_mean = float(np.mean(per_star))
    surv = coords[keep]
    # 30 mas of RA and nothing in Dec.
    moved_ra_deg = 30.0 / 3.6e6 / float(np.cos(np.radians(DEC0)))
    stage = SkyCoord(
        ra=(surv.ra.deg + moved_ra_deg) * u.deg,
        dec=(surv.dec.deg + _zero_mean_jitter(len(surv))) * u.deg, frame="icrs")
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    _write_m2_record(basepath, 0.0, m2_full_mean, cat)
    _install_fakes(monkeypatch, coords, per_star, stage)

    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "the two consensi share" in msg, msg
    assert "30.0" in msg or "29.9" in msg or "30.1" in msg, msg


def test_survivor_matching_is_by_sky_position_not_row_order(tmp_path, monkeypatch):
    """Shuffling the stage consensus changes nothing.  An index-based
    implementation passes everything else and fails this."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    shuffled = _ac.build_visit_consensus(None)["coords"]
    order = np.random.default_rng(0).permutation(len(shuffled))
    inner = _ac.build_visit_consensus

    def _shuffled_consensus(tables, context="", **kw):
        out = dict(inner(tables, context=context, **kw))
        out["coords"] = shuffled[order]
        return out

    monkeypatch.setattr(_ac, "build_visit_consensus", _shuffled_consensus)
    rec = _run(basepath)
    assert rec["passed"]
    assert _sym(rec)["n_survivors"] == N_M2_STARS - N_DROPOUTS


def test_the_match_tolerance_has_to_be_wide_enough_to_pair(tmp_path, monkeypatch):
    """The stage's stars sit POS_JITTER_MAS from m2's, as real ones do.  A match
    tolerance below that pairs nothing and the comparison refuses -- which is
    what makes the constant testable at the tight end.  With coordinates
    identical on both sides, every tolerance down to 1e-9 mas would pass."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    monkeypatch.setattr(_ac, "SURVIVOR_MATCH_TOL_MAS", POS_JITTER_MAS / 2.0)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    assert "are shared within" in str(ex.value), str(ex.value)


# ---------------------------------------------------------------------------
# refusals -- each leaves the raw comparison standing and names its reason
# ---------------------------------------------------------------------------

def test_no_consensus_catalog_raises_and_says_why(tmp_path, monkeypatch):
    """Without m2's catalog the re-measure cannot run.  That is not evidence the
    solution stayed put, so the stage fails closed -- and names what was
    missing, since "MOVED 2.23 mas" on its own sent two earlier investigations
    to the wrong cause (#368)."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch,
                                            write_catalog=False)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "was unavailable" in msg, msg
    assert "no m2 consensus catalog" in msg, msg


def test_too_small_a_consensus_raises_and_says_why(tmp_path, monkeypatch):
    """A re-measure on a handful of stars is not a measurement."""
    n = _ac.SURVIVOR_MIN_STARS + 10
    basepath, _m, _c, _p, _k = _sickle_case(
        tmp_path, monkeypatch, n_stars=n,
        n_dropouts=n - (_ac.SURVIVOR_MIN_STARS - 5))
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "was unavailable" in msg, msg
    assert "too few stars to re-measure" in msg, msg


def test_a_small_shared_FRACTION_raises_even_when_the_count_is_large(
        tmp_path, monkeypatch):
    """Two large catalogs sharing a small fraction of their stars clear any
    absolute floor while the intersection says nothing about either.  Here both
    consensi hold 400 stars and share 40 -- 40 is under SURVIVOR_MIN_STARS=50
    only by accident of this fixture, so the count that decides is spelled out
    in the message: the floor is 200, half of the smaller catalog."""
    coords = _m2_star_grid()
    keep = np.zeros(N_M2_STARS, dtype=bool)
    keep[-40:] = True                      # only 40 m2 stars are re-detected
    per_star = np.where(keep, SHARED_DDEC, DROPOUT_DDEC)
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    _write_m2_record(basepath, 0.0, float(np.mean(per_star)), cat)
    surv = coords[keep]
    # Pad to 400 with stars a degree away, so the stage clears the size guard
    # and the SHARED count is what refuses.
    extra_ra = coords.ra.deg[:N_M2_STARS - 40] + 1.0
    stage = SkyCoord(
        ra=np.concatenate([surv.ra.deg, extra_ra]) * u.deg,
        dec=np.concatenate([surv.dec.deg + _zero_mean_jitter(len(surv)),
                            np.full(len(extra_ra), DEC0)]) * u.deg,
        frame="icrs")
    _install_fakes(monkeypatch, coords, per_star, stage)

    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "below the floor of 200" in msg, msg
    assert "50% of the smaller catalog" in msg, msg


def test_a_re_measure_that_did_not_sign_off_cannot_waive_the_failure(
        tmp_path, monkeypatch):
    """``apply_ok=False`` means the estimator refused its own number.  Using it
    to overturn a blocking failure is the mistake bulk_offset_step0 names."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch,
                                            remeasure_apply_ok=False)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    assert "did not sign off on itself" in str(ex.value), str(ex.value)


def test_a_swept_re_measure_cannot_waive_the_failure(tmp_path, monkeypatch):
    """A swept peak is a large-offset search result, not a small-tie
    measurement, and the frozen gate is a small-tie question."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch,
                                            remeasure_swept=True)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    assert "had to SWEEP the window" in str(ex.value), str(ex.value)


def test_two_different_estimators_cannot_be_differenced(tmp_path, monkeypatch):
    """A same-star bulk and a histogram bulk differ by several mas against a
    dense reference.  Differencing one of each measures the method, not a
    shift."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch,
                                            stage_bulk_source="histogram")
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    assert "different estimators" in str(ex.value), str(ex.value)


# ---------------------------------------------------------------------------
# what the record has to show about the population it certified
# ---------------------------------------------------------------------------

def test_the_magnitude_split_of_the_shared_sample_is_recorded(
        tmp_path, monkeypatch):
    """The intersection is a biased sample and the direction is not fixed --
    sickle's F335M drop-outs are ~1.1 mag fainter than its survivors, its F187N
    drop-outs ~2.2 mag brighter.  A pass has to say which population it was
    measured on."""
    coords = _m2_star_grid()
    keep = np.ones(N_M2_STARS, dtype=bool)
    keep[:N_DROPOUTS] = False
    per_star = np.where(keep, SHARED_DDEC, DROPOUT_DDEC)
    mags = np.where(keep, 16.0, 13.8)          # drop-outs 2.2 mag brighter
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords, refmag=mags)
    _write_m2_record(basepath, 0.0, float(np.mean(per_star)), cat)
    surv = coords[keep]
    stage = SkyCoord(ra=surv.ra.deg * u.deg,
                     dec=(surv.dec.deg + _zero_mean_jitter(len(surv))) * u.deg,
                     frame="icrs")
    _install_fakes(monkeypatch, coords, per_star, stage)

    rec = _run(basepath)
    split = _sym(rec)["mag_split"]
    assert split["n_dropped"] == N_DROPOUTS
    assert split["kept_median"] == pytest.approx(16.0)
    assert split["dropped_median"] == pytest.approx(13.8)
    assert split["dropped_minus_kept"] == pytest.approx(-2.2, abs=1e-6)


def test_a_POOLED_multi_visit_m2_catalog_refuses(tmp_path, monkeypatch):
    """The m2 side is the pooled per-filter catalog, whose positions are each
    star's direction averaged over its visits; the stage side is ONE visit.
    Restricting to shared stars does not remove that, because one side stays
    averaged, and on a two-visit field the substitution alone is most of the
    2 mas budget with no real movement -- brick f115w_o004 pools to 1.678 mas
    from visit 1 and 1.911 from visit 2.  Refuse rather than difference two
    different quantities."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_path
    path = consensus_path(basepath, "F212N", obs_token="")
    tbl = Table.read(path)
    tbl.meta["NVISITS"] = 2
    tbl.write(path, overwrite=True)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    assert "pools 2 visits" in msg, msg
    assert "visit-averaged" in msg, msg


def test_a_SINGLE_visit_pooled_catalog_is_fine(tmp_path, monkeypatch):
    """The refusal must not become "no multi-exposure field may be checked":
    sickle is single-visit, which is why it is the case this branch verified."""
    basepath, _m, _c, _p, _k = _sickle_case(tmp_path, monkeypatch)
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_path
    path = consensus_path(basepath, "F212N", obs_token="")
    tbl = Table.read(path)
    tbl.meta["NVISITS"] = 1
    tbl.write(path, overwrite=True)
    rec = _run(basepath)
    assert rec["passed"], rec["failures"]
    assert _sym(rec)["m2_n_visits"] == 1


def test_the_shared_FRACTION_is_of_the_LARGER_catalog(tmp_path, monkeypatch):
    """`min(n_m2, n_stage)` leaves the asymmetric case its own comment cites:
    90,000 m2 stars, 60 stage stars, 55 shared is accepted -- a 55-star tie
    standing in for 0.06% of the m2 catalog.  The larger catalog decides."""
    coords = _m2_star_grid()
    keep = np.zeros(N_M2_STARS, dtype=bool)
    keep[-60:] = True                       # 60 of 400 re-detected
    per_star = np.where(keep, SHARED_DDEC, DROPOUT_DDEC)
    basepath = str(tmp_path)
    cat = _write_m2_consensus_catalog(basepath, coords)
    _write_m2_record(basepath, 0.0, float(np.mean(per_star)), cat)
    surv = coords[keep]
    stage = SkyCoord(ra=surv.ra.deg * u.deg,
                     dec=(surv.dec.deg + _zero_mean_jitter(len(surv))) * u.deg,
                     frame="icrs")
    _install_fakes(monkeypatch, coords, per_star, stage)
    with pytest.raises(AstrometryRegressionError) as ex:
        _run(basepath)
    msg = str(ex.value)
    # 60 clears SURVIVOR_MIN_STARS=50 and 50% of the SMALLER catalog (30);
    # it does not clear 50% of the larger (200).
    assert "below the floor of 200" in msg, msg


def test_a_frozen_stage_with_NO_basepath_still_runs(tmp_path, monkeypatch):
    """`m2_n_visits` is bound inside `if not correcting and basepath:` and read
    unconditionally by the frozen-stage branch, so a frozen stage called WITHOUT
    a basepath raised `UnboundLocalError` before it could raise anything about
    astrometry.

    That is the combination the callers in `cataloging.py` avoid but every
    direct caller of `run_visit_checkpoint` can reach -- including the CLI
    `run_astrometry_checkpoint.py` -- and the failure is an interpreter error,
    not a checkpoint verdict.
    """
    _m2_baseline_only = _write_m2_record(str(tmp_path), 0.0, 0.0, None)
    coords = _m2_star_grid()
    keep = np.ones(N_M2_STARS, dtype=bool)
    _install_fakes(monkeypatch, coords, np.full(N_M2_STARS, 40.0), coords[keep])
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")     # no basepath
