"""The m7 cross-filter bulk gate's 3-sigma clause has to leave a record.

`run_crossfilter_checkpoint` fails a filter whose bulk offset exceeds
`tol_mas` only when the offset is also larger than 3x its own combined error
bar.  That rule is defensible -- an offset the error bar covers is not
evidence of a misalignment -- but before issue #396 it wrote nothing at the
verdict level when it fired: `failures` and `unverified` were both empty,
`passed` and `all_verified` were both true, and the only trace of a filter
measured far from the anchor was the raw numbers in `filters[i].bulk`, which
a reader had to recompute the condition from.

The comment in the source claimed the neighbouring branch caught this case.
It could not: that branch is guarded on `swept or off >= 100`, and a
suppressed failure is by construction unswept with `tol < off < 100`.  Pinned
here so the claim cannot drift back.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import astrometry_checkpoint
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    run_crossfilter_checkpoint)

from .test_visit_consensus import COSD, DEC0, RA0

TOL_MAS = 5.0
OFFSET_MAS = 20.0        # 4x the tolerance -- the number in issue #396
INFLATED_ERR_MAS = 8.0   # hypot(8, 8) = 11.3 mas, so 3*err = 34 > 20


@pytest.fixture(autouse=True)
def _enforce_at_the_stage(monkeypatch):
    """Same reason as test_astrometry_checkpoint.py: the default enforcement is
    `release`, and these assertions are about what the stage measures."""
    monkeypatch.setenv("ASTROM_CHECKPOINT_ENFORCE", "stage")
    monkeypatch.delenv("ALLOW_CROSSFILTER_ASTROM_FAIL", raising=False)


def _catalogs(offset_mas, n=6000, extent=60.0, seed=5):
    """Two filters, the second shifted rigidly in RA by `offset_mas`."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, extent, n)
    y = rng.uniform(0, extent, n)
    ra = RA0 + x / 3600.0 / COSD
    dec = DEC0 + y / 3600.0

    def _tbl(ra_deg, dec_deg, noise_mas=0.5):
        t = Table()
        nn = len(ra_deg)
        t["skycoord"] = SkyCoord(
            ra=(ra_deg + rng.normal(0, noise_mas, nn) / 3.6e6 / COSD) * u.deg,
            dec=(dec_deg + rng.normal(0, noise_mas, nn) / 3.6e6) * u.deg,
            frame="icrs")
        t["flux_fit"] = rng.uniform(1e3, 1e5, nn)
        t["flux_err"] = t["flux_fit"] / 100.0
        t["qfit"] = rng.uniform(0.01, 0.05, nn)
        return t

    return {"F212N": _tbl(ra, dec),
            "F405N": _tbl(ra + offset_mas / 3.6e6 / COSD, dec)}


def _inflate_error_bars(monkeypatch, err_mas):
    """Widen the reported error bars without touching the measured offset.

    The real estimator on 6000 stars reports ~0.01 mas, which is why the
    3-sigma clause has never fired on a live record (largest combined error bar
    across 141 m7 filter entries: 0.196 mas).  Reaching the branch at all needs
    an error bar the data cannot produce, so it is injected -- the offset, the
    sweep flag and the coordinates all stay real, and the local residual map
    below still runs on the true bulk.
    """
    real = astrometry_checkpoint.measure_offset

    def _wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        if isinstance(out, dict):
            out = dict(out, dra_err=err_mas, ddec_err=err_mas)
        return out

    monkeypatch.setattr(astrometry_checkpoint, "measure_offset", _wrapped)


@pytest.fixture(scope="module")
def suppressed_record(tmp_path_factory):
    """The 20 mas / wide-error-bar case, measured once.

    `run_crossfilter_checkpoint` on 6000 stars costs ~100 s, so the two tests
    that ask different questions of the same record share it.
    """
    mp = pytest.MonkeyPatch()
    mp.setenv("ASTROM_CHECKPOINT_ENFORCE", "stage")
    mp.delenv("ALLOW_CROSSFILTER_ASTROM_FAIL", raising=False)
    _inflate_error_bars(mp, INFLATED_ERR_MAS)
    try:
        record = run_crossfilter_checkpoint(
            _catalogs(OFFSET_MAS),
            record_dir=str(tmp_path_factory.mktemp("suppressed")),
            cell_min_stars=15, tol_mas=TOL_MAS, context="test")
    finally:
        # UNDO BEFORE HANDING THE RECORD OUT.  A module-scoped fixture that
        # yields inside the patch keeps it live for every test in the file,
        # including the ones that must run against the REAL error bars.
        mp.undo()
    return record


def test_suppressed_bulk_failure_is_recorded_as_unverified(suppressed_record):
    record = suppressed_record

    # The 3-sigma clause did its job: 20 mas against a 34 mas 3-sigma is not
    # significant, so it is not a failure and the stage does not stop.
    assert record["passed"], record["failures"]
    assert not record["failures"], record["failures"]

    # ... and the record now says what was measured and why it was let through.
    suppressed = [w for w in record["unverified"] if "NOT significant" in w]
    assert suppressed, (
        "an offset 4x the tolerance passed and the verdict records nothing: "
        f"unverified={record['unverified']}")
    assert len(suppressed) == 1, suppressed
    msg = suppressed[0]
    assert f"{OFFSET_MAS:.2f} mas" in msg, msg      # the measured offset
    assert "3-sigma" in msg, msg                    # and the bar it cleared
    assert record["all_verified"] is False


def test_the_swept_branch_cannot_reach_a_suppressed_failure(suppressed_record):
    """The reason the record was silent: the branch whose comment claimed to
    cover this case is guarded on `swept or off >= 100`."""
    record = suppressed_record
    frec = [f for f in record["filters"]
            if f["filtername"] != record["anchor_filter"]][0]
    assert frec["bulk"]["swept"] is False
    assert TOL_MAS < frec["bulk"]["off"] < 100.0
    # so nothing from that branch is in the list -- the entry above is the
    # only thing keeping the verdict from being empty
    assert not [w for w in record["unverified"]
                if "skipped the local cell map" in w], record["unverified"]


def test_a_significant_over_tolerance_offset_still_fails(tmp_path, monkeypatch):
    """The new branch is an `elif`, so it must not intercept a real failure."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        CrossFilterAstrometryError)
    with pytest.raises(CrossFilterAstrometryError) as exc:
        run_crossfilter_checkpoint(
            _catalogs(OFFSET_MAS), record_dir=str(tmp_path), cell_min_stars=15,
            tol_mas=TOL_MAS, context="test")
    assert "bulk offset" in str(exc.value)


def test_an_in_tolerance_offset_records_nothing(tmp_path, monkeypatch):
    """A wide error bar on an offset BELOW the tolerance is an ordinary pass and
    must not be reported as unverified."""
    _inflate_error_bars(monkeypatch, INFLATED_ERR_MAS)
    record = run_crossfilter_checkpoint(
        _catalogs(0.0), record_dir=str(tmp_path), cell_min_stars=15,
        tol_mas=TOL_MAS, context="test")
    assert record["passed"], record["failures"]
    assert not [w for w in record["unverified"] if "NOT significant" in w], \
        record["unverified"]
