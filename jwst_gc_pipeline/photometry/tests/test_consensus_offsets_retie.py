"""Consensus offsets re-tie (Fix B): seed a sparse per-exposure table from the m2
checkpoint corrections and look up a single exposure's shift.

The table is SPARSE -- only off-tolerance exposures get a row.  The critical
regression is the SINGLE-ROW case: a visit/filter whose only off-consensus
exposure has one row must NOT leak that shift onto the visit's OTHER
(within-tolerance) exposures -- those must return (0, 0).  (An earlier version
narrowed by Exposure/Module only when ``match.sum() > 1``, so a lone row was
applied to every exposure of the visit -- the per-exposure-corruption class the
astrometry checkpoint exists to prevent.)
"""
import numpy as np
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    lookup_consensus_offset, seed_offsets_table_from_consensus)

DEC = -29.44


def _corr(visit, exposure, module, dra_mas, ddec_mas, filt="F115W"):
    return dict(visit=visit, exposure=exposure, module=module, filtername=filt,
                dra_onsky_mas=dra_mas, ddec_onsky_mas=ddec_mas, dec_deg=DEC,
                source="m2 visit-consensus")


def test_seed_round_trip_convention(tmp_path):
    """dra (arcsec) = (dra_onsky_mas/1000)/cos(dec); ddec 1:1; visit token."""
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "012",
        [_corr("1", 2, "nrcb2", 7.78, -2.0)], stage="m2")
    t = Table.read(p)
    assert t["Visit"][0] == "jw04147012001"
    assert t["Exposure"][0] == 2 and t["Module"][0] == "nrcb2"
    exp_dra = (7.78 / 1000.0) / np.cos(np.radians(DEC))
    assert abs(float(t["dra (arcsec)"][0]) - exp_dra) < 1e-9
    assert abs(float(t["ddec (arcsec)"][0]) + 0.002) < 1e-12


def test_single_row_does_not_leak_to_other_exposures():
    """THE regression: one off-tolerance exposure (exp 2) in visit/filter; the
    within-tolerance exposures (1, 3, 4) have no row and must get (0, 0)."""
    t = Table([dict(Visit="jw04147012001", Filter="F115W", Module="nrcb2",
                    Exposure=2, **{"dra (arcsec)": 0.00894, "ddec (arcsec)": -0.002})])
    # the listed exposure gets its shift
    dra, ddec = lookup_consensus_offset(t, "jw04147012001", 2, "nrcb2", "F115W")
    assert abs(dra - 0.00894) < 1e-9 and abs(ddec + 0.002) < 1e-12
    # every OTHER exposure of the same visit/filter -> (0, 0), NOT exp 2's shift
    for other in (1, 3, 4):
        assert lookup_consensus_offset(
            t, "jw04147012001", other, "nrcb2", "F115W") == (0.0, 0.0)
    # a different module of the listed exposure also gets nothing
    assert lookup_consensus_offset(
        t, "jw04147012001", 2, "nrcb1", "F115W") == (0.0, 0.0)


def test_lw_module_family_matches():
    """LW frames carry 'nrcblong'; the seed stores the family 'nrcb'."""
    t = Table([dict(Visit="jw04147012001", Filter="F405N", Module="nrcb",
                    Exposure=1, **{"dra (arcsec)": 0.005, "ddec (arcsec)": 0.001})])
    dra, ddec = lookup_consensus_offset(t, "jw04147012001", 1, "nrcblong", "F405N")
    assert abs(dra - 0.005) < 1e-12 and abs(ddec - 0.001) < 1e-12


def test_duplicate_rows_raise():
    t = Table([dict(Visit="jw04147012001", Filter="F115W", Module="nrcb2",
                    Exposure=2, **{"dra (arcsec)": 0.001, "ddec (arcsec)": 0.0}),
               dict(Visit="jw04147012001", Filter="F115W", Module="nrcb2",
                    Exposure=2, **{"dra (arcsec)": 0.002, "ddec (arcsec)": 0.0})])
    try:
        lookup_consensus_offset(t, "jw04147012001", 2, "nrcb2", "F115W")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on duplicate rows")


def test_seed_then_lookup_sparse(tmp_path):
    """End-to-end: seed a table with two off-tolerance exposures across two
    visits, then confirm lookups hit only the listed frames."""
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "012",
        [_corr("1", 2, "nrcb2", 7.78, -2.0),
         _corr("2", 5, "nrcb3", -3.1, 1.4)], stage="m2")
    t = Table.read(p)
    assert lookup_consensus_offset(t, "jw04147012001", 2, "nrcb2", "F115W")[0] != 0.0
    assert lookup_consensus_offset(t, "jw04147012002", 5, "nrcb3", "F115W")[1] != 0.0
    # cross terms are all zero (listed exposures live in different visits)
    assert lookup_consensus_offset(t, "jw04147012001", 5, "nrcb3", "F115W") == (0.0, 0.0)
    assert lookup_consensus_offset(t, "jw04147012002", 2, "nrcb2", "F115W") == (0.0, 0.0)


def test_upsert_inserts_newly_flagged_exposure(tmp_path):
    """Regression for the re-tie loop hang: iter 2 flags an exposure that was
    within tolerance on iter 1 (no row).  seed_offsets_table_from_consensus must
    INSERT it, not hard-fail like update_offsets_table ('matches NO row')."""
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "012", [_corr("1", 2, "nrcb2", 7.78, -2.0)])
    # iter 2: a DIFFERENT exposure (3/nrca1) crosses the 2 mas line
    p2 = seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "012", [_corr("1", 3, "nrca1", 2.22, -0.77)])
    assert p2 == p
    t = Table.read(p)
    keys = {(r["Visit"], r["Module"], int(r["Exposure"])) for r in t}
    assert ("jw04147012001", "nrcb2", 2) in keys  # original survives
    assert ("jw04147012001", "nrca1", 3) in keys  # newly inserted
    dra, _ = lookup_consensus_offset(t, "jw04147012001", 3, "nrca1", "F115W")
    assert abs(dra - (2.22 / 1000.0) / np.cos(np.radians(DEC))) < 1e-9


def test_upsert_accumulates_residual_on_existing_row(tmp_path):
    """A correction for an exposure already in the table ADDS (the correction is
    the RESIDUAL after the previous tie), it does not replace."""
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "012", [_corr("1", 2, "nrcb2", 8.0, -2.0)])
    seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "012", [_corr("1", 2, "nrcb2", 1.5, 0.5)])
    t = Table.read(p)
    row = t[(t["Visit"] == "jw04147012001") & (t["Exposure"] == 2)
            & (t["Module"] == "nrcb2")]
    assert len(row) == 1  # accumulated in place, not duplicated
    cosd = np.cos(np.radians(DEC))
    assert abs(float(row["dra (arcsec)"][0]) - (8.0 + 1.5) / 1000.0 / cosd) < 1e-9
    assert abs(float(row["ddec (arcsec)"][0]) - (-2.0 + 0.5) / 1000.0) < 1e-9
    assert abs(float(row["prov_dra_added_mas"][0]) - 9.5) < 1e-9


def _bulk(visit, dra_mas, ddec_mas, filt="F115W"):
    """A consensus->reference bulk correction (per-visit; exposure/module None)."""
    return dict(visit=visit, exposure=None, module=None, filtername=filt,
                dra_onsky_mas=dra_mas, ddec_onsky_mas=ddec_mas, dec_deg=DEC,
                source="m2 consensus->reference")


def test_bulk_applies_to_every_exposure(tmp_path):
    """The consensus->reference bulk (exposure/module None) must reach EVERY
    frame of the visit/filter -- not a dead exposure=0 row.  This was the sgrc
    canary's persistent 37 mas that never converged."""
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "012",
        [_bulk("1", -35.2, -12.5), _corr("1", 2, "nrcb2", 7.78, -2.0)])
    t = Table.read(p)
    cosd = np.cos(np.radians(DEC))
    bulk_dra = (-35.2 / 1000.0) / cosd
    bulk_ddec = -12.5 / 1000.0
    # a frame WITH a jitter row -> jitter + bulk
    dra, ddec = lookup_consensus_offset(t, "jw04147012001", 2, "nrcb2", "F115W")
    assert abs(dra - ((7.78 / 1000.0) / cosd + bulk_dra)) < 1e-9
    assert abs(ddec - (-0.002 + bulk_ddec)) < 1e-9
    # a frame with NO jitter row -> bulk only (not zero!)
    dra2, ddec2 = lookup_consensus_offset(t, "jw04147012001", 4, "nrcb3", "F115W")
    assert abs(dra2 - bulk_dra) < 1e-9 and abs(ddec2 - bulk_ddec) < 1e-9
    # a different visit -> nothing
    assert lookup_consensus_offset(t, "jw04147012002", 4, "nrcb3", "F115W") == (0.0, 0.0)


def test_bulk_accumulates_on_sentinel_row(tmp_path):
    """A second bulk correction ADDS to the per-visit sentinel row (residual
    after the prior reference tie), staying a single row."""
    p = seed_offsets_table_from_consensus(str(tmp_path), "4147", "012",
                                          [_bulk("1", -35.0, -12.0)])
    seed_offsets_table_from_consensus(str(tmp_path), "4147", "012",
                                      [_bulk("1", -3.0, 1.0)])
    t = Table.read(p)
    sentinel = t[(t["Exposure"] == -1) & (t["Module"] == "all")]
    assert len(sentinel) == 1
    cosd = np.cos(np.radians(DEC))
    assert abs(float(sentinel["dra (arcsec)"][0]) - (-38.0 / 1000.0) / cosd) < 1e-9
    assert abs(float(sentinel["ddec (arcsec)"][0]) - (-11.0 / 1000.0)) < 1e-9
