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
