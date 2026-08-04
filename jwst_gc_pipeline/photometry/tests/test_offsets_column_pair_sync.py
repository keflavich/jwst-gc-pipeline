"""An offsets table's two dra/ddec column pairs are one quantity, not two.

``build_virac2_offsets`` ends with ``t['dra (arcsec)'] = t['dra']`` -- a copy --
so a builder-shaped table carries both conventions with identical values.
``unified_alignment`` reads the ``(arcsec)`` pair; ``update_offsets_table`` used
to write only that one. Every correction therefore froze the plain pair one step
further behind, silently and without limit:

    field     rows   rows where dra/ddec != dra/ddec (arcsec)   worst
    cloudef    128                    96                        7329.1 mas
    cloudc     192                    95                        7876.8 mas
    sickle     120                     24                         95.6 mas
    sgrc        96                      7                          6.1 mas

The reductions were right -- fix_alignment reads the maintained pair -- but the
plain pair is ``validate_offsets_table``'s fallback and the first thing a person
reads off the table.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    OffsetsTableUpdateError, update_offsets_table)


def _table(tmp_path, both_pairs=True, diverge=0.0, prov=None):
    t = Table()
    t["Visit"] = ["jw02092005001"] * 4
    t["Exposure"] = [1, 2, 3, 4]
    t["Filter"] = ["F360M"] * 4
    t["Module"] = ["nrcblong"] * 4
    t["dra"] = np.array([0.10, 0.11, 0.12, 0.13])
    t["ddec"] = np.array([-0.20, -0.21, -0.22, -0.23])
    if both_pairs:
        t["dra (arcsec)"] = t["dra"] + diverge
        t["ddec (arcsec)"] = t["ddec"] + diverge
    if prov is not None:
        # the divergence as the provenance records it: on-sky mas on the Dec axis
        t["prov_stage"] = ["m2"] * 4
        t["prov_date"] = ["2026-08-03T00:00:00Z"] * 4
        t["prov_source"] = ["test"] * 4
        t["prov_dra_added_mas"] = np.full(4, prov)
        t["prov_ddec_added_mas"] = np.full(4, prov)
    p = tmp_path / "Offsets_JWST_Brick2092_VIRAC2locked.csv"
    t.write(p, format="ascii.csv", overwrite=True)
    return str(p)


def _corr(exposure, dra=10.0, ddec=-20.0):
    return dict(visit="jw02092005001", exposure=exposure, module="nrcblong",
                filtername="F360M", dra_onsky_mas=dra, ddec_onsky_mas=ddec,
                dec_deg=-28.8, source="test")


def test_both_column_pairs_are_corrected(tmp_path):
    """The whole point: a correction must land in every pair the table carries."""
    p = _table(tmp_path)
    before = Table.read(p, format="ascii.csv")
    update_offsets_table(p, [_corr(2)], stage="m2")
    after = Table.read(p, format="ascii.csv")

    row = 1                                     # Exposure == 2
    cosd = np.cos(np.radians(-28.8))
    d_ra, d_dec = (10.0 / 1000.0) / cosd, -20.0 / 1000.0
    for dc, cc in (("dra", "ddec"), ("dra (arcsec)", "ddec (arcsec)")):
        assert after[dc][row] == pytest.approx(before[dc][row] + d_ra, abs=1e-9), dc
        assert after[cc][row] == pytest.approx(before[cc][row] + d_dec, abs=1e-9), cc


def test_the_pairs_stay_equal_after_a_correction(tmp_path):
    """They are two names for one number; correcting must not separate them."""
    p = _table(tmp_path)
    update_offsets_table(p, [_corr(2)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    assert np.allclose(np.asarray(t["dra"], dtype=float),
                       np.asarray(t["dra (arcsec)"], dtype=float), atol=1e-12)
    assert np.allclose(np.asarray(t["ddec"], dtype=float),
                       np.asarray(t["ddec (arcsec)"], dtype=float), atol=1e-12)


def test_uncorrected_rows_are_untouched(tmp_path):
    p = _table(tmp_path)
    before = Table.read(p, format="ascii.csv")
    update_offsets_table(p, [_corr(2)], stage="m2")
    after = Table.read(p, format="ascii.csv")
    for row in (0, 2, 3):
        for c in ("dra", "ddec", "dra (arcsec)", "ddec (arcsec)"):
            assert after[c][row] == pytest.approx(before[c][row], abs=1e-12)


def test_a_single_pair_table_still_works(tmp_path):
    """Consensus tables carry only dra/ddec; they must not regress."""
    p = _table(tmp_path, both_pairs=False)
    before = Table.read(p, format="ascii.csv")
    update_offsets_table(p, [_corr(3)], stage="m2")
    after = Table.read(p, format="ascii.csv")
    assert "dra (arcsec)" not in after.colnames
    assert after["ddec"][2] == pytest.approx(before["ddec"][2] - 0.020, abs=1e-9)


def test_an_explained_divergence_is_HEALED_not_refused(tmp_path, capsys):
    """The gap equals the accumulated provenance on every table on disk
    (0.000000 mas across all ten), so the plain pair is provably the as-built
    value and carries nothing the (arcsec) pair lacks.  Re-sync it by proof.

    Refusing instead would stop the m2 checkpoint on EVERY locked-channel field
    -- all ten live tables diverge, and no caller catches the exception."""
    p = _table(tmp_path, diverge=0.050, prov=50.0)     # 50 mas, and prov says 50
    update_offsets_table(p, [_corr(2)], stage="m2")
    t = Table.read(p, format="ascii.csv")
    # the TOUCHED row (Exposure == 2) is healed and then corrected; the others are
    # out of scope by design -- see test_healing_is_scoped_to_the_rows...
    assert float(t["ddec"][1]) == pytest.approx(float(t["ddec (arcsec)"][1]),
                                                abs=1e-12)
    assert "re-syncing" in capsys.readouterr().out


def test_an_UNEXPLAINED_divergence_is_still_refused(tmp_path):
    """A gap the provenance does not account for means something edited one pair
    outside this function, so which one is right is not on record."""
    p = _table(tmp_path, diverge=0.050, prov=5.0)      # 50 mas gap, prov says 5
    with pytest.raises(OffsetsTableUpdateError) as exc:
        update_offsets_table(p, [_corr(2)], stage="m2")
    msg = str(exc.value)
    assert "more than the recorded provenance explains" in msg
    assert "NOT writing" in msg


def test_the_refusal_leaves_the_table_alone(tmp_path):
    """A guard that half-writes is worse than no guard."""
    p = _table(tmp_path, diverge=0.050, prov=5.0)
    before = open(p).read()
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(p, [_corr(2)], stage="m2")
    assert open(p).read() == before


def test_healing_is_scoped_to_the_rows_a_correction_touches(tmp_path):
    """sgrb2 has ONE stale filter and ten clean ones; sgrc blocks eight filters
    over seven rows.  A table-wide sweep blocks them together and breaks the
    recovery path, since rebuilding one filter re-equalises only its own rows."""
    p = _table(tmp_path, diverge=0.050, prov=50.0)
    before = Table.read(p, format="ascii.csv")
    update_offsets_table(p, [_corr(2)], stage="m2")   # touches Exposure == 2 only
    after = Table.read(p, format="ascii.csv")
    # the untouched rows keep their divergence rather than being swept
    for row in (0, 2, 3):
        assert after["ddec"][row] == pytest.approx(before["ddec"][row], abs=1e-12)
        assert abs(float(after["ddec"][row])
                   - float(after["ddec (arcsec)"][row])) > 1e-9


def test_an_integer_offset_column_cannot_lock_the_table(tmp_path):
    """With dra int64 and dra (arcsec) float, one write gives 0 vs 0.0114 and
    every later write is refused forever.  Coerce before comparing or applying."""
    t = Table()
    t["Visit"] = ["jw02092005001"] * 4
    t["Exposure"] = [1, 2, 3, 4]
    t["Filter"] = ["F360M"] * 4
    t["Module"] = ["nrcblong"] * 4
    t["dra"] = np.zeros(4, dtype=np.int64)
    t["ddec"] = np.zeros(4, dtype=np.int64)
    t["dra (arcsec)"] = np.zeros(4, dtype=float)
    t["ddec (arcsec)"] = np.zeros(4, dtype=float)
    p = str(tmp_path / "Offsets_JWST_Brick2092_VIRAC2locked.csv")
    t.write(p, format="ascii.csv", overwrite=True)

    update_offsets_table(p, [_corr(2)], stage="m2")
    after = Table.read(p, format="ascii.csv")
    assert after["ddec"][1] == pytest.approx(-0.020, abs=1e-9)
    assert np.allclose(np.asarray(after["ddec"], dtype=float),
                       np.asarray(after["ddec (arcsec)"], dtype=float), atol=1e-12)


def test_a_table_without_Visit_raises_the_right_exception(tmp_path):
    """Three real tables carry both pairs and no Visit column; a KeyError is the
    wrong class to escape from a guarded writer."""
    t = Table()
    t["Exposure"] = [1, 2]
    t["Filter"] = ["F360M"] * 2
    t["dra"] = np.zeros(2)
    t["ddec"] = np.zeros(2)
    t["dra (arcsec)"] = np.zeros(2)
    t["ddec (arcsec)"] = np.zeros(2)
    p = str(tmp_path / "Offsets_JWST_Brick2221_average.csv")
    t.write(p, format="ascii.csv", overwrite=True)
    with pytest.raises(OffsetsTableUpdateError, match="no Visit column"):
        update_offsets_table(p, [_corr(2)], stage="m2")


def test_round_trip_noise_is_not_called_divergence(tmp_path):
    """CSV float round-trip must not trip the guard; the tolerance is 0.1 mas
    against a tree whose tightest real gate is 2 mas."""
    p = _table(tmp_path, diverge=1e-9)
    update_offsets_table(p, [_corr(2)], stage="m2")   # must not raise


def _table_ra_only(tmp_path, ra_gap, prov_dra):
    """A row whose DEC pairs agree but whose RA pairs do not."""
    t = Table()
    t["Visit"] = ["jw02092005001"] * 4
    t["Exposure"] = [1, 2, 3, 4]
    t["Filter"] = ["F360M"] * 4
    t["Module"] = ["nrcblong"] * 4
    t["dra"] = np.array([0.10, 0.11, 0.12, 0.13])
    t["ddec"] = np.array([-0.20, -0.21, -0.22, -0.23])
    t["dra (arcsec)"] = t["dra"] + ra_gap
    t["ddec (arcsec)"] = t["ddec"]                      # Dec agrees exactly
    t["prov_stage"] = ["m2"] * 4
    t["prov_date"] = ["2026-08-03T00:00:00Z"] * 4
    t["prov_source"] = ["test"] * 4
    t["prov_dra_added_mas"] = np.full(4, prov_dra)
    t["prov_ddec_added_mas"] = np.zeros(4)
    p = tmp_path / "Offsets_JWST_Brick2092_VIRAC2locked.csv"
    t.write(p, format="ascii.csv", overwrite=True)
    return str(p)


def test_an_RA_only_gap_the_provenance_does_not_explain_is_refused(tmp_path):
    """The heal writes BOTH columns, so both must be proved first.

    A row whose Dec gap is explained (here: zero) and whose RA gap is not would
    otherwise have its `dra` silently overwritten -- the case the Dec-only
    precondition let through.
    """
    p = _table_ra_only(tmp_path, ra_gap=0.050, prov_dra=0.0)   # 50 mas, prov says 0
    with pytest.raises(OffsetsTableUpdateError) as exc:
        update_offsets_table(p, [_corr(2)], stage="m2")
    msg = str(exc.value)
    assert "RA axis" in msg
    assert "prov_dra_added_mas" in msg
    assert "NOT writing" in msg


def test_an_RA_gap_inside_the_cosdec_bound_is_healed(tmp_path):
    """The bound is a window, not an equality: the apply loop divided on-sky mas
    by a cos(dec) this function cannot recover, so any coordinate gap between
    prov/1000 and prov/1000/COS_DEC_MIN is explained."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import COS_DEC_MIN
    prov = 50.0
    mid = 0.5 * (prov / 1000.0 + prov / 1000.0 / COS_DEC_MIN)
    p = _table_ra_only(tmp_path, ra_gap=mid, prov_dra=prov)
    update_offsets_table(p, [_corr(2)], stage="m2")            # must not raise
    t = Table.read(p, format="ascii.csv")
    assert float(t["dra"][1]) == pytest.approx(float(t["dra (arcsec)"][1]),
                                               abs=1e-12)
