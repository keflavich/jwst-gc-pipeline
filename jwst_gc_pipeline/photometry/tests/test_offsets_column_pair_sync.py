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


def _table(tmp_path, both_pairs=True, diverge=0.0):
    t = Table()
    t["Visit"] = ["jw02092005001"] * 4
    t["Exposure"] = [1, 2, 3, 4]
    t["Filter"] = ["F360M"] * 4
    t["Module"] = ["nrcblong"] * 4
    t["dra"] = np.array([0.10, 0.11, 0.12, 0.13])
    t["ddec"] = np.array([-0.20, -0.21, -0.22, -0.23])
    if both_pairs:
        t["dra (arcsec)"] = t["dra"] + diverge
        t["ddec (arcsec)"] = t["ddec"]
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


def test_an_already_diverged_table_is_refused(tmp_path):
    """Adding the same increment to both pairs PRESERVES a pre-existing gap.

    A table that arrives diverged would keep its divergence forever with a fresh
    provenance date on top -- 'looks maintained, isn't'.  The diverged tables on
    disk (cloudc, cloudef) have to be reconciled deliberately, and the message
    has to say which pair the pixels agree with.
    """
    p = _table(tmp_path, diverge=0.05)          # 50 mas apart
    with pytest.raises(OffsetsTableUpdateError) as exc:
        update_offsets_table(p, [_corr(2)], stage="m2")
    msg = str(exc.value)
    assert "disagree" in msg
    assert "unified_alignment reads" in msg
    assert "NOT writing" in msg


def test_the_refusal_leaves_the_table_alone(tmp_path):
    """A guard that half-writes is worse than no guard."""
    p = _table(tmp_path, diverge=0.05)
    before = open(p).read()
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(p, [_corr(2)], stage="m2")
    assert open(p).read() == before


def test_round_trip_noise_is_not_called_divergence(tmp_path):
    """CSV float round-trip must not trip the guard; the tolerance is 0.1 mas
    against a tree whose tightest real gate is 2 mas."""
    p = _table(tmp_path, diverge=1e-9)
    update_offsets_table(p, [_corr(2)], stage="m2")   # must not raise
